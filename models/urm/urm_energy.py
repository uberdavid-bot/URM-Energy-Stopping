from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, replace
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
from models.common import trunc_normal_init_
from models.layers import rms_norm, ConvSwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding


@dataclass
class URMCarry:
    current_hidden: torch.Tensor
    steps: Optional[torch.Tensor] = None
    halted: Optional[torch.Tensor] = None
    current_data: Optional[Dict[str, torch.Tensor]] = None
    prev_energy: Optional[torch.Tensor] = None


class URMConfig(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int
    num_layers: int
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str
    attn_dropout: float = 0.0
    mlp_dropout: float = 0.0
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    loops: int
    L_cycles: int = 1
    H_cycles: int = 1
    forward_dtype: str = "bfloat16"
    # Energy-specific
    energy_threshold: float = 0.005
    min_steps: int = 8
    dsm_noise_scales: List[float] = [0.1, 0.3, 0.5, 1.0]
    dsm_weight: float = 0.0
    contrastive_weight: float = 1.0
    contrastive_margin: float = 0.5
    refine_steps: int = 0
    refine_step_size: float = 0.01
    # MCMC training (second-order gradients through MCMC into energy head)
    mcmc_steps: int = 0
    mcmc_step_size: float = 0.01
    mcmc_training: bool = False
    # Dual reconstruction loss weights (only used when MCMC active)
    unrefined_loss_weight: float = 0.5
    refined_loss_weight: float = 0.5
    # Trajectory-supervised energy
    trajectory_loss_weight: float = 0.0
    trajectory_margin: float = 0.1
    trajectory_max_steps: int = 4

class URMBlock(nn.Module):
    def __init__(self, config: URMConfig) -> None:
        super().__init__()
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
        )
        self.mlp = ConvSwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        seq_len = hidden_states.shape[1]
        cos_sin = tuple(c[:seq_len] for c in cos_sin)
        attn_output = self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states)
        hidden_states = rms_norm(hidden_states + attn_output, variance_epsilon=self.norm_eps)
        mlp_output = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + mlp_output, variance_epsilon=self.norm_eps)
        return hidden_states


class URM_Inner(nn.Module):
    def __init__(self, config: URMConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)
        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)

        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )

        self.rotary_emb = RotaryEmbedding(
            dim=self.config.hidden_size // self.config.num_heads,
            # compute_joint_energy concatenates input_emb [seq_len + puzzle_emb_len] + predicted_emb [seq_len]
            max_position_embeddings=2 * self.config.seq_len + self.puzzle_emb_len,
            base=self.config.rope_theta,
        )

        self.layers = nn.ModuleList([URMBlock(self.config) for _ in range(self.config.num_layers)])

        self.init_hidden = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        embedding = self.embed_tokens(input.to(torch.int32))

        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding),
                dim=-2,
            )
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int) -> URMCarry:
        return URMCarry(
            current_hidden=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: URMCarry) -> URMCarry:
        new_hidden = torch.where(
            reset_flag.view(-1, 1, 1),
            self.init_hidden,
            carry.current_hidden
        )
        return replace(carry, current_hidden=new_hidden)

    def forward(
        self,
        carry: URMCarry,
        batch: Dict[str, torch.Tensor],
        capture_trajectory: bool = False,
    ) -> Tuple[URMCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(cos_sin=self.rotary_emb())
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        hidden_states = carry.current_hidden
        trajectory: List[torch.Tensor] = []

        for _ in range(self.config.loops):
            hidden_states = hidden_states + input_embeddings
            for layer in self.layers:
                hidden_states = layer(hidden_states=hidden_states, **seq_info)
            if capture_trajectory:
                trajectory.append(hidden_states.detach())

        new_carry = replace(carry, current_hidden=hidden_states.detach())
        output = self.lm_head(hidden_states)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(hidden_states[:, 0]).to(torch.float32)
        if capture_trajectory:
            return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), trajectory
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class URM_Energy(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = URMConfig(**config_dict)
        self.inner = URM_Inner(self.config)
        self.energy_head = nn.Linear(self.config.hidden_size, 1, dtype=self.inner.forward_dtype)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def compute_joint_energy(self, input_embeddings: torch.Tensor, output_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute joint energy E(input, output).
        Concatenates input and output embeddings, passes through transformer layers,
        pools to scalar energy per example. Lower energy = better prediction.
        """
        input_embeddings = input_embeddings.to(self.inner.forward_dtype)
        output_embeddings = output_embeddings.to(self.inner.forward_dtype)

        all_embeddings = torch.cat((input_embeddings, output_embeddings), dim=1)

        cos_sin = self.inner.rotary_emb()

        hidden_states = all_embeddings
        for layer in self.inner.layers:
            hidden_states = layer(cos_sin=cos_sin, hidden_states=hidden_states)

        h_pooled = hidden_states.mean(dim=1)
        energy = self.energy_head(h_pooled)
        return energy.squeeze(-1)

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> URMCarry:
        batch_size = batch["inputs"].shape[0]
        base = self.inner.empty_carry(batch_size)
        return URMCarry(
            current_hidden=base.current_hidden,
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=batch["inputs"].device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=batch["inputs"].device),
            current_data={k: v.clone().detach() for k, v in batch.items()},
        )

    def forward(
        self,
        carry: URMCarry,
        batch: Dict[str, torch.Tensor],
        compute_target_q=False
    ) -> Tuple[URMCarry, Dict[str, torch.Tensor]]:
        # 1. Standard carry management
        new_carry = self.inner.reset_carry(carry.halted, carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v
            )
            for k, v in carry.current_data.items()
        }

        # 2. Run URM inner recurrence (the actual thinking)
        capture = self.training and self.config.trajectory_loss_weight > 0
        inner_result = self.inner(new_carry, new_current_data, capture_trajectory=capture)
        if capture:
            new_carry_inner, logits, (q_halt_logits, q_continue_logits), trajectory = inner_result
        else:
            new_carry_inner, logits, (q_halt_logits, q_continue_logits) = inner_result

        # 3. Compute energy on current output for stopping criterion
        input_embeddings = self.inner._input_embeddings(
            new_current_data["inputs"], new_current_data["puzzle_identifiers"]
        )
        output_hidden = new_carry_inner.current_hidden[:, self.inner.puzzle_emb_len:]
        current_energy = self.compute_joint_energy(input_embeddings, output_hidden)

        # 3b. MCMC refinement with second-order gradients (training only)
        # Reconstruction loss on refined logits flows back through MCMC into energy head,
        # teaching it to produce gradients that improve predictions.
        unrefined_logits = None
        if self.config.mcmc_steps > 0 and self.config.mcmc_training and self.training:
            unrefined_logits = logits
            # Convert logits to soft embeddings
            probs = F.softmax(logits.float(), dim=-1)
            predicted_emb = (probs @ self.inner.embed_tokens.embedding_weight.data.float()).to(
                self.inner.forward_dtype
            )
            # MCMC loop with create_graph=True for second-order gradients
            for _ in range(self.config.mcmc_steps):
                predicted_emb = predicted_emb.detach().requires_grad_(True)
                energy = self.compute_joint_energy(input_embeddings, predicted_emb)
                grad = torch.autograd.grad(
                    energy.sum(), predicted_emb, create_graph=True
                )[0]
                grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                grad = grad / grad_norm
                predicted_emb = predicted_emb - self.config.mcmc_step_size * grad
            # Convert refined embeddings back to logits
            logits = self.inner.lm_head(predicted_emb.to(self.inner.forward_dtype))
        elif self.config.mcmc_steps > 0 and not self.training:
            # Inference: use refine_with_mcmc without create_graph
            unrefined_logits = logits
            logits = self.refine_with_mcmc(
                logits, input_embeddings,
                steps=self.config.mcmc_steps,
                step_size=self.config.mcmc_step_size,
            )

        # 4. Halting logic
        with torch.no_grad():
            new_steps = new_steps + 1
            halted = (new_steps >= self.config.loops)

            # Energy-based early stopping only during training (experimental)
            # During eval, use fixed iterations for fair comparison with baseline
            if self.training and carry.prev_energy is not None:
                energy_change = torch.abs(current_energy - carry.prev_energy)
                converged = energy_change < self.config.energy_threshold
                halted = halted | (converged & (new_steps >= self.config.min_steps))

        # 5. Outputs
        # Energy model uses energy convergence for stopping, not Q-halt.
        # Zero Q values provided for evaluator compatibility.
        outputs = {
            "logits": logits,
            "current_energy": current_energy,
            "output_hidden": output_hidden,
            "q_halt_logits": torch.zeros_like(current_energy),
            "q_continue_logits": torch.zeros_like(current_energy),
        }
        if unrefined_logits is not None:
            outputs["unrefined_logits"] = unrefined_logits
        if capture:
            outputs["trajectory"] = trajectory
            outputs["input_embeddings"] = input_embeddings

        return (
            URMCarry(
                current_hidden=new_carry_inner.current_hidden,
                steps=new_steps,
                halted=halted,
                current_data=new_current_data,
                prev_energy=current_energy.detach(),
            ),
            outputs,
        )

    def refine_with_mcmc(self, logits: torch.Tensor, input_embeddings: torch.Tensor,
                         steps: int = 8, step_size: float = 0.01) -> torch.Tensor:
        """
        Post-hoc MCMC refinement at inference time only.
        Uses DSM-trained energy gradients to polish URM predictions.
        """
        probs = F.softmax(logits.float(), dim=-1)
        predicted_emb = (probs @ self.inner.embed_tokens.embedding_weight.data.float()).to(
            self.inner.forward_dtype
        ).detach().requires_grad_(True)

        for _ in range(steps):
            with torch.enable_grad():
                energy = self.compute_joint_energy(input_embeddings, predicted_emb)
                grad = torch.autograd.grad(energy.sum(), predicted_emb)[0]
                grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                grad = grad / grad_norm
            predicted_emb = (predicted_emb - step_size * grad).detach().requires_grad_(True)

        refined_logits = self.inner.lm_head(predicted_emb.to(self.inner.forward_dtype))
        return refined_logits