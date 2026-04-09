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
    # Mode: "urm" (implicit recurrence), "ebt" (explicit MCMC), "hybrid" (URM then MCMC)
    mode: str = "urm"
    # MCMC training (second-order gradients through MCMC into energy head)
    mcmc_steps: int = 0
    mcmc_step_size: float = 0.01
    mcmc_training: bool = False
    # Hybrid: number of URM steps before switching to MCMC (must be < loops)
    mcmc_start_step: int = 0

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

    def _mcmc_steps(self, hidden: torch.Tensor, input_embeddings: torch.Tensor,
                    num_steps: int, step_size: float, full_hidden: bool = False) -> torch.Tensor:
        """Run MCMC gradient descent in hidden space.

        Training: create_graph=True, no detach (second-order gradients flow).
        Inference: detach between steps to save memory.

        If full_hidden=True, hidden includes puzzle_emb_len prefix positions
        which are stripped before energy computation (they get zero gradient).
        """
        P = self.inner.puzzle_emb_len if full_hidden else 0

        if self.training:
            if not hidden.requires_grad:
                hidden = hidden.requires_grad_(True)
            for _ in range(num_steps):
                energy = self.compute_joint_energy(input_embeddings, hidden[:, P:])
                grad = torch.autograd.grad(energy.sum(), hidden, create_graph=True)[0]
                grad = grad / grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                hidden = hidden - step_size * grad
        else:
            hidden = hidden.detach().requires_grad_(True)
            for _ in range(num_steps):
                with torch.enable_grad():
                    energy = self.compute_joint_energy(input_embeddings, hidden[:, P:])
                    grad = torch.autograd.grad(energy.sum(), hidden)[0]
                    grad = grad / grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                hidden = (hidden - step_size * grad).detach().requires_grad_(True)

        return hidden

    def forward(
        self,
        carry: URMCarry,
        batch: Dict[str, torch.Tensor],
        compute_target_q=False
    ) -> Tuple[URMCarry, Dict[str, torch.Tensor]]:
        # 1. Carry management
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

        input_embeddings = self.inner._input_embeddings(
            new_current_data["inputs"], new_current_data["puzzle_identifiers"]
        )

        if self.config.mode == "ebt":
            # Pure EBT: MCMC in hidden space from input_embeddings (no URM recurrence)
            hidden = input_embeddings.clone()
            unrefined_logits = self.inner.lm_head(hidden)[:, self.inner.puzzle_emb_len:]
            hidden = self._mcmc_steps(
                hidden, input_embeddings, self.config.loops,
                self.config.mcmc_step_size, full_hidden=True,
            )
            logits = self.inner.lm_head(hidden)[:, self.inner.puzzle_emb_len:]
            output_hidden = hidden[:, self.inner.puzzle_emb_len:].detach()
            current_energy = self.compute_joint_energy(input_embeddings, output_hidden)
            carry_hidden = hidden.detach()

        elif self.config.mode == "hybrid":
            # Hybrid: M URM recurrence steps, then (N-M) MCMC steps
            seq_info = dict(cos_sin=self.inner.rotary_emb())
            hidden = new_carry.current_hidden
            for _ in range(self.config.mcmc_start_step):
                hidden = hidden + input_embeddings
                for layer in self.inner.layers:
                    hidden = layer(hidden_states=hidden, **seq_info)
            unrefined_logits = self.inner.lm_head(hidden)[:, self.inner.puzzle_emb_len:]
            mcmc_steps = self.config.loops - self.config.mcmc_start_step
            hidden = self._mcmc_steps(
                hidden, input_embeddings, mcmc_steps,
                self.config.mcmc_step_size, full_hidden=True,
            )
            logits = self.inner.lm_head(hidden)[:, self.inner.puzzle_emb_len:]
            output_hidden = hidden[:, self.inner.puzzle_emb_len:].detach()
            current_energy = self.compute_joint_energy(input_embeddings, output_hidden)
            carry_hidden = hidden.detach()

        else:
            # URM mode (existing behavior)
            new_carry_inner, logits, (q_halt_logits, q_continue_logits) = self.inner(
                new_carry, new_current_data
            )
            output_hidden = new_carry_inner.current_hidden[:, self.inner.puzzle_emb_len:]
            current_energy = self.compute_joint_energy(input_embeddings, output_hidden)

            unrefined_logits = None
            if self.config.mcmc_steps > 0 and self.config.mcmc_training and self.training:
                unrefined_logits = logits
                hidden = output_hidden
                if not hidden.requires_grad:
                    hidden = hidden.requires_grad_(True)
                for _ in range(self.config.mcmc_steps):
                    energy = self.compute_joint_energy(input_embeddings, hidden)
                    grad = torch.autograd.grad(
                        energy.sum(), hidden, create_graph=True
                    )[0]
                    grad = grad / grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    hidden = hidden - self.config.mcmc_step_size * grad
                logits = self.inner.lm_head(hidden)
            elif self.config.mcmc_steps > 0 and not self.training:
                unrefined_logits = logits
                logits = self.refine_with_mcmc(
                    output_hidden, input_embeddings,
                    steps=self.config.mcmc_steps,
                    step_size=self.config.mcmc_step_size,
                )
            carry_hidden = new_carry_inner.current_hidden

        # Halting
        with torch.no_grad():
            new_steps = new_steps + 1
            if self.config.mode == "urm":
                halted = (new_steps >= self.config.loops)
                if self.training and carry.prev_energy is not None:
                    energy_change = torch.abs(current_energy - carry.prev_energy)
                    converged = energy_change < self.config.energy_threshold
                    halted = halted | (converged & (new_steps >= self.config.min_steps))
            else:
                # EBT/hybrid: all steps in one call, always halt
                halted = torch.ones(new_steps.shape, dtype=torch.bool, device=new_steps.device)

        # Outputs (zero Q values for evaluator compatibility)
        outputs = {
            "logits": logits,
            "current_energy": current_energy,
            "output_hidden": output_hidden,
            "q_halt_logits": torch.zeros_like(current_energy),
            "q_continue_logits": torch.zeros_like(current_energy),
        }
        if unrefined_logits is not None:
            outputs["unrefined_logits"] = unrefined_logits

        return (
            URMCarry(
                current_hidden=carry_hidden,
                steps=new_steps,
                halted=halted,
                current_data=new_current_data,
                prev_energy=current_energy.detach(),
            ),
            outputs,
        )

    def refine_with_mcmc(self, hidden: torch.Tensor, input_embeddings: torch.Tensor,
                         steps: int = 8, step_size: float = 0.01) -> torch.Tensor:
        """
        Post-hoc MCMC refinement at inference time only.
        Uses energy gradients in hidden space to refine URM predictions.
        Detaches between steps to save memory (no second-order gradients needed).
        """
        hidden = hidden.detach().requires_grad_(True)

        for _ in range(steps):
            with torch.enable_grad():
                energy = self.compute_joint_energy(input_embeddings, hidden)
                grad = torch.autograd.grad(energy.sum(), hidden)[0]
                grad = grad / grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            hidden = (hidden - step_size * grad).detach().requires_grad_(True)

        return self.inner.lm_head(hidden.to(self.inner.forward_dtype))