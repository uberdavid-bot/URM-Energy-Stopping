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
    predicted_embeddings: Optional[torch.Tensor] = None  # For energy-based updates


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
    L_cycles: int
    H_cycles: int
    forward_dtype: str = "bfloat16"
    # Energy-specific
    energy_threshold: float = 0.01
    langevin_noise_std: float = 0.01
    replay_buffer_size: int = 1000
    random_steps_min: int = 5
    random_steps_max: int = 50

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
            max_position_embeddings=self.config.seq_len * 3 + self.puzzle_emb_len,  # Support input + output + buffer
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
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[URMCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(cos_sin=self.rotary_emb())
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        hidden_states = carry.current_hidden
        if self.config.H_cycles > 1:
            with torch.no_grad():
                for _ in range(self.config.H_cycles - 1):
                    for _ in range(self.config.L_cycles):
                        hidden_states = hidden_states + input_embeddings
                        for layer in self.layers:
                            hidden_states = layer(hidden_states=hidden_states, **seq_info)

        for _ in range(self.config.L_cycles):
            hidden_states = hidden_states + input_embeddings
            for layer in self.layers:
                hidden_states = layer(hidden_states=hidden_states, **seq_info)

        new_carry = replace(carry, current_hidden=hidden_states.detach())
        output = self.lm_head(hidden_states)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(hidden_states[:, 0]).to(torch.float32)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class URM_Energy(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = URMConfig(**config_dict)
        self.inner = URM_Inner(self.config)
        # Energy-specific components
        self.energy_head = nn.Linear(self.config.hidden_size, 1, dtype=self.inner.forward_dtype)
        self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32), requires_grad=False)  # MCMC step size
        self.langevin_noise_std = nn.Parameter(torch.tensor(float(self.config.langevin_noise_std), dtype=torch.float32))
        # Replay buffer removed for now - add later if needed

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    # @torch._dynamo.disable
    # @torch.compiler.disable
    def compute_joint_energy(self, input_embeddings: torch.Tensor, predicted_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute joint energy E(input, predicted_output).
        For ARC-AGI: input_embeddings and predicted_embeddings are flattened grids.
        Concatenate, pass through transformer, pool to scalar energy per example.
        """
        # Ensure embeddings are in the correct dtype
        input_embeddings = input_embeddings.to(self.inner.forward_dtype)
        predicted_embeddings = predicted_embeddings.to(self.inner.forward_dtype)
        
        all_embeddings = torch.cat((input_embeddings, predicted_embeddings), dim=1)  # [batch_size, total_seq_len, hidden_size]
        
        # Use the existing rotary_emb from inner (already configured correctly)
        cos_sin = self.inner.rotary_emb()
        
        hidden_states = all_embeddings
        for layer in self.inner.layers:
            hidden_states = layer(cos_sin=cos_sin, hidden_states=hidden_states)
        # Pool over ALL positions to get scalar energy per example
        h_pooled = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        energy = self.energy_head(h_pooled)  # [batch_size, 1]
        return energy.squeeze(-1)  # [batch_size] - scalar per example

    # @torch.compiler.disable
    def mcmc_update(self, carry: URMCarry, batch: Dict[str, torch.Tensor], training: bool = True) -> Tuple[URMCarry, torch.Tensor]:
        """
        Perform one MCMC step: update predicted_embeddings via energy minimization.
        """
        input_embeddings = self.inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        predicted_embeddings = carry.predicted_embeddings.detach().requires_grad_(True)

        energy = self.compute_joint_energy(input_embeddings, predicted_embeddings)
        total_energy = energy.sum()

        # CRITICAL: create_graph=True during training for second-order gradients
        if training:
            grad = torch.autograd.grad(
                total_energy, 
                predicted_embeddings, 
                create_graph=True
            )[0]
        else:
            grad = torch.zeros_like(predicted_embeddings)

        # Add gradient normalization
        grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        grad = grad / grad_norm  # Normalize to unit vectors
        grad = grad * 0.1  # Scale to reasonable magnitude

        # Langevin dynamics: add noise (only during training)
        if training:
            noise = torch.randn_like(predicted_embeddings) * self.langevin_noise_std
        else:
            noise = 0  # Scalar 0 broadcasts correctly
        
        # alpha = torch.clamp(self.alpha, min=0.0001)
        alpha = self.alpha.clamp(min=0.0001)

        # Update
        updated_predicted = predicted_embeddings - alpha * grad + noise

        # Update carry
        new_carry = replace(carry, predicted_embeddings=updated_predicted.detach())

        # # DON'T detach during training!
        # if training:
        #     new_carry = replace(carry, predicted_embeddings=updated_predicted)
        # else:
        #     new_carry = replace(carry, predicted_embeddings=updated_predicted.detach())
    
        return new_carry, energy

    def embeddings_to_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Convert predicted embeddings back to logits over vocab.
        embeddings: [batch_size, seq_len, hidden_size]
        returns: [batch_size, seq_len, vocab_size]
        """
        return self.inner.lm_head(embeddings)

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> URMCarry:
        batch_size = batch["inputs"].shape[0]
        base = self.inner.empty_carry(batch_size)
        # Initialize predicted_embeddings for OUTPUT grid (no puzzle embeddings)
        output_seq_len = batch["labels"].shape[1]  # Use labels to get output length
        predicted_embeddings = torch.randn(
            batch_size, 
            output_seq_len, 
            self.config.hidden_size, 
            dtype=self.inner.forward_dtype,
            device=batch["inputs"].device
        )
        return URMCarry(
            current_hidden=base.current_hidden,
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=batch["inputs"].device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=batch["inputs"].device),
            current_data={k: v.clone().detach() for k, v in batch.items()},  # Deep copy + detach
            predicted_embeddings=predicted_embeddings,
        )

    def forward(
        self,
        carry: URMCarry,
        batch: Dict[str, torch.Tensor],
        compute_target_q=False
    ) -> Tuple[URMCarry, Dict[str, torch.Tensor]]:
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

        # Track MCMC trajectory for loss
        energies_trajectory = []
        predictions_trajectory = []
        
        prev_energy = None
        for step in range(self.config.loops):
            # MCMC update
            new_carry, current_energy = self.mcmc_update(
                new_carry, 
                new_current_data,
                training=self.training
            )
            energies_trajectory.append(current_energy)
            
            # Convert embeddings to logits for this step
            logits = self.embeddings_to_logits(new_carry.predicted_embeddings)
            # No need to remove puzzle positions - predicted_embeddings doesn't include them
            predictions_trajectory.append(logits)
            
            # Check convergence
            if prev_energy is not None:
                energy_change = torch.abs(current_energy - prev_energy).mean()
                min_steps = 6  # Force at least 8 MCMC steps
                if energy_change < self.config.energy_threshold and step >= min_steps:
                    break
            prev_energy = current_energy
            new_steps = new_steps + 1

        # Standard URM forward for Q values - commented out since using energy-based stopping
        # new_carry_urm, _, (q_halt_logits, q_continue_logits) = self.inner(new_carry, new_current_data)

        # Set dummy Q values since we're using energy-based stopping
        batch_size = new_current_data["inputs"].shape[0]
        q_halt_logits = torch.zeros(batch_size, device=new_current_data["inputs"].device)
        q_continue_logits = torch.zeros(batch_size, device=new_current_data["inputs"].device)

        outputs = {
            "logits": predictions_trajectory[-1],  # Final prediction
            "predictions_trajectory": predictions_trajectory,  # For loss
            "energies_trajectory": energies_trajectory,  # For loss
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "final_energy": energies_trajectory[-1],
        }

        with torch.no_grad():
            new_steps = new_steps + 1
            halted = (new_steps >= self.config.loops)

            # Energy-based halting: already handled in the loop above
            # No need for Q-halt logic
            # if self.training and (self.config.max_steps > 1):
            #     halted = halted | (q_halt_logits > 0)
            #     halt_exploration_prob = 0.1
            #     min_halt_steps = (torch.rand_like(q_halt_logits) < halt_exploration_prob) * torch.randint_like(new_steps, low=self.config.random_steps_min, high=self.config.random_steps_max + 1)
            #     halted = halted & (new_steps >= min_halt_steps)

        return (
            URMCarry(
                current_hidden=new_carry.current_hidden,
                steps=new_steps,
                halted=halted,
                current_data=new_current_data,
                predicted_embeddings=new_carry.predicted_embeddings,
            ),
            outputs,
        )