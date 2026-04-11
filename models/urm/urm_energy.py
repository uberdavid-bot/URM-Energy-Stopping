from typing import Dict, Optional
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
from models.common import trunc_normal_init_
from models.layers import rms_norm, ConvSwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding


class ARCModelConfig(BaseModel):
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
    # Total recurrence steps before halting
    loops: int
    forward_dtype: str = "bfloat16"
    # Energy-specific
    energy_threshold: float = 0.005
    min_steps: int = 8
    # Refinement: "urm" (implicit recurrence), "ebt" (explicit MCMC), "hybrid" (URM then MCMC)
    refinement: str = "urm"
    # Stopping criterion: "qhalt" (learned halt signal) | "energy" (energy convergence)
    stopping: str = "qhalt"
    # Ranking signal for pass@K: "qhalt" (Q-halt confidence) | "energy" (negative energy)
    ranking: str = "qhalt"
    # MCMC step size for EBT/hybrid refinement
    mcmc_step_size: float = 0.01
    # Hybrid: number of URM steps before switching to MCMC (must be < loops)
    mcmc_start_step: int = 0


class ARCBlock(nn.Module):
    def __init__(self, config: ARCModelConfig) -> None:
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


class ARCBackbone(nn.Module):
    def __init__(self, config: ARCModelConfig) -> None:
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

        self.layers = nn.ModuleList([ARCBlock(self.config) for _ in range(self.config.num_layers)])

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


class ARCModel(nn.Module):
    """Unified model supporting URM, EBT, and hybrid refinement modes.

    Primary entry point: forward_trajectory() runs N recurrence steps with
    full gradient flow (no detach between steps) for deep supervision.
    """

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = ARCModelConfig(**config_dict)
        self.inner = ARCBackbone(self.config)
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

    def _mcmc_step(self, hidden: torch.Tensor, input_embeddings: torch.Tensor,
                   step_size: float, full_hidden: bool = False) -> torch.Tensor:
        """Single MCMC gradient step in hidden space.

        Training: create_graph=True (second-order gradients flow into energy head).
        Inference: detach to save memory.
        """
        P = self.inner.puzzle_emb_len if full_hidden else 0

        if self.training:
            if not hidden.requires_grad:
                hidden = hidden.requires_grad_(True)
            energy = self.compute_joint_energy(input_embeddings, hidden[:, P:])
            grad = torch.autograd.grad(energy.sum(), hidden, create_graph=True)[0]
            grad = grad / grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            hidden = hidden - step_size * grad
        else:
            hidden = hidden.detach().requires_grad_(True)
            with torch.enable_grad():
                energy = self.compute_joint_energy(input_embeddings, hidden[:, P:])
                grad = torch.autograd.grad(energy.sum(), hidden)[0]
                grad = grad / grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            hidden = (hidden - step_size * grad).detach().requires_grad_(True)

        return hidden

    def forward_trajectory(self, batch: Dict[str, torch.Tensor], N: Optional[int] = None):
        """Run N recurrence steps with full gradient flow (no detach between steps).

        Returns per-step logits, per-step Q-halt logits, per-step hidden states,
        and input_embeddings (for energy computation).
        """
        N = N or self.config.loops
        input_embeddings = self.inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        seq_info = dict(cos_sin=self.inner.rotary_emb())
        P = self.inner.puzzle_emb_len

        # Initialize hidden state
        hidden = self.inner.init_hidden.expand(
            batch["inputs"].shape[0],
            self.inner.config.seq_len + P, -1
        ).clone()

        all_logits = []
        all_q_logits = []
        all_hidden = []

        for step in range(N):
            if self.config.refinement == "urm" or (
                self.config.refinement == "hybrid" and step < self.config.mcmc_start_step
            ):
                # URM update: input re-injection + transformer pass
                hidden = hidden + input_embeddings
                for layer in self.inner.layers:
                    hidden = layer(hidden_states=hidden, **seq_info)
            else:
                # EBT update: energy gradient step
                hidden = self._mcmc_step(
                    hidden, input_embeddings, self.config.mcmc_step_size, full_hidden=True
                )

            logits = self.inner.lm_head(hidden)[:, P:]
            q_logits = self.inner.q_head(hidden[:, 0]).to(torch.float32)
            all_logits.append(logits)
            all_q_logits.append(q_logits)
            all_hidden.append(hidden)

        return all_logits, all_q_logits, all_hidden, input_embeddings
