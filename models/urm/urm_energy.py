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
    # R4a: number of learnable register tokens inserted between puzzle_emb and input
    num_registers: int = 0
    # Additive Gaussian noise stddev applied to hidden states after each URM recurrence pass (training only)
    recurrence_noise: float = 0.0
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
    # Langevin noise scale added to hidden after each MCMC update (training only)
    mcmc_langevin_noise: float = 0.0
    # If True, sample step size uniformly from [mcmc_step_size/3, mcmc_step_size] per training step
    mcmc_random_step_size: bool = False
    # Energy head architecture: "linear" | "position_mlp" | "position_conv_mlp" | "position_attn_mlp"
    energy_head_type: str = "linear"
    energy_head_hidden: int = 32


class ARCBlock(nn.Module):
    def __init__(self, config: ARCModelConfig) -> None:
        super().__init__()
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
            attn_dropout=config.attn_dropout,
        )
        self.mlp = ConvSwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.mlp_dropout = nn.Dropout(config.mlp_dropout)
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        seq_len = hidden_states.shape[1]
        cos_sin = tuple(c[:seq_len] for c in cos_sin)
        attn_output = self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states)
        hidden_states = rms_norm(hidden_states + attn_output, variance_epsilon=self.norm_eps)
        mlp_output = self.mlp_dropout(self.mlp(hidden_states))
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
        self.q_head = CastedLinear(self.config.hidden_size, 1, bias=True)
        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)

        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )

        # R4a: learnable register tokens inserted between puzzle_emb and input_tokens.
        # Initialized at std=1/embed_scale so that the trailing self.embed_scale * embedding
        # in _input_embeddings rescales them to std~1, matching embed_tokens output.
        if self.config.num_registers > 0:
            self.register_emb = nn.Parameter(
                trunc_normal_init_(
                    torch.empty(self.config.num_registers, self.config.hidden_size, dtype=self.forward_dtype),
                    std=embed_init_std,
                )
            )

        # compute_joint_energy concatenates input_emb [P + R + seq_len] + output_emb [R + seq_len]
        # (output_emb = hidden[:, P:] keeps the register slots). Total length = 2*seq_len + P + 2*R.
        self.rotary_emb = RotaryEmbedding(
            dim=self.config.hidden_size // self.config.num_heads,
            max_position_embeddings=(
                2 * self.config.seq_len
                + self.puzzle_emb_len
                + 2 * self.config.num_registers
            ),
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

        # R4a: insert registers between puzzle_emb and input_tokens.
        # Layout becomes [puzzle_emb (P), registers (R), input_tokens (seq_len)].
        if self.config.num_registers > 0:
            B = embedding.shape[0]
            registers = self.register_emb.unsqueeze(0).expand(B, -1, -1)
            embedding = torch.cat(
                (embedding[:, : self.puzzle_emb_len], registers, embedding[:, self.puzzle_emb_len :]),
                dim=1,
            )

        return self.embed_scale * embedding


class PositionEnergyHead(nn.Module):
    """Per-position energy projection, summed to scalar.

    Replaces mean_pool -> Linear(H, 1) with per-position MLP -> sum.
    Preserves spatial information that mean pooling destroys.
    """

    def __init__(self, hidden_size: int, mlp_hidden: int = 32,
                 use_conv: bool = False, use_attn: bool = False,
                 num_heads: int = 4, dtype=torch.bfloat16):
        super().__init__()
        self.use_conv = use_conv
        self.use_attn = use_attn

        if use_attn:
            self.attn = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                batch_first=True,
                dtype=dtype,
            )
            self.attn_norm = nn.LayerNorm(hidden_size, dtype=dtype)

        if use_conv:
            self.conv = nn.Conv1d(
                hidden_size, hidden_size,
                kernel_size=3, padding=1,
                groups=min(16, hidden_size),
                bias=True,
                dtype=dtype,
            )
            self.conv_act = nn.SiLU()

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden, dtype=dtype),
            nn.SiLU(),
            nn.Linear(mlp_hidden, 1, dtype=dtype),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """[B, seq_len, H] -> [B] scalar energy per example."""
        if self.use_attn:
            attn_out, _ = self.attn(hidden_states, hidden_states, hidden_states)
            hidden_states = self.attn_norm(hidden_states + attn_out)

        if self.use_conv:
            conv_out = self.conv(hidden_states.transpose(1, 2)).transpose(1, 2)
            hidden_states = hidden_states + self.conv_act(conv_out)

        energy_per_pos = self.mlp(hidden_states)  # [B, seq_len, 1]
        energy = energy_per_pos.sum(dim=1).squeeze(-1)  # [B]
        return energy


class ARCModel(nn.Module):
    """Unified model supporting URM, EBT, and hybrid refinement modes.

    Primary entry point: forward_trajectory() runs N recurrence steps with
    full gradient flow (no detach between steps) for deep supervision.
    """

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = ARCModelConfig(**config_dict)
        self.inner = ARCBackbone(self.config)

        if self.config.energy_head_type == "linear":
            self.energy_head = nn.Linear(self.config.hidden_size, 1, dtype=self.inner.forward_dtype)
        else:
            self.energy_head = PositionEnergyHead(
                hidden_size=self.config.hidden_size,
                mlp_hidden=self.config.energy_head_hidden,
                use_conv=(self.config.energy_head_type == "position_conv_mlp"),
                use_attn=(self.config.energy_head_type == "position_attn_mlp"),
                num_heads=self.config.num_heads,
                dtype=self.inner.forward_dtype,
            )

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

        if self.config.energy_head_type == "linear":
            h_pooled = hidden_states.mean(dim=1)
            energy = self.energy_head(h_pooled)
            return energy.squeeze(-1)
        else:
            return self.energy_head(hidden_states)

    def _mcmc_step(self, hidden: torch.Tensor, input_embeddings: torch.Tensor,
                   step_size: float, full_hidden: bool = False) -> torch.Tensor:
        """Single MCMC gradient step in hidden space.

        Training: create_graph=True (second-order gradients flow into energy head).
        Optional Langevin noise + randomized step size during training (EBT-style
        landscape regularization).
        Inference: detach to save memory, no noise, deterministic step size.
        """
        P = self.inner.puzzle_emb_len if full_hidden else 0

        if self.training:
            if self.config.mcmc_random_step_size:
                r = float(torch.rand(1).item())  # [0, 1)
                effective_step_size = step_size * (1.0 / 3.0 + (2.0 / 3.0) * r)
            else:
                effective_step_size = step_size

            if not hidden.requires_grad:
                hidden = hidden.requires_grad_(True)
            energy = self.compute_joint_energy(input_embeddings, hidden[:, P:])
            grad = torch.autograd.grad(energy.sum(), hidden, create_graph=True)[0]
            grad = grad / grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            hidden = hidden - effective_step_size * grad

            if self.config.mcmc_langevin_noise > 0:
                hidden = hidden + self.config.mcmc_langevin_noise * torch.randn_like(hidden)
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
        R = self.config.num_registers

        # Initialize hidden state. Layout matches input_embeddings: [P, R, seq_len].
        hidden = self.inner.init_hidden.expand(
            batch["inputs"].shape[0],
            self.inner.config.seq_len + P + R, -1
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

                # Recurrence noise: additive Gaussian on hidden state (training only)
                if self.config.recurrence_noise > 0 and self.training:
                    hidden = hidden + self.config.recurrence_noise * torch.randn_like(hidden)
            else:
                # EBT update: energy gradient step
                hidden = self._mcmc_step(
                    hidden, input_embeddings, self.config.mcmc_step_size, full_hidden=True
                )

            # lm_head slice skips both puzzle_emb (P) and register (R) prefixes.
            logits = self.inner.lm_head(hidden)[:, P + R:]
            # Q-halt reads position 0 (puzzle_emb slot, unchanged by registers).
            q_logits = self.inner.q_head(hidden[:, 0]).to(torch.float32).squeeze(-1)
            all_logits.append(logits)
            all_q_logits.append(q_logits)
            all_hidden.append(hidden)

        return all_logits, all_q_logits, all_hidden, input_embeddings
