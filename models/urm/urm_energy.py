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
    # R4b: mask attention diagonal so each token attends exclusively to others
    exclusive_attention: bool = False
    # R6a: per-head learnable sink logit in softmax denominator (allows attn weights to sum < 1)
    attention_sink: bool = False
    # R6b: per-head learnable temperature (multiplicative scaling of pre-softmax logits)
    attention_temperature: bool = False
    # R6g: fraction of head_dim to apply RoPE to (rest is content-only attention)
    rope_fraction: float = 1.0
    # R6h: number of KV heads for grouped-query attention (None = num_heads, i.e., standard MHA)
    num_kv_heads: Optional[int] = None
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
    # R7a GRAM: stochastic latent transitions (prior/posterior VAE-style perturbation per recurrence step)
    gram_enabled: bool = False
    gram_beta: float = 0.1
    gram_logvar_min: float = -5.0
    gram_logvar_max: float = 2.0
    gram_mlp_hidden: int = 64
    gram_num_samples: int = 8
    # R7c: per-example bottleneck latent dimension (0 = legacy per-position full-H eps)
    gram_latent_dim: int = 0
    # R7b: decode logits from pre-eps deterministic state (breaks one-pass posterior leak)
    gram_predecode: bool = False
    # R7b: stop recon-loss gradients from flowing into posterior MLP (KL-only posterior shaping)
    gram_detach_posterior_recon: bool = False
    # R7c2: KL warmup — linear ramp beta: 0 -> gram_beta over N steps; 0 = no warmup
    gram_kl_warmup_steps: int = 0
    # R7c2: free-bits floor (nats); KL penalized only above this threshold
    gram_free_bits: float = 0.0
    # R7d: delta-norm-proportional sigma — eps magnitude = alpha * ||delta_t|| (per-example)
    # When > 0, overrides learned sigma: eps is normalized to unit norm and rescaled by alpha * delta_norm_t.
    # Provides calibration (eps is a small fraction of the update) and annealing (shrinks with convergence).
    gram_sigma_alpha: float = 0.0
    # Diagnostic: run posterior-conditioned eval pass to confirm leak (temporary probe)
    gram_posterior_eval_probe: bool = False


class ARCBlock(nn.Module):
    def __init__(self, config: ARCModelConfig) -> None:
        super().__init__()
        head_dim = config.hidden_size // config.num_heads
        num_kv_heads = config.num_kv_heads if config.num_kv_heads is not None else config.num_heads
        rope_dims = round(head_dim * config.rope_fraction)
        rope_dims = rope_dims - (rope_dims % 2)  # must be even for RoPE
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=head_dim,
            num_heads=config.num_heads,
            num_key_value_heads=num_kv_heads,
            causal=False,
            attn_dropout=config.attn_dropout,
            exclusive_attention=config.exclusive_attention,
            attention_sink=config.attention_sink,
            attention_temperature=config.attention_temperature,
            rope_dim=rope_dims if config.rope_fraction < 1.0 else None,
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

        assert not (config.attention_sink and config.attention_temperature), \
            "attention_sink and attention_temperature are mutually exclusive"
        if config.num_kv_heads is not None:
            assert config.num_heads % config.num_kv_heads == 0, \
                f"num_heads ({config.num_heads}) must be divisible by num_kv_heads ({config.num_kv_heads})"
        assert config.hidden_size % config.num_heads == 0, \
            f"hidden_size ({config.hidden_size}) must be divisible by num_heads ({config.num_heads})"

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

        head_dim = self.config.hidden_size // self.config.num_heads
        rope_dims = round(head_dim * self.config.rope_fraction)
        rope_dims = rope_dims - (rope_dims % 2)
        self.rotary_emb = RotaryEmbedding(
            dim=rope_dims,
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

        if self.config.gram_enabled:
            H = self.config.hidden_size
            gh = self.config.gram_mlp_hidden
            k = self.config.gram_latent_dim
            dt = self.inner.forward_dtype
            if k > 0:
                # R7c: per-example bottleneck latent of dim k, projected up to H and broadcast
                self.gram_prior_mlp = nn.Sequential(
                    nn.Linear(H, gh, dtype=dt),
                    nn.SiLU(),
                    nn.Linear(gh, 2 * k, dtype=dt),
                )
                self.gram_posterior_mlp = nn.Sequential(
                    nn.Linear(2 * H, gh, dtype=dt),
                    nn.SiLU(),
                    nn.Linear(gh, 2 * k, dtype=dt),
                )
                self.gram_up_proj = nn.Linear(k, H, bias=False, dtype=dt)
            else:
                # R7a/R7b: per-position full-H eps
                self.gram_prior_mlp = nn.Sequential(
                    nn.Linear(H, gh, dtype=dt),
                    nn.SiLU(),
                    nn.Linear(gh, 2 * H, dtype=dt),
                )
                self.gram_posterior_mlp = nn.Sequential(
                    nn.Linear(2 * H, gh, dtype=dt),
                    nn.SiLU(),
                    nn.Linear(gh, 2 * H, dtype=dt),
                )

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def _gram_reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        logvar = logvar.clamp(self.config.gram_logvar_min, self.config.gram_logvar_max)
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(mu)

    def _gram_split(self, out: torch.Tensor):
        d = out.shape[-1] // 2
        mu, logvar = out[..., :d], out[..., d:]
        logvar = logvar.clamp(self.config.gram_logvar_min, self.config.gram_logvar_max)
        return mu, logvar

    def _gram_kl(self, mu_q: torch.Tensor, logvar_q: torch.Tensor,
                 mu_p: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
        """Closed-form KL(q || p) for diagonal Gaussians, summed over last dim."""
        return 0.5 * (
            logvar_p - logvar_q
            + (logvar_q.exp() + (mu_q - mu_p).pow(2)) / logvar_p.exp().clamp(min=1e-8)
            - 1.0
        ).sum(dim=-1)

    def _gram_label_embeddings(self, labels: torch.Tensor) -> torch.Tensor:
        """Embed labels for posterior conditioning. Returns [B, P+seq, H]."""
        safe_labels = labels.clone()
        safe_labels[safe_labels == -100] = 0
        label_emb = self.inner.embed_scale * self.inner.embed_tokens(safe_labels.to(torch.int32))
        B = labels.shape[0]
        P = self.inner.puzzle_emb_len
        R = self.config.num_registers
        pad = torch.zeros(B, P + R, self.config.hidden_size,
                          device=labels.device, dtype=label_emb.dtype)
        return torch.cat([pad, label_emb], dim=1)

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

    def forward_trajectory(self, batch: Dict[str, torch.Tensor], N: Optional[int] = None,
                           labels: Optional[torch.Tensor] = None,
                           force_posterior: bool = False):
        """Run N recurrence steps with full gradient flow (no detach between steps).

        Returns per-step logits, per-step Q-halt logits, per-step hidden states,
        input_embeddings (for energy computation), and per-step KL list (GRAM only).

        force_posterior: if True, use posterior (with labels) even at eval. For leak probe only.
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

        # GRAM: precompute label embeddings for posterior
        gram_label_emb = None
        use_posterior = (self.config.gram_enabled and
                         (self.training or force_posterior) and
                         labels is not None)
        if use_posterior:
            gram_label_emb = self._gram_label_embeddings(labels)

        all_logits = []
        all_q_logits = []
        all_hidden = []
        all_gram_kl = []
        all_gram_effective_sigma = []  # R7d: per-step effective injection scale

        for step in range(N):
            if self.config.refinement == "urm" or (
                self.config.refinement == "hybrid" and step < self.config.mcmc_start_step
            ):
                # URM update: input re-injection + transformer pass
                pre_t = hidden + input_embeddings

                # Deterministic forward: u_t = transformer(pre_t)
                u_t = pre_t
                for layer in self.inner.layers:
                    u_t = layer(hidden_states=u_t, **seq_info)

                # Decode from clean u_t (never from perturbed state)
                logits = self.inner.lm_head(u_t)[:, P + R:]
                q_logits = self.inner.q_head(u_t[:, 0]).to(torch.float32).squeeze(-1)

                # GRAM: stochastic perturbation applied to carry only
                if self.config.gram_enabled:
                    k = self.config.gram_latent_dim
                    if k > 0:
                        # R7c: per-example bottleneck — condition on u_t (post-transformer)
                        pooled = u_t.mean(dim=1)  # [B, H]
                        mu_p, logvar_p = self._gram_split(self.gram_prior_mlp(pooled))

                        if use_posterior:
                            pooled_label = gram_label_emb.mean(dim=1)  # [B, H]
                            posterior_input = torch.cat([pooled, pooled_label], dim=-1)
                            mu_q, logvar_q = self._gram_split(self.gram_posterior_mlp(posterior_input))
                            z = self._gram_reparam(mu_q, logvar_q)
                            if self.config.gram_detach_posterior_recon:
                                z = z.detach()
                            kl_t = self._gram_kl(mu_q, logvar_q, mu_p, logvar_p)
                            all_gram_kl.append(kl_t.mean())
                        else:
                            z = self._gram_reparam(mu_p, logvar_p)

                        eps_t = self.gram_up_proj(z).unsqueeze(1).expand_as(u_t)
                    else:
                        # Per-position full-H eps — condition on u_t
                        mu_p, logvar_p = self._gram_split(self.gram_prior_mlp(u_t))

                        if use_posterior:
                            posterior_input = torch.cat([u_t, gram_label_emb], dim=-1)
                            mu_q, logvar_q = self._gram_split(self.gram_posterior_mlp(posterior_input))
                            eps_t = self._gram_reparam(mu_q, logvar_q)
                            if self.config.gram_detach_posterior_recon:
                                eps_t = eps_t.detach()
                            kl_t = self._gram_kl(mu_q, logvar_q, mu_p, logvar_p)
                            all_gram_kl.append(kl_t.mean())
                        else:
                            eps_t = self._gram_reparam(mu_p, logvar_p)

                    # Rescale eps: σ = α · ‖u_t − hidden‖ (delta-norm-proportional)
                    if self.config.gram_sigma_alpha > 0:
                        delta_norm_t = (u_t - hidden).norm(dim=-1).mean(dim=-1)  # [B]
                        eps_flat = eps_t.reshape(eps_t.shape[0], -1)
                        eps_norm = eps_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                        eps_unit = eps_flat / eps_norm
                        scale = self.config.gram_sigma_alpha * delta_norm_t
                        eps_t = (eps_unit * scale.unsqueeze(-1)).reshape_as(eps_t)
                        all_gram_effective_sigma.append(scale.mean().detach())

                    # Carry: perturbed for next step, clean at last step
                    hidden = u_t + eps_t if step < N - 1 else u_t
                else:
                    hidden = u_t

                # Recurrence noise: additive Gaussian on hidden state (training only)
                if self.config.recurrence_noise > 0 and self.training:
                    hidden = hidden + self.config.recurrence_noise * torch.randn_like(hidden)
            else:
                # EBT update: energy gradient step
                hidden = self._mcmc_step(
                    hidden, input_embeddings, self.config.mcmc_step_size, full_hidden=True
                )

                logits = self.inner.lm_head(hidden)[:, P + R:]
                q_logits = self.inner.q_head(hidden[:, 0]).to(torch.float32).squeeze(-1)

            all_logits.append(logits)
            all_q_logits.append(q_logits)
            all_hidden.append(hidden)

        gram_kl = all_gram_kl if all_gram_kl else None
        gram_effective_sigma = all_gram_effective_sigma if all_gram_effective_sigma else None
        return all_logits, all_q_logits, all_hidden, input_embeddings, gram_kl, gram_effective_sigma

    def forward_gram_samples(self, batch: Dict[str, torch.Tensor], M: int):
        """Run M independent prior-sampled trajectories (eval only).

        Returns:
            all_preds: [M, B, seq_len] — argmax grid predictions per trajectory
            all_q_logits: [M, B] — final-step Q-halt logits per trajectory
        """
        all_preds = []
        all_q_logits = []
        for _ in range(M):
            logits_list, q_list, _, _, _, _ = self.forward_trajectory(batch)
            final_preds = torch.argmax(logits_list[-1], dim=-1)
            final_q = q_list[-1]
            all_preds.append(final_preds)
            all_q_logits.append(final_q)
        return torch.stack(all_preds, dim=0), torch.stack(all_q_logits, dim=0)
