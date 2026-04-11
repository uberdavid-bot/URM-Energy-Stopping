# Architecture

## Overview

This project compares two iterative refinement mechanisms for ARC-AGI puzzle solving, sharing a common backbone:

- **URM mode (implicit)**: Shared transformer layers applied recurrently, refining hidden states via attention + input re-injection. Stopping via Q-halt or fixed steps.
- **EBT mode (explicit)**: Energy gradient descent in hidden space, refining hidden states via ∇E. Stopping via energy convergence.

Both modes produce hidden states → lm_head → logits → predictions. The only difference is how hidden states are updated at each step.

```
input grid → tokenize → embed_tokens → input_embeddings [B, seq_len, hidden_dim]

URM step:  hidden = transformer_layers(hidden + input_embeddings)
EBT step:  energy = E(input_embeddings, hidden); hidden = hidden - α∇E

After N steps: logits = lm_head(hidden) → predictions
```

## Common Components

### Transformer Backbone
- Shared transformer layers (weight-tied across steps)
- Target config: to be confirmed in Phase 1 (testing depth=2 at various hidden/expansion)
- Bidirectional attention (non-causal) — ARC needs global pattern recognition
- RoPE positional encodings

### Input/Output Encoding
- ARC grids flattened to seq_len tokens (100 for 10×10)
- Vocabulary: 12 tokens (2 special + 10 colors)
- Per-puzzle learned embeddings prepended to sequence
- `input_embeddings = embed_scale * embed_tokens(input)`

### lm_head
- Linear(hidden_dim, vocab_size) projecting hidden states to logits
- Trained on transformer hidden states in both modes

### Energy Head
- `compute_joint_energy(input_embeddings, hidden)`:
  - Concatenates input_embeddings and hidden → [B, 2*seq_len, hidden_dim]
  - Runs through backbone transformer layers (reuses weights)
  - Mean pools → [B, hidden_dim]
  - Linear(hidden_dim, 1) → scalar energy per example
- Lower energy = better prediction
- RoPE sized for 2 × seq_len + puzzle_emb_len
- **Optional upgrade**: position-aware energy head (per-position energy contributions summed rather than pool-then-project). Try if energy-accuracy correlation is weak in R2.

## URM Mode

```python
hidden = init_hidden  # learned constant buffer
for step in range(N):
    hidden = hidden + input_embeddings        # re-inject input
    hidden = transformer_layers(hidden)       # shared weights
    logits = lm_head(hidden)
    q_logits = q_head(hidden[:, 0])           # halt/continue prediction
    if should_halt(q_logits): break
```

Implicit energy minimization: each transformer pass pushes hidden toward a fixed point that balances input structure with learned priors. Q-halt detects convergence. Q-halt confidence used for pass@K ranking.

## EBT Mode (Hybrid: M URM + K MCMC)

The primary EBT configuration is hybrid — M URM steps followed by K MCMC steps, with M+K = N (matched compute). Controlled by `ARCModelConfig.refinement="hybrid"` and `mcmc_start_step`.

```python
# Phase 1: URM initialization (M steps)
hidden = init_hidden
for step in range(M):
    hidden = hidden + input_embeddings
    hidden = transformer_layers(hidden)

# Phase 2: MCMC refinement (K steps)
for step in range(K):
    hidden = hidden.requires_grad_(True)      # (don't detach during training)
    energy = compute_joint_energy(input_embeddings, hidden)
    grad = autograd.grad(energy.sum(), hidden, create_graph=True)[0]
    grad = grad / grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    step_size_t = step_size * (1 - step / K)  # annealed step size
    hidden = hidden - step_size_t * grad
    # Langevin noise (training only):
    # hidden = hidden + noise_scale * torch.randn_like(hidden)
    if energy_converged(energy, prev_energy): break
logits = lm_head(hidden)
```

Pure EBT (N MCMC steps from input_embeddings, no URM steps) is also supported via `refinement="ebt"` but is an optional ablation (R3b), not the primary experiment.

### Why hybrid, not pure EBT?
- IREM (Du et al., 2022) showed energy minimization struggles when initialization is far from solution.
- ARC grids have discrete, spatially structured outputs — a cold start from unprocessed embeddings may never reach the right basin.
- URM steps provide attention-processed spatial structure that gives MCMC a viable starting point.

### MCMC Stabilization (from literature review)
- **Step size annealing**: Large steps early (coarse), small late (fine). Approximates IRED's multi-scale energy landscapes within a single energy head.
- **Langevin noise**: `hidden = hidden - step_size * grad + noise_scale * randn_like(hidden)` during training. Prevents collapse to sharp narrow minima. Drop noise at inference. (Pang et al., 2020; Nijkamp et al., 2022)
- **Separate gradient clipping**: energy head max_norm=1.0, backbone max_norm=5.0. Second-order gradients through the energy head have much larger magnitude.

### Why hidden space, not soft-embedding space?
The previous implementation converted logits → softmax → embed_tokens to create "soft embeddings" for MCMC. This is wrong for our architecture because:
1. Collapses hidden_dim to an 11-dimensional simplex (12 vocab tokens)
2. lm_head expects transformer hidden states, not convex combinations of token embeddings
3. compute_joint_energy is trained on hidden states, not soft embeddings

Hidden-space MCMC avoids all three issues. Gradients operate in the full hidden_dim space, and both lm_head and compute_joint_energy receive inputs from the distribution they were trained on.

### Gradient flow across MCMC steps
During training, do NOT detach hidden between MCMC steps. The energy head must learn from multi-step trajectories to shape a landscape that supports multi-step optimization. Detaching limits learning to single-step gradient quality.

During inference, detach between steps (no create_graph needed, saves memory).

## Training

### Losses

| Loss | What it trains | When used |
|------|---------------|-----------|
| Reconstruction (per-step) | URM backbone / embeddings | Every URM step |
| Trajectory ranking | Energy head (first-order) | R2+: all-pairs ranking across URM trajectory |
| Reconstruction (refined) | Energy head (via MCMC backprop) | R3: on logits after MCMC refinement |

**Dual reconstruction loss** (0.5/0.5 weighting) is mandatory when training with MCMC (R3): loss on both pre-MCMC and post-MCMC logits. The unrefined loss keeps the backbone learning cleanly. The refined loss trains the energy head through second-order gradients.

### Trajectory Ranking Loss (primary energy training signal)

```python
# During URM forward pass, capture hidden states at each step
trajectory_hiddens = [h_1, h_2, ..., h_N]  # [N, B, seq_len, hidden]
trajectory_qualities = [q_1, q_2, ..., q_N]  # per-step accuracy vs labels

# All-pairs weighted margin loss
loss = 0
for i, j in all_pairs where q_i > q_j:
    gap = q_i - q_j
    loss += gap * F.relu(E(h_i) - E(h_j) + margin)
```

**Properties:**
- N steps → N*(N-1)/2 ordered pairs (28 for N=8, 120 for N=16)
- Weighted by quality gap: large accuracy differences contribute more
- Natural anti-collapse: 28+ pairs requiring different energy values makes constant-output impossible
- First-order only: no create_graph=True needed, trains at URM speed (~3.5 it/s)
- Co-trained with URM: sees diverse trajectories while URM is still learning

**Why not contrastive loss?** Exp 1 showed contrastive loss alone (E(true) < E(pred)) collapses. Trajectory ranking provides dense, ordered supervision across many quality levels, not just binary correct/incorrect.

### Config fields
Three explicit fields control operational mode:
- **`refinement`**: `"urm"` | `"ebt"` | `"hybrid"` — how hidden states are updated each step.
- **`stopping`**: `"qhalt"` | `"energy"` — when to stop iterating.
- **`ranking`**: `"qhalt"` | `"energy"` — confidence signal for pass@K reranking.

Training requirements by refinement mode:
- **"urm"**: Reconstruction + Q-halt loss, no energy head needed. First-order only.
- **"urm" + trajectory energy (R2)**: Reconstruction + Q-halt + trajectory ranking loss. First-order only. Co-trained.
- **"ebt"**: Dual reconstruction loss. N MCMC steps from input_embeddings. Second-order gradients via create_graph=True.
- **"hybrid" (R3)**: Dual reconstruction + trajectory ranking. M URM steps then (N-M) MCMC steps (controlled by `mcmc_start_step`). Second-order gradients through MCMC phase.

## Energy Reranking (R2.5 — eval only)

At inference, energy can be used as a post-hoc verifier without MCMC:
1. Run URM with multiple seeds/augmentations → K candidate predictions
2. Score each candidate: `energy = compute_joint_energy(input_emb, hidden_final)`
3. Rank candidates by energy (lower = better)
4. Compare pass@K to Q-halt ranking

This requires no MCMC at inference — just one forward pass through the energy head per candidate.

## Recurrence Simplification

Single recurrence loop — every step gets gradients and can be evaluated. The legacy H_cycles/L_cycles memory optimization has been removed from the config and model. This makes "one URM step" and "one MCMC step" directly comparable units of compute.

## Evaluator

Collects predictions across augmented puzzle versions, votes, computes pass@K. Reports both Q-halt-ranked and energy-ranked pass@K for direct comparison.

## File Map

- `models/urm/urm_energy.py` — Unified URM: config, forward (urm/ebt/hybrid modes), energy computation, MCMC refinement
- `models/losses.py` — Loss heads
- `models/trajectory_loss.py` — Trajectory ranking loss function
- `evaluators/arc.py` — ARC evaluation with energy reranking
- `pretrain.py` — Training loop
- `data/build_arc_dataset.py` — Dataset preparation
