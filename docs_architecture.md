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

## Refinement Modes

All modes use `forward_trajectory(batch, N)` which runs N recurrence steps in a single call with **full gradient flow** (no detach between steps). Training uses deep supervision: every step gets a weighted reconstruction loss.

### URM Mode (`refinement="urm"`)
```python
# forward_trajectory runs N steps:
for step in range(N):
    hidden = hidden + input_embeddings        # re-inject input
    hidden = transformer_layers(hidden)       # shared weights
    logits_t = lm_head(hidden)
    q_logits_t = q_head(hidden[:, 0])
# Returns: all_logits, all_q_logits, all_hidden, input_embeddings
```
Implicit energy minimization: each transformer pass pushes hidden toward a fixed point. No detach between steps — gradients flow through the full trajectory for deep supervision.

### EBT Mode (`refinement="ebt"`)
```python
# forward_trajectory runs N steps:
for step in range(N):
    energy = compute_joint_energy(input_embeddings, hidden)
    grad = autograd.grad(energy.sum(), hidden, create_graph=True)[0]
    hidden = hidden - step_size * normalized(grad)
    logits_t = lm_head(hidden)
```
Explicit energy minimization: each step follows the energy gradient in hidden space. create_graph=True during training for second-order gradient flow into the energy head. Detach between steps at inference only.

### Hybrid Mode (`refinement="hybrid"`)
```python
# forward_trajectory runs N steps:
for step in range(N):
    if step < mcmc_start_step:
        hidden = hidden + input_embeddings
        hidden = transformer_layers(hidden)
    else:
        grad = autograd.grad(energy.sum(), hidden, create_graph=True)[0]
        hidden = hidden - step_size * normalized(grad)
    logits_t = lm_head(hidden)
```
M URM steps then (N-M) MCMC steps, controlled by `mcmc_start_step`. Transitions cleanly mid-trajectory.

Pure EBT (N MCMC steps, no URM) is supported via `refinement="ebt"`. Hybrid is the primary comparison (R3).

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

### Gradient flow across steps
During training, `forward_trajectory()` does NOT detach hidden between steps for any mode. Gradients flow through the full N-step trajectory. For MCMC steps, `create_graph=True` enables second-order gradient flow into the energy head.

During inference (EBT/hybrid only), MCMC steps detach between iterations to save memory.

## Training

### Deep Supervision

Training always runs N fixed steps with deep supervision. At every step t, the loss is:
- **Reconstruction**: `loss_fn(logits_t, labels)` weighted by `(t+1)/N` (linear ramp — later steps matter more)
- **Q-halt**: `BCE(q_head(hidden_t), correct_t)` weighted by the same linear ramp

The final loss is `lm_loss + 0.5 * qhalt_loss`, where both are normalized by weight sum.

```python
for t in range(N):
    w = (t + 1) / N
    total_recon += w * loss_fn(logits_t, labels)
    total_qhalt += w * BCE(q_logits_t, correct_t)
lm_loss = total_recon / weight_sum
total_loss = lm_loss + 0.5 * (total_qhalt / weight_sum)
```

Stopping criteria (Q-halt convergence, energy convergence) are **eval-only** — computed post-hoc on the full trajectory, never used during training.

### Losses

| Loss | What it trains | When used |
|------|---------------|-----------|
| Deep supervision (per-step reconstruction) | URM backbone / embeddings | Every step, weighted by (t+1)/N |
| Q-halt BCE (per-step) | q_head | Every step, weighted by (t+1)/N |
| Trajectory ranking | Energy head (first-order) | R2+: all-pairs ranking across trajectory |

For MCMC modes (EBT/hybrid), the reconstruction loss at MCMC steps trains the energy head through second-order gradients (create_graph=True).

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
- **`refinement`**: `"urm"` | `"ebt"` | `"hybrid"` — what one step does.
- **`stopping`**: `"qhalt"` | `"energy"` — when to stop iterating.
- **`ranking`**: `"qhalt"` | `"energy"` — confidence signal for pass@K reranking.

Training requirements by refinement mode:
- **"urm"**: Deep supervision (reconstruction + Q-halt) at every step. First-order only.
- **"urm" + trajectory energy (R2, planned)**: Deep supervision + trajectory ranking loss. First-order only. Co-trained.
- **"ebt"**: Deep supervision. Second-order gradients via create_graph=True.
- **"hybrid" (R3, planned)**: Deep supervision + trajectory ranking. URM then MCMC (controlled by `mcmc_start_step`). Second-order gradients through MCMC phase.

## Energy Reranking (R2.5 — eval only)

At inference, energy can be used as a post-hoc verifier without MCMC:
1. Run URM with multiple seeds/augmentations → K candidate predictions
2. Score each candidate: `energy = compute_joint_energy(input_emb, hidden_final)`
3. Rank candidates by energy (lower = better)
4. Compare pass@K to Q-halt ranking

This requires no MCMC at inference — just one forward pass through the energy head per candidate.

## Architecture Simplification

`forward_trajectory(batch, N)` runs the full N-step recurrence in a single call. No carry state, no per-sample halting, no outer loop. Deep supervision provides per-step learning signal through undetached gradient flow. The carry-based outer loop and ModelCarry dataclass have been removed entirely. Stopping criteria are eval-only metrics computed post-hoc on the full trajectory.

## Evaluator

Collects predictions across augmented puzzle versions, votes, computes pass@K. Reports both Q-halt-ranked and energy-ranked pass@K for direct comparison.

## File Map

- `models/urm/urm_energy.py` — Unified model: config, forward_trajectory (urm/ebt/hybrid modes), energy computation, MCMC step
- `models/losses.py` — EnergyLossHead: deep supervision, per-step metrics, eval stopping metrics
- `models/trajectory_loss.py` — Trajectory ranking loss function
- `evaluators/arc.py` — ARC evaluation with energy reranking
- `pretrain.py` — Training loop
- `data/build_arc_dataset.py` — Dataset preparation
