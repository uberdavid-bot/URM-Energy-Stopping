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
- Target config: 2 layers, hidden=64, 4 heads (to be confirmed in Phase 1)
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

## EBT Mode

```python
hidden = input_embeddings  # start from unprocessed input representation
for step in range(N):
    hidden = hidden.requires_grad_(True)      # (don't detach during training)
    energy = compute_joint_energy(input_embeddings, hidden)
    grad = autograd.grad(energy.sum(), hidden, create_graph=True)[0]
    grad = grad / grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    hidden = hidden - step_size * grad        # gradient descent on energy
    if energy_converged(energy, prev_energy): break
logits = lm_head(hidden)
```

Explicit energy minimization: each step follows the energy gradient downhill in hidden space. Energy convergence = principled stopping. Energy value used for pass@K ranking. **Implemented** via `ARCModelConfig.refinement="ebt"`.

### Why start from input_embeddings?
Starting from init_hidden (learned constant) means MCMC must discover the input structure from scratch via energy gradients alone. Starting from input_embeddings gives MCMC a representation that encodes *what the input is* without attention-based processing. This is analogous to the EBT paper providing context and only optimizing the prediction.

### Why hidden space, not soft-embedding space?
The previous implementation converted logits → softmax → embed_tokens to create "soft embeddings" for MCMC. This is wrong for our architecture because:
1. Collapses 128d to an 11-dimensional simplex (12 vocab tokens)
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
| Reconstruction (unrefined) | URM backbone / embeddings | Always: on logits before MCMC (or only logits if no MCMC) |
| Reconstruction (refined) | Energy head (via MCMC backprop) | EBT/hybrid mode, on logits after MCMC |

**Dual reconstruction loss** (0.5/0.5 weighting) is mandatory when training with MCMC: loss on both pre-MCMC and post-MCMC logits. The unrefined loss keeps the backbone learning cleanly. The refined loss trains the energy head through second-order gradients.

### Config fields
Three explicit fields control operational mode:
- **`refinement`**: `"urm"` | `"ebt"` | `"hybrid"` — what one step does.
- **`stopping`**: `"qhalt"` | `"energy"` — when to stop iterating.
- **`ranking`**: `"qhalt"` | `"energy"` — confidence signal for pass@K reranking.

Refinement modes (all do one step per `forward()` call):
- **"urm"**: One transformer pass (input re-injection + shared-weight layers). First-order only.
- **"ebt"**: One MCMC gradient step in hidden space. Second-order gradients via create_graph=True during training.
- **"hybrid"**: URM pass if `steps < mcmc_start_step`, else MCMC step. Transitions cleanly from URM to MCMC mid-sequence.

## Recurrence Structure

One config field controls recurrence:
- **`loops`**: Total recurrence steps before halting. The outer loop (pretrain.py) calls `ARCModel.forward()` repeatedly until `steps >= loops`.

Each `forward()` call does exactly one step of work (one transformer pass for URM, one gradient step for EBT). Per-step exact accuracy and delta norms are computed in the outer eval loop in pretrain.py, giving step_1..step_N metrics that reflect each recurrence step.

All three modes use the same halting logic and the same outer loop, making per-step metrics directly comparable across URM, EBT, and hybrid.

## Evaluator

Collects predictions across augmented puzzle versions, votes, computes pass@K. Reports both Q-halt-ranked and energy-ranked pass@K for direct comparison.

## File Map

- `models/urm/urm_energy.py` — Unified URM: config, forward (urm/ebt/hybrid modes), energy computation, MCMC refinement
- `models/losses.py` — Loss heads
- `evaluators/arc.py` — ARC evaluation with energy reranking
- `pretrain.py` — Training loop
- `data/build_arc_dataset.py` — Dataset preparation
