# Research Strategy

## Core Research Question
For discrete abstract reasoning (ARC-AGI), is explicit energy-based refinement (EBT-style MCMC in hidden space) competitive with implicit iterative refinement (URM recurrence), and under what conditions does each approach have advantages?

## Framing
URM recurrence is *implicit* energy minimization — shared transformer weights applied repeatedly perform gradient-descent-like updates in hidden space, guided by re-injected input embeddings. Q-halt is a learned stopping detector bolted on top.

EBT-style refinement is *explicit* energy minimization — a learned scalar energy function E(input, hidden) provides gradients directly in hidden space, with energy convergence as a principled stopping criterion.

This project implements both approaches in a shared architecture (same backbone, same embeddings, same lm_head) and compares them with matched compute budgets. The goal is not to prove one is universally better, but to characterize when explicit energy adds value over implicit recurrence for discrete reasoning tasks.

## Current Best Result
**Baseline URM (urm_small)**: 15.9% composite at 10K steps (hidden=128, depth=2).
This was with an over-parameterized model that converges in 1-2 recurrence steps on 10×10 grids.

## Phase 1: Find the Right Scale
Strip H_cycles/L_cycles to a single loop with per-step evaluation. Reduce model capacity until URM needs most of its step budget to converge.

Starting point: depth=2, hidden=64, 8 total steps, 10×10 grids.

Success criterion: URM accuracy should be meaningfully improving between steps 4 and 8, not plateauing at step 2. If hidden=64 still converges too fast, try hidden=48 or hidden=32.

## Phase 2: Fix MCMC Implementation
Two critical bugs in the current implementation:
1. **MCMC operates in soft-embedding space, not hidden space.** The current code does `softmax(logits) @ embed_tokens.weight` to create a "soft embedding," then takes energy gradients w.r.t. that. This collapses 128d hidden states to an 11-dimensional simplex (12 vocab tokens), and feeds lm_head an input distribution it was never trained on. Fix: take energy gradients w.r.t. hidden states directly, which is what compute_joint_energy already expects.
2. **Detached gradients between MCMC steps.** The current code calls `predicted_emb.detach().requires_grad_(True)` inside the MCMC loop, breaking the computational graph across steps. The energy head only learns from single-step gradients, not multi-step trajectories. Fix: make detach optional (default off for training, on for inference if memory-constrained).

## Phase 3: Core Comparison
Given N total steps of compute, compare:
- **N URM steps** (baseline): implicit refinement via recurrence, Q-halt stopping, Q-halt confidence for pass@K ranking.
- **N MCMC steps** (pure EBT): explicit refinement via energy gradient descent in hidden space, starting from input_embeddings (not init_hidden — give MCMC the input representation, just unprocessed by attention). Energy convergence stopping, energy confidence for pass@K ranking.

Both use the same backbone, energy head, and lm_head. The only difference is the refinement mechanism.

Metrics to compare: pass@K (all K values), composite metric, per-step accuracy curves, energy landscape quality (gap between correct/incorrect predictions).

## Phase 4: Hybrid (if pure EBT underperforms)
If pure EBT can't match URM (likely — ARC may require attention-based processing that gradients alone can't replicate), test:
- **M URM steps + (N-M) MCMC steps**: URM provides a reasonable initialization, MCMC refines explicitly.
- Trajectory supervision from the URM steps can provide additional training signal for the energy landscape.

This becomes the interesting result — explicit refinement as complement to implicit recurrence.

## Dead Ends (from prior experiments)
- **Contrastive loss as sole energy training signal**: energy collapse to trivial solution (Exp 1)
- **Refined-only reconstruction loss with MCMC**: destroys URM learning (Exp 3a)
- **MCMC on top of fully-converged URM**: no room for improvement (Exp 3b) — though soft-embedding bug may have contributed
- **Over-parameterized model on easy grids**: converges too fast for any refinement to help
- **DSM (denoising score matching)**: unnecessary given tractable second-order gradients, trains on wrong distribution

## Key Architectural Decisions
- **Energy in hidden space**: E(input_embeddings, hidden) where both are [B, seq_len, hidden_dim]. No soft-embedding bridge.
- **Dual reconstruction loss**: mandatory when training with MCMC. Unrefined + refined logits both get reconstruction loss.
- **Single recurrence loop**: no H_cycles/L_cycles distinction. Every step gets gradients and evaluation.
- **Matched compute comparison**: always compare with equal total steps, never "free" extra compute.

## Open Questions
- Does the energy landscape over hidden states learn useful structure for ARC puzzles?
- Is energy-based reranking better than Q-halt reranking given a well-trained energy function?
- How many MCMC steps does it take to match one URM recurrence step? (compute efficiency)
- Does explicit energy help more on harder puzzles within the 10×10 distribution?
