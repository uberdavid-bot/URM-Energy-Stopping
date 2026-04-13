# Research Strategy

## Core Research Question
For discrete abstract reasoning (ARC-AGI), is explicit energy-based refinement (EBT-style MCMC in hidden space) competitive with implicit iterative refinement (URM recurrence), and under what conditions does each approach have advantages?

## Framing
URM recurrence is *implicit* energy minimization — shared transformer weights applied repeatedly perform gradient-descent-like updates in hidden space, guided by re-injected input embeddings. Q-halt is a learned stopping detector bolted on top.

EBT-style refinement is *explicit* energy minimization — a learned scalar energy function E(input, hidden) provides gradients directly in hidden space, with energy convergence as a principled stopping criterion.

This project implements both approaches in a shared architecture (same backbone, same embeddings, same lm_head) and compares them with matched compute budgets. The goal is not to prove one is universally better, but to characterize when explicit energy adds value over implicit recurrence for discrete reasoning tasks.

## Current Best Result
**R2c URM with position-aware energy co-training + dropout (d=1, h=64, exp=2)**: 4.5% pass@1, 79.9% token accuracy, 6.95% exact accuracy (peak at step 6) at 80K steps on 10×10 grids. 37K params, 8 recurrence steps. Q-halt pass@100: 26.0%, pass@1000: 29.2%. Per-step convergence: 67.3% → 79.9% token accuracy step 1→6.

Prior baselines: R1h (no dropout, no energy): 3.76% eval exact, 5.2% pass@1. R1i (dropout=0.1, no energy): 5.33% eval exact, 5.2% pass@1. The energy co-training acts as multi-task regularization that improves the backbone even though energy ranking itself fails.

Original baseline without deep supervision: R1-full URM (h=128, exp=4, depth=2) achieved 25.3% pass@1 / 85.9% token accuracy but with a completely flat per-step curve (0.6% variation). That model was 15× larger (530K params) and converged in 1 step.

## Experiment Sequence

### Phase 1: Find the Right Scale (R1) — COMPLETE
**Validated config**: depth=1, h=64, expansion=2, 8 recurrence steps, 10×10 grids, ~35K params. Uses `forward_trajectory()` with deep supervision (weighted per-step reconstruction + Q-halt loss, linear ramp `(t+1)/N`, no detach between steps).

R1h confirmed monotonic per-step improvement: 0.13% → 3.76% exact accuracy step 1→6 (29× gain). Delta norms decrease monotonically (0.008 → 0.0009). This resolves the Phase 1 prerequisite — multi-step convergence exists, and trajectory quality spread is sufficient for energy training.

Key finding: The flat per-step curves in R1a–R1g were caused by `.detach()` between carry-based steps, not by model capacity or problem difficulty. Deep supervision + cross-step gradient flow is the fix. The TRM paper (Jolicoeur-Martineau, 2025) predicted this — they found deep supervision to be the single largest contributor to recurrence benefit.

### Phase 2: Train Energy as Verifier via Trajectory Supervision (R2) — COMPLETE, FAILED
Co-trained energy head alongside URM using trajectory ranking loss. First-order only.

**Result: Within-trajectory ranking succeeds, cross-input ranking fails.**
- R2 (no dropout): Train Spearman -1.0, eval Spearman -0.48. Energy pass@K near zero.
- R2b (dropout backbone): Eval Spearman improved to -0.585. Energy pass@K still near zero.
- R2c (position-aware MLP): Eval Spearman collapsed to -0.069. Energy pass@K still near zero. BUT: best URM eval exact (6.95%) and Q-halt pass@K (26.0% @100, 29.2% @1000) — energy co-training helps backbone as regularizer.
- Eval-only ranking comparison (energy drop, best-step energy): all strategies fail equally.

**Root cause:** Trajectory ranking teaches step ordering within a single puzzle's trajectory (step 6 > step 1). This doesn't transfer to ranking predictions across different puzzles. The energy head learns puzzle-specific hidden state patterns, not abstract prediction quality.

**Positive finding:** Energy co-training (especially position-aware R2c) significantly improves the backbone via multi-task regularization, even though the energy ranking signal itself doesn't generalize.

### Phase 2.5: Energy Reranking + Stopping (R2.5) — BLOCKED
Energy reranking cannot beat Q-halt with current training signal. The R2 series gate for R2.5 was not met. R2.5 as originally designed (eval-only comparison) was partially done via `scripts/eval_energy_ranking.py` on R2b — confirmed failure at all K values.

### Phase 3: MCMC Refinement (R3) — CONDITIONAL ON R2.5
Now that the energy landscape has validated structure, test whether its gradients can drive refinement.

Trade off URM steps for MCMC steps: M URM + K MCMC at matched total compute (M+K = N). Start with M = N-2 (most URM, little MCMC) and decrease M.

This is where second-order gradient tricks become relevant:
- Langevin noise during training: `hidden = hidden - step_size * grad + noise_scale * randn_like(hidden)` (drop noise at inference)
- Step size annealing across MCMC trajectory: `step_size * (1 - t/T)` (IRED-inspired coarse-to-fine)
- Separate gradient clipping for energy head
- No detach between MCMC steps during training (create_graph=True)

Success criterion: mcmc_improvement > 0 (refined predictions better than unrefined at same checkpoint). Hybrid accuracy > pure URM accuracy at matched total steps.

### Phase 3b: Pure EBT Ablation (R3b) — OPTIONAL
Pure EBT (all MCMC, no URM steps) only if R3 shows MCMC refinement works. Use 1 URM step as initialization (counted in total budget) since cold start from input_embeddings is extremely unlikely to work on discrete ARC grids. This is an ablation for the paper, not a primary experiment.

## Dead Ends (from prior experiments)
- **Contrastive loss as sole energy training signal**: energy collapse to trivial solution (Exp 1). Trajectory ranking subsumes contrastive and doesn't collapse.
- **Refined-only reconstruction loss with MCMC**: destroys URM learning (Exp 3a)
- **MCMC on top of fully-converged URM**: no room for improvement (Exp 3b, Exp 4)
- **Over-parameterized model on easy grids**: converges too fast for any refinement to help. Quality spread collapses → trajectory signal dies.
- **DSM (denoising score matching)**: unnecessary given tractable second-order gradients, trains on wrong distribution
- **Sequential training (freeze URM, then train energy)**: quality spread only exists during URM learning. Co-training is required.
- **Trajectory ranking for cross-input energy verification** (R2/R2b/R2c): Within-trajectory ordering doesn't transfer to cross-input ranking. Energy head memorizes puzzle-specific patterns. More capacity (position-aware MLP) makes eval Spearman worse, not better.
- **Mean pooling energy head** (R2/R2b): destroys spatial information. But position-aware heads (R2c) didn't fix ranking either — the problem is the training signal, not the architecture.

## Key Architectural Decisions
- **Energy in hidden space**: E(input_embeddings, hidden) where both are [B, seq_len, hidden_dim]. No soft-embedding bridge.
- **Trajectory ranking loss** as primary energy training signal: dense, ordered, anti-collapse.
- **Single recurrence loop**: no H_cycles/L_cycles distinction. Every step gets gradients and evaluation.
- **Matched compute comparison**: always compare with equal total steps, never "free" extra compute.
- **Co-train energy head with URM**: energy head sees diverse trajectories while URM is still learning.

## Open Questions
- **Can MCMC (R3) force generalizable energy features?** MCMC generates diverse hidden states beyond fixed URM trajectories. This changes the energy head's training distribution. The energy gradient cosine similarity is weakly negative on train (-0.10), suggesting some directional signal exists.
- **Is cross-puzzle contrastive training needed?** Trajectory ranking only teaches within-puzzle ordering. A loss that compares hidden states *across* different puzzles might force the energy head to learn abstract quality features.
- **Can energy co-training be improved as a regularizer?** R2c accidentally discovered that energy co-training improves backbone quality (6.95% vs 5.33% eval exact). Can this be amplified — e.g., higher energy_loss_weight, different energy head architectures optimized for regularization rather than ranking?
- **Is Q-halt fundamentally better than energy for verification?** Q-halt is trained per-sample (binary: is this exact sequence correct?) while energy is trained per-trajectory (ranking: which step is better?). Q-halt's per-sample signal may simply be more informative.
- **Should the energy head have separate parameters from backbone?** Shared backbone means the energy head sees features optimized for reconstruction, not verification. A dedicated verification network might learn different features.
