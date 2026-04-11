# Research Strategy

## Core Research Question
For discrete abstract reasoning (ARC-AGI), is explicit energy-based refinement (EBT-style MCMC in hidden space) competitive with implicit iterative refinement (URM recurrence), and under what conditions does each approach have advantages?

## Framing
URM recurrence is *implicit* energy minimization — shared transformer weights applied repeatedly perform gradient-descent-like updates in hidden space, guided by re-injected input embeddings. Q-halt is a learned stopping detector bolted on top.

EBT-style refinement is *explicit* energy minimization — a learned scalar energy function E(input, hidden) provides gradients directly in hidden space, with energy convergence as a principled stopping criterion.

This project implements both approaches in a shared architecture (same backbone, same embeddings, same lm_head) and compares them with matched compute budgets. The goal is not to prove one is universally better, but to characterize when explicit energy adds value over implicit recurrence for discrete reasoning tasks.

## Current Best Result
**R1h URM with deep supervision (d=1, h=64, exp=2)**: 5.2% pass@1, 78.9% token accuracy, 3.76% exact accuracy (peak at step 6) at 80K steps on 10×10 grids. 35K params, 8 recurrence steps. Per-step convergence is monotonically increasing: 66.8% → 78.9% token accuracy step 1→6 (12.1% gain). This is the first run with cross-step gradient flow and deep supervision.

Prior baseline without deep supervision: R1-full URM (h=128, exp=4, depth=2) achieved 25.3% pass@1 / 85.9% token accuracy but with a completely flat per-step curve (0.6% variation). That model was 15× larger (530K params) and converged in 1 step. The R1h model is deliberately small so it needs all 8 steps.

## Experiment Sequence

### Phase 1: Find the Right Scale (R1) — COMPLETE
**Validated config**: depth=1, h=64, expansion=2, 8 recurrence steps, 10×10 grids, ~35K params. Uses `forward_trajectory()` with deep supervision (weighted per-step reconstruction + Q-halt loss, linear ramp `(t+1)/N`, no detach between steps).

R1h confirmed monotonic per-step improvement: 0.13% → 3.76% exact accuracy step 1→6 (29× gain). Delta norms decrease monotonically (0.008 → 0.0009). This resolves the Phase 1 prerequisite — multi-step convergence exists, and trajectory quality spread is sufficient for energy training.

Key finding: The flat per-step curves in R1a–R1g were caused by `.detach()` between carry-based steps, not by model capacity or problem difficulty. Deep supervision + cross-step gradient flow is the fix. The TRM paper (Jolicoeur-Martineau, 2025) predicted this — they found deep supervision to be the single largest contributor to recurrence benefit.

### Phase 2: Train Energy as Verifier via Trajectory Supervision (R2)
Co-train the energy head alongside URM using trajectory ranking loss. No MCMC, no second-order gradients — first-order only.

**Why trajectory supervision, not MCMC-based training:**
- MCMC improvement > 0 is the hardest thing to achieve and has the most uncertainty.
- Trajectory ranking is supervised learning — straightforward and stable.
- Exp 4 proved the concept works when quality spread exists; Exp 4 failed only because the over-parameterized model converged too fast (quality spread collapsed). R1 fixes this prerequisite.
- Co-training (not sequential) is required because quality spread is highest when the model is still learning. A frozen URM that already converges has no spread to supervise on.

**Trajectory ranking loss design (from Exp 4, refined):**
- All-pairs weighted margin loss across URM steps: quality_gap * F.relu(E(better) - E(worse) + margin)
- N steps → N*(N-1)/2 ordered pairs (120 pairs for N=16)
- Natural anti-collapse: 120+ pairs requiring different energy values makes collapse structurally hard
- No contrastive loss needed as primary signal (trajectory ranking subsumes it)

**Stabilization tricks from literature:**
- Log energy-accuracy Spearman correlation from step 0 as early diagnostic (IREM Fig. 5)
- Separate gradient clipping for energy head (max_norm=1.0) vs backbone (max_norm=5.0)
- Position-aware energy head (optional, try if correlation is weak after 2K steps)

Success criterion: Energy-accuracy correlation remains strong throughout training (not just first few K steps). Active pairs in trajectory ranking loss stay high.

### Phase 2.5: Energy Reranking + Stopping (R2.5) — EVAL ONLY
No new training. Use the R2-trained energy head to:
1. Rerank URM pass@K candidates by energy score. Compare to Q-halt ranking.
2. Use energy convergence as stopping criterion. Compare to Q-halt stopping.

This is a single eval script — score existing URM samples with the energy head.

Success criterion: Energy reranking improves pass@K over Q-halt on at least some K values. If so, this is the first publishable result: "learned energy verification beats learned halting for iterative reasoning."

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

## Key Architectural Decisions
- **Energy in hidden space**: E(input_embeddings, hidden) where both are [B, seq_len, hidden_dim]. No soft-embedding bridge.
- **Trajectory ranking loss** as primary energy training signal: dense, ordered, anti-collapse.
- **Single recurrence loop**: no H_cycles/L_cycles distinction. Every step gets gradients and evaluation.
- **Matched compute comparison**: always compare with equal total steps, never "free" extra compute.
- **Co-train energy head with URM**: energy head sees diverse trajectories while URM is still learning.

## Open Questions
- Does the energy landscape over hidden states learn useful structure for ARC puzzles when trained via trajectory ranking?
- Is energy-based reranking better than Q-halt reranking given a well-trained energy function?
- How many MCMC steps does it take to match one URM recurrence step? (compute efficiency)
- Does explicit energy help more on harder puzzles within the 10×10 distribution?
- Can MCMC gradients from a trajectory-trained energy head actually improve predictions, or is the energy function only useful as a verifier?
