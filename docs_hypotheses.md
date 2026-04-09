# Experiment Hypotheses Log

## Prior Experiments (archived — conducted under previous project framing)

These experiments used an over-parameterized model (hidden=128) on 10×10 grids where URM converged in 1-2 steps. MCMC was implemented incorrectly in soft-embedding space with detached gradients between steps. Results are preserved for reference but should not directly inform expectations for the redesigned experiments.

- **Exp 0 (Baseline URM, hidden=128)**: 15.9% composite at 10K steps, 48.2% converged. Strong baseline, but model converges too fast for refinement to help.
- **Exp 1 (Contrastive-only energy)**: Energy gap collapsed to ~0. Contrastive loss alone is insufficient.
- **Exp 2 (DSM)**: Superseded. DSM trains on wrong distribution; unnecessary given tractable second-order gradients.
- **Exp 3a (MCMC refined-only loss)**: 0% accuracy. Refined-only reconstruction loss destroys URM learning. Dual loss is mandatory.
- **Exp 3b (MCMC dual loss)**: mcmc_improvement=0. But MCMC was in soft-embedding space with detached gradients — two critical bugs.
- **Exp 4 (Trajectory supervision)**: Energy gap stable at 0.85 but trajectory quality spread collapsed as URM converged too fast.

See `docs_hypotheses_archived.md` for full details of prior experiments.

---

## New Experiment Series — Implicit vs Explicit Iterative Refinement

### Experiment R1 — Right-sized baseline (find the scale)
Date: TBD
Config: depth=2, hidden=64, 4 heads, 8 recurrence steps, H_cycles=1, L_cycles=1, batch 512, 10×10 grids, 10K training steps
Hypothesis: Reducing hidden from 128 to 64 (~4x fewer transformer params) will make the model need most of its 8 steps to converge on 10×10 grids, providing headroom for refinement experiments.
Expected outcome: Per-step accuracy should show meaningful improvement between steps 4 and 8. If the model plateaus by step 2-3 (as with hidden=128), we need to go smaller (hidden=48 or hidden=32).
Key measurements: per-step accuracy curve, total VRAM usage, throughput (it/s), pass@K at 10K steps.
Risk: hidden=64 may be too small for the transformer to learn ARC patterns at all, or too large and still converges fast.

### Result
TBD

---

### Experiment R2 — Hidden-space MCMC (fix the implementation)
Date: TBD
Config: Same backbone as R1. Add energy head. MCMC in hidden space, no detach between steps during training, create_graph=True, dual reconstruction loss, N MCMC steps.
Hypothesis: With MCMC correctly implemented in hidden space (not soft-embedding space) and with gradient flow across steps (not detached), the energy head will learn to produce useful gradients that improve predictions.
Expected outcome: mcmc_improvement > 0 (refined predictions better than unrefined). Energy gap maintained (no collapse). VRAM should be well within 24GB budget at hidden=64.
Inspired by: EBT paper's optimization-based training (Algorithm 1). Fixes two critical bugs from Exp 3b.
Risk: Second-order gradients through multiple non-detached MCMC steps may cause training instability. Energy landscape may still not be useful for discrete reasoning.

### Result
TBD

---

### Experiment R3 — Core comparison: URM vs EBT
Date: TBD
Config: Two models sharing the same backbone from R1:
- **URM model**: N recurrence steps (transformer + input re-injection), Q-halt stopping, Q-halt pass@K ranking.
- **EBT model**: N MCMC steps in hidden space starting from input_embeddings (no URM steps), energy convergence stopping, energy pass@K ranking.
Both trained for 10K steps with matched compute budgets.
Hypothesis: Pure EBT (explicit energy minimization) will underperform URM (implicit recurrence) on ARC reasoning because attention-based processing provides inductive biases (spatial relationships, pattern matching) that scalar energy gradients cannot replicate. However, energy-based pass@K reranking may outperform Q-halt reranking even if per-prediction accuracy is lower.
Expected outcome: URM accuracy > EBT accuracy at matched steps. Energy pass@K ranking may beat Q-halt ranking. Energy convergence may correlate better with prediction correctness than Q-halt confidence.
Inspired by: EBT paper Section 2.1 — verification generalizes better than generation. Even if EBT generates worse predictions, its energy function may be a better verifier.
Risk: Pure EBT may fail catastrophically on ARC (0% accuracy), making comparison meaningless. In that case, proceed to R4.

### Result
TBD

---

### Experiment R4 — Hybrid: URM initialization + MCMC refinement (contingent on R3)
Date: TBD
Config: M URM steps + (N-M) MCMC steps, where M+total_MCMC = N (matched compute with R3). Try M=2 as starting point.
Hypothesis: URM provides a good initialization via attention-based processing, then MCMC refinement in hidden space provides complementary explicit optimization that improves predictions beyond what additional URM steps would achieve.
Expected outcome: Hybrid accuracy > pure URM accuracy at matched total steps. The URM trajectory (steps 1→M) provides natural training signal for the energy landscape via trajectory supervision.
Inspired by: All prior experiments showed MCMC adds nothing on top of *fully converged* URM. With M=2, URM is far from converged, giving MCMC genuine room to contribute.
Risk: MCMC refinement may just be a worse version of additional URM steps — same hidden space, similar update direction, but without attention's inductive bias.

### Result
TBD

---

## Lessons Carried Forward
1. **Dual reconstruction loss is mandatory** for MCMC training (Exp 3a).
2. **Contrastive loss alone causes energy collapse** (Exp 1). Use reconstruction-through-MCMC as primary energy training signal.
3. **Right-size the model for the problem.** The model must struggle enough that refinement has room to help.
4. **MCMC must operate in hidden space**, not soft-embedding space.
5. **Don't detach between MCMC steps during training.** Energy head needs multi-step trajectory gradients.
6. **Budget in steps, not epochs.**
