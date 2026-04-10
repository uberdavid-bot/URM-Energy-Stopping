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

**R1a — hidden=64** (too small)
Date: 2026-04-09
Script: `scripts/train_r1_scale.sh`
Config: `config/arch/urm_energy_r1.yaml` — depth=2, hidden=64, 4 heads, 8 recurrence steps, expansion=4, H_cycles=1, L_cycles=1, mode="urm", puzzle_emb_ndim=64, batch 512, 10×10 grids, ~10K training steps (4000 epochs), 5 eval checkpoints (every 800 epochs), lr=3e-4, stablemax_cross_entropy, bfloat16, EMA 0.999.

Result: **0.6% pass@1** — model too small to learn ARC patterns on 10×10 grids. ~130K transformer params insufficient for the task. Need to increase hidden size.

**R1b — hidden=96** (next trial)
Date: TBD
Script: `scripts/train_r1_h96.sh`
Config: `config/arch/urm_r1_h96.yaml` — depth=2, hidden=96, 4 heads (head_dim=24), 8 recurrence steps, expansion=4, puzzle_emb_ndim=96, otherwise same as R1a. ~300K transformer params — between h64 (too small) and h128 (converges too fast, 15.9% composite from Exp 0).
Hypothesis: hidden=96 provides enough capacity to learn ARC patterns while still needing most of the 8-step budget. If it converges in 1-2 steps like h128, we have our answer: the sweet spot is closer to h64 and the problem is capacity, not convergence speed.
Expected outcome: Per-step accuracy should show meaningful improvement between steps 4 and 8. pass@1 should be significantly above 0.6%.
Key measurements: per-step accuracy curve, pass@K, VRAM, throughput.
Risk: h96 may still converge too fast (like h128), or still be too small (like h64).

### Result
TBD

**R1c — expansion=2 at h=128** (reduce MLP, keep representation width)
Date: 2026-04-09
Script: `scripts/train_r1_exp2.sh`
Config: `config/arch/urm_r1_exp2.yaml` — depth=2, hidden=128, 4 heads (head_dim=32), 8 recurrence steps, expansion=2 (down from 4), refinement="urm", stopping="qhalt", ranking="qhalt", puzzle_emb_ndim=128, batch 512, 10x10 grids, ~10K training steps (4000 epochs), 5 eval checkpoints (every 800 epochs), lr=3e-4, stablemax_cross_entropy, bfloat16, EMA 0.999.

Hypothesis: Reducing expansion from 4 to 2 cuts ~38% of transformer params (mostly MLP) while preserving the h=128 representation width and depth=2 attention structure that we know works (Exp 0: 0.482 composite). This should degrade capacity enough that the model needs more recurrence steps to converge, without falling off the capacity cliff we saw at h=96 (0.016 composite). The MLP intermediate size lands at 256 due to `_find_multiple` rounding (`round(2 * 128 * 2/3) = 171` → rounded up to 256).

Expected outcome: Per-step accuracy should show meaningful improvement between steps 4 and 8. Composite should be between h=96 result (0.016) and h=128 baseline (0.482).

Param budget: ~330K transformer params. MLP per layer: 128×512 (gate_up) + 256×128 (down) + dwconv ≈ 98K. Attention per layer: 4×(128×128) = 65K. Total per layer ≈ 163K, ×2 layers ≈ 330K.

Reference: R1a (h=64, 0.6% pass@1) and R1b (h=96, 0.016 composite) both capacity-limited. Expansion knob targets MLP which is 75% of params at expansion=4.

### Result
**0.0% pass@1, 332K params.** Wandb: `R1c-exp2-h128-260409` ([link](https://wandb.ai/uberdavid-personal/arcagi/runs/3tax2qv5))

Per-step accuracy is essentially flat: 60.3% → 59.8% → 60.1% → 60.9% → 61.9% → **62.1%** → 61.7% → 61.1% (steps 1-8). Only ~2% variation across all steps — model converges immediately with no meaningful step-by-step improvement. Delta norms are tiny and roughly constant (~0.003-0.004).

pass@K (Q-halt): 0% @1, 1.9% @2, 5.2% @5, 9.7% @10, 16.9% @100, 23.4% @1000.
pass@K (energy): 0% @1, 0% @10, 0.6% @100, 13.0% @1000.
Token-level accuracy: 61.1%. Exact puzzle accuracy: 0.7%.

**Conclusion:** Expansion=2 at h=128 is **not the right-sizing sweet spot**. The model has decent token-level accuracy (~61%) but near-zero exact puzzle solve rate. Per-step curve is flat — no room for refinement to help. The capacity reduction went too far for puzzle solving while still not producing the convergence curve we need. The h=128 attention structure alone (without sufficient MLP capacity) can approximate per-token patterns but can't assemble them into correct full-puzzle solutions.

Key insight: hidden dim vs expansion may not be independent knobs. At h=128, the model's attention layers provide enough representational width to converge quickly regardless of MLP size. The recurrence convergence speed may be driven more by attention capacity than MLP capacity.

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
