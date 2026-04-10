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

**R1d — 15×15 grids at h=128 exp=4** (harder problem, known-good model)
Date: 2026-04-09
Script: `scripts/train_r1_15x15.sh`
Config: `config/arch/urm_r1_15x15.yaml` — depth=2, hidden=128, 4 heads (head_dim=32), 8 recurrence steps, expansion=4, refinement="urm", stopping="qhalt", ranking="qhalt", puzzle_emb_ndim=128, batch 256 (VRAM constraint: 22GB at batch 512 too tight for 24GB card), 15×15 grids (225 tokens), ~10K training steps (1050 epochs), 5 eval checkpoints (every 210 epochs), lr=3e-4, stablemax_cross_entropy, bfloat16, EMA 0.999.

Dataset: `data/arc1concept-aug-1000-size-15/` — 563/960 tasks included (58.6% pass grid size filter), 507K puzzle identifiers.

Hypothesis: The h=128 exp=4 model converges in 1-2 steps on 10×10 because the problem is too easy, not because the model is too big. 15×15 grids have 2.25× more tokens and quadratically more spatial relationships, which should require more recurrence passes to resolve. This will create the smooth step-by-step convergence curve needed for refinement experiments. Pivoting from "weaker model" to "harder problem" after R1a-R1c showed a sharp capacity cliff: anything below h=128/exp=4 produces flat per-step curves with near-zero pass@1.

Expected outcome: Per-step accuracy should show meaningful improvement between steps 4 and 8. Composite should be lower than the 10×10 baseline (0.482) but with a much better convergence profile. Delta norms should decrease gradually across steps rather than being flat from step 1.

Reference: R1a (h=64), R1b (h=96), R1c (h=128 exp=2) all showed that reducing model capacity creates a ceiling, not refinement headroom.

### Result
**0.0% pass@1, 530K params, too hard.** Wandb: `R1d-15x15-h128-260409` ([link](https://wandb.ai/uberdavid-personal/arcagi/runs/2z1dha2n))

Per-step accuracy: 53.6% → 52.9% → 53.2% → 54.6% → 56.1% → **56.9%** → 56.5% → 55.1% (steps 1-8). The curve shows some improvement (~4% from step 1 to peak at step 6) but still very modest and with a decline at steps 7-8. Delta norms are larger than 10×10 (0.010-0.017 vs 0.003-0.004) indicating the model is working harder per step.

pass@K (Q-halt): 0% @1, 0.5% @2, 0.8% @5, 1.1% @10, 4.2% @100, 6.6% @1000.
pass@K (energy): 0% @1, 0% @10, 0.5% @100, 5.6% @1000.
Token-level accuracy: 55.1%. Exact puzzle accuracy: 0.07%.

**Conclusion:** 15×15 is **too hard** — near-zero pass@1 and much lower token accuracy than 10×10 (55% vs 61%). The problem difficulty jumped too far. The small per-step improvement (4%) is encouraging directionally but insufficient for our needs. Need to search between 10×10 (too easy) and 15×15 (too hard) — grid sizes 11-14.

**Next:** Exhaustive sweep of grid sizes 11, 12, 13, 14 with same h=128/exp=4 model to find the sweet spot.

### R1 Grid Size Sweep — h=128, exp=4, depth=2, 4 heads, 8 loops, batch 512, ~10K steps each

All runs use `config/arch/urm_r1_15x15.yaml` (h=128/exp=4/depth=2) with `scripts/train_r1_grid_sweep.sh`.

**R1-grid10 (10×10, confirmation):**
Wandb: `R1-grid10-h128-260409` ([link](https://wandb.ai/uberdavid-personal/arcagi/runs/1nh7wwjr))
epochs=3950, eval_interval=790, 294 groups, seq=100.
Per-step accuracy: 58.2% → 58.3% → 59.1% → 60.0% → **60.3%** → 59.9% → 59.2% → 58.5% (steps 1-8). Flat — only 2.1% variation. Delta norms ~0.003.
pass@K (Q-halt): 0% @1, 2.6% @2, 8.4% @5, 11.0% @10, 14.9% @100, 21.4% @1000.
**Conclusion:** Baseline confirmed post-refactor. Model converges by step 1-2 on 10×10. Too easy.

**R1-grid11 (11×11):**
Wandb: `R1-grid11-h128-260409` ([link](https://wandb.ai/uberdavid-personal/arcagi/runs/koy26yun))
epochs=3520, eval_interval=704, 329 groups, seq=121.
Per-step accuracy: 63.2% → 62.7% → 62.7% → 62.7% → 63.0% → 63.5% → **63.7%** → 63.4% (steps 1-8). Flat — only 1% variation. Delta norms ~0.004-0.006.
pass@K (Q-halt): 0% @1, 0.6% @2, 1.7% @5, 5.6% @10, 14.6% @100, 19.7% @1000.
**Conclusion:** Still too easy. Converges by step 1.

**R1-grid12 (12×12):**
Wandb: `R1-grid12-h128-260409` ([link](https://wandb.ai/uberdavid-personal/arcagi/runs/doxywq6v))
epochs=2885, eval_interval=577, 399 groups, seq=144.
Per-step accuracy: 60.4% → 59.5% → 59.3% → 60.1% → 61.1% → 61.6% → **61.7%** → 61.2% (steps 1-8). 2.4% variation. Delta norms ~0.003-0.005.
pass@K (Q-halt): 0.9% @1, 1.3% @2, 2.6% @5, 6.6% @10, 12.7% @100, 15.8% @1000.
**Conclusion:** Still too easy. Slight dip at steps 2-3 then recovery — hints of needing recurrence but variation too small.

**R1-grid13 (13×13):**
Wandb: `R1-grid13-h128-260410` ([link](https://wandb.ai/uberdavid-personal/arcagi/runs/gkglg4x5))
epochs=2595, eval_interval=519, 446 groups, seq=169.
Per-step accuracy: 60.1% → 59.7% → 60.1% → 60.9% → **61.2%** → 61.0% → 60.8% → 60.5% (steps 1-8). 1.4% variation. Delta norms ~0.004-0.007.
pass@K (Q-halt): 0.7% @1, 1.1% @2, 2.9% @5, 4.3% @10, 8.0% @100, 9.4% @1000.
**Conclusion:** Still flat. pass@K declining with grid size (harder puzzles, fewer solved) but per-step curve stays flat.

**R1-grid14 (14×14):**
Wandb: `R1-grid14-h128-260410` ([link](https://wandb.ai/uberdavid-personal/arcagi/runs/ma32mka7))
epochs=2375, eval_interval=475, 487 groups, seq=196.
Per-step accuracy: 58.8% → 58.1% → 58.3% → 59.2% → 60.0% → **60.3%** → 60.1% → 59.6% (steps 1-8). 2.1% variation. Delta norms ~0.004-0.007.
pass@K (Q-halt): 0% @1, 0% @2, 2.0% @5, 3.3% @10, 7.2% @100, 9.5% @1000.
**Conclusion:** Same pattern — flat curve, declining pass@K.

### Grid Sweep Summary

| Grid | seq | Token acc | Step variation | Peak step | pass@1 | pass@100 |
|------|-----|-----------|---------------|-----------|--------|----------|
| 10×10 | 100 | 58.5% | 2.1% | 5 | 0.0% | 14.9% |
| 11×11 | 121 | 63.4% | 1.0% | 7 | 0.0% | 14.6% |
| 12×12 | 144 | 61.2% | 2.4% | 7 | 0.9% | 12.7% |
| 13×13 | 169 | 60.5% | 1.4% | 5 | 0.7% | 8.0% |
| 14×14 | 196 | 59.6% | 2.1% | 6 | 0.0% | 7.2% |
| 15×15 | 225 | 55.1% | 3.8% | 6 | 0.0% | 4.2% |

**Key finding:** Per-step variation is consistently 1-4% across all grid sizes 10-15 at h=128/exp=4. The model converges in 1-2 recurrence steps regardless of problem difficulty. Increasing grid size makes the problem harder (pass@K drops steadily) but does NOT create the multi-step convergence curve needed for refinement experiments.

**Interpretation:** The flat per-step curve appears to be a fundamental property of this architecture at h=128, not a problem difficulty issue. The URM recurrence reaches its attractor after 1-2 passes through shared-weight transformer layers, and additional passes provide negligible improvement. The 15×15 result shows the largest variation (3.8%) but still insufficient.

**Next steps to consider:**
1. **Depth=1 instead of depth=2** — fewer layers per recurrence step forces more reliance on repeated iteration. This is the most direct way to make each step "weaker."
2. **Remove input re-injection** — without re-injecting input_embeddings at each step, the model must carry all information in hidden states, potentially requiring more steps to converge.
3. **Accept the flat curve** and proceed to R2/R3 anyway — the energy head comparison may still reveal differences in stopping/ranking even if per-step accuracy is flat.

**R1-full — 80K steps on 10×10 (proper training budget)**
Date: 2026-04-10
Script: `scripts/train_r1_full.sh`
Config: h=128, exp=4, depth=2, 4 heads, 8 loops, 10×10 grids, batch 512, **80K training steps** (31590 epochs), 10 eval checkpoints (every 3159 epochs), lr=3e-4 with warmup=800, lr_min_ratio=0.1, EMA 0.999.

Hypothesis: The 10K-step grid sweep runs were undertrained. With cosine LR decay over only 10K steps, the LR had already decayed to near-minimum before the model finished learning — we were measuring convergence of an underfitted model, not a trained one. 80K steps gives the model a proper training budget (the original Exp 0 baseline used similar scale). The per-step convergence profile of a fully-trained model may look very different from the flat curves we saw at 10K steps.

Expected outcome: Higher pass@1 (Exp 0 baseline was 15.9% composite), clearer per-step convergence curve once the model is properly trained. This is the actual R1 measurement that should have been done first.

### Result
**25.3% pass@1, 85.9% token accuracy, 16.3% exact accuracy.** Wandb: `R1-full-h128-10x10-260410` ([link](https://wandb.ai/uberdavid-personal/arcagi/runs/vimsv39t))

Per-step accuracy: **85.9%** → 85.8% → 85.6% → 85.4% → 85.4% → 85.7% → 85.8% → 85.9% (steps 1-8). **0.6% variation — completely flat.** The model peaks at step 1 and step 8 equally. Delta norms show a U-shape (0.009 → 0.005 → 0.008) suggesting the model oscillates around a fixed point rather than converging toward one.

pass@K (Q-halt): 25.3% @1, 37.0% @2, 49.4% @5, 51.9% @10, 59.7% @100, 64.9% @1000.
pass@K (energy): 0% @1, 7.1% @100, 40.9% @1000. Energy ranking far worse than Q-halt.

**Conclusion:** With proper training (80K steps), the model is much stronger (25.3% pass@1 vs 0% at 10K steps, 85.9% token acc vs 58%) but the per-step convergence curve is even flatter (0.6% vs 2.1%). This confirms definitively that at h=128/depth=2, URM recurrence converges in a single pass. The 10K-step runs weren't showing a flat convergence curve due to undertrained models — the flat curve is the ground truth.

This is the correct R1 baseline. The model is strong and fully trained, but there is no multi-step convergence to exploit for refinement experiments.

---

### Experiment R1e — expansion=2 full training
Date: 2026-04-10
Script: `scripts/train_r1e_exp2_full.sh`
Config: depth=2, h=128, 4 heads, expansion=2, 8 steps, batch 512, 10×10 grids, 80K steps (31590 epochs), constant LR (3e-4) after 100-step warmup, EMA 0.999. ~330K transformer params (vs ~530K at exp=4). eval_interval=2106 epochs (15 checkpoints).

Hypothesis: The full 80K baseline at exp=4 converges in 1 step with 85.9% token accuracy. Expansion=2 has ~38% fewer params (mostly MLP). With full training duration and constant LR, we want to determine: (a) does expansion=2 reach high enough accuracy for exact puzzle solves, and (b) does a multi-step convergence curve emerge at any point during training. The per-step accuracy and delta norm logs across the full run will show exactly when (if ever) step 1 stops being sufficient and later steps start contributing.

Expected outcome: Three possible outcomes: (1) multi-step convergence emerges mid-training then collapses as the model gets stronger — this tells us the right training duration for refinement experiments. (2) Multi-step convergence persists at end of training — we've found our right-sizing. (3) Flat curve throughout, same as exp=4 — expansion=2 still converges in 1 step and we need to reduce capacity further.

Reference: R1c (exp=2, 10K cosine) was undertrained. R1-full (exp=4, 80K) proved full training is needed for ground truth.

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
