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

### Critical bug fix: inner/outer loop simplification (2026-04-10/11)

Experiments R1a–R1g used `loops=8` which controlled BOTH the inner recurrence loop in `ARCBackbone.forward()` AND the outer halting threshold in `ARCModel.forward()`. This meant each puzzle got 8×8=64 total recurrence passes, and per-step metrics only captured the last 8 (passes 57–64), which appeared flat because the model had already converged.

**Fix**: Simplified to one step per `forward()` call for all modes (URM, EBT, hybrid). Removed the inner loop entirely — `ARCBackbone.forward()` now does a single transformer pass. The outer loop in pretrain.py calls `forward()` repeatedly until halted, computing per-step exact accuracy and delta norms at each step. All three modes use the same halting logic, making per-step metrics directly comparable. All R1a–R1g results are invalid and are being re-run.

### Root cause of flat per-step accuracy + deep supervision fix (2026-04-11)

Root cause of flat per-step accuracy: `ARCBackbone.forward()` detached hidden states between carry-based steps (`hidden_states.detach()` at line 171), preventing cross-step gradient flow. Each step was independently trained as a single-pass model with shared weights — there was no learning signal for "produce intermediate representations that later steps can improve."

**Fix**: Replaced carry-based outer loop with flat `forward_trajectory()` that runs all N steps in a single call with NO `.detach()` between steps. Gradients flow through the full trajectory. Added deep supervision: every step t gets reconstruction loss weighted by `(t+1)/N` (linear ramp), plus Q-halt BCE loss at every step. This gives the model a direct signal to improve across steps — earlier steps produce coarser representations, later steps refine them.

Key changes:
- `ARCModel.forward_trajectory(batch, N)` — new primary entry point, runs N recurrence steps with full gradient flow
- `EnergyLossHead.forward(batch)` — computes deep supervision loss over the full trajectory in a single call
- Carry-based loop removed entirely (ModelCarry, initial_carry, ARCBackbone.forward(), halting logic)
- Stopping criteria (Q-halt convergence, energy convergence) are evaluated post-hoc on the full trajectory at eval time — not used during training
- Per-step metrics (accuracy, exact accuracy, delta norm) computed inside EnergyLossHead for both train and eval
- Reference: TRM paper (Jolicoeur-Martineau, 2025) found deep supervision to be the single largest contributor to recurrence benefit, doubling accuracy

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
**Killed at ~54K/80K steps (67%). Per-step convergence flat throughout.** Wandb: `R1e-exp2-full-h128-260410` ([link](https://wandb.ai/uberdavid-personal/arcagi/runs/d1fnymz8))

10 eval checkpoints (5.3K–53.5K steps). Per-step variation 0.1–0.5% at every checkpoint — completely flat from the first eval onward. The model always peaks at step 1 or 2, with negligible change through step 8.

| Step | Tok% | Exact% | P@1 | P@10 | S1→S8 variation |
|------|------|--------|-----|------|-----------------|
| 5.3K | 57.8% | 0.2% | 0.0% | 6.5% | 0.4% |
| 10.7K | 65.0% | 1.7% | 3.2% | 20.8% | 0.1% |
| 21.4K | 74.1% | 6.8% | 11.7% | 31.8% | 0.4% |
| 32.1K | 78.8% | 11.3% | 15.6% | 37.7% | 0.4% |
| 42.8K | 81.9% | 14.8% | 19.5% | 44.2% | 0.3% |
| 53.5K | 83.8% | 16.8% | 23.4% | 46.8% | 0.3% |

At 53.5K steps: 83.8% token acc, 23.4% pass@1 — still climbing (vs R1-full 85.9% / 25.3% at 80K), but the per-step curve is flat at every training stage. No multi-step convergence emerged at any point during training. Constant LR (vs R1-full's cosine decay to 0.1×) made no difference to the convergence pattern.

**Conclusion:** Reducing MLP capacity (expansion=2) does not create multi-step convergence at depth=2. The flat per-step curve is a property of having two transformer layers per recurrence step, not the MLP width. Each depth=2 step provides enough nonlinear processing to reach the fixed point in a single pass.

---

### Experiment R1f — depth=1, h=128, expansion=2
Date: 2026-04-10
Script: `scripts/train_r1f_d1.sh`
Config: depth=1, h=128, 4 heads, expansion=2, 8 steps, batch 512, 10×10 grids, 80K steps (31590 epochs), constant LR (3e-4) after 100-step warmup, EMA 0.999. ~164K transformer params (vs 330K at depth=2, 530K at baseline). eval_interval=2106 epochs (15 checkpoints).

Hypothesis: Every depth=2 configuration tested (h=128/exp=4, h=128/exp=2, h=96/exp=4, h=64/exp=4) converges in a single recurrence step regardless of MLP capacity. The two-layer transformer performs too much nonlinear processing per step for the 10×10 problem. Reducing to depth=1 halves the processing per step — each recurrence pass is now a single attention + MLP block. This should force the model to distribute computation across multiple recurrence steps. At h=128 exp=2 this gives ~164K transformer params (vs 530K baseline).

Expected outcome: Per-step accuracy should show meaningful improvement across steps, not one-step convergence. If it still one-steps, next candidate is depth=1, h=64, exp=2 (~66K params).

Reference: R1e (depth=2, h=128, exp=2) killed — confirmed one-step convergence persists at depth=2 regardless of MLP size.

### Result
**Killed at ~37.5K/80K steps (47%). Per-step convergence flat throughout.** Wandb: `R1f-d1-h128-exp2-260410` ([link](https://wandb.ai/uberdavid-personal/arcagi/runs/rs5u8pma))

7 eval checkpoints (5.3K–37.5K steps). Per-step variation 0.3–0.9% — flat, same pattern as depth=2.

| Step | Tok% | Exact% | P@1 | P@10 | S1→S8 variation |
|------|------|--------|-----|------|-----------------|
| 5.3K | 55.8% | 0.2% | 0.0% | 2.6% | 0.5% |
| 16.1K | 66.9% | 1.3% | 2.6% | 10.4% | 0.7% |
| 26.8K | 73.5% | 3.1% | 5.2% | 19.5% | 0.9% |
| 37.5K | 76.7% | 5.1% | 13.0% | 24.7% | 0.3% |

At 37.5K steps: 76.7% token acc, 13.0% pass@1 — weaker than depth=2/exp=2 (83.8% / 23.4% at 53.5K) as expected from fewer params, but per-step curve is still flat. Depth=1 with h=128/exp=2 (~164K params) still one-step converges.

**Conclusion:** depth=1 alone is not sufficient to break one-step convergence at h=128. The representation width (128-dim hidden states) provides enough capacity for a single attention+MLP block to extract the answer. Need to reduce h as well.

---

### Experiment R1g — depth=1, h=64, exp=2, corrected rounding
Date: 2026-04-10
Script: `scripts/train_r1g_d1_h64.sh`
Config: depth=1, h=64, 4 heads, expansion=2, 8 steps, batch 512, 10×10 grids, 80K steps (31590 epochs), constant LR (3e-4) after 100-step warmup, EMA 0.999. MLP rounding granularity reduced from 256 to 8 — at h=64/exp=2, inter=_find_multiple(85, 8)=88 instead of 256. eval_interval=2106 epochs (15 checkpoints).

Hypothesis: All depth=2 and depth=1/h=128 configs one-step converge. Reducing to h=64 with corrected MLP rounding (inter=88 instead of 256) gives a much smaller model that should need multiple recurrence steps. Previous h=64 results were invalid due to (a) undertrained from LR decay and (b) inflated MLP from 256 rounding.

Expected outcome: Per-step accuracy should show meaningful improvement across steps. If it still one-steps, may need to go even smaller (h=32) or consider that the one-step convergence is fundamental to the architecture's input re-injection mechanism.

Reference: R1f (depth=1, h=128, exp=2) killed — still one-step convergence.

### Result
TBD

---

### Experiment R1h — depth=1, h=64, exp=2, deep supervision (first run with cross-step gradient flow)
Date: 2026-04-11
Script: `scripts/train_r1h_deep_sup.sh`
Config: `config/arch/urm_r1g_d1_h64.yaml` — depth=1, h=64, 4 heads (head_dim=16), expansion=2, 8 recurrence steps, batch 512, 10×10 grids, 80K steps (31590 epochs), constant LR (3e-4) after 100-step warmup, EMA 0.999. MLP inter=88 (granularity=8). ~35K transformer params. eval_interval=2106 epochs (15 checkpoints). stablemax_cross_entropy, bfloat16.

**This is the first run with deep supervision + cross-step gradient flow.** All prior R1 experiments (R1a–R1g) used the old carry-based outer loop that detached hidden states between steps — each step was independently trained as a single-pass model with shared weights. R1h uses `forward_trajectory()` which runs all 8 steps in a single call with NO `.detach()` between steps. Deep supervision applies weighted reconstruction loss at every step (linear ramp `(t+1)/N`) plus Q-halt BCE at every step.

Hypothesis: Deep supervision provides a per-step learning signal and cross-step gradient flow enables the model to learn to distribute computation across steps. Unlike the old detached training where each step had identical behavior (producing flat per-step curves), gradients from later steps now flow into earlier steps, creating a direct learning signal for "produce intermediate representations that later steps can improve." We expect to see a monotonic accuracy ramp from step 1 to step 8, unlike all prior R1 experiments which showed 0.3–3.8% variation (flat).

Expected outcome: Per-step accuracy should show clear monotonic improvement from step 1 to step 8. The step-8 accuracy should exceed what R1g achieves (TBD — R1g not yet run with old code). If the ramp emerges, this validates that the flat curves were caused by detached gradients, not architectural limitations, and opens the path to R2/R3 refinement comparisons.

Reference: R1g (same config, old carry-based code) — pending. TRM paper (Jolicoeur-Martineau, 2025) found deep supervision to be the single largest contributor to recurrence benefit, doubling accuracy.

### Result
**Deep supervision works — monotonic per-step accuracy ramp confirmed. Phase 1 complete.** 35,275 params. Wandb: `R1h-deepsup-d1-h64-260411` ([link](https://wandb.ai/uberdavid-personal/arcagi/runs/tyn03n4l))

**Eval per-step accuracy (final checkpoint, 80K steps):**

| Step | Token Acc | Exact Acc | Delta Norm |
|------|-----------|-----------|------------|
| 1 | 66.8% | 0.13% | — |
| 2 | 74.7% | 0.38% | 0.0080 |
| 3 | 77.3% | 1.43% | 0.0042 |
| 4 | 78.4% | 3.04% | 0.0027 |
| 5 | 78.8% | 3.61% | 0.0018 |
| 6 | **78.9%** | **3.76%** | 0.0014 |
| 7 | 78.9% | 3.61% | 0.0011 |
| 8 | 78.7% | 3.16% | 0.0009 |

**12.1% token accuracy gain step 1→6. 29× exact accuracy improvement (0.13% → 3.76% peak at step 6).** Monotonically increasing through step 6, slight decline at 7–8. Prior R1 experiments had <2% step variation.

Train per-step exact accuracy (final batch): 3.9% → 8.6% → 12.3% → 17.2% → 19.9% → 19.5% → 20.7% → 19.1%. Train overfits significantly vs eval (20.7% vs 3.76% peak exact).

pass@K (Q-halt): 5.2% @1, 5.8% @2, 10.4% @5, 15.6% @10, 22.7% @100, 24.0% @1000.
pass@K (energy): 0% @1, 0% @10, 1.9% @100, 14.3% @1000. Energy ranking far worse — energy head is untrained in URM mode.

Stopping metrics: qhalt_stop_step=7.5 (2.6% acc), energy_stop_step=6.3 (3.2% acc). Energy stopping is measuring hidden state convergence (via shared backbone layers), not learned energy structure.

**Conclusion:** Deep supervision + cross-step gradient flow is the fix for flat per-step curves. The model learns to distribute computation across 8 steps, with clear monotonic improvement from step 1 through step 6. The slight decline at steps 7–8 suggests the model may benefit from fewer steps or that the linear weight ramp over-weights late steps where overfitting occurs. The train/eval gap (20.7% vs 3.76% exact) indicates significant overfitting at this model size — a capacity increase or regularization may help generalization. Phase 1 is complete: we have a validated architecture and training setup where multi-step convergence emerges, enabling Phase 2 (energy training via trajectory supervision).

---

### Experiment R1i — Dropout regularization (prerequisite for R2 retry)
Date: 2026-04-11
Script: `scripts/train_r1i_dropout.sh`
Config: `config/arch/urm_r1i_dropout.yaml` — same as R1h (d=1, h=64, exp=2, 8 steps) + attn_dropout=0.1, mlp_dropout=0.1.
Hypothesis: Dropout closes the train/eval exact accuracy gap (23.2% vs 2.9% in R1h / 3.8% in R1h eval peak). The energy head's failure to generalize in R2 was caused by backbone overfitting, not energy head design. With a regularized backbone, trajectory ranking should produce eval Spearman correlation closer to train.
Expected outcome: Train exact accuracy decreases (expected with dropout). Eval exact accuracy increases or stays similar. Train/eval gap shrinks substantially. Per-step monotonic improvement preserved.
Success criterion: Train/eval exact accuracy ratio < 3:1 (vs ~8:1 in R1h). If met, re-run R2 on this backbone.
Fallback: If dropout=0.1 kills learning (eval accuracy drops below R1h), retry with dropout=0.05. If that also fails, scale to h=96 with dropout.

### Result
TBD

---

### Experiment R2 — Energy verifier via trajectory ranking (first-order)
Date: 2026-04-11
Script: `scripts/train_r2_trajectory.sh`
Config: `config/arch/urm_r2_trajectory.yaml` — Same backbone as R1h (d=1, h=64, exp=2, 8 steps). Co-train energy head alongside URM using trajectory ranking loss. First-order only — no MCMC, no create_graph=True. All-pairs weighted margin loss across URM trajectory steps: quality_gap * F.relu(E(better) - E(worse) + margin). energy_loss_weight=0.1, ranking_margin=0.1. Separate gradient clipping: energy head max_norm=1.0, backbone max_norm=5.0.
Hypothesis: The energy head, trained via trajectory ranking on URM hidden states, learns to assign lower energy to better hidden states (higher reconstruction accuracy). This produces a verifier that outperforms Q-halt for pass@K reranking — without requiring any MCMC or second-order gradients.
Expected outcome: Energy-accuracy Spearman correlation > 0.5 throughout training. Active ranking pairs stay > 30% of 28 total. Energy reranking improves pass@K over Q-halt at some K values.
Key diagnostics to log:
- `trajectory_quality_first`, `trajectory_quality_last`, `active_pairs` (from step 0)
- Energy-accuracy Spearman correlation per eval
- Cosine similarity between ∇_hidden E and direction toward best trajectory hidden state (R3 readiness signal — does energy gradient point toward better states?)
Inspired by: Exp 4 (trajectory supervision worked when quality spread existed), EBT paper Section 2.1 (verification easier than generation).
Risk: Shared backbone means trajectory ranking gradients modify URM weights — monitor for URM regression. Quality spread on training data may collapse faster than on eval data due to train/eval gap (20.7% vs 3.76% exact in R1h).

### Result
**Energy head learns correct ordering — Spearman ρ = -0.48, all 28 pairs active. URM NOT regressed. Energy reranking far worse than Q-halt.** 35,210 params. Wandb: `R2-trajectory-d1-h64-260411` ([link](https://wandb.ai/uberdavid-personal/arcagi/runs/m3fzbof3))

**Eval per-step accuracy (final checkpoint, 80K steps):**

| Step | Token Acc | Exact Acc | Delta Norm |
|------|-----------|-----------|------------|
| 1 | 66.9% | 0.10% | — |
| 2 | 74.6% | 0.35% | 0.0080 |
| 3 | 77.4% | 1.38% | 0.0043 |
| 4 | 78.5% | 3.25% | 0.0026 |
| 5 | 78.9% | 3.89% | 0.0018 |
| 6 | **79.0%** | **3.87%** | 0.0014 |
| 7 | 78.9% | 3.57% | 0.0011 |
| 8 | 78.8% | 2.93% | 0.0009 |

**No URM regression vs R1h baseline.** Per-step accuracy curve is nearly identical to R1h (66.8%→78.9% vs 66.9%→78.8% token acc; 0.13%→3.76% vs 0.10%→3.87% peak exact). The trajectory ranking loss at weight=0.1 does not interfere with URM training. Training speed: 7.0 it/s (R1h: ~5.8 it/s — faster due to different batch scheduling).

**Energy head diagnostics (training):**

| Metric | Step 336 | Step 7K | Step 39K | Step 78K |
|--------|----------|---------|----------|----------|
| Spearman ρ | -0.22 | -0.69 | -0.98 | -1.00 |
| Active pairs | 28/28 | 28/28 | 28/28 | 28/28 |
| Cosine sim | -0.12 | -0.05 | -0.10 | -0.11 |

Train Spearman rapidly reaches -0.98 to -1.0 (perfect negative correlation: lower energy = higher quality). All 28 pairs remain active throughout training. Cosine similarity is consistently negative (-0.05 to -0.12), indicating energy gradients point toward better hidden states (R3 favorable).

**Energy head diagnostics (eval):**

| Checkpoint | Spearman ρ | Active | Cosine | Energy pass@100 | Energy pass@1K |
|------------|-----------|--------|--------|-----------------|----------------|
| 1 (5.3K) | -0.13 | 28 | +0.01 | 2.6% | 4.5% |
| 3 (16K) | -0.57 | 28 | -0.01 | 1.3% | 7.8% |
| 6 (32K) | -0.55 | 28 | -0.02 | 2.6% | 14.9% |
| 9 (48K) | -0.52 | 28 | -0.01 | 1.3% | 13.0% |
| 12 (64K) | -0.50 | 28 | -0.01 | 1.3% | 12.3% |
| 15 (80K) | -0.48 | 28 | -0.01 | 1.3% | 12.3% |

Eval Spearman stabilizes at -0.48 to -0.55 (moderately negative). Significantly weaker than training (-0.98), indicating the energy head generalizes poorly to eval data. Cosine similarity is near zero on eval (vs -0.10 on train), suggesting the energy gradient signal doesn't generalize well.

**pass@K comparison (final checkpoint):**

| K | Q-halt | Energy |
|---|--------|--------|
| 1 | 5.8% | 0% |
| 10 | 16.9% | 0% |
| 100 | 22.7% | 1.3% |
| 1000 | 26.0% | 12.3% |

Energy reranking is **dramatically worse** than Q-halt at all K values. At pass@100: 22.7% vs 1.3%. At pass@1000: 26.0% vs 12.3%. The energy head learns the correct ordering on training data (ρ = -1.0) but doesn't generalize to eval — this is a verification overfitting problem.

**Stopping comparison:**
- Q-halt stop: step 7.4, accuracy 2.68%
- Energy stop: step 7.1, accuracy 3.07%

Energy stopping slightly outperforms Q-halt stopping (3.07% vs 2.68%) by stopping earlier (7.1 vs 7.4 steps). This is marginal.

**Conclusion:** The trajectory ranking loss successfully trains the energy head without regressing URM performance. On training data, the energy head achieves near-perfect correlation with hidden state quality (ρ → -1.0). However, the energy head **overfits badly** to training trajectories — eval Spearman is only -0.48, and energy reranking is far worse than Q-halt for pass@K. The shared backbone architecture means the energy head sees training-specific hidden state patterns that don't transfer to eval.

Key finding: verification via trajectory ranking works mechanistically (the energy head learns to rank steps correctly) but doesn't generalize across puzzles. The energy function doesn't capture abstract quality — it memorizes training-specific patterns. This suggests trajectory ranking alone is insufficient; the energy head needs exposure to diverse hidden states (which MCMC in R3 would provide by generating varied trajectories) or a different training signal that encourages generalization (e.g., puzzle-independent features).

The cosine similarity diagnostic is weakly negative on train (-0.10) and near-zero on eval, suggesting MCMC gradients would provide marginal directional signal at best. R3 may still be worth trying because MCMC training would give the energy head a fundamentally different training distribution (gradient-optimized hidden states vs fixed URM trajectories).

---

### Experiment R2b — Energy verifier via trajectory ranking (dropout backbone)
Date: 2026-04-12
Script: `scripts/train_r2b_trajectory_dropout.sh`
Config: `config/arch/urm_r2b_trajectory_dropout.yaml` — same as R2 + attn_dropout=0.1, mlp_dropout=0.1 from R1i.
Hypothesis: R2's energy head overfitting (train Spearman -1.0 vs eval -0.48) was caused by backbone overfitting (train/eval exact accuracy ratio 6:1). R1i reduced this to 3.7:1 via dropout. With a regularized backbone, the energy head should learn more generalizable quality features, improving eval Spearman and energy pass@K.
Expected outcome: Eval Spearman correlation improves from -0.48 toward -0.7+. Energy pass@K becomes competitive with Q-halt at some K values. URM eval metrics match or exceed R1i (5.3% exact accuracy). Active pairs remain high.
Success criterion: Energy reranking improves pass@K over Q-halt at some K values (R2.5 gate). Eval Spearman < -0.6.
Risk: Dropout may not be sufficient — energy head generalization may require fundamentally different architecture (separate parameters from backbone) rather than just backbone regularization.

### Result
TBD (full R2b results pending — see energy ranking comparison below)

#### R2b Energy Ranking Strategy Comparison (eval-only)
Tested alternative eval-time ranking signals from R2b checkpoint (step 80K):

| Ranking Strategy | pass@1 | pass@2 | pass@5 | pass@10 | pass@100 | pass@1000 |
|---|---|---|---|---|---|---|
| Q-halt | 12.3% | 12.3% | 16.9% | 20.8% | 23.4% | 23.4% |
| -E_final (current) | 0.6% | 0.6% | 1.9% | 1.9% | 14.9% | 23.4% |
| Energy drop (E1-E8) | 0.6% | 0.6% | 0.6% | 0.6% | 15.6% | 23.4% |
| -E_step6 (best step) | 0.0% | 0.0% | 1.9% | 1.9% | 14.9% | 23.4% |

**Interpretation:** All energy ranking strategies converge at pass@1000 (23.4%, same as Q-halt) — the energy head doesn't filter out correct predictions, it just can't rank them. Q-halt dominates at all practical K values (12.3% pass@1 vs 0.0-0.6%). Energy drop (E1-E8) is slightly better at pass@100 (15.6% vs 14.9%) but no better otherwise. The energy head with trajectory ranking doesn't learn a useful ranking signal regardless of which step's energy we use. The problem isn't which energy value to use — it's that the energy function doesn't capture prediction quality in a generalizable way.

---

### Experiment R2.5 — Energy reranking + stopping (eval only)
Date: TBD
Config: No new training. Use R2-trained checkpoint. Score URM pass@K candidates by energy. Compare to Q-halt ranking and stopping.
Hypothesis: Learned energy function is a better confidence signal than Q-halt for selecting among multiple URM predictions.
Expected outcome: Energy reranking improves pass@K over Q-halt on at least some K values. Energy convergence stopping selects a step closer to peak accuracy than Q-halt stopping.
Success criterion: This is the first publishable result if it works — "learned energy verification beats learned halting for iterative reasoning."
Gate for R3: If energy reranking shows no improvement AND cosine similarity diagnostic from R2 is low (energy gradients don't point toward better states), skip R3 (MCMC won't help either). If reranking works but cosine similarity is low, energy is a good verifier but not a good landscape — R3 may still fail.

### Result
TBD

---

### Experiment R3 — MCMC refinement: hybrid URM + MCMC (conditional on R2.5)
Date: TBD
Config: M URM steps + K MCMC steps at matched total compute (M+K = N). Start with M = N-2 (most URM, little MCMC) and decrease M. Second-order gradients via create_graph=True. EBT-style regularization: Langevin noise during training (drop at inference), step size annealing, randomized step count. Dual reconstruction loss (unrefined + refined) — non-negotiable per Exp 3a.
Hypothesis: MCMC refinement in hidden space, guided by the energy landscape trained in R2, provides complementary optimization that improves predictions beyond what additional URM steps achieve. The energy landscape, now trained through both trajectory ranking (R2) and MCMC optimization (R3), develops smooth gradient structure.
Expected outcome: mcmc_improvement > 0. Hybrid accuracy > pure URM accuracy at matched total steps.
Inspired by: EBT paper Section 3.3 (energy landscape regularization), IREM (initialization matters for energy minimization). All prior experiments showed MCMC fails on fully-converged URM; M < N ensures URM is not converged when MCMC starts.
Risk: Second-order gradients may cause training instability. MCMC may be a worse version of additional URM steps — same hidden space but without attention's inductive bias.

### Result
TBD

---

### Experiment R3b — Pure EBT ablation (optional, contingent on R3)
Date: TBD
Config: All MCMC steps, no URM. Use 1 URM step as initialization (counted in budget) since cold start from input_embeddings unlikely to work on discrete ARC grids.
Hypothesis: Pure energy minimization alone cannot solve ARC — attention-based processing provides essential inductive biases for spatial pattern recognition.
Expected outcome: Pure EBT << URM accuracy. This is an ablation for the paper, not a primary experiment.

### Result
TBD

---

## Lessons Carried Forward
1. **Dual reconstruction loss is mandatory** for MCMC training (Exp 3a).
2. **Contrastive loss alone causes energy collapse** (Exp 1). Use trajectory ranking as primary energy training signal (first-order, R2). MCMC reconstruction provides second-order signal in R3.
3. **Right-size the model for the problem.** The model must struggle enough that refinement has room to help.
4. **MCMC must operate in hidden space**, not soft-embedding space.
5. **Don't detach between MCMC steps during training.** Energy head needs multi-step trajectory gradients.
6. **Budget in steps, not epochs.**
