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
**Dropout closes the gap — eval exact accuracy 5.33% (vs 3.76% without dropout).** 35,210 params. Wandb: `R1i-dropout-d1-h64-260411` ([link](https://wandb.ai/uberdavid-personal/arcagi/runs/lmzfasu2))

| Step | Token Acc | Exact Acc | Delta Norm |
|------|-----------|-----------|------------|
| 1 | 67.4% | 0.06% | — |
| 2 | 74.8% | 0.26% | 0.0075 |
| 3 | 78.0% | 1.91% | 0.0040 |
| 4 | 79.1% | 4.02% | 0.0024 |
| 5 | 79.6% | 5.04% | 0.0016 |
| 6 | **79.7%** | **5.33%** | 0.0011 |
| 7 | 79.7% | 5.32% | 0.0009 |
| 8 | 79.6% | 5.12% | 0.0007 |

Train exact accuracy: 19.7% (vs R1h 19.1%). Train/eval ratio: 3.9:1 (vs R1h 6.0:1). Per-step monotonic improvement preserved. pass@K (Q-halt): 5.2% @1, 16.2% @10, 21.4% @100, 22.1% @1000.

**Conclusion:** Dropout=0.1 successfully reduces overfitting without hurting learning. Eval exact accuracy improved 42% (3.76% → 5.33%). Train/eval ratio compressed from 6:1 to 3.9:1. This is the new standard backbone for energy experiments.

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
**Eval Spearman improved to -0.585 (vs R2's -0.484). Energy ranking still fails. URM matches R1i baseline.** 35,210 params. Wandb: `R2b-traj-dropout-d1-h64-260412` ([link](https://wandb.ai/uberdavid-personal/arcagi/runs/d21szemv))

| Step | Token Acc | Exact Acc | Delta Norm |
|------|-----------|-----------|------------|
| 1 | 67.4% | 0.04% | — |
| 2 | 75.2% | 0.60% | 0.0076 |
| 3 | 78.2% | 2.27% | 0.0040 |
| 4 | 79.3% | 4.72% | 0.0024 |
| 5 | 79.6% | 5.41% | 0.0016 |
| 6 | **79.7%** | **5.49%** | 0.0011 |
| 7 | 79.7% | 5.35% | 0.0009 |
| 8 | 79.6% | 5.04% | 0.0007 |

Train/eval ratio: 3.9:1 (matches R1i). Eval Spearman: -0.585 (improved from R2's -0.484). Train Spearman: -0.905. Cosine similarity: -0.013 (eval), -0.057 (train). pass@K (Q-halt): 3.9% @1, 19.5% @10, 22.7% @100, 24.7% @1000. pass@K (Energy): 0% @1, 0% @10, 1.3% @100, 11.7% @1000.

**Conclusion:** Dropout improved eval Spearman modestly (-0.484 → -0.585) but energy pass@K is still dramatically worse than Q-halt. The energy head can rank within-trajectory steps but cannot discriminate across different inputs/puzzles. Dropout helped the backbone but didn't solve the fundamental energy ranking problem.

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

### Experiment R2c/d/e — Position-aware energy head cascade
Date: 2026-04-12

Cascade of increasing energy head complexity. Run in order, stop when energy pass@K beats Q-halt.

**Root cause of R2/R2b failure:** The energy head is `mean_pool -> Linear(64,1)` (65 params). Mean pooling destroys spatial information critical for ARC — whether a prediction is correct depends on *which positions* are wrong, not the average hidden state. Alternative eval-time ranking strategies (energy drop, best-step energy) confirmed this isn't a calibration issue — the features themselves lack discriminative power.

**R2c — Per-position MLP -> sum (~2K new params)**
Config: `urm_r2c_pos_mlp.yaml`. Replace mean_pool -> linear with per-position `Linear(64,32) -> SiLU -> Linear(32,1)` then sum across positions. Tests whether spatial preservation plus nonlinearity is sufficient.
energy_head_type: position_mlp

**R2d — Conv1d + per-position MLP -> sum (~3K new params)** — conditional on R2c failure
Config: `urm_r2d_pos_conv.yaml`. Add `Conv1d(64,64,k=3,groups=16)` before per-position MLP. Adds local neighbor context — "is this position consistent with neighbors?"
energy_head_type: position_conv_mlp

**R2e — Dedicated attention + per-position MLP -> sum (~10K new params)** — conditional on R2d failure
Config: `urm_r2e_pos_attn.yaml`. Add dedicated `nn.MultiheadAttention` layer (own parameters, not shared with backbone) before per-position MLP. Enables cross-position comparison optimized for verification.
energy_head_type: position_attn_mlp

Success criterion (same for all): Energy pass@K > Q-halt pass@K at any K.
Key diagnostics: eval Spearman, energy pass@K, URM per-step accuracy (monitor for regression).

### Result
**R2c: Best URM yet (6.95% eval exact) but energy ranking still fails. Eval Spearman collapsed to -0.069.** 37,258 params. Wandb: `R2c-pos-mlp-d1-h64-260412` ([link](https://wandb.ai/uberdavid-personal/arcagi/runs/rbbrk0fh))

| Step | Token Acc | Exact Acc | Delta Norm |
|------|-----------|-----------|------------|
| 1 | 67.3% | 0.07% | — |
| 2 | 75.2% | 0.57% | 0.0076 |
| 3 | 78.3% | 3.18% | 0.0041 |
| 4 | 79.5% | 5.86% | 0.0024 |
| 5 | 79.8% | 6.80% | 0.0016 |
| 6 | **79.9%** | **6.95%** | 0.0011 |
| 7 | 79.9% | 6.87% | 0.0009 |
| 8 | 79.8% | 6.51% | 0.0007 |

Train/eval ratio: 3.3:1 (best yet). Eval Spearman: **-0.069** (collapsed vs R2b's -0.585). Train Spearman: -0.958. pass@K (Q-halt): 4.5% @1, 20.1% @10, **26.0%** @100, **29.2%** @1000. pass@K (Energy): 0% @1, 0% @10, 2.6% @100, 16.2% @1000.

**Paradox:** The position-aware energy head produces the best backbone (6.95% eval exact, Q-halt pass@100=26.0% vs R2b's 22.7%) while completely failing at its intended job (eval Spearman near zero). The energy co-training gradient helps as a multi-task regularizer even though the energy function doesn't generalize. Energy pass@K improved slightly (16.2% vs 11.7% at pass@1000) but is still far from Q-halt (29.2%).

**Decision:** Energy ranking has failed across linear (R2/R2b) and position-aware (R2c) heads. The within-trajectory ranking signal (Spearman) doesn't transfer to cross-input ranking (pass@K). R2d/R2e are unlikely to change this — more energy head capacity makes the Spearman *worse* on eval (more overfitting). The cascade should stop here.

R2d: Skipped — R2c showed more capacity worsens eval Spearman.
R2e: Skipped — same reasoning.

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

### Ablation Series A — Why does energy co-training improve the backbone?

R2c discovered that energy co-training with trajectory ranking improves the backbone by 30% (6.95% vs 5.33% eval exact) even though the energy function itself doesn't generalize (0% energy pass@10, eval Spearman -0.069). These ablations isolate the mechanism.

**Hypothesis:** The trajectory ranking loss provides useful gradient diversity through the shared backbone layers, acting as multi-task regularization. This is tested by varying the loss weight (A1), replacing the signal with random noise (A2), and blocking gradient flow to the backbone (A3).

#### A1 — Energy loss weight sweep
Configs: `ablation_a1_elw001.yaml` (0.01), `ablation_a1_elw005.yaml` (0.05), `ablation_a1_elw020.yaml` (0.2), `ablation_a1_elw050.yaml` (0.5). R2c used 0.1.
Question: Is there a sweet spot for `energy_loss_weight`, or does the backbone improvement scale monotonically? If flat across a wide range → gradient direction matters, not magnitude. If peaked → optimal regularization strength exists. If monotonically increasing → more energy gradient = more regularization.

#### A2 — Random auxiliary head (shuffled quality labels)
Config: `ablation_a2_random.yaml`. Same as R2c but quality labels randomly permuted each forward pass.
Question: Does the backbone improvement require *correct* trajectory ordering, or does *any* auxiliary gradient through the energy head help? If backbone improves similarly → mechanism is pure gradient diversity / multi-task regularization. If backbone improvement disappears → the trajectory ordering signal specifically helps.

#### A3 — Frozen energy head (detached hidden states)
Config: `ablation_a3_detach.yaml`. Same as R2c but hidden states detached before energy head — energy head trains but gradients don't flow into backbone.
Question: Does the backbone improvement require gradient flow from the energy head *into* the backbone? If backbone improvement disappears → the mechanism is gradient flow through shared layers (the energy loss modifies backbone weights). If backbone still improves → having extra parameters / forward computation alone helps (unlikely but would be surprising).

#### Expected interpretation matrix:

| A1 shape | A2 result | A3 result | Interpretation |
|----------|-----------|-----------|---------------|
| Peaked ~0.1 | A2 < R2c | A3 ≈ R1i | Trajectory-specific gradient regularization at optimal strength |
| Flat | A2 ≈ R2c | A3 ≈ R1i | Any auxiliary gradient through energy head helps; not trajectory-specific |
| Flat | A2 < R2c | A3 ≈ R1i | Trajectory ordering needed, but robust to weight |
| Any | A2 ≈ R2c | A3 ≈ R2c | Extra parameters help (not gradient flow) — very unlikely |

### Results

#### A1 — Energy loss weight sweep (partial: A1a–A1c complete, A1d running)

| Experiment | elw | Eval Exact | Train Exact | Train/Eval | Spearman | Q-halt pass@1 | Q-halt pass@10 | Q-halt pass@100 |
|-----------|-----|-----------|------------|-----------|----------|--------------|---------------|----------------|
| R1i (baseline) | 0.0 | 5.33% | 20.8% | 3.9x | — | — | — | 22.1% |
| A1a | 0.01 | 6.17% | 20.90% | 3.39x | -0.189 | 5.19% | 22.08% | 26.62% |
| A1b | 0.05 | 6.06% | 22.46% | 3.71x | -0.087 | 6.49% | 20.13% | 24.68% |
| R2c (baseline) | 0.1 | 6.95% | 22.9% | 3.3x | -0.069 | — | — | 29.2% |
| A1c | 0.2 | 6.80% | 21.88% | 3.22x | +0.050 | 5.19% | 20.13% | 25.97% |
| A1d | 0.5 | **4.67%** | 14.84% | 3.17x | +0.213 | 7.79% | 15.58% | 20.13% |

**A1 interpretation (complete):** The A1 sweep reveals an **inverse-U shape** with a peak around elw=0.1–0.2:
- **elw ∈ [0.01, 0.2]**: Broad plateau at 6.06–6.95% eval exact, all above R1i's 5.33%. Even 0.01 weight helps substantially (+16% over R1i).
- **elw = 0.5**: Catastrophic regression to 4.67% — below R1i baseline. Train exact also collapses (14.8% vs 20–23% at lower weights). The energy loss dominates and starves the reconstruction objective.
- **Energy Spearman flips sign** across the sweep: -0.189 (0.01) → -0.087 (0.05) → -0.069 (0.1, R2c) → +0.050 (0.2) → +0.213 (0.5). Higher weight actively anti-aligns energy with quality on eval, even though train Spearman stays near -1.0 — classic memorization without generalization.
- **A1d energy pass@100 jumped to 12.3%** (vs ~1–3% elsewhere). Interesting because the backbone is worse overall — suggests extreme weight forces some energy structure at the cost of LM quality.
- **Conclusion:** R2c's choice of elw=0.1 was near-optimal. The mechanism is genuine regularization with a finite capacity budget, not a direction you can push arbitrarily.

#### A2 — Random auxiliary head (shuffled quality labels, elw=0.1)

| Metric | R1i (baseline) | R2c (real labels, elw=0.1) | **A2 (random labels, elw=0.1)** |
|---|---|---|---|
| Eval exact | 5.33% | 6.95% | **1.31%** |
| Train exact | 20.8% | 22.9% | 6.25% |
| Train/eval ratio | 3.9x | 3.3x | 4.8x |
| Pass@1 / @10 / @100 / @1000 | — | — / — / — / 29.2% | 0.65% / 11.0% / 14.9% / 15.6% |
| Train Spearman | — | ~-1.0 | +0.167 |
| Eval Spearman | — | -0.069 | -0.029 |
| Energy pass@1 / @5 / @100 / @1000 | — | ~0 / 0 / ~0 / ~0 | 0 / 0 / 0.013 / 0.156 |
| Q-halt accuracy (eval) | — | — | 98.5% |

**A2 interpretation: catastrophic regression.** Random trajectory labels don't just fail to help — they actively destroy the backbone. Eval exact collapses to 1.31% (75% below R1i), train exact collapses to 6.25%, and pass@1000 drops to 15.6% (below R1i's 22.1%). The random-label gradient injects destructive noise into shared backbone features that the reconstruction objective cannot fully correct over 80K steps. This cleanly rules out the "any auxiliary gradient helps" hypothesis — maps to the `A2 < R2c` row of the interpretation matrix.

#### A3 — Frozen energy head (detached hidden states, elw=0.1)

| Metric | R1i (baseline) | R2c (coupled, elw=0.1) | **A3 (detached, elw=0.1)** |
|---|---|---|---|
| Eval exact | 5.33% | 6.95% | **5.67%** |
| Train exact | 20.8% | 22.9% | 21.3% |
| Train/eval ratio | 3.9x | 3.3x | 3.76x |
| Pass@1 / @10 / @100 / @1000 | — / — / — / 22.1% | — / — / — / 29.2% | 5.19% / 17.5% / 24.7% / 28.6% |
| Train Spearman | — | ~-1.0 | -0.976 |
| Eval Spearman | — | -0.069 | -0.264 |
| Energy pass@1 / @5 / @100 / @1000 | — | ~0 / 0 / ~0 / ~0 | 0 / 0 / 0.013 / 0.162 |
| Q-halt accuracy (eval) | — | — | 92.5% |

**A3 interpretation: eval exact snaps back to R1i baseline.** Detaching hidden states before the energy head eliminates R2c's backbone gain — eval exact drops from 6.95% to 5.67% (within noise of R1i's 5.33%), and train/eval ratio returns to baseline. The energy head still learns its within-trajectory ranking task near-perfectly (train Spearman -0.976) using frozen backbone features, confirming the detach is working as designed. Notable secondary finding: **eval Spearman is actually better when detached** (-0.264 vs R2c's -0.069) — coupling to the backbone drags the energy head toward puzzle-specific features the backbone is learning, hurting cross-input generalization. Still not discriminative enough to recover energy pass@1/5 > 0. Asterisk: pass@1000 stays high at 28.6% (between R1i's 22.1% and R2c's 29.2%), suggesting the detached auxiliary still perturbs Q-halt dynamics slightly — worth caveating as a secondary effect on one seed.

#### Combined A1/A2/A3 conclusion

The three ablations together give a tight mechanistic explanation for R2c's ~1.6pp eval-exact gain over R1i:

| Alternative hypothesis | Ruled out by |
|---|---|
| "Wrong weight, just retune" | **A1** — inverse-U peaked at elw=0.1–0.2; elw=0.5 collapses the backbone |
| "Any auxiliary gradient regularizes" | **A2** — random-label gradient drops eval exact 75% (5.33% → 1.31%) |
| "Extra parameters / independent head helps" | **A3** — detached head matches R1i, not R2c |

**Conclusion: energy co-training is *structured* multi-task regularization.** It requires both (a) a correctly-ordered trajectory ranking signal and (b) that signal flowing as gradient into the shared backbone at roughly the right strength. The trajectory-ordering task shapes backbone features in a way generic auxiliary losses cannot, but only when the backbone is on the receiving end of that gradient. This is the `A2 < R2c | A3 ≈ R1i | Peaked A1` row of the interpretation matrix — the tightest possible story.

---

### Experiment A4 — Recurrence noise as backbone regularizer

Dropout=0.1 improved eval exact from 3.76% to 5.33% (R1i). This tests whether additive Gaussian noise in the recurrence loop is an alternative or complementary mechanism.

**Motivation:** Dropout zeros random features; noise perturbs all features continuously. Noise also has a connection to Langevin dynamics — if the recurrent process is implicitly minimizing an energy landscape, noise injection regularizes that landscape. This may have downstream benefits for energy head training if we later add trajectory ranking.

#### A4a — Recurrence noise only (no dropout)
Config: `ablation_a4a_noise.yaml`. σ=0.005, attn_dropout=0, mlp_dropout=0. No energy head.
Hypothesis: Recurrence noise provides regularization comparable to dropout by preventing the model from relying on precise hidden state features that don't generalize.
Success criterion: Eval exact accuracy > 4.5% (within 85% of R1i's 5.33%).

#### A4b — Recurrence noise + dropout (reduced strength)
Config: `ablation_a4b_noise_dropout.yaml`. σ=0.003, attn_dropout=0.05, mlp_dropout=0.05. No energy head.
Hypothesis: Noise and dropout are complementary — dropout regularizes individual features, noise regularizes the trajectory dynamics. Combined at reduced individual strength, they should outperform either alone.
Success criterion: Eval exact accuracy > 5.33% (beats R1i).

#### Hyperparameter rationale
- **A4a σ=0.005**: R1h delta norms range from 0.008 (step 2) to 0.0009 (step 8), averaging ~0.004 at mid-trajectory. σ=0.005 means noise magnitude is comparable to one refinement step — enough to regularize without overwhelming the signal.
- **A4b combined**: Both mechanisms reduced to half-ish of full strength. dropout=0.05 (half of R1i's 0.1), noise=0.003 (60% of A4a's 0.005). Splits the regularization budget between two complementary mechanisms.

#### Key metrics
- Eval exact accuracy (peak step and step 8)
- Train/eval ratio
- Per-step accuracy curve shape (is the monotonic ramp preserved?)
- Delta norms (does noise change convergence dynamics?)

### Results

Both A4 experiments completed at 80K steps, 35,210 params.

| Config | σ noise | Dropout | Eval exact (step 8) | Peak eval exact | Train exact | T/E ratio | Pass@1000 |
|--------|---------|---------|--------------------|-----------------|-----------  |-----------|-----------|
| R1h (no reg) | 0 | 0 | 3.76% | — | ~22.6% | 6.0× | — |
| A4a (noise only) | 0.005 | 0 | 3.34% | 3.96% (step 6) | 22.27% | 6.67× | 23.38% |
| A4b (noise+dropout) | 0.003 | 0.05 | 4.19% | 4.80% (step 6) | 21.29% | 5.08× | 28.57% |
| R1i (dropout only) | 0 | 0.1 | 5.33% | — | 20.8% | 3.9× | 22.1% |

Per-step eval exact (monotonic ramp preserved, peak at step 6 for both):

| Step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|------|---|---|---|---|---|---|---|---|
| A4a | 0.14% | 0.37% | 1.26% | 3.26% | 3.85% | 3.96% | 3.72% | 3.34% |
| A4b | 0.08% | 0.36% | 2.28% | 4.49% | 4.75% | 4.80% | 4.61% | 4.19% |

Delta norms essentially unchanged from R1h — noise at σ=0.005 doesn't disrupt convergence dynamics.

**Conclusion: Noise is not a useful backbone regularizer at this scale.** A4a (noise only) performs at or below the unregularized R1h baseline. A4b (noise + reduced dropout) sits between R1h and R1i but below R1i — trading dropout strength for noise is a net loss. Dropout=0.1 remains the standard backbone regularizer.

**Key secondary finding: A4b pass@1000 = 28.6%** — matching A3's 28.6% and approaching R2c's 29.2%, despite worse single-shot eval exact than R1i. This is the second time (after A3) that adding stochasticity to hidden states improves candidate diversity under Q-halt reranking without improving the mode. Pass@K and single-shot accuracy respond to different regularization mechanisms — a useful finding for the paper.

**Important distinction: A4 does NOT test ranking noise.** A4 asked "does noise in the recurrence loop regularize the backbone?" — answer: no, dropout is better. The ranking noise idea (R2g) asks a different question: "does noising hidden states before the energy head scores them break the step-index shortcut in trajectory ranking?" That's about the energy head's training signal, not backbone regularization. The energy head never existed in A4. The step-index shortcut (confirmed by A3's better eval Spearman when decoupled: -0.264 vs R2c's -0.069) is a property of how the ranking loss interacts with deterministic trajectories. Ranking noise remains untested and its motivation is fully intact.

The Langevin-dynamics analogy for backbone regularization is dead. But noise injection into the energy head's view of hidden states (ranking noise) is a separate mechanism targeting a different failure mode and should be tested after scaling experiments confirm the capacity story.

---

### Experiment R1-h96 — Scaled baseline (h=96, depth=1, dropout=0.1)
Date: 2026-04-14
Script: `scripts/train_h96_scale.sh`
Config: `config/arch/urm_r1_h96_baseline.yaml` — d=1, h=96, 4 heads (head_dim=24, flash-attn verified), exp=2, 8 steps, attn_dropout=0.1, mlp_dropout=0.1. No energy head. **~76.6K params (2.2× R1i's 35K)**. VRAM at batch=512 smoke-tested at 3.64 GB — well within budget.

**Motivation:** A1d showed the energy head can learn cross-input ranking at h=64 but only by sacrificing reconstruction quality (eval exact 4.67% vs 6.95% at elw=0.1). The 35K-param model can't do both tasks. Scaling to h=96 (~80K params) tests whether additional capacity resolves this trade-off.

**Critical gate:** Must show multi-step convergence (monotonic per-step accuracy ramp, >5% variation step 1→6). R1f showed h=128/d=1 (without deep supervision) one-step converged. h=96/d=1 with deep supervision is untested — if it one-step converges, R2c-h96 is pointless (no trajectory to rank) and we'll need h=80 or more steps or reduced expansion.

Hypothesis: h=96 with deep supervision maintains multi-step convergence (like h=64) while achieving higher absolute accuracy due to increased capacity. Expected eval exact > 7% (scaling from R1i's 5.33%).

Key measurements: per-step exact accuracy curve, eval exact (peak and step 8), pass@K, VRAM usage, throughput (it/s).

### Result
**Massive scaling win — eval exact 15.59%, pass@1000 40.91%.** 76K params. Runtime: 1h 42m.

| Step | Token Acc | Exact Acc | Delta Norm |
|------|-----------|-----------|------------|
| 1 | 67.58% | 0.24% | — |
| 2 | 78.89% | 5.44% | 0.0105 |
| 3 | 82.43% | 11.60% | 0.0055 |
| 4 | 83.47% | 13.97% | 0.0034 |
| 5 | 83.79% | 15.22% | 0.0023 |
| 6 | 83.83% | 16.08% | 0.0016 |
| 7 | 83.75% | **16.09%** | 0.0012 |
| 8 | 83.61% | 15.59% | 0.00098 |

Multi-step convergence confirmed: +15.85pp exact accuracy step 1→7. Train exact: 36.33%. Train/eval ratio: 2.33× (best yet — overfitting decreased at larger scale). pass@K (Q-halt): 20.78% @1, 31.17% @10, 34.42% @100, 40.91% @1000.

**Conclusion:** h=96 is the validated operating point. 2.2× params (35K → 76K) → 2.9× eval exact (5.33% → 15.59%). The capacity ceiling at h=64 was real and severe. All future experiments should use h=96 as the baseline.

---

### Experiment R2c-h96 — Trajectory ranking at h=96 (position-aware energy head)
Date: 2026-04-14
Script: `scripts/train_h96_scale.sh` (second experiment)
Config: `config/arch/urm_r2c_h96_pos_mlp.yaml` — same backbone as R1-h96 + energy_loss_weight=0.1, position_mlp energy head, energy_head_hidden=32.

**Contingent on R1-h96 showing multi-step convergence.**

Hypothesis: At 2× capacity, the model can serve both reconstruction and energy ranking without the trade-off seen at h=64. Energy pass@K should be non-trivial (>5% at pass@100) while eval exact matches or exceeds R2c's 6.95%.

**Key comparison:**
- vs R2c (h=64): Does energy ranking improve with more capacity?
- vs A1d (h=64, elw=0.5): Does scaling resolve the capacity trade-off that A1d exposed?

Success criteria (ordered by importance):
1. Eval exact ≥ R2c's 6.95% (capacity helps reconstruction)
2. Energy pass@100 > 5% (first practical cross-input energy ranking)
3. Eval Spearman < -0.3 (energy head generalizes better than R2c's -0.069)
4. Per-step monotonic ramp preserved

Key diagnostics: eval exact, energy pass@K, eval Spearman, train Spearman, backbone eval vs R1-h96 (isolate energy co-training effect at new scale).

### Result
**Energy co-training HURTS at h=96 — eval exact 11.78% vs R1-h96's 15.59%.** 80K params. Runtime: 4h 39m.

| Step | Exact Acc |
|------|-----------|
| 1 | 0.12% |
| 2 | 4.73% |
| 3 | 10.47% |
| 4 | 11.93% |
| 5 | **12.50%** |
| 6 | 12.42% |
| 7 | 12.20% |
| 8 | 11.78% |

Train exact: 33.98%. Train/eval ratio: 2.88×. pass@K (Q-halt): 18.18% @1, 29.87% @10, 34.42% @100, 35.71% @1000. Energy pass@100: 4.55% (vs R2c h=64 ~1.3%). Eval Spearman: +0.007 (statistically independent of quality, worse than R2c's -0.069).

**Inversion finding:** At h=64, R2c beat R1i by +1.6pp. At h=96, R2c-h96 lags R1-h96 by -3.8pp. The trajectory ranking benefit was a capacity-starvation effect — the auxiliary loss provided gradient regularization that helped an underpowered backbone, but at h=96 where the backbone has sufficient capacity, the energy objective actively competes for representational budget. Train exact ALSO drops (36.33% → 33.98%), confirming the energy head isn't trading train for eval — it's consuming capacity that would otherwise serve reconstruction. This is NOT a universal multi-task regularization mechanism; it was a crutch for a capacity-starved model.

**Energy head still fails:** Eval Spearman moved from -0.069 (h=64) to +0.007 (h=96). More capacity gave the energy head more parameters to overfit the within-trajectory ranking without transferring to cross-input quality. The step-index shortcut is robust to scale.

**Implication:** elw=0.1 at h=96 is in the "starving reconstruction" regime (like elw=0.3-0.5 at h=64). The trajectory ranking training signal itself needs to change — either ranking noise to break the shortcut, or cross-trajectory pairs where the shortcut doesn't apply.

---

### Experiment R2g-h96 — Ranking noise at h=96
Date: 2026-04-14
Script: `scripts/train_r2g_r2i_h96.sh`
Config: `config/arch/urm_r2g_h96_ranking_noise.yaml` — h=96 backbone + trajectory ranking with `ranking_noise_sigma=0.01`, `energy_loss_weight=0.05`, position_mlp energy head.

**Motivation:** The step-index shortcut (confirmed by A3, unchanged at h=96) lets the energy head learn "step 6 > step 1" without inspecting hidden state quality. Ranking noise adds Gaussian noise to hidden states before the energy head scores them, corrupting the precise features that encode step identity while preserving coarse quality differences. The energy head is forced to learn noise-invariant quality features.

**Key design choices:**
- σ ~ Uniform(0, 0.01) per step (fresh sample each step): average σ=0.005, comparable to mid-trajectory delta norms (0.003-0.01 at h=96).
- Noise only affects energy head inputs — reconstruction loss, Q-halt loss, and quality labels all use the clean hidden states.
- elw=0.05 (half of R2c-h96's 0.1) to limit reconstruction impact in a regime where any energy weight already hurts.
- Quality labels computed from the clean forward pass — noise doesn't change the ranking targets, only the features used to predict them.

Hypothesis: Ranking noise breaks the step-index shortcut, forcing the energy head to learn features that generalize across inputs. Eval Spearman should improve significantly from R2c-h96's +0.007.

Success criteria:
1. Eval exact ≥ 14% (backbone not significantly hurt)
2. Eval Spearman < -0.2 (energy head generalizes)
3. Energy pass@100 > 10% (first practical cross-input energy ranking)

### Result
**Ranking noise partially works — first-ever meaningful cross-input energy generalization.** 79,650 params. Runtime: 4h 40m.

| Step | Exact Acc |
|------|-----------|
| 1 | 0.29% |
| 2 | 3.29% |
| 3 | 10.27% |
| 4 | 12.83% |
| 5 | 13.62% |
| 6 | **13.88%** |
| 7 | 13.82% |
| 8 | 13.59% |

Train exact: 32.62%. Train/eval ratio: 2.40× (nearly matches R1-h96's 2.33×, much better than R2c-h96's 2.88×). pass@K (Q-halt): 16.23% @1, 28.57% @10, 31.82% @100, 37.01% @1000. Energy pass@K: 0 @1, 0 @5, 0 @10, **5.84% @100**, 25.97% @1000. Train Spearman: −0.958. **Eval Spearman: −0.227** (vs R2c-h96's +0.007).

**Success criterion scorecard:**
- ✅ Eval Spearman < −0.2 (achieved −0.227 — the mechanism works)
- ❌ Eval exact ≥ 14% (achieved 13.59%, missed by 0.4pp)
- ❌ Energy pass@100 > 10% (achieved 5.84%, short of the "practical ranking" bar)

**Interpretation:** Ranking noise is directionally correct and measurably breaks the step-index shortcut — the first energy head in the project with meaningful negative eval Spearman at h=96. But at σ=0.01 / elw=0.05 the backbone still pays −2.0pp vs R1-h96 (on training capacity, not overfitting — train exact also drops from 36.3% to 32.6%), and the ranking precision is still well below Q-halt. Energy pass@K stays below practical reranking thresholds because the negative Spearman is weak (−0.23 vs Q-halt's implicit strong ordering).

**What this unlocks:** A tighter elw sweep (0.02, 0.03, 0.05-done, 0.075) at h=96 should find a setting where the backbone recovers to R1-h96 levels while preserving the negative Spearman. A σ sweep (0.005, 0.01-done, 0.02, 0.03) could sharpen the ranking at the cost of more backbone noise. R2g is the most promising energy-head follow-up in the R2 series.

---

### Experiment R2i-h96 — Cross-trajectory ranking at h=96 (cross-traj only, no within-traj)
Date: 2026-04-14
Script: `scripts/train_r2g_r2i_h96.sh` (second experiment)
Config: `config/arch/urm_r2i_h96_cross_traj.yaml` — h=96 backbone + `cross_trajectory=true`, `energy_loss_weight=0.1`, position_mlp energy head.

**Motivation:** Cross-trajectory ranking creates pairs where step index is held constant by construction. For the same recurrence step t, compare hidden state quality across different augmentations of the same puzzle. The energy head literally cannot use step depth as a shortcut because both items in every pair share the same step.

**Cross-trajectory only — within-trajectory ranking is dropped entirely.** Deep supervision in the reconstruction path already teaches step-wise improvement (weighted per-step reconstruction + Q-halt losses with `(t+1)/N` linear ramp). The within-trajectory ranking term is redundant for the backbone AND is precisely the source of the step-index shortcut diagnosed in A3 and R2c-h96 (eval Spearman +0.007 despite perfect train Spearman). Keeping within-traj alongside cross-traj would re-inject the shortcut the cross-traj term is designed to avoid. So R2i runs cross-trajectory as the *only* energy-head training signal.

**Why elw bumped to 0.1 (from the initial R2g setting of 0.05):** Cross-trajectory gradients are constructive — they push the backbone toward features that discriminate quality across same-depth augmentations, which is aligned with the reconstruction objective (same-depth augmentations that solve the puzzle should look different from augmentations that don't, regardless of step). Within-trajectory gradients at h=96 competed for capacity (R2c-h96 inversion), but cross-trajectory gradients should reinforce rather than compete. The higher weight pushes harder on the cross-traj signal when it's the only ranking term in play.

**Pipeline finding that makes this free:** `puzzle_dataset._sample_batch` already fills each training batch with augmentations of a single puzzle (batch=512, num_aug=1000, append_size=512 on the first iteration, loop exits). So intra-batch same-step pairs are naturally same-puzzle cross-augmentation pairs. No dataloader change, no extra forward passes — cross-trajectory pairs reuse the per-example energies already computed for the within-trajectory ranking loss. VRAM cost is a [B, B]=512² pairwise matrix per step (negligible vs the transformer forward pass).

**Implementation:** At each step t, compute the [B, B] all-pairs quality-diff and energy-diff matrices. For every `quality_diff > 0` pair, penalize `F.relu(energy_diff + margin)`, weighted by `quality_diff` as the gap. Mean over valid pairs per step, then mean over active steps. In the code path, when `cross_trajectory=True` the within-trajectory loop in `trajectory_ranking_loss` is skipped entirely — `total_loss = recon_loss + 0.5 * qhalt_loss + elw * cross_traj_loss`.

**Why this is orthogonal to ranking noise:** Ranking noise corrupts shortcut features but keeps the same within-trajectory pairs (step i vs step j). Cross-trajectory creates entirely new same-step pairs where the shortcut doesn't apply. If hidden states genuinely lack quality-discriminative features at matched depth, both fail. But A3's eval Spearman of -0.264 with the detached head suggests discriminative features DO exist in natural hidden states — the coupled head just doesn't learn to use them because the shortcut is easier.

**Key design choices:**
- Cross-trajectory only, no within-trajectory ranking.
- [B, B] all-pairs at each step. At B=512 this is ~131K potential pairs per step × 8 steps. Most will be ties or small gaps; `quality_diff > 0` filters strict pairs, `quality_diff` weights them.
- elw=0.1 (higher than R2g's 0.05) — cross-trajectory gradients are constructive so the bigger weight should help, not hurt.
- Quality measured against each augmentation's own ground truth.
- Diagnostics logged: `cross_traj_active_pairs` (strictly-beaten pair count) and `cross_traj_quality_std` (std of quality across the B augmentations per step, averaged across steps). If `cross_traj_quality_std` ≈ 0 across training, augmentations have nearly identical quality and the signal is empty — kill early.

Hypothesis: Cross-trajectory pairs force the energy head to learn quality features that transfer across inputs, since the step-index shortcut is structurally eliminated.

Success criteria: same as R2g-h96.

### Result
**Cross-trajectory-only fails via train/eval distribution mismatch — eval Spearman +0.118 (anti-aligned).** 79,650 params. Runtime: 4h 40m.

| Step | Exact Acc |
|------|-----------|
| 1 | 0.21% |
| 2 | 3.69% |
| 3 | 10.08% |
| 4 | 11.38% |
| 5 | 12.46% |
| 6 | **12.84%** |
| 7 | 12.57% |
| 8 | 11.99% |

Train exact: 33.40%. Train/eval ratio: 2.79×. pass@K (Q-halt): 18.18% @1, 29.87% @10, 34.42% @100, 37.01% @1000. Energy pass@K: 0 @1, 0 @5, 0 @10, **2.60% @100**, 17.53% @1000. Train cross_traj_active_pairs per forward: 962K. Train cross_traj_quality_std per forward: 0.113. Eval cross_traj_active_pairs: 807K. Eval cross_traj_quality_std: 0.065. **Eval Spearman: +0.118** (worse than R2c-h96's +0.007 — the head is anti-aligned with eval quality).

**Mechanism working but misdirected.** All diagnostic evidence shows the cross-trajectory mechanism ran as designed: within_active_pairs = 0 (within-trajectory correctly disabled), cross_traj_active_pairs ≈ 962K/forward at train and 807K at eval, cross_traj_quality_std > 0 at both train and eval (there IS real quality variance across augmentations to train on). Train-time diagnostics look fine. The problem is not implementation — it's distribution.

**Root cause: train ≠ eval distribution for the energy head.** `puzzle_dataset._sample_batch` fills each *training* batch with 512 augmentations of one puzzle. Cross-trajectory ranking teaches the head "within this puzzle's augmentation family, which augmentation is better-solved at this step?" — features that discriminate rotations/reflections of one specific puzzle can be highly puzzle-specific. Eval batches, however, contain heterogeneous puzzles, and `energy_accuracy_spearman` ranks across *different* inputs. The features the head learned (within-puzzle augmentation discriminators) do not transfer to cross-puzzle quality discrimination, and at +0.118 they actively *anti*-align.

**Success criterion scorecard:**
- ❌ Eval exact ≥ 14% (achieved 11.99%, essentially tied with R2c-h96's 11.78%)
- ❌ Eval Spearman < −0.2 (achieved +0.118 — worse than baseline R2c-h96)
- ❌ Energy pass@100 > 10% (achieved 2.60% — regression from R2c-h96's 4.55%)

**Interpretation:** Cross-trajectory ranking is a **clean negative result** for the paper. It demonstrates that structural shortcut removal is insufficient when the training distribution is narrower than the eval distribution. Fixing the distribution mismatch requires re-plumbing `_sample_batch` to mix multiple puzzles per training batch, which (a) halves effective examples-per-puzzle and (b) is a substantial data pipeline change. The elw=0.1 bump (based on the "cross-traj gradients are constructive" hypothesis) did not help — train exact also dropped by 2.9pp vs R1-h96, confirming cross-traj gradients are no more constructive than within-trajectory gradients when the train distribution is wrong.

**Comparison to R2g-h96**: maps cleanly to the interpretation matrix row "noise works, cross-traj doesn't" → R2g is the mechanism for breaking the shortcut. Cross-trajectory is dead without a dataloader change.

---

### Experiment R3-diag — Eval-only MCMC diagnostic using Q-halt as energy function
Date: 2026-04-15
Script: `scripts/eval_qhalt_mcmc.py`
Checkpoint: R1-h96-baseline-260414 (step 80011, EMA weights — the best backbone: 15.59% eval exact)
Results TSV: `results/qhalt_mcmc_eval.tsv`

**Motivation:** The energy head failed as a verifier (R2 series). But Q-halt already provides a well-calibrated pointwise quality signal — so rather than training a new energy landscape, use Q-halt's pre-sigmoid logit as the energy function for MCMC. Take gradient of Q-halt confidence w.r.t. hidden states, step toward higher confidence. This isolates the MCMC-refinement question from the energy-training question: does *any* gradient-based hidden-space optimization improve predictions beyond URM recurrence, using the best available quality signal?

**Grid:** M ∈ {4, 5, 6} URM steps, K ∈ {1, 2, 4} MCMC steps, step_size ∈ {1e-4, 1e-3, 1e-2, 0.1, 0.3, 1.0}, ±normalized gradient. 108 conditions total.

**Two implementation issues that had to be resolved before the grid ran:**
1. **Gradient localization.** A naive `grad(q_head(hidden[:, 0]).sum(), hidden)` gives a gradient nonzero only at position 0 (the puzzle-embedding slice). `lm_head(hidden)[:, P:]` then discards position 0, so the MCMC update cannot change decoded tokens by construction — the literal spec produces zero change in predictions regardless of step size (verified in a 5-batch sanity run). **Fix:** route Q-halt through one transformer probe at each MCMC step: `grad(q_head(transformer(hidden + input_emb)[:, 0]).sum(), hidden)`. Autograd propagates the gradient through attention back to every position, so the update reaches the token positions lm_head decodes. One forward per MCMC step, preserving matched compute with (M+K) URM.
2. **Step-size grid too small.** Initial sanity run with the spec grid {1e-4, 1e-3, 1e-2} flipped 0.05% of tokens at the largest value — far below anything that can move exact accuracy. The gradient magnitude was still dominated by position 0 (L2 = 18.9) and only weakly leaked to token positions (L2 = 0.38). Extended to {1e-4, 1e-3, 1e-2, 0.1, 0.3, 1.0} to find the regime where predictions actually change. 1.0 with normalized gradient flips ~2.6% of tokens.

### Result
**Q-halt is not a useful energy function for MCMC refinement.** The failure criterion was hit exactly as specified:

- **Best delta vs URM-M (does MCMC improve the starting point?): +0.0001 exact.** Across all 108 conditions, no configuration improves over M-step URM by more than a single puzzle. Most conditions produce 0.0000 or negative deltas.
- **Best delta vs URM-(M+K) (matched-compute): +0.0203 at M=6, K=4, step_size=0.1, norm=0.** This is a **red herring, not an MCMC win**. URM-10 exact (0.1406) is worse than URM-6 (0.1608) because R1-h96 peaks at step 6 and degrades at steps 7–8+ (confirmed in the R1-h96 per-step table). So MCMC at M=6 "wins" vs URM-(6+4) only by *failing to move the hidden state* — it effectively freezes at step 6 while pure URM runs past the peak. Not a refinement effect.
- **Adversarial Q-halt optimization confirmed.** At large step sizes, Q-halt confidence soars while exact accuracy drops:

| M | K | step_size | norm | q_before → q_after | exact_before → exact_after |
|---|---|---|---|---|---|
| 4 | 4 | 1.0 | 1 | −3.749 → **+5.368** | 0.1397 → **0.1303** |
| 5 | 4 | 1.0 | 1 | −3.626 → **+5.491** | 0.1522 → **0.1440** |
| 6 | 4 | 1.0 | 1 | −3.601 → **+5.539** | 0.1608 → **0.1534** |

At step_size=1.0 K=4 normalized, Q-halt's sigmoid moves from ~0.023 to ~0.996 — a 40× increase in stated confidence — while exact accuracy drops ~0.7–0.9pp. Q-halt is being successfully fooled into high confidence in regions of hidden-state space where lm_head decoding is equal or worse.

- **The "sweet spot" is no movement.** The smallest step sizes (1e-4, 1e-3) produce 0.0000 delta because the hidden state barely changes; step_size=0.1 is approximately neutral; step_size ≥ 0.3 begins hurting. There is no regime where Q-halt-directed MCMC helps.

**Key diagnostic evidence on the two failure modes:**
- **Gradient imbalance:** Even with the transformer probe, gradient L2 at position 0 (puzzle-emb slice, ~18.9) is ~50× larger than at token positions (~0.38). Attention leaks signal to token positions but weakly. Normalized gradient fixes the per-position magnitude but the *direction* at token positions is still dominated by features optimized for Q-halt rather than lm_head decoding.
- **Manifold mismatch:** Q-halt's gradient direction does not align with the direction that improves lm_head decoding. Q-halt was trained to predict "is this hidden state's argmax correct?" pointwise — it reads out from the puzzle-emb token after a full URM step. The features that make Q-halt confident are NOT the features that make lm_head decode the correct answer. Optimizing Q-halt's readout is orthogonal (or adversarial) to improving the decoded prediction.

### Conclusion

**Pure Q-halt gradient MCMC does not improve predictions at any (M, K, step_size) condition on the R1-h96 backbone.** The literal failure criterion from the experiment spec is satisfied: MCMC accuracy ≤ URM accuracy at all conditions AND Q-halt confidence increases adversarially as step size grows. This rules out the cheapest version of hidden-space refinement using the existing quality signal.

**What this means for R3.** R3 (trained MCMC with dual loss and second-order gradients) is NOT obviated by this result — the failure here is specifically about using a *pointwise classifier trained on unmoved hidden states* as an energy function after-the-fact. An energy function trained *with MCMC in the loop* (create_graph=True, dual reconstruction loss) would be co-adapted to its own gradient direction, which is exactly the property Q-halt lacks. The docs_architecture MCMC design (energy head + dual loss) is the right path; the shortcut of "reuse Q-halt as energy" does not work.

**What this tells us about R1-h96's trajectory.** The best step is step 6 (0.1608 exact) and the model degrades over steps 7–8. The fact that Q-halt-optimized MCMC can't recover this suggests the degradation isn't fixable by staying in the same "quality" basin — moving toward higher Q-halt confidence takes you *away* from the lm_head decoding optimum. The degradation is a fundamental backbone issue (likely: later steps over-refine into puzzle-specific features that lose generality), not an off-manifold wandering that MCMC could reverse.

**Recommendation:** Do not pursue Q-halt-gradient MCMC further. If R3 is pursued, it must train a dedicated energy function with MCMC in the training loop. Otherwise, focus on R2g-series extensions (elw/σ sweeps) or accept that the energy-refinement path is paper-ready as a negative result with clean ablations.

---

### Experiment R3 — Hybrid URM + MCMC training at h=96
Date: 2026-04-15
Script: `scripts/train_r3_hybrid_h96.sh`
Config: `config/arch/urm_r3_hybrid_h96.yaml` — depth=1, h=96, 4 heads, expansion=2, 8 total steps (5 URM + 3 MCMC), dropout=0.1. Linear energy head (mean pool → Linear(96, 1)). MCMC step_size=0.01 (normalized, randomized per training step in [step/3, step]). Langevin noise σ=0.003 during training only. No trajectory ranking loss (`energy_loss_weight=0.0`) — energy head trained purely through MCMC reconstruction loss via `create_graph=True`. Separate gradient clipping: backbone=5.0, energy_head=1.0.

**This is the core experiment of the project.** Every prior energy-head failure had a specific, diagnosable cause:
1. Contrastive collapse (Exp 1) → solved by trajectory ranking / deep supervision.
2. Step-index shortcut (R2c/R2c-h96) → not applicable, no trajectory ranking.
3. Train/eval distribution mismatch (R2i-h96) → not applicable, no cross-trajectory ranking.
4. Position-0 gradient bottleneck (R3-diag) → solved by full-sequence `compute_joint_energy` (mean pool over all positions, not the puzzle-emb token alone).
5. Manifold mismatch between Q-halt readout and lm_head decoding (R3-diag) → solved by `create_graph=True` training on MCMC-moved states, which forces the energy landscape to be co-adapted to its own gradient direction.

R3 addresses all five by construction. The energy head learns through its own optimization process: reconstruction loss on MCMC-refined predictions flows back through `create_graph` into the energy head, forcing it to learn a landscape where gradient descent produces better hidden states.

**Training signal flow:**
- Steps 1–5 (URM): standard deep supervision (reconstruction + Q-halt BCE, weighted by `(t+1)/8`). First-order gradients only. These steps are protected from MCMC interference.
- Steps 6–8 (MCMC): hidden states are refined by energy gradient descent with `create_graph=True`. Deep supervision applies reconstruction loss at each MCMC step. This reconstruction loss has a computational graph that includes the energy head — backpropagation through this graph trains the energy head to produce a landscape where descent improves predictions.
- Dual loss is automatic: URM steps get their own reconstruction loss (protecting URM learning), and MCMC steps get separate reconstruction loss (training the energy landscape). No explicit dual loss term needed.

**Implementation notes:** `_mcmc_step` now takes `mcmc_langevin_noise` and `mcmc_random_step_size` from config. Training path: `effective_step = step * Uniform(1/3, 1)` when random, then `hidden = hidden - effective_step * normalized(grad) + langevin_noise * randn`. Eval path: deterministic step, no noise, detach between steps to save memory. `EnergyLossHead` gained hybrid-only metrics: `mcmc_improvement` (Δ exact accuracy from URM endpoint to final MCMC step), `mcmc_urm_end_energy`, `energy_decrease` (final energy − URM endpoint energy), and per-MCMC-step energy `mcmc_step_{k}_energy`.

**VRAM safety:** 5-batch smoke test at batch 512 completed one train-eval cycle without OOM. Full-run estimate: ~8–12 GB based on create_graph overhead. Fallback config (batch 256, epochs 63180, eval_interval 4212) documented in the training script header if OOM appears mid-run.

Hypothesis: The energy head, trained through MCMC optimization (`create_graph=True`), learns a landscape where gradient descent in hidden space improves predictions beyond what URM recurrence achieves at matched total steps.

Expected outcome: `mcmc_improvement > 0` — accuracy after 3 MCMC steps exceeds accuracy at URM step 5. Per-MCMC-step energy should decrease monotonically (landscape is being optimized). URM steps 1–5 should not degrade vs R1-h96 (dual loss protection).

Key comparison: R3 per-step accuracy at steps 6–8 vs R1-h96 per-step accuracy at steps 6–8. R1-h96 peaks at step 7 (16.09%) and slightly degrades at step 8 (15.59%). If R3's MCMC steps maintain or improve beyond step 6, explicit refinement helps.

Success criterion: `mcmc_improvement > 0` AND eval exact at step 8 ≥ R1-h96's step 7 peak (16.09%).
Partial success: `mcmc_improvement > 0` but eval exact < 16.09% — MCMC helps but not enough to beat best URM step.
Failure: `mcmc_improvement ≤ 0` — MCMC degrades predictions, energy landscape is not useful for optimization.

Risk: Second-order gradients may cause training instability (gradient explosion through energy head). Monitored via separate gradient clipping and energy value tracking. MCMC may also be a worse version of additional URM steps — same hidden space but without attention's inductive bias.

### Result
**MCMC is a no-op at step_size=0.01 — updates too small relative to URM step magnitude, NOT a precision issue.** Killed at step 37,775/80,011 (47%, ~5.5h of 11h ETA) once the diagnosis was clear.

Per-step exact accuracy at eval #7 (step 37.5K, pre-kill): 0.33% → 1.00% → 4.94% → 7.87% → **8.40%** (URM end) → 8.39% → 8.39% → 8.39% (MCMC). mcmc_improvement = −7.2e-5 (statistical noise, one puzzle). Reconstruction loss: 0.973 → 0.506 across 7 evals (URM backbone training healthy). Q-halt pass@100 25.97%; energy pass@100 1.95%.

**Energy head learned a directionally correct landscape.** Strictly monotonic energy decrease across MCMC steps at every single eval: urm_end → mcmc_6 → mcmc_7 → mcmc_8 descending by ~0.002-0.003 per step at every checkpoint. `create_graph=True` training through MCMC reconstruction loss works as designed — the landscape descent is real.

**Initial diagnosis (BF16 underflow) was wrong.** Reported `step_6_delta_norm = 1.6e-5` vs `step_5_delta_norm = 2.8e-3` made it look like MCMC updates were rounding to zero in BF16. Verification by running `EnergyLossHead` directly on R3's step_37486 checkpoint with a real eval batch showed the raw per-batch-mean `step_6_delta_norm = 0.00818` (post-aggregation) — essentially the designed `step_size = 0.01`. The small reported value was a metric aggregation bug in `pretrain.py`: `delta_norm` was a per-batch `.mean()` but the eval loop sums-then-divides-by-count, under-reporting by ~batch_size. Fixed in a separate commit.

**The real diagnosis: step_size=0.01 is a ~150× negligible perturbation.** URM step 5 per-position delta is ~1.5 on R3's partially-trained backbone; MCMC step_size=0.01 produces ~0.01-per-position updates. The ratio (≈150:1) means MCMC can't flip any argmax that URM's previous pass didn't already move.

Q-halt reranking: pass@1 3.90%, pass@100 25.97%, pass@1000 27.27%. Energy reranking: pass@1 0%, pass@100 1.95%, pass@1000 16.23%.

**Key finding:** Trained energy-gradient MCMC at create_graph=True produces a directionally correct descent landscape, but a step_size of 0.01 is too small to compete with URM's natural step magnitude. R3b tests step_size=0.1.

**Metric aggregation bug:** `step_k_delta_norm` and `current_energy` were under-reported by ~batch_size (~512×) across ALL prior experiments. Fixed in the commit immediately preceding R3b. Historical values in this log for these two metrics should be multiplied by ~512 for correct magnitudes. Accuracy, reconstruction loss, Q-halt, exact_accuracy, and trajectory ranking metrics were unaffected.

---

### Experiment R3b — Meaningful MCMC magnitude (step_size=0.1)
Date: 2026-04-15
Script: `scripts/train_r3b_hybrid_h96.sh`
Config: `config/arch/urm_r3b_hybrid_h96.yaml` — identical to R3 except `mcmc_step_size=0.1` (was 0.01). With `mcmc_random_step_size=true`, training samples from [0.033, 0.1]; eval uses the fixed 0.1. All other fields unchanged: depth=1, h=96, 4 heads, expansion=2, 8 total steps (5 URM + 3 MCMC), dropout=0.1, linear energy head, Langevin noise σ=0.003 during training, `energy_loss_weight=0.0` (no trajectory ranking), grad clips 5.0 / 1.0.

**Motivation.** R3 proved the energy head learns a directionally correct landscape (monotonic descent at every eval) but step_size=0.01 produces ~0.01-per-position updates vs URM step 5's ~1.5 per-position — a ~150× negligible perturbation. At step_size=0.1, per-position updates become ~0.1 vs URM's ~1.5, i.e. ~7% of URM magnitude. That's the regime where MCMC can actually flip argmax decisions.

**R3-diag context.** R3-diag showed Q-halt gradient MCMC going adversarial at step_size ≥ 0.3 on R1-h96. But Q-halt had position-0 gradient concentration (50× imbalance) and was never trained for MCMC. R3b's energy head operates on the full sequence (no position-0 bottleneck) and is trained through `create_graph=True` on MCMC-moved hidden states (no manifold mismatch). step_size=0.1 is the first honest test of a co-adapted energy landscape at a meaningful magnitude.

**Hypothesis:** The create_graph-trained energy landscape directs hidden states toward decoding quality. At step_size=0.1, the landscape's descent produces accuracy improvements rather than adversarial degradation.

Success criterion: `mcmc_improvement > 0` AND eval exact at step 8 ≥ R1-h96's step 7 peak (16.09%).
Partial success: `mcmc_improvement > 0` but eval exact < 16.09%.
Failure mode A (adversarial): `mcmc_improvement < 0` with `energy_decrease < 0` — landscape direction doesn't align with decoding quality.
Failure mode B (collapse): energy values flat across MCMC steps — head collapsed under larger gradient magnitudes.
Failure mode C (instability): NaN or divergent training — second-order grads at 10× step size overwhelm gradient clipping.

Risk: 10× step size means the energy head's gradient is sent through a 10× louder signal, which could destabilize training. Mitigations: grad clip at 1.0 on energy head params, randomized step size during training, Langevin noise, `ema_rate=0.999` for eval.

### Result
**Clean negative: energy landscape descends correctly but orthogonally to decoding quality. MCMC at step_size=0.1 produces meaningful hidden perturbations but zero accuracy improvement.** 76,610 params. Runtime: 12h 11m.

Per-step exact accuracy (final eval, step 80K):

| Step | Exact Acc | Phase |
|------|-----------|-------|
| 1 | 0.31% | URM |
| 2 | 3.24% | URM |
| 3 | 10.54% | URM |
| 4 | 12.97% | URM |
| 5 | **13.15%** | URM endpoint |
| 6 | 13.14% | MCMC |
| 7 | 13.14% | MCMC |
| 8 | 13.13% | MCMC |

Train exact: 32.03%. Train/eval ratio: 2.44×. pass@1: 16.88%, pass@100: 35.71%, pass@1000: 38.31%. Energy pass@100: 3.25%, energy pass@1000: 20.13%. `mcmc_improvement = −0.000204` (−0.02%, 17 puzzles worse). `energy_decrease = −0.158` (monotonic at all 15 evals).

**Success criterion scorecard:**
- ❌ mcmc_improvement > 0 (achieved −0.02% — slightly negative, not improving)
- ❌ Eval exact ≥ 16.09% (achieved 13.15% — 2.94pp below R1-h96 peak, backbone capacity cost)
- ✅ No training instability (clean 80K run, no NaN)
- ✅ Energy landscape monotonically descending at every eval (energy_decrease −0.145 to −0.158)
- ✅ MCMC delta_norm = 0.10 ≈ step_size (meaningful perturbation at 7% of URM magnitude)

**Failure mode: A (orthogonal landscape).** Energy descent ≠ decoding improvement. mcmc_improvement oscillated across 15 evals: 4 positive, 9 negative, median −0.016%. The energy head learns to minimize a function of hidden states, but the minimum-energy direction in the 96-dim hidden space does not point toward better lm_head argmax. This is not adversarial (would show large negative mcmc_improvement) — it's orthogonal (small random perturbation effect).

**Backbone cost:** R3b's URM (13.15% step 5) lags R1-h96 (15.59% step 8, 16.09% peak at step 7) by 2.94pp. The second-order gradient signal from MCMC steps consumes backbone capacity. Energy co-training is a net negative: backbone pays 2.94pp cost, MCMC provides no compensating accuracy improvement. Compare to R2c-h96 where energy co-training also cost backbone accuracy (−3.81pp vs R1-h96 at h=96) — the "structured multi-task regularization" benefit seen at h=64 does not survive at h=96 regardless of training method.

**What R3 + R3b prove together:**
1. `create_graph=True` through MCMC successfully trains the energy head to produce a descending landscape (R3 at step_size=0.01 showed −0.008 descent, R3b at 0.1 showed −0.158 descent — both strictly monotonic at every eval).
2. The descent direction is not aligned with reconstruction quality at either step size. At 0.01: updates are too small to test (R3's metric bug finding). At 0.1: updates are meaningful (delta_norm = 0.1, 7% of URM magnitude) but don't flip argmax in the quality-improving direction.
3. The energy head, even when co-adapted to its own gradient through second-order training, does not learn a landscape where gradient descent in hidden space improves lm_head decoding. The fundamental obstacle is that the energy surface and the reconstruction quality surface are different functions of hidden states with different critical points.

### Experiment R4 — Backbone architecture probes at h=96

Motivation: R2/R3 concluded that energy-based refinement cannot compete with transformer recurrence at matched compute. Next question: does the URM backbone itself have headroom?

**Failure mode analysis (pre-experiment diagnostic, complete).** Bucketed all R1-h96 eval items:
- Correct: 16.08%
- Oscillating (wrong + step-8 delta_norm > 0.003): 83.92%
- Stuck (wrong + converged, qhalt didn't fire): ~0%
- Halted early (wrong + qhalt fired): small
- 88% of wrong predictions correctly don't fire Q-halt — Q-halt is well-calibrated
- Top ~20 puzzles wrong on ~2000 augs each (systematically un-learned patterns)

See `logs/failure_modes_R1h96.txt`. This diagnosis redirected the R4 plan: oscillation, not stuck-minima, is the dominant failure mode.

### R4a — Learnable register tokens [SKIPPED]
Config: `config/arch/urm_r4a_registers_h96.yaml` (committed but not run).
Mechanism: extra scratchpad capacity to escape local minima.
Rationale for skipping: Failure mode analysis showed ~0% of wrong predictions are stuck-in-minima. The mechanism registers target doesn't match the observed failure distribution. Code changes retained (num_registers defaults to 0, no-op) as documented architectural option for future work.

### R4b — Exclusive self-attention (XSA)
Config: `config/arch/urm_r4b_xsa_h96.yaml`. Self-attention diagonal masked (a_ii = -inf); each token attends exclusively to other tokens.
Hypothesis: removing self-attention concentration allows attention heads to do more context-integration work per step, sharpening hidden state representations. Softer motivation post-failure-mode-analysis (still plausibly attacks self-reinforcing attention patterns that could drive oscillation).
Success criterion: eval exact > 16.1%.
Implementation: SDPA fallback (flash_attn doesn't support arbitrary masks). ~10-20% throughput hit acceptable at this scale. No registers (clean isolation: test XSA independently first).

### Result
TBD

### R4c — Doubled refinement budget (loops=16)
Config: `config/arch/urm_r4c_loops16_h96.yaml`. Identical to R1-h96 baseline except loops=16 (was 8) and min_steps=16. Zero code changes.
Hypothesis: If oscillation at step 8 is slow-convergence (refinement-limited), doubling the budget should contract delta norms and lift accuracy. If oscillation is a genuine limit cycle (capacity-limited), step-16 behavior will mirror step-8 and accuracy will plateau.
Expected outcomes (diagnostic, not pass/fail):
  - Convergence scenario: step-16 delta_norm < 0.001, eval exact > 20%
  - Limit-cycle scenario: step-16 delta_norm ≈ step-8 delta_norm, accuracy flat
  - Mixed: partial decay, modest accuracy gain
Runtime note: ~20-22h at loops=16 (vs ~10-12h at loops=8). Primary diagnostic (delta_norm trajectory) visible at first eval (~1.5h), so early kill is viable if the limit-cycle scenario is clearly in play.

### Result
TBD

---

## Lessons Carried Forward
1. **Dual reconstruction loss is mandatory** for MCMC training (Exp 3a).
14. **Energy gradient is orthogonal to decoding quality** (R3/R3b). A create_graph-trained energy head learns a monotonically descending landscape, but the descent direction doesn't improve lm_head argmax. MCMC at step_size=0.1 produces 0.1-per-position perturbations that are meaningful in magnitude but neutral-to-negative for accuracy. The energy surface and reconstruction quality surface have different geometries in hidden space.
15. **Metric aggregation matters** (R3 → R3b). Batch-mean metrics (delta_norm, current_energy) divided by sample-count in pretrain.py were under-reported by ~batch_size (~512×). This caused a multi-hour diagnostic detour (false BF16 underflow hypothesis). Fixed by returning batch-sums instead.
2. **Contrastive loss alone causes energy collapse** (Exp 1). Use trajectory ranking as primary energy training signal (first-order, R2). MCMC reconstruction provides second-order signal in R3.
3. **Right-size the model for the problem.** The model must struggle enough that refinement has room to help.
4. **MCMC must operate in hidden space**, not soft-embedding space.
5. **Don't detach between MCMC steps during training.** Energy head needs multi-step trajectory gradients.
6. **Budget in steps, not epochs.**
7. **Deep supervision is the fix for flat per-step curves** (R1h). Without cross-step gradient flow, each recurrence step trains independently and convergence is flat. Reference: TRM paper.
8. **Dropout=0.1 reduces backbone overfitting** (R1i). Train/eval exact accuracy ratio compressed from 6:1 to 3.9:1. Eval exact improved 42% (3.76% → 5.33%). Use as standard backbone for all future experiments.
9. **Within-trajectory ranking ≠ cross-input ranking** (R2/R2b/R2c). The energy head learns to rank steps within a trajectory perfectly (train Spearman → -1.0) but completely fails to rank predictions across different inputs (energy pass@K near zero). This is the fundamental failure mode of trajectory ranking for verification.
10. **Energy co-training helps URM even when energy ranking fails** (R2c). Position-aware energy head produced best eval exact accuracy (6.95%) and best Q-halt pass@K despite near-zero eval Spearman. The multi-task gradient acts as a regularizer.
11. **More energy head capacity ≠ better generalization** (R2c vs R2b). Position-aware MLP (2K params) had worse eval Spearman (-0.069) than linear head (-0.585). The problem is the training signal (trajectory ranking), not the architecture.
12. **Mean pooling destroys spatial info but that's not the bottleneck** (R2c). Preserving per-position information didn't help cross-input ranking. The energy head sees backbone features that encode puzzle-specific patterns, not abstract quality.
13. **Energy co-training is structured multi-task regularization, not generic gradient noise** (A1/A2/A3). R2c's backbone gain requires both a correctly-ordered trajectory signal (A2 random labels → catastrophic regression, 5.33% → 1.31%) and gradient flow into the shared backbone (A3 detach → eval exact snaps back to R1i baseline). A1 additionally shows an inverse-U in loss weight peaked at elw=0.1–0.2 — weight 0.5 collapses reconstruction. Any story that reduces R2c to "extra parameters" or "any auxiliary gradient helps" is ruled out.
