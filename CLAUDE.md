# Agent Workflow Rules
## Before Refactoring
- Before any structural refactor on a file >300 LOC, first remove dead props, unused exports, unused imports, and debug logs. Commit cleanup separately before the real work.

## Verification
- Never report a task as complete until you have run the project's available verification steps — type-checking, linting, unit tests, and E2E tests — and fixed all resulting errors. If a verification layer is not configured, state that explicitly rather than claiming success.

## Edit Safety
- Re-read any file immediately before editing it. After editing, read the file again to confirm the change applied correctly. Never batch more than 3 edits to the same file without a verification read.
- In long conversations (10+ messages), always re-read files before editing. Do not trust your memory of file contents.
- If any search or command returns suspiciously few results, re-run with narrower scope (single directory, stricter glob) and state when you suspect truncation.

## Renaming and Refactoring
When renaming or changing any function, type, or variable, search separately for:
- Direct calls and references
- Type-level references (interfaces, generics)
- String literals containing the name
- Dynamic imports and `require()` calls
- Re-exports and barrel file entries
- Test files and mocks
Do not assume a single grep caught everything.

## Code Quality
- Do not defend or patch around a workaround when the correct solution is straightforward. If data should be stored, store it — do not estimate it from derived values. If a schema change, new field, or additional table is the right fix, propose it directly rather than adding complexity to preserve a hack. When a fix feels "large," verify that assumption before pushing back — state the actual scope of the change. When in doubt, state the tradeoff plainly: "We could patch the formula, but storing the value directly is cleaner because X."

## Scope Management
- For tasks touching >5 independent files, use subagents.
- Flag architectural concerns when you see them. Propose fixes as separate tasks rather than silently expanding scope.

# Project Identity

## What We're Building
**Implicit vs Explicit Iterative Refinement for ARC-AGI**: a controlled comparison of URM recurrence (implicit energy minimization via shared-weight transformer passes) against EBT-style MCMC (explicit energy minimization via gradient descent on a learned energy function) for abstract reasoning.

The current project codebase is located at: https://github.com/uberdavid-bot/URM-Energy-Stopping

## One-Line Summary
On discrete abstract reasoning tasks, does explicit energy minimization (MCMC in hidden space) offer measurable advantages over implicit iterative refinement (recurrent transformer passes), and under what conditions?

## Research Goal
Characterize the tradeoffs between implicit and explicit iterative refinement for ARC-AGI puzzle solving using a shared transformer backbone. URM recurrence and EBT-style MCMC are two mechanisms for iteratively improving predictions — this project determines when each has advantages and whether they are complementary or redundant.

The specific hypotheses under test:
1. **Explicit vs implicit refinement**: Given matched compute (N total steps), how does URM recurrence compare to MCMC gradient descent in hidden space?
2. **Energy reranking**: Does a learned energy function E(input, hidden) provide a better confidence signal for pass@K reranking than Q-halt?
3. **Energy-based stopping**: Is energy convergence a more principled stopping criterion than Q-halt, particularly for adaptive compute allocation?
4. **Complementarity**: Can MCMC refinement improve predictions beyond what URM recurrence achieves when used as a second stage (hybrid architecture)?

## Hardware Environment
Single NVIDIA RTX 3090 (24GB VRAM), home lab. All experiments must fit in this envelope. Training budget: 80K steps per experiment (constant LR after 100-step warmup). Second-order gradients (create_graph=True) are tractable at this model scale. Current baseline: depth=1, h=64, exp=2, ~35K params, 10×10 grids, ~7 it/s, ~5.8GB VRAM.

## Success Criteria
- **Minimum**: Clear characterization of when explicit energy helps vs. when implicit recurrence is sufficient. Energy reranking improves pass@K over Q-halt on matched architecture.
- **Target**: Hybrid architecture (M URM + K MCMC steps) outperforms pure URM at matched total compute.
- **Stretch**: Pure EBT (MCMC from input_embeddings, no URM steps) achieves non-trivial accuracy on ARC, demonstrating that energy gradients alone can drive abstract reasoning.
- **Publishable result**: Controlled ablation comparing implicit vs explicit iterative refinement on discrete reasoning, with analysis of why each succeeds or fails, reproducible on consumer hardware.

## Design Philosophy
- **Shared backbone, isolated mechanism.** URM mode and EBT mode use the same transformer layers, embeddings, lm_head, and energy head. The only difference is the refinement step: transformer pass vs energy gradient step. This isolates the comparison.
- **MCMC in hidden space.** Energy gradients operate on transformer hidden states directly — not soft-embedding space. The energy head and lm_head both expect hidden states; don't introduce distribution mismatches.
- **Train and test the same way.** MCMC at inference requires create_graph=True during training. No detaching between MCMC steps during training.
- **Right-size the model so it struggles.** The model must need most of its step budget to converge. If it plateaus in 1-2 steps, refinement has no room to help.
- **Iterate fast at small scale.** 10×10 grids, ~35K params, 80K steps per experiment.

## Researcher Context
Solo research project by David Colmenares (CMU Robotics PhD, Research Scientist at Meta). Running overnight experiments on a home 3090. The agent should optimize for autonomous reliability over cleverness. When in doubt, run the simpler experiment.

## Key References
- URM paper: arXiv:2512.14693 (Gao et al., 2024)
- Energy-Based Transformers: arXiv:2507.02092 (Gladstone et al., 2025)
- ARC-AGI: arcprize.org (Chollet, 2019)

## Doc Pointers
- `docs_strategy.md` — Research direction, phasing, key decisions
- `docs_architecture.md` — System design: URM mode, EBT mode, shared components
- `docs_constraints.md` — Hardware limits, known failures, anti-patterns
- `docs_hypotheses.md` — Experiment log (R1-R4 series)

## Critical Implementation Notes

These are hard-won lessons. Violating any of these will waste experiment cycles.

### MCMC operates in hidden space (FIXED)
MCMC gradients are taken w.r.t. transformer hidden states directly — the output of the URM recurrence loop. Both compute_joint_energy and lm_head expect hidden states; never convert to soft-embeddings (logits → softmax → embed_tokens). Previous implementation was buggy; now fixed.

### No detach between MCMC steps during training (FIXED)
The energy head learns from multi-step MCMC trajectories via create_graph=True. Do not call `hidden.detach().requires_grad_(True)` inside the MCMC loop during training — this breaks the computational graph. Detach only at inference to save memory. Previous implementation was buggy; now fixed.

### Deep supervision is mandatory
Training uses deep supervision: every recurrence step t gets reconstruction loss weighted by `(t+1)/N` (linear ramp) plus Q-halt BCE with the same weighting. Gradients flow undetached through the full N-step trajectory. This is the primary mechanism for learning multi-step refinement — without it, each step trains independently as a single-pass model. Reference: TRM paper (Jolicoeur-Martineau, 2025).

### Legacy code removed
The codebase has been streamlined to only the active experiment pipeline:
- **Deleted models**: `models/hrm/`, `models/trm/`, `models/urm/urm.py` — the unified `models/urm/urm_energy.py` handles all modes via `ARCModelConfig.refinement`.
- **Deleted losses**: `models/dsm_loss.py`, `ACTLossHead` — DSM was unnecessary; contrastive-only caused energy collapse; ACTLossHead was the old carry-based loss head. Note: `models/trajectory_loss.py` was re-created for R2 (trajectory ranking loss for energy head co-training).
- **Deleted carry infrastructure**: `ModelCarry` dataclass, `ARCBackbone.forward()`, `ARCBackbone.empty_carry()`, `ARCBackbone.reset_carry()`, `ARCModel.forward()`, `ARCModel.initial_carry()`, per-sample halting logic. Replaced by `forward_trajectory()`.
- **MLP rounding fix**: `_find_multiple` granularity changed from 256 to 8 in `models/layers.py` to allow smooth capacity scaling at small model sizes (at h=64/exp=2, inter=88 instead of 256).
- **Active configs**: `config/arch/urm_qhalt.yaml`, `config/arch/ebt_energy.yaml`; h=64 series: `urm_r1i_dropout.yaml`, `urm_r2_trajectory.yaml`, `urm_r2b_trajectory_dropout.yaml`, `urm_r2c_pos_mlp.yaml`, ablation_a1/a2/a3/a4 variants; **h=96 series (current standard)**: `urm_r1_h96_baseline.yaml`, `urm_r2c_h96_pos_mlp.yaml`, `urm_r2g_h96_ranking_noise.yaml`, `urm_r2i_h96_cross_traj.yaml`.
- **Active scripts**: `scripts/train_h96_scale.sh` (R1-h96 + R2c-h96), `scripts/train_r2g_r2i_h96.sh` (R2g + R2i), `scripts/eval_qhalt_mcmc.py` (R3-diag eval-only MCMC), `scripts/eval_energy_ranking.py` (energy ranking comparison), plus legacy per-experiment scripts.

### Flat trajectory forward architecture
`ARCModel.forward_trajectory(batch, N)` is the primary entry point. Runs N recurrence steps in a single call with full gradient flow. `EnergyLossHead.forward(batch)` calls `forward_trajectory` and computes deep supervision loss, per-step metrics, and eval stopping metrics. No carry state, no per-sample halting, no outer loop.

Three model config fields control behavior:
- **`refinement`**: `"urm"` (one transformer pass) | `"ebt"` (one MCMC gradient step) | `"hybrid"` (URM if step < mcmc_start_step, else MCMC) — what one step does.
- **`stopping`**: `"qhalt"` (learned halt signal) | `"energy"` (energy convergence) — eval-only stopping criterion.
- **`ranking`**: `"qhalt"` (Q-halt confidence) | `"energy"` (negative energy) — confidence signal for pass@K reranking.

Three model config fields control regularization (R1i+, A4+):
- **`attn_dropout`**: float (default 0.0). Dropout applied inside flash attention. Passed through `ARCBlock` → `Attention` → `flash_attn_func(dropout_p=...)`. Training only.
- **`mlp_dropout`**: float (default 0.0). Dropout applied after MLP output, before residual add in `ARCBlock`. Standard `nn.Dropout`.
- **`recurrence_noise`**: float (default 0.0). Additive Gaussian noise σ applied to hidden states after each URM transformer pass, training only. A4 experiments tested this as a dropout alternative — it did not help at σ=0.005 (R1h-equivalent) or σ=0.003+dropout=0.05 (below R1i). Documented negative. Dropout=0.1 remains the standard.

Two model config fields control energy head architecture (R2c+):
- **`energy_head_type`**: `"linear"` (default, mean_pool→Linear(H,1), 65 params) | `"position_mlp"` (per-position MLP→sum, ~2K params at h=64 / ~3K at h=96) | `"position_conv_mlp"` (conv+MLP) | `"position_attn_mlp"` (attention+MLP). Implemented via `PositionEnergyHead` class.
- **`energy_head_hidden`**: int (default 32). Hidden dim for per-position MLP in non-linear energy heads.

Four loss config fields control energy head co-training (R2+/R2g/R2i):
- **`energy_loss_weight`**: float (default 0.0). When > 0, adds trajectory ranking loss to total loss. Weight 0.1 used in R2 series at h=64 and is now in the "starving reconstruction" regime at h=96. R2g uses 0.05.
- **`ranking_margin`**: float (default 0.1). Margin for pair ranking loss.
- **`ranking_noise_sigma`**: float (default 0.0). R2g: σ sampled per step ~ Uniform(0, ranking_noise_sigma), applied to hidden states before the energy head scores them. Reconstruction path uses clean hidden states. Training only. Breaks the step-index shortcut (R2g-h96: eval Spearman +0.007 → −0.227).
- **`cross_trajectory`**: bool (default False). R2i: replace within-trajectory ranking with [B, B] same-step all-pairs ranking across augmentations within a batch. When True, the within-trajectory loop in `trajectory_ranking_loss` is skipped entirely; `total_loss = recon + 0.5*qhalt + elw * cross_traj`. Documented negative at h=96 due to train (same-puzzle augs) ≠ eval (heterogeneous puzzles) distribution mismatch.

Two pretrain config fields control gradient clipping:
- **`grad_clip_backbone`**: float (default 5.0). Max gradient norm for backbone parameters.
- **`grad_clip_energy_head`**: float (default 1.0). Max gradient norm for energy head parameters. Name-based filtering (`'energy_head' in param_name`) covers all energy head variants.

### Per-step metrics
Per-step metrics are computed inside `EnergyLossHead` for both train and eval: `step_k_accuracy`, `step_k_exact_accuracy`, `step_k_delta_norm`. Eval mode additionally computes stopping metrics: `qhalt_stop_step`, `qhalt_stop_accuracy`, `energy_stop_step`, `energy_stop_accuracy`.

### Standard backbone: R1-h96 (the validated operating point)
**Current best backbone**: depth=1, **h=96**, expansion=2, 8 steps, num_heads=4 (head_dim=24), attn_dropout=0.1, mlp_dropout=0.1, recurrence_noise=0.0. **~76.6K params (transformer)**. `config/arch/urm_r1_h96_baseline.yaml`. Results: **eval exact 15.59%** (step 8), **16.09% peak (step 7)**, train exact 36.33%, train/eval ratio 2.33× (best seen), pass@1000 40.91%. VRAM at batch=512: ~3.6 GB. Monotonic per-step ramp: 0.24% → 5.44% → 11.60% → 13.97% → 15.22% → 16.08% → 16.09% → 15.59%. Peaks at step 7, slight degradation at step 8.

All new experiments should scale h=96 unless specifically testing scaling. Checkpoint: `checkpoints/R1-h96-baseline-260414/step_80011.pt` (EMA-saved).

**h=64 (R1i) is legacy** — 35K params, 5.33% eval exact, 3.9× train/eval. Scaling h=64 → h=96 delivered 2.9× eval exact from 2.2× params. The capacity ceiling at h=64 was real and severe.

### Scale-dependent findings (h=64 → h=96 inversion)
The R2 energy-head story at h=64 does NOT transfer to h=96:

1. **R2c inverted at scale.** At h=64, R2c (position_mlp energy head, elw=0.1) beat R1i by +1.6pp via multi-task regularization. At h=96, R2c-h96 LAGS R1-h96 by **−3.81pp** (11.78% vs 15.59%). Train exact also drops (36.3% → 34.0%), so the energy objective is consuming backbone capacity, not trading train for eval. The "structured multi-task regularization" benefit was a capacity-starvation crutch, not a universal mechanism.
2. **Ablation series A (A1/A2/A3) is still valid mechanistically** — random labels (A2) catastrophically destroy h=64 backbone, detach (A3) matches R1i baseline — but all A-series experiments were at h=64 and the conclusions about *strength* of regularization are scale-dependent.

### Energy head training status — ranking noise is the mechanism; cross-trajectory is dead; trained energy function needed for MCMC
R2/R2b/R2c (h=64) and R2c-h96/R2g-h96/R2i-h96 (h=96) comprehensively tested trajectory ranking loss for energy head co-training. The story is now complete:

1. **Within-trajectory ranking has a step-index shortcut.** The head learns "step 6 > step 1" without inspecting hidden state quality. Train Spearman → −1.0, eval Spearman ≈ 0 at h=96 (+0.007). A3 (detach) confirmed: when gradient flow is blocked, eval Spearman is *better* (−0.264) — coupling drags the head toward step-identity features.
2. **Scale does not break the shortcut.** h=64 → h=96 made eval Spearman *worse* (−0.069 → +0.007). More head capacity just overfits the within-trajectory ordering harder.
3. **Ranking noise breaks the shortcut (R2g-h96).** Adding σ ~ U[0, 0.01] Gaussian noise to hidden states before the energy head scores them (reconstruction path unchanged) drove eval Spearman to **−0.227** — the first meaningful cross-input generalization in the project. Backbone cost 13.59% vs R1-h96's 15.59% (−2.0pp); energy pass@100 reached 5.84% but still below Q-halt's 31.82%. R2g is the mechanism; elw/σ sweeps at h=96 are the obvious next step if the energy head matters.
4. **Cross-trajectory ranking fails due to train/eval distribution mismatch (R2i-h96).** `puzzle_dataset._sample_batch` fills each training batch with 512 augmentations of ONE puzzle. Cross-trajectory ranking teaches within-puzzle augmentation discrimination, which anti-aligns with cross-puzzle eval ranking (eval Spearman +0.118, worse than R2c-h96). Clean negative result — structural shortcut removal without distribution matching is not enough. Fixing requires dataloader re-plumbing.
5. **Q-halt cannot serve as an energy function for MCMC (R3-diag).** Eval-only MCMC using `∂q_head(transformer(h + input_emb))/∂h` on R1-h96 shows: (a) no condition improves over URM-M exact accuracy (best delta +0.0001), (b) Q-halt is adversarially optimizable — at step_size=1.0 normalized, q shifts from −3.749 to +5.368 (sigmoid 0.023 → 0.996) while exact accuracy *drops*. Q-halt's gradient direction is orthogonal or adversarial to lm_head's decoding direction. A trained energy function with MCMC in the loop (second-order grads, dual reconstruction loss) is required.

**Implication for R3:** R3 must train a dedicated energy head *with MCMC in the training loop* (create_graph=True for second-order grads; dual reconstruction loss on both pre-MCMC and post-MCMC states). Reusing Q-halt as energy does not work. The EBT refinement mode in `models/urm/urm_energy.py` (`_mcmc_step`, `compute_joint_energy`) is the existing scaffolding for this.

## Setup

```bash
conda create -n urm python=3.10
conda activate urm
pip install -r requirements.txt
```

### Data preparation
Raw ARC-AGI JSON files live in `kaggle/combined/` (gitignored). To regenerate datasets at different grid sizes:
```bash
# 10x10 grids (100 tokens)
python -m data.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000-size-10 \
  --subsets training evaluation concept \
  --test-set-name evaluation \
  --num-aug 1000 \
  --max-grid-size 10

# 15x15 grids (225 tokens)
python -m data.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000-size-15 \
  --subsets training evaluation concept \
  --test-set-name evaluation \
  --num-aug 1000 \
  --max-grid-size 15
```

### Running tests
```bash
conda activate urm && python -m pytest tests/ -v
```