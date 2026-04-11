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
Single NVIDIA RTX 3090 (24GB VRAM), home lab. All experiments must fit in this envelope. Training budget: 80K steps per experiment (constant LR after 100-step warmup). Second-order gradients (create_graph=True) are tractable at this model scale. Current R1 sweep: depth=1-2, hidden=64-128, expansion=2-4 on 10×10 grids.

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
- **Iterate fast at small scale.** 10×10 grids, ~130K params, 10K steps per experiment.

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
- **Deleted losses**: `models/dsm_loss.py`, `models/trajectory_loss.py`, `ACTLossHead` — DSM was unnecessary; contrastive-only caused energy collapse; ACTLossHead was the old carry-based loss head.
- **Deleted carry infrastructure**: `ModelCarry` dataclass, `ARCBackbone.forward()`, `ARCBackbone.empty_carry()`, `ARCBackbone.reset_carry()`, `ARCModel.forward()`, `ARCModel.initial_carry()`, per-sample halting logic. Replaced by `forward_trajectory()`.
- **MLP rounding fix**: `_find_multiple` granularity changed from 256 to 8 in `models/layers.py` to allow smooth capacity scaling at small model sizes (at h=64/exp=2, inter=88 instead of 256).
- **Active configs**: `config/arch/urm_qhalt.yaml`, `config/arch/ebt_energy.yaml`, plus R1 experiment configs.
- **Active scripts**: `scripts/run_r1_rerun.sh` (current sweep), plus individual experiment scripts in `scripts/`.

### Flat trajectory forward architecture
`ARCModel.forward_trajectory(batch, N)` is the primary entry point. Runs N recurrence steps in a single call with full gradient flow. `EnergyLossHead.forward(batch)` calls `forward_trajectory` and computes deep supervision loss, per-step metrics, and eval stopping metrics. No carry state, no per-sample halting, no outer loop.

Three config fields control behavior:
- **`refinement`**: `"urm"` (one transformer pass) | `"ebt"` (one MCMC gradient step) | `"hybrid"` (URM if step < mcmc_start_step, else MCMC) — what one step does.
- **`stopping`**: `"qhalt"` (learned halt signal) | `"energy"` (energy convergence) — eval-only stopping criterion.
- **`ranking`**: `"qhalt"` (Q-halt confidence) | `"energy"` (negative energy) — confidence signal for pass@K reranking.

Active configs: `config/arch/urm_qhalt.yaml`, `config/arch/ebt_energy.yaml`, plus R1 experiment configs (`urm_r1g_d1_h64.yaml`, `urm_r1f_d1.yaml`, `urm_r1h_d1_h128_exp4.yaml`, `urm_r1_15x15.yaml`).

### Per-step metrics
Per-step metrics are computed inside `EnergyLossHead` for both train and eval: `step_k_accuracy`, `step_k_exact_accuracy`, `step_k_delta_norm`. Eval mode additionally computes stopping metrics: `qhalt_stop_step`, `qhalt_stop_accuracy`, `energy_stop_step`, `energy_stop_accuracy`.

### Right-size the model for the problem
Validated config: depth=1, h=64, expansion=2, 8 steps, 10×10 grids, ~35K params. R1h confirmed monotonic per-step accuracy ramp (0.13% → 3.76% exact accuracy step 1→6). Prior R1 experiments showed flat per-step curves because hidden states were detached between steps; deep supervision + cross-step gradient flow fixed this.

### Energy head is untrained in URM mode
In URM refinement mode, `compute_joint_energy` is only called inside `torch.no_grad()` blocks (for eval metrics). The energy head receives zero training gradient. The `energy_stop_step` and `energy_stop_accuracy` eval metrics currently measure backbone hidden state convergence (since the energy head reuses backbone transformer layers), not learned energy structure. Training the energy head requires either: (a) trajectory ranking loss (Phase 2, first-order), or (b) EBT/hybrid mode where `_mcmc_step` calls `compute_joint_energy` with `create_graph=True`.

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