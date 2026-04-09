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
Single NVIDIA RTX 3090 (24GB VRAM), home lab. All experiments must fit in this envelope. Training budget: 10K steps per experiment. Second-order gradients (create_graph=True) are tractable at this model scale. Target model: depth=2, hidden=64, ~130K transformer params on 10×10 grids.

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

### Dual reconstruction loss is mandatory with MCMC
When training with MCMC refinement, compute reconstruction loss on BOTH unrefined logits (before MCMC) and refined logits (after MCMC), weighted 0.5/0.5. The unrefined loss keeps the backbone learning cleanly. The refined loss trains the energy head through second-order gradients. Refined-only loss destroys URM learning (confirmed in Exp 3a).

### Legacy code removed
DSM (denoising score matching), contrastive loss, and trajectory supervision have been removed. DSM was unnecessary given tractable second-order gradients. Contrastive-only loss caused energy collapse. Trajectory supervision is a future Phase 4 extension. The only energy training signal is reconstruction-through-MCMC (dual loss). Deleted files: `models/dsm_loss.py`, `models/trajectory_loss.py`, and their associated configs/scripts.

### Three forward modes implemented
`URMConfig.mode` controls the refinement mechanism:
- **"urm"** (default): N iterations of shared-weight transformer recurrence (implicit refinement). Q-halt or fixed-step stopping.
- **"ebt"**: N steps of MCMC gradient descent in hidden space starting from input_embeddings (explicit refinement). Energy convergence stopping. Enables Experiment R3.
- **"hybrid"**: M URM recurrence steps then (N-M) MCMC steps, controlled by `mcmc_start_step`. Enables Experiment R4.

EBT and hybrid modes always halt after one forward() call (all compute happens in that call). Both use dual reconstruction loss (unrefined + refined) when training.

### Per-step accuracy logging
At eval time, URM mode captures logits at every recurrence step. EnergyLossHead computes `step_k_accuracy` metrics for each step k=1..loops, logged to wandb. Use this to determine at which step the model's predictions plateau — critical for right-sizing the model.

### Right-size the model for the problem
The model must need most of its step budget to converge. At hidden=128, the URM converges in 1-2 steps on 10×10 grids, leaving no room for refinement to help. Target: hidden=64, depth=2, where accuracy should meaningfully improve between steps 4 and 8. Run R1 experiment: `scripts/train_r1_scale.sh`.

## Setup

```bash
conda create -n urm python=3.10
conda activate urm
pip install -r requirements.txt
```

### Data preparation (10x10 grids)
```bash
python -m data.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000-size-10 \
  --subsets training evaluation concept \
  --test-set-name evaluation \
  --num-aug 1000 \
  --max-grid-size 10
```

### Running tests
```bash
conda activate urm && python -m pytest tests/ -v
```