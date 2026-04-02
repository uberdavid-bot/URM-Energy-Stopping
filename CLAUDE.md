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

# Project 
# URM-Energy: Energy-Based Stopping for Universal Reasoning Models on ARC-AGI

Fork of the [Universal Reasoning Model (URM)](https://github.com/UbiquantAI/URM) extended with an energy-based transformer (EBT) stopping mechanism as an alternative to Adaptive Computation Time (ACT).

## Motivation

The URM paper ([arXiv:2512.14693](https://arxiv.org/abs/2512.14693)) showed that recurrent inductive bias and strong nonlinearity are the key ingredients for reasoning on ARC-AGI, achieving 53.8% pass@1 on ARC-AGI 1. URM uses ACT (a learned halting probability) to decide when to stop iterating its recurrent computation.

This project explores replacing ACT with an **energy-based stopping criterion**, inspired by [Energy-Based Transformers (Hoover et al., 2024)](https://arxiv.org/abs/2410.09197). Instead of learning a binary "should I stop?" signal, we learn an energy function E(input, output) that scores prediction quality. Iteration stops when energy converges. This provides:

- **Principled stopping**: energy convergence rather than a learned halting head
- **Iterative refinement via MCMC**: gradient descent in output embedding space
- **Built-in verification**: energy score as a confidence measure

## What's Implemented

### New files
- `models/urm/urm_energy.py` — Energy-based URM model with joint energy function E(input, output), MCMC refinement loop with learnable step size (alpha), and Langevin dynamics noise
- `config/arch/urm_energy.yaml` — Hydra config for energy model hyperparameters
- `scripts/URM_energy_arcagi1.sh` — Launch script for energy experiments
- `tests/test_mcmc_inference.py` — Tests for inference-time MCMC and alpha step size

### Modified files
- `models/losses.py` — Added contrastive energy loss (margin-based, pushes E(input, correct) < E(input, predicted))
- `evaluators/arc.py` — Fixed InferenceMode compatibility for single-GPU evaluation
- `pretrain.py` — Integrated energy model forward pass, MCMC loop, and energy metric logging

## Architecture
```
Standard URM:  input → [shared layer × T iterations] → ACT halt? → output
URM-Energy:    input → [shared layer × T iterations] → energy E(input, output)
                                                      → MCMC refinement via ∇E
                                                      → stop when ΔE < threshold
```

The energy function processes concatenated (input, predicted_output) embeddings through the transformer layers, pools, and projects to a scalar energy. Lower energy = better prediction. During inference, predicted embeddings are iteratively refined by descending the energy gradient.

## Experiments

Trained on ARC-AGI-1 with 10×10 downscaled grids on a single RTX 3090 (24GB).

| Run | Config | Notes |
|-----|--------|-------|
| URM baseline | `arch=urm`, batch=32, 10×10 grids | Converged quickly, severe overfitting (train acc ~20-30%, val ~0.6% pass@1) |
| Energy v0 | `arch=urm_energy`, batch=12, 10×10 | Initial energy collapse — energy head output constant for all inputs |
| Energy v1 (+ contrastive loss) | Added margin-based contrastive loss | Fixed collapse: energy function learned to separate correct vs incorrect outputs |
| Energy v2 (tuning) | Removed ACT q_halt_loss, energy-only stopping | MCMC taking only 1-2 steps; needs minimum step count and threshold tuning |

### Key findings
- **Contrastive loss is essential**: Without it, the energy head collapses to a constant. The margin-based contrastive loss (E_true < E_predicted - margin) creates the energy separation needed for MCMC to have useful gradients.
- **Model capacity mismatch**: The full URM architecture is heavily overparameterized for 10×10 grids, causing severe overfitting. Right-sizing the model (smaller hidden dim, fewer layers) or using full 30×30 grids is needed.
- **MCMC step count matters**: With a low energy threshold, MCMC converges in 1-2 steps before the energy landscape is well-shaped. Forcing a minimum number of steps (8+) is needed to give refinement room to work.

## Reproduction

### Setup
```bash
conda create -n urm python=3.10
conda activate urm
pip install -r requirements.txt
wandb login  # optional, for experiment tracking
```

### Data preparation
```bash
# Download ARC-AGI data from Kaggle into kaggle/ directory
python -m data.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-10 \
  --subsets training evaluation concept \
  --test-set-name evaluation
```

### Training (baseline URM)
```bash
torchrun --nproc-per-node 1 pretrain.py \
  data_path=data/arc1concept-aug-10 \
  arch=urm arch.loops=16 arch.H_cycles=2 arch.L_cycles=6 arch.num_layers=4 \
  epochs=100 eval_interval=10 global_batch_size=32 \
  puzzle_emb_lr=1e-2 weight_decay=0.1 +ema=True
```

### Training (energy-based URM)
```bash
torchrun --nproc-per-node 1 pretrain.py \
  data_path=data/arc1concept-aug-10 \
  arch=urm_energy arch.loops=16 arch.H_cycles=4 arch.L_cycles=3 arch.num_layers=4 \
  arch.energy_threshold=0.005 arch.langevin_noise_std=0.01 \
  epochs=200000 eval_interval=1000 global_batch_size=12 \
  puzzle_emb_lr=1e-2 lr=1e-4 weight_decay=0.1 +ema=True
```

## Status & Next Steps

This is early-stage research. The energy-based mechanism shows promise (learned energy separation, working MCMC gradients) but needs further work:

- [x] Fix inference-time MCMC (was producing zero gradients; now uses `torch.enable_grad()`)
- [x] Replace `torch.inference_mode()` with `torch.no_grad()` in `evaluate()` so MCMC can compute gradients
- [x] Force minimum MCMC steps (8+) to allow iterative refinement
- [x] Remove redundant hard-coded 0.1 gradient scaling — alpha now solely controls MCMC step size
- [x] Remove dead code (commented-out ACTLossHead in losses.py, unused replay_buffer.py)
- [x] Lower alpha default to 0.01 (preserves effective step size after 0.1 scaling removal)
- [x] Lower energy_threshold to 0.005 (was 0.1, triggered immediately after min_steps)
- [x] Align training script/README with actual experiment config (H_cycles=4, L_cycles=3)
- [x] Fix tests to run on real CUDA with flash_attn (no mocks); requires `conda activate urm`
- [ ] Right-size model for small grids (hidden_dim 64-128, 2 layers) or scale to 30×30
- [ ] Increase data augmentation for small grids to reduce overfitting
- [ ] Hyperparameter sweep on contrastive loss weight, margin, alpha init
- [ ] Compare energy stopping vs ACT on matched architectures

## References

- [Universal Reasoning Model (URM)](https://arxiv.org/abs/2512.14693) — Gao et al., 2024
- [Energy-Based Transformers](https://arxiv.org/abs/2410.09197) — Hoover et al., 2024
- [ARC-AGI](https://arcprize.org/) — Chollet, 2019

## License

Same as the original URM repository.
