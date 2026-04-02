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
- **Iterative refinement via MCMC**: gradient descent in output embedding space at inference
- **Built-in verification**: energy score as a confidence measure

## What's Implemented

### New files
- `models/urm/urm_energy.py` — Energy-based URM model: URM inner recurrence + energy scoring + inference MCMC refinement
- `models/dsm_loss.py` — Multi-scale Denoising Score Matching loss for training the energy head
- `config/arch/urm_energy.yaml` — Hydra config for energy model hyperparameters
- `scripts/URM_energy_arcagi1.sh` — Launch script for energy experiments
- `tests/test_mcmc_inference.py` — Tests for DSM loss, forward pass, energy halting, MCMC refinement, backward pass

### Modified files
- `models/losses.py` — EnergyLossHead: reconstruction loss + DSM loss (replaces old contrastive+MCMC approach)
- `evaluators/arc.py` — Fixed InferenceMode compatibility for single-GPU evaluation
- `pretrain.py` — Integrated energy model forward pass and DSM metric logging

## Architecture

### Training
```
input → [URM inner: shared layer × H_cycles × L_cycles] → logits + hidden states
                                                         → energy E(input, output_hidden)
Loss = reconstruction_loss(logits, labels) + DSM_loss(energy_head)
```

The URM inner recurrence produces logits and hidden states. The energy head scores E(input, output). DSM trains the energy head's gradients to point from corrupted toward clean outputs — no MCMC backprop needed (O(1) cost, not O(steps)).

### Inference
```
input → [URM inner × T iterations] → energy E(input, output)
                                    → stop when ΔE < threshold & steps >= min_steps
                                    → optional: refine_with_mcmc(logits, ∇E)
```

The outer loop calls forward() repeatedly. Energy convergence across iterations drives halting. Post-hoc MCMC refinement uses the DSM-trained energy gradients to polish predictions.

### Why DSM instead of MCMC-backprop?
The old approach trained the energy head by backpropping through sequential MCMC steps (create_graph=True through the chain). This was ~300x slower and blew up memory on a 3090. DSM trains the same gradient signal (energy pointing toward correct outputs) with a single gradient computation per training step.

## Experiments

Trained on ARC-AGI-1 with 10×10 downscaled grids on a single RTX 3090 (24GB).

| Run | Config | Notes |
|-----|--------|-------|
| URM baseline | `arch=urm`, batch=32, 10×10 grids | Converged quickly, severe overfitting (train acc ~20-30%, val ~0.6% pass@1) |
| Energy v0 (MCMC) | `arch=urm_energy`, batch=12, 10×10 | Initial energy collapse — energy head output constant for all inputs |
| Energy v1 (+ contrastive) | Added margin-based contrastive loss | Fixed collapse but MCMC backprop too slow for 3090 |
| Energy v2 (DSM) | Reconstruction + DSM loss, no MCMC in training | Current architecture — efficient energy head training |

### Key findings
- **Contrastive loss was a stepping stone**: It proved the energy head can learn separation, but backpropping through MCMC was impractical. DSM achieves the same goal more efficiently.
- **Model capacity mismatch**: The full URM architecture is heavily overparameterized for 10×10 grids, causing severe overfitting. Right-sizing the model (smaller hidden dim, fewer layers) or using full 30×30 grids is needed.
- **DSM directly trains useful gradients**: Instead of hoping MCMC backprop teaches the energy landscape, DSM explicitly trains ∇E to point toward correct answers at multiple noise scales.

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

### Training (energy-based URM with DSM)
```bash
torchrun --nproc-per-node 1 pretrain.py \
  data_path=data/arc1concept-aug-10 \
  arch=urm_energy arch.loops=16 arch.H_cycles=4 arch.L_cycles=3 arch.num_layers=4 \
  arch.energy_threshold=0.005 arch.min_steps=8 \
  epochs=200000 eval_interval=1000 global_batch_size=12 \
  puzzle_emb_lr=1e-2 lr=1e-4 weight_decay=0.1 +ema=True
```

### Running tests
```bash
conda activate urm && python -m pytest tests/ -v
```

## Status & Next Steps

### Completed
- [x] Energy-based stopping with energy convergence across outer loop iterations
- [x] DSM training for energy head (replaces contrastive loss + MCMC backprop)
- [x] Inference-time MCMC refinement via refine_with_mcmc()
- [x] All bugfixes from previous rounds (inference gradients, threshold, config alignment, etc.)

### Phase 2: Evaluation & Tuning
- [ ] Integrate refine_with_mcmc into evaluation pipeline (evaluate with and without refinement)
- [ ] Tune DSM noise scales for ARC grid embeddings
- [ ] Tune DSM loss weight relative to reconstruction loss
- [ ] Right-size model for small grids (hidden_dim 64-128, 2 layers) or scale to 30×30
- [ ] Compare energy stopping vs ACT on matched architectures
- [ ] Increase data augmentation for small grids to reduce overfitting

## References

- [Universal Reasoning Model (URM)](https://arxiv.org/abs/2512.14693) — Gao et al., 2024
- [Energy-Based Transformers](https://arxiv.org/abs/2410.09197) — Hoover et al., 2024
- [Denoising Score Matching](https://www.iro.umontreal.ca/~vin101/publications/smdae_techreport.pdf) — Vincent, 2011
- [ARC-AGI](https://arcprize.org/) — Chollet, 2019

## License

Same as the original URM repository.
