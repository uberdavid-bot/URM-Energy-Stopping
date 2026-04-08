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

Fork of the [Universal Reasoning Model (URM)](https://github.com/UbiquantAI/URM) extended with an energy-based stopping mechanism as an alternative to Adaptive Computation Time (ACT).

## Motivation

The URM paper ([arXiv:2512.14693](https://arxiv.org/abs/2512.14693)) showed that recurrent inductive bias and strong nonlinearity are the key ingredients for reasoning on ARC-AGI, achieving 53.8% pass@1 on ARC-AGI 1. URM uses ACT (a learned halting probability) to decide when to stop iterating its recurrent computation.

This project replaces ACT with an **energy-based stopping criterion**. Instead of learning a binary "should I stop?" signal, we learn an energy function E(input, output) that scores prediction quality. This provides:

- **Principled stopping**: energy convergence rather than a learned halting head
- **Built-in reranking**: energy score ranks predictions for pass@K (lower energy = more confident)
- **Optional MCMC refinement**: gradient descent in output embedding space at inference (experimental)

## What's Implemented

### New files
- `models/urm/urm_energy.py` — Energy-based URM: inner recurrence + energy scoring + optional MCMC refinement
- `models/dsm_loss.py` — Multi-scale Denoising Score Matching loss (for training energy gradients, optional)
- `config/arch/urm_energy.yaml` — Hydra config with two modes: default and MCMC refinement
- `config/arch/urm_energy_small.yaml` — Energy URM for 10x10 grids (matches urm_small arch, +129 params for energy head)
- `config/arch/urm_energy_mcmc_small.yaml` — MCMC refinement with second-order gradients (mcmc_steps=4, create_graph=True)
- `config/arch/urm_small.yaml` — Right-sized baseline URM for 10x10 grids (2 layers, hidden=128, 530K params)
- `scripts/URM_energy_arcagi1.sh` — Launch script for energy experiments
- `scripts/energy_small.sh` — Launch script for energy model on 10x10 grids (direct baseline comparison)
- `scripts/energy_mcmc_small.sh` — Launch script for MCMC energy model on 10x10 grids
- `scripts/baseline_small.sh` — Launch script for baseline URM on 10x10 grids
- `tests/test_mcmc_inference.py` — Tests for losses, forward pass, energy halting, MCMC refinement, evaluator ranking

### Modified files
- `models/losses.py` — EnergyLossHead: reconstruction + contrastive loss (+ optional DSM)
- `evaluators/arc.py` — Energy-based reranking (`energy_pass@K` alongside q-based `pass@K`)
- `pretrain.py` — Energy model forward pass and loss metric logging

## Architecture

### Training (default mode)
```
input → [URM inner: shared layer × H_cycles × L_cycles] → logits + hidden states
                                                         → energy E(input, output_hidden)
Loss = reconstruction(logits, labels) + contrastive_weight * contrastive(E)
```

Two losses:
- **Reconstruction**: standard LM loss on URM output logits — trains the main predictor
- **Contrastive**: trains energy values so E(true) < E(predicted) - margin — for stopping and reranking

### Training (with MCMC refinement via second-order gradients)
```
input ��� [URM inner: shared layer × H_cycles × L_cycles] → initial logits
      → convert to soft embeddings: softmax(logits) @ embed_tokens.weight
      → MCMC loop (N steps, create_graph=True):
            energy = E(input_emb, predicted_emb)
            grad = ∇E w.r.t. predicted_emb (create_graph=True)
            predicted_emb = predicted_emb - alpha * normalized_grad
      → refined logits = lm_head(refined_emb)
Loss = reconstruction(refined_logits, labels) + contrastive_weight * contrastive(E)
```

The reconstruction loss on MCMC-refined output flows back through the MCMC steps into the energy head via second-order gradients. This teaches the energy head to produce gradients that actually improve predictions. No DSM loss needed — reconstruction through MCMC is strictly more informative. Contrastive loss is kept as auxiliary at reduced weight (0.5).

### Training (with DSM, legacy approach)
```
Loss = reconstruction + contrastive + dsm_weight * DSM(energy_head)
```

DSM (Denoising Score Matching) trains energy *gradients* to point from corrupted toward clean outputs. This was the original approach for MCMC refinement at inference. Superseded by second-order gradient training. Enable with `dsm_weight: 1.0`.

### Inference
```
input → [URM inner × T iterations] → energy E(input, output)
                                    → stop when ΔE < threshold & steps >= min_steps
                                    → rerank predictions by energy (lower = better)
                                    → optional: refine_with_mcmc(logits, ∇E) if refine_steps > 0
```

The outer loop calls forward() repeatedly. Energy convergence across iterations drives halting. The ARC evaluator reports both `pass@K` (q-based, baseline) and `energy_pass@K` (energy-ranked) for comparison.

## Key Findings

- **Contrastive loss trains energy separation**: E(true) vs E(predicted) gap enables both stopping and reranking
- **DSM trains energy gradients (optional)**: Only needed for MCMC refinement; trains ∇E to point toward correct answers. Single gradient computation (O(1) cost), not O(steps) like MCMC backprop
- **Model capacity mismatch**: Full URM architecture is overparameterized for 10×10 grids. Right-sizing needed.
- **Energy reranking is free**: Once trained, energy scoring adds minimal compute but provides pass@K ranking

## Reproduction

### Setup
```bash
conda create -n urm python=3.10
conda activate urm
pip install -r requirements.txt
wandb login  # optional, for experiment tracking
```

### Data preparation

30x30 grids (full ARC, 960 tasks):
```bash
python -m data.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation \
  --num-aug 1000
```

10x10 grids (294/960 tasks, faster iteration):
```bash
python -m data.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000-size-10 \
  --subsets training evaluation concept \
  --test-set-name evaluation \
  --num-aug 1000 \
  --max-grid-size 10
```

### Training (baseline URM on 10x10, right-sized)
```bash
bash scripts/baseline_small.sh
# urm_small: 2 layers, hidden=128, 530K params + 33M puzzle_emb
# 32000 epochs, eval every 2000, batch 512
```

### Training (energy URM on 10x10, matching baseline)
```bash
bash scripts/energy_small.sh
# urm_energy_small: same arch as urm_small + 129-param energy head
# 4000 epochs, eval every 200, batch 512, same hyperparams as baseline
```

### Training (energy-based URM on 30x30)
```bash
bash scripts/URM_energy_arcagi1.sh
```

### Training (energy URM with MCMC refinement on 10x10)
```bash
bash scripts/energy_mcmc_small.sh
# urm_energy_mcmc_small: same arch + MCMC refinement (4 steps, create_graph=True)
# 4000 epochs, eval every 200, batch 512, contrastive_weight=0.5
```

### Training (with DSM refinement, legacy)
```bash
# Same as energy training, plus:
#   arch.dsm_weight=1.0 arch.refine_steps=8 arch.refine_step_size=0.01
```

### Running tests
```bash
conda activate urm && python -m pytest tests/ -v
```

## Status & Next Steps

### Completed
- [x] Energy-based stopping with energy convergence across outer loop iterations
- [x] Contrastive loss for energy value separation (stopping + reranking)
- [x] Energy-based reranking in ARC evaluator (energy_pass@K alongside pass@K)
- [x] DSM training for energy head gradients (optional, for MCMC refinement)
- [x] Inference-time MCMC refinement via refine_with_mcmc() (optional, off by default)
- [x] Configurable loss weights and training modes
- [x] Configurable grid size via --max-grid-size (10x10 dataset: 294/960 tasks, 1.15M samples)
- [x] All bugfixes from previous rounds
- [x] Right-sized baseline model (urm_small: 2 layers, hidden=128, 530K params, ~38 it/s on 3090)
- [x] Fix evaluator _crop() for variable grid sizes (was hardcoded to 30x30)
- [x] Energy small config + training script for direct baseline comparison (urm_energy_small, +129 params)
- [x] Fixed-iteration eval: energy model uses same `loops` count as baseline during eval (energy convergence stopping only active during training)
- [x] Safety cap (max_inference_steps=100) in evaluate() while loop
- [x] Verified: eval completes in exactly 16 steps/batch, energy metrics + ARC/energy_pass@K flow to wandb
- [x] MCMC refinement with second-order gradients: reconstruction loss flows back through MCMC into energy head
- [x] MCMC training config (urm_energy_mcmc_small) and launch script (energy_mcmc_small.sh)
- [x] MCMC improvement metrics (mcmc_improvement, unrefined_accuracy) in EnergyLossHead + pretrain.py logging
- [x] Tests for MCMC training: creates_graph, energy_head_gets_gradients, improves_logits, eval_no_create_graph, disabled_by_default

### Experimental Plan

#### Experiment 1: Fixed iterations, energy reranking vs Q-halt reranking (current)
Both models run the same number of inner loop iterations (`loops=16`). The only difference is how pass@K predictions are ranked:
- **Baseline (urm_small)**: Q-halt head ranks predictions (learned halting probability as confidence)
- **Energy (urm_energy_small)**: Energy function ranks predictions (lower energy = more confident)
- Both `ARC/pass@K` (Q-based) and `ARC/energy_pass@K` (energy-based) are reported for the energy model
- Energy convergence stopping is disabled during eval — both models use identical fixed iterations

Steps:
1. **Run baseline URM** on 10x10 (`bash scripts/baseline_small.sh`) — establish ACT pass@K — **RUNNING**
2. **Run energy URM** on 10x10 (`bash scripts/energy_small.sh`) — compare energy_pass@K vs pass@K — **READY**

#### Experiment 2: Adaptive stopping comparison (future)
Compare energy convergence stopping vs ACT halting, with matched total compute budget.

#### Experiment 3: MCMC refinement with second-order gradients (current)
Core thesis: EBT-style MCMC refinement on a small recurrent model. URM inner recurrence produces initial prediction, then MCMC steps refine it by descending the energy gradient in output embedding space. Second-order gradients through the MCMC steps train the energy head to produce gradients that actually improve predictions.

Three-way comparison (all on 10x10 grids):
- **Baseline (urm_small)**: Q-halt head ranks predictions, no energy
- **Energy reranking (urm_energy_small)**: Energy function ranks predictions, no MCMC
- **Energy MCMC (urm_energy_mcmc_small)**: Energy MCMC refines predictions + energy reranking

Key differences from old DSM-based MCMC:
- Reconstruction loss on MCMC-refined output provides direct training signal to energy head
- No DSM loss needed — reconstruction through MCMC is strictly more informative
- Contrastive loss kept as auxiliary (weight=0.5) for energy value separation
- `mcmc_training=True` enables `create_graph=True` during training; eval uses `create_graph=False`

Steps:
1. **Run MCMC energy URM** on 10x10 (`bash scripts/energy_mcmc_small.sh`) — **READY**
2. Compare `energy_pass@K` across all three models
3. Monitor `mcmc_improvement` metric (did MCMC steps help predictions?)

#### Experiment 4: Ablations (future)
- +N MCMC refinement steps vs +N extra URM passes
- Tune contrastive_margin and contrastive_weight
- Scale to 30×30 grids once 10×10 pipeline validated

### Model configs
| Config | Layers | Hidden | Heads | Params | Speed (3090) | Data |
|--------|--------|--------|-------|--------|-------------|------|
| urm_small | 2 | 128 | 4 | 530K (+33M puzzle_emb) | ~38 it/s | 10x10 |
| urm_energy_small | 2 | 128 | 4 | 531K (+33M puzzle_emb) | ~3.5 it/s | 10x10 |
| urm_energy_mcmc_small | 2 | 128 | 4 | 531K (+33M puzzle_emb) | TBD | 10x10 |
| urm | 8 | 512 | 8 | ~40M (+33M puzzle_emb) | ~2 it/s | 30x30 |

Note: `eval_interval` is in *epochs* not steps. Each epoch = ~294 puzzle groups (~40 steps, ~1 min) at batch_size=32 on the 10x10 dataset.

## References

- [Universal Reasoning Model (URM)](https://arxiv.org/abs/2512.14693) — Gao et al., 2024
- [Energy-Based Transformers](https://arxiv.org/abs/2410.09197) — Hoover et al., 2024
- [Denoising Score Matching](https://www.iro.umontreal.ca/~vin101/publications/smdae_techreport.pdf) — Vincent, 2011
- [ARC-AGI](https://arcprize.org/) — Chollet, 2019

## License

Same as the original URM repository.
