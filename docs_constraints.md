# Constraints

## Hardware

- **GPU**: NVIDIA RTX 3090, 24GB VRAM
- **CUDA**: 13.0, Driver 580.126.09
- **Environment**: `conda activate urm` (Python 3.10). Always activate before any command.

### Memory Budget
| Configuration | Approx VRAM | Notes |
|--------------|-------------|-------|
| URM only, no energy head | ~2-3 GB (est.) | Confirm in Phase 1 |
| URM + trajectory energy (first-order) | ~3-5 GB (est.) | R2: energy head adds minimal overhead |
| URM + MCMC (create_graph=True) | ~8-12 GB (est.) | R3: second-order gradients scale with MCMC steps |
| hidden=128 depth=2, batch 512, create_graph=True | ~19 GB | Measured (previous experiments) |

Note: R2 (trajectory supervision, first-order only) should train at nearly URM speed. Second-order gradients are only needed in R3.

### Throughput
Measured at current R1 experiment scale (URM mode, batch 512, 10×10):
| Configuration | Speed | 80K steps in |
|--------------|-------|-------------|
| d=1, h=64, exp=2 (~34K params) | ~5.8 it/s | ~3.8 hrs |
| d=1, h=128, exp=2 (~134K params) | ~5.6 it/s | ~4.0 hrs |
| d=2, h=128, exp=4 (~398K params) | ~5.5 it/s | ~4.0 hrs |

EBT/hybrid modes with create_graph=True will be slower. Re-measure when starting R2/R3.

### Long-running training
Use `nohup` to launch training — background bash processes from the agent get killed otherwise:
```bash
nohup bash scripts/run_r1_rerun.sh > logs/R1_rerun.log 2>&1 &
```

## Training Budget
- **Standard**: 80,000 steps per experiment, constant LR (3e-4) after 100-step warmup
- **Target wall time**: ~3-4 hours per run on 3090
- **Epochs**: `total_steps = epochs * total_groups * mean_puzzle_examples / global_batch_size` — metadata in `data/*/train/dataset.json`. For 10×10 data at batch 512: ~31586 epochs ≈ 80K steps.
- **eval_interval must divide epochs evenly** (assertion in pretrain.py line 792). Use factors of the epoch count, not round numbers.
- **Eval checkpoints**: ~15-17 per run (eval_interval chosen as a divisor of epochs giving 13-18 evals)

### Epoch calculation for different datasets
Each dataset has different `total_groups` and `mean_puzzle_examples` (in `train/dataset.json`). When changing grid size or dataset, recalculate epochs for 80K steps:
```python
epochs_needed = round(80000 * batch_size / (total_groups * mean_puzzle_examples))
```
Then find a nearby value with a good eval_interval divisor.

## Known Failure Modes

### Energy collapse
- **Symptom**: Energy gap → 0, E(true) ≈ E(predicted)
- **Cause**: Contrastive loss alone finds trivial solution (Exp 1)
- **Fix**: Use trajectory ranking loss (dense, ordered, anti-collapse by construction)

### URM degradation from refined-only loss
- **Symptom**: 0% test accuracy, train accuracy drops
- **Cause**: Reconstruction loss only on MCMC-refined logits
- **Fix**: Dual reconstruction loss (unrefined + refined). Non-negotiable.

### Trajectory quality spread collapse
- **Symptom**: Active ranking pairs → 0, trajectory_quality_first ≈ trajectory_quality_last
- **Cause**: Over-parameterized model converges in 1-2 steps, leaving no quality spread for energy head training (Exp 4)
- **Fix**: Right-size the model (Phase 1 / R1) so quality spread persists throughout training. Co-train energy head with URM (not sequential).

### Soft-embedding MCMC (ELIMINATED)
- **Symptom**: mcmc_improvement = 0 despite correct training setup
- **Cause**: MCMC operating on softmax(logits) @ embed_weight instead of hidden states. 11-dimensional simplex constraint, distribution mismatch with lm_head and energy head.
- **Fix**: MCMC in hidden space. This has been fixed in the architecture redesign.

### Detached MCMC gradients (ELIMINATED)
- **Symptom**: Energy head doesn't learn from multi-step MCMC trajectories
- **Cause**: `hidden.detach().requires_grad_(True)` inside MCMC loop breaks computational graph
- **Fix**: Don't detach during training. Detach only at inference.

### Infinite eval steps
- **Symptom**: Eval hangs
- **Cause**: Energy convergence threshold too tight, no cap
- **Fix**: max_inference_steps cap. Use fixed steps at eval.

## Wandb API Queries

When querying wandb for in-progress or completed run metrics:
```python
import wandb
api = wandb.Api()
run = api.run('uberdavid-personal/arcagi/<run_id>')

# Eval metrics use flattened dot notation: all.step_1_exact_accuracy, not all/step_1_exact_accuracy
# Use scan_history without key filter — key filtering returns empty on nested keys
rows = list(run.scan_history(page_size=500))

# Find eval rows by checking for ARC/pass@1 key
eval_rows = [r for r in rows if r.get('ARC/pass@1') is not None]
for row in eval_rows:
    step_acc = row.get('all.step_1_exact_accuracy', 0)
```

The `run.summary` dict contains latest values and is faster than scanning full history.

## Anti-Patterns

- **Do not use `torch.inference_mode()` with energy models.** Use `torch.no_grad()` — can be overridden by `enable_grad()`.
- **Do not hardcode grid dimensions.** Infer from tensor shapes.
- **Do not install new packages.**
- **Do not use `(base)` conda environment.** Always `conda activate urm`.
- **Do not do MCMC in soft-embedding space.** Always in hidden space.
- **Do not detach between MCMC steps during training.** Energy head needs multi-step gradient flow.
- **Do not train energy head sequentially on frozen URM.** Quality spread only exists while URM is learning. Co-train.
