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
To be re-measured at new model scale. Previous measurements at hidden=128:
| Mode | Speed | 10K steps in |
|------|-------|-------------|
| URM (no energy) | ~6.5 it/s | ~26 min |
| Energy (first-order, no MCMC) | ~3.5 it/s | ~48 min |
| Energy + MCMC (create_graph) | ~0.66 it/s | ~4.2 hrs |

Expect all to be faster at right-sized model. Re-measure after Phase 1.

## Training Budget
- **Fixed**: 10,000 steps per experiment
- **Target wall time**: under 2.5 hours (should be well within budget at smaller model)
- **Epochs**: ~4000 epochs ≈ 10K steps at batch 512 on 10×10 data
- **Eval checkpoints**: 5 per run

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

## Anti-Patterns

- **Do not use `torch.inference_mode()` with energy models.** Use `torch.no_grad()` — can be overridden by `enable_grad()`.
- **Do not hardcode grid dimensions.** Infer from tensor shapes.
- **Do not install new packages.**
- **Do not use `(base)` conda environment.** Always `conda activate urm`.
- **Do not do MCMC in soft-embedding space.** Always in hidden space.
- **Do not detach between MCMC steps during training.** Energy head needs multi-step gradient flow.
- **Do not train energy head sequentially on frozen URM.** Quality spread only exists while URM is learning. Co-train.
