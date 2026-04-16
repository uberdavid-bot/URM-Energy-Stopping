#!/bin/bash
# R4 backbone experiments — pivoted after failure mode analysis.
#
# Failure mode analysis on R1-h96 (complete, see logs/failure_modes_R1h96.txt):
#   84% of wrong predictions are OSCILLATING (step-8 delta_norm > 0.003)
#   0% are stuck (wrong + converged)
#   88% of wrong predictions don't fire Q-halt (Q-halt well-calibrated)
#
# Implication: registers (stuck-in-minima targeting) and multi-register Q-halt
# were dropped. Two experiments remain that address oscillation directly.
#
# Sequence:
#   1. R4b: Exclusive self-attention at h=96 (~10-12h via SDPA fallback)
#   2. R4c: loops=16 at h=96 (~20-22h, 2x forward compute per step)
#
# Baseline: R1-h96 (15.59% eval exact, 40.91% pass@1000, step-6 peak 16.09%)
# Success criterion: Eval exact > 16.1% for either experiment.
# Diagnostic value of R4c (independent of accuracy): is step-16 delta_norm
# substantially smaller than step-8? If yes, refinement is the bottleneck.
# If no, oscillation is a genuine limit cycle and capacity is the bottleneck.

set -e

# Use `conda run -n urm ...` for every python/torchrun invocation so this
# script can be launched non-interactively (e.g. nohup). `conda activate` does
# not work in a plain bash -c.
CONDA_RUN="conda run -n urm --no-capture-output"

COMMON_ARGS="data_path=data/arc1concept-aug-1000-size-10 \
  epochs=31590 \
  eval_interval=2106 \
  global_batch_size=512 \
  lr=3e-4 \
  lr_min_ratio=1.0 \
  lr_warmup_steps=100 \
  beta1=0.9 \
  beta2=0.95 \
  puzzle_emb_lr=1e-2 \
  puzzle_emb_weight_decay=0.01 \
  weight_decay=0.1 \
  target_q_update_every=10 \
  grad_clip_backbone=5.0 \
  +ema=True"

# [1/2] R4b: Exclusive self-attention
run_name="R4b-xsa-h96-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path
echo "=== [1/2] R4b: XSA at h=96 ==="
DISABLE_COMPILE=1 $CONDA_RUN torchrun --nproc-per-node 1 pretrain.py \
  arch=urm_r4b_xsa_h96 \
  $COMMON_ARGS \
  +run_name=$run_name \
  +checkpoint_path=$checkpoint_path
echo "=== R4b complete ==="

# [2/2] R4c: loops=16
run_name="R4c-loops16-h96-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path
echo "=== [2/2] R4c: loops=16 at h=96 ==="
DISABLE_COMPILE=1 $CONDA_RUN torchrun --nproc-per-node 1 pretrain.py \
  arch=urm_r4c_loops16_h96 \
  $COMMON_ARGS \
  +run_name=$run_name \
  +checkpoint_path=$checkpoint_path
echo "=== R4c complete ==="

echo "=== All R4 experiments complete ==="
echo "Key R4c metrics to check on wandb:"
echo "  - all.step_8_delta_norm vs all.step_16_delta_norm (does it decay?)"
echo "  - all.step_N_exact_accuracy across steps 8-16 (does accuracy climb?)"
echo "  - Final eval exact vs R1-h96 baseline (16.09% step-6 peak)"
