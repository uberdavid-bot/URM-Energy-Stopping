#!/bin/bash
# R4d/R4e/R4f — width scaling and regularization probe.
#
# Motivation: R4c diagnosed h=96 as capacity-bound (generalization collapse
# under extended refinement, not limit-cycle oscillation). Three experiments
# test whether width or regularization addresses this.
#
# Baseline: R1-h96 (15.59% eval exact, 40.91% pass@1000, step-6 peak 16.09%)
# Success criterion: Eval exact > 17% for R4d or R4e.
# R4f is diagnostic (is h=128 train/eval gap regularizable?).
#
# Runtimes (3090): R4d ~2h, R4e ~3h, R4f ~2h. Total ~7-8h — fits in one night.

set -e

# Use `conda run -n urm ...` so this script can be launched non-interactively
# (e.g. nohup). `conda activate` does not work in a plain bash -c.
CONDA_RUN="conda run -n urm --no-capture-output"

# Pre-flight: verify GPU is idle and no leftover pretrain processes
if pgrep -f "pretrain.py" > /dev/null; then
    echo "ERROR: pretrain.py already running. Kill before launching R4d sweep."
    exit 1
fi
mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$mem_used" -gt 2000 ]; then
    echo "WARNING: GPU has ${mem_used}MB allocated — previous run may not have cleaned up."
fi

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
  target_q_update_every=10 \
  grad_clip_backbone=5.0 \
  +ema=True"

# [1/3] R4d: h=128, loops=8, standard regularization
run_name="R4d-h128-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path
echo "=== [1/3] R4d: h=128, loops=8, standard regularization ==="
DISABLE_COMPILE=1 $CONDA_RUN torchrun --nproc-per-node 1 pretrain.py \
  arch=urm_r4d_h128 \
  weight_decay=0.1 \
  $COMMON_ARGS \
  +run_name=$run_name \
  +checkpoint_path=$checkpoint_path
echo "=== R4d complete ==="

# [2/3] R4e: h=160, loops=8, standard regularization
run_name="R4e-h160-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path
echo "=== [2/3] R4e: h=160, loops=8, standard regularization ==="
DISABLE_COMPILE=1 $CONDA_RUN torchrun --nproc-per-node 1 pretrain.py \
  arch=urm_r4e_h160 \
  weight_decay=0.1 \
  $COMMON_ARGS \
  +run_name=$run_name \
  +checkpoint_path=$checkpoint_path
echo "=== R4e complete ==="

# [3/3] R4f: h=128 + stronger regularization
run_name="R4f-h128-reg-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path
echo "=== [3/3] R4f: h=128, loops=8, dropout=0.15 + wd=0.15 ==="
DISABLE_COMPILE=1 $CONDA_RUN torchrun --nproc-per-node 1 pretrain.py \
  arch=urm_r4f_h128_regularized \
  weight_decay=0.15 \
  $COMMON_ARGS \
  +run_name=$run_name \
  +checkpoint_path=$checkpoint_path
echo "=== R4f complete ==="

echo "=== All R4d/e/f experiments complete ==="
echo "Key metrics to compare on wandb:"
echo "  - all.exact_accuracy (final) vs R1-h96 baseline 15.59%, R4c baseline 13.38%"
echo "  - train/eval ratio (R1-h96: 2.33x, R4c: 2.73x)"
echo "  - Peak per-step accuracy vs final per-step accuracy (late-step degradation?)"
echo "  - num_params — confirms hidden_size scaling"
