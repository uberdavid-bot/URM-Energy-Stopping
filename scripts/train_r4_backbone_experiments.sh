#!/bin/bash
# R4 backbone experiments — three tasks run sequentially overnight.
#
# Sequence:
#   1. Failure mode analysis on latest R1-h96 checkpoint (~30 min)
#   2. R4a: 4 registers at h=96 (~10h)
#   3. R4b: Exclusive self-attention at h=96 (~10h, possibly ~12h due to SDPA fallback)
#
# Baseline: R1-h96 (15.59% eval exact, 40.91% pass@1000)
# Success criterion for R4a/R4b: eval exact > 16.1% (baseline step-6 peak)

set -e

# Use `conda run -n urm ...` for every python/torchrun invocation so this
# script can be launched non-interactively (e.g. nohup). `conda activate` does
# not work in a plain bash -c, and the sibling train scripts assume the user
# already activated the env in their shell.
CONDA_RUN="conda run -n urm --no-capture-output"

# Find latest R1-h96 checkpoint for failure mode analysis
R1_H96_CKPT=$(ls -td checkpoints/R1-scale-h96-* checkpoints/R1-h96-* 2>/dev/null | head -1)
if [ -z "$R1_H96_CKPT" ]; then
    echo "ERROR: No R1-h96 checkpoint found in checkpoints/"
    exit 1
fi
echo "=== Using R1-h96 checkpoint: $R1_H96_CKPT ==="

# Task 1: Failure mode analysis
echo "=== [1/3] Failure mode analysis ==="
$CONDA_RUN python scripts/analyze_failure_modes.py \
    --checkpoint "$R1_H96_CKPT" \
    --data_path data/arc1concept-aug-1000-size-10 \
    --output logs/failure_modes_R1h96.txt
echo "=== Failure mode analysis complete — see logs/failure_modes_R1h96.txt ==="

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

# Task 2: R4a registers
run_name="R4a-registers-h96-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path
echo "=== [2/3] R4a: 4 registers at h=96 ==="
DISABLE_COMPILE=1 $CONDA_RUN torchrun --nproc-per-node 1 pretrain.py \
  arch=urm_r4a_registers_h96 \
  $COMMON_ARGS \
  +run_name=$run_name \
  +checkpoint_path=$checkpoint_path
echo "=== R4a complete ==="

# Task 3: R4b XSA
run_name="R4b-xsa-h96-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path
echo "=== [3/3] R4b: XSA at h=96 ==="
DISABLE_COMPILE=1 $CONDA_RUN torchrun --nproc-per-node 1 pretrain.py \
  arch=urm_r4b_xsa_h96 \
  $COMMON_ARGS \
  +run_name=$run_name \
  +checkpoint_path=$checkpoint_path
echo "=== R4b complete ==="

echo "=== All R4 experiments complete ==="
echo "Compare wandb runs: R1-h96 baseline vs R4a-registers-h96 vs R4b-xsa-h96"
