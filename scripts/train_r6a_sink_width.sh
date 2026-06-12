#!/bin/bash
# R6a follow-up — Attention sinks at h=160 and h=192
# R6a (sink at h=128) was the only positive R6 result: +0.95pp over R4d baseline,
# with train/eval ratio compression (2.37x -> 2.12x).
# This tests whether sinks compound with width or wash out.
#
# Baselines:
#   R4e h=160: 23.72% eval exact, 48.05% pass@1000
#   R5a h=192: 24.64% eval exact, 53.25% pass@1000
#
# Total runtime: ~5-6h (2.5-3h each). Safe daytime run.

set -e

CONDA_RUN="conda run -n urm --no-capture-output"

# Pre-flight
if pgrep -f "pretrain.py" > /dev/null; then
    echo "ERROR: pretrain.py already running. Kill before launching."
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
  weight_decay=0.1 \
  target_q_update_every=10 \
  grad_clip_backbone=5.0 \
  +ema=True"

# [1/2] R6a2: sink + h=160
run_name="R6a2-sink-h160-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p "$checkpoint_path"
echo "=== [1/2] R6a2: sink + h=160, 80K steps ==="
DISABLE_COMPILE=1 $CONDA_RUN torchrun --nproc-per-node 1 pretrain.py \
  arch=urm_r6a2_sink_h160 \
  $COMMON_ARGS \
  +run_name="$run_name" \
  +checkpoint_path="$checkpoint_path" \
  2>&1 | tee "logs/${run_name}.log"
echo "=== R6a2 complete ==="

# [2/2] R6a3: sink + h=192
run_name="R6a3-sink-h192-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p "$checkpoint_path"
echo "=== [2/2] R6a3: sink + h=192, 80K steps ==="
DISABLE_COMPILE=1 $CONDA_RUN torchrun --nproc-per-node 1 pretrain.py \
  arch=urm_r6a3_sink_h192 \
  $COMMON_ARGS \
  +run_name="$run_name" \
  +checkpoint_path="$checkpoint_path" \
  2>&1 | tee "logs/${run_name}.log"
echo "=== R6a3 complete ==="

echo "All R6a follow-up experiments complete at $(date)"
