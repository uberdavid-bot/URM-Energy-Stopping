#!/bin/bash
# R7e — GRAM-faithful clean decode, full 80K training run.
#
# Fixes R7d structural bugs: (1) decode always from clean u_t (never perturbed),
# (2) σ calibrated at injection point (u_t − hidden, not pre_t − hidden).
# Perturbation only affects carry to next step.
#
# Baseline: R4d h=128 (21.25% eval exact, 44.81% pass@1000).
# R7d negative: 11.83% eval exact (−9.42pp), delta norms 5-7× larger.
# Key question: does clean decode recover accuracy lost to R7d's decode corruption?
#
# Runtime (3090): ~2h (h=128, loops=8). GRAM adds ~10% overhead for VAE ops.

set -e

CONDA_RUN="conda run -n urm --no-capture-output"

if pgrep -f "pretrain.py" > /dev/null; then
    echo "ERROR: pretrain.py already running. Kill before launching R7e."
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

run_name="R7e-gram-cleandecode-a02-h128-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p "$checkpoint_path"

echo "=== R7e: GRAM clean decode, alpha=0.2, h=128 ==="
DISABLE_COMPILE=1 $CONDA_RUN torchrun --nproc-per-node 1 pretrain.py \
  arch=urm_r7e_gram_cleandecode_h128 \
  weight_decay=0.1 \
  $COMMON_ARGS \
  +run_name="$run_name" \
  +checkpoint_path="$checkpoint_path"
echo "=== R7e complete ==="
