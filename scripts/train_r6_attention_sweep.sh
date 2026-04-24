#!/bin/bash
# R6 Attention Architecture Sweep at h=128
# Sequential execution — each experiment ~2h on 3090
# Total: ~16h for 8 experiments
# Baseline comparison: R4d (urm_r4d_h128.yaml) = 21.25% eval exact, 24.03% pass@1

set -e

CONDA_RUN="conda run -n urm --no-capture-output"

# Pre-flight
if pgrep -f "pretrain.py" > /dev/null; then
    echo "ERROR: pretrain.py already running. Kill before launching R6 sweep."
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

CONFIGS=(
    "urm_r6a_sink_h128"
    "urm_r6b_temperature_h128"
    "urm_r6c_registers_h128"
    "urm_r6d_heads8_h128"
    "urm_r6e_heads16_h128"
    "urm_r6f_heads2_h128"
    "urm_r6g_partial_rope_h128"
    "urm_r6h_gqa_h128"
)

NAMES=(
    "R6a-sink-h128"
    "R6b-temperature-h128"
    "R6c-reg4-h128"
    "R6d-heads8-h128"
    "R6e-heads16-h128"
    "R6f-heads2-h128"
    "R6g-partialrope-h128"
    "R6h-gqa8q2kv-h128"
)

for i in "${!CONFIGS[@]}"; do
    run_name="${NAMES[$i]}-$(date +%y%m%d)"
    checkpoint_path="checkpoints/${run_name}"
    mkdir -p "$checkpoint_path"

    echo "========================================="
    echo "[$(( i + 1 ))/${#CONFIGS[@]}] Starting ${NAMES[$i]} at $(date)"
    echo "Config: ${CONFIGS[$i]}"
    echo "========================================="

    DISABLE_COMPILE=1 $CONDA_RUN torchrun --nproc-per-node 1 pretrain.py \
      arch="${CONFIGS[$i]}" \
      $COMMON_ARGS \
      +run_name="$run_name" \
      +checkpoint_path="$checkpoint_path" \
      2>&1 | tee "logs/${run_name}.log"

    echo "Finished ${NAMES[$i]} at $(date)"
    echo ""
done

echo "All R6 experiments complete at $(date)"
