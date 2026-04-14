#!/bin/bash
# A4 Noise experiments: recurrence noise as backbone regularizer
#
# A4a: noise only (sigma=0.005, no dropout)        ~1.5 hours
# A4b: noise + reduced dropout (sigma=0.003, p=0.05) ~1.5 hours
#
# Baselines:
#   R1h (no reg):       3.76% eval exact
#   R1i (dropout=0.1):  5.33% eval exact

set -e

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
  grad_clip_energy_head=1.0 \
  +ema=True"

# A4a: recurrence noise only
run_name="A4a-noise-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path
echo "=== Starting A4a: recurrence noise only (sigma=0.005, no dropout) ==="
DISABLE_COMPILE=1 torchrun --nproc-per-node 1 pretrain.py \
  arch=ablation_a4a_noise \
  $COMMON_ARGS \
  +run_name=$run_name \
  +checkpoint_path=$checkpoint_path

# A4b: noise + dropout (both reduced)
run_name="A4b-noise-dropout-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path
echo "=== Starting A4b: noise + dropout (sigma=0.003, dropout=0.05) ==="
DISABLE_COMPILE=1 torchrun --nproc-per-node 1 pretrain.py \
  arch=ablation_a4b_noise_dropout \
  $COMMON_ARGS \
  +run_name=$run_name \
  +checkpoint_path=$checkpoint_path

echo "=== A4 experiments complete ==="
