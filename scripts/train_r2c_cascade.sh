#!/bin/bash
# R2c/d/e cascade: position-aware energy head experiments
# Run R2c first. If energy pass@K beats Q-halt, stop.
# Otherwise run R2d, then R2e.
#
# R2c: per-position MLP (~2K params) — spatial preservation
# R2d: conv1d + per-position MLP (~3K params) — local context
# R2e: dedicated attention + per-position MLP (~10K params) — cross-position
#
# Success criterion: energy pass@K > Q-halt pass@K at any K value.
# Key metrics: energy_accuracy_spearman, energy_pass@K vs Q-halt pass@K.

set -e

# ---- R2c: Per-position MLP ----
run_name="R2c-pos-mlp-d1-h64-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path

echo "=== Starting R2c: Per-position MLP ==="
DISABLE_COMPILE=1 torchrun --nproc-per-node 1 pretrain.py \
  data_path=data/arc1concept-aug-1000-size-10 \
  arch=urm_r2c_pos_mlp \
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
  +run_name=$run_name \
  +checkpoint_path=$checkpoint_path \
  +ema=True

echo "=== R2c complete. Check wandb for energy pass@K results. ==="
echo "=== If energy pass@K beats Q-halt, STOP HERE. ==="
echo ""

# ---- R2d: Conv + per-position MLP ----
# Uncomment to run after checking R2c results:
# run_name="R2d-pos-conv-d1-h64-$(date +%y%m%d)"
# checkpoint_path="checkpoints/${run_name}"
# mkdir -p $checkpoint_path
#
# echo "=== Starting R2d: Conv + Per-position MLP ==="
# DISABLE_COMPILE=1 torchrun --nproc-per-node 1 pretrain.py \
#   data_path=data/arc1concept-aug-1000-size-10 \
#   arch=urm_r2d_pos_conv \
#   epochs=31590 \
#   eval_interval=2106 \
#   global_batch_size=512 \
#   lr=3e-4 \
#   lr_min_ratio=1.0 \
#   lr_warmup_steps=100 \
#   beta1=0.9 \
#   beta2=0.95 \
#   puzzle_emb_lr=1e-2 \
#   puzzle_emb_weight_decay=0.01 \
#   weight_decay=0.1 \
#   target_q_update_every=10 \
#   grad_clip_backbone=5.0 \
#   grad_clip_energy_head=1.0 \
#   +run_name=$run_name \
#   +checkpoint_path=$checkpoint_path \
#   +ema=True
#
# echo "=== R2d complete. Check wandb for energy pass@K results. ==="

# ---- R2e: Attention + per-position MLP ----
# Uncomment to run after checking R2d results:
# run_name="R2e-pos-attn-d1-h64-$(date +%y%m%d)"
# checkpoint_path="checkpoints/${run_name}"
# mkdir -p $checkpoint_path
#
# echo "=== Starting R2e: Attention + Per-position MLP ==="
# DISABLE_COMPILE=1 torchrun --nproc-per-node 1 pretrain.py \
#   data_path=data/arc1concept-aug-1000-size-10 \
#   arch=urm_r2e_pos_attn \
#   epochs=31590 \
#   eval_interval=2106 \
#   global_batch_size=512 \
#   lr=3e-4 \
#   lr_min_ratio=1.0 \
#   lr_warmup_steps=100 \
#   beta1=0.9 \
#   beta2=0.95 \
#   puzzle_emb_lr=1e-2 \
#   puzzle_emb_weight_decay=0.01 \
#   weight_decay=0.1 \
#   target_q_update_every=10 \
#   grad_clip_backbone=5.0 \
#   grad_clip_energy_head=1.0 \
#   +run_name=$run_name \
#   +checkpoint_path=$checkpoint_path \
#   +ema=True
#
# echo "=== R2e complete. Check wandb for energy pass@K results. ==="
