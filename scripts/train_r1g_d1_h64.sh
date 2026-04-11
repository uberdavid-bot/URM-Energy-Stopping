#!/bin/bash
# R1g: depth=1, h=64, exp=2 — small model with corrected MLP rounding
# MLP inter=88 (granularity=8) instead of 256 (old granularity=256).
# Previous h=64 results were invalid: undertrained + inflated MLP.
# eval_interval=2106 gives 15 checkpoints (must divide 31590 evenly).

run_name="R1g-d1-h64-exp2-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path

DISABLE_COMPILE=1 torchrun --nproc-per-node 1 pretrain.py \
  data_path=data/arc1concept-aug-1000-size-10 \
  arch=urm_r1g_d1_h64 \
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
  +run_name=$run_name \
  +checkpoint_path=$checkpoint_path \
  +ema=True
