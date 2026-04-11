#!/bin/bash
# R1h: depth=1, h=64, exp=2 — FIRST RUN WITH DEEP SUPERVISION
# Same config as R1g (urm_r1g_d1_h64) but now with:
#   - forward_trajectory(): full gradient flow across all 8 steps (no detach)
#   - Deep supervision: per-step reconstruction loss weighted (t+1)/N
#   - Q-halt BCE at every step
# This is the critical test: does deep supervision + cross-step gradient flow
# produce the monotonic per-step accuracy ramp that all prior R1 runs lacked?
# ~35K transformer params, 80K training steps, 10x10 grids.

run_name="R1h-deepsup-d1-h64-$(date +%y%m%d)"
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
