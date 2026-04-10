#!/bin/bash
# R1 full training: h=128 exp=4 on 10x10, 80K steps (vs previous 10K)
# The 10K-step runs were undertrained — LR decayed before learning completed.
# 10 eval checkpoints over training for convergence profile.

run_name="R1-full-h128-10x10-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path

DISABLE_COMPILE=1 torchrun --nproc-per-node 1 pretrain.py \
  data_path=data/arc1concept-aug-1000-size-10 \
  arch=urm_r1_15x15 \
  epochs=31590 \
  eval_interval=3159 \
  global_batch_size=512 \
  lr=3e-4 \
  lr_min_ratio=0.1 \
  lr_warmup_steps=800 \
  beta1=0.9 \
  beta2=0.95 \
  puzzle_emb_lr=1e-2 \
  puzzle_emb_weight_decay=0.01 \
  weight_decay=0.1 \
  target_q_update_every=10 \
  +run_name=$run_name \
  +checkpoint_path=$checkpoint_path \
  +ema=True
