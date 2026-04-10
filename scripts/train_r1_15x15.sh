#!/bin/bash
# Experiment R1d: 15x15 grids at h=128 exp=4
# Goal: Make the problem harder (2.25x more tokens) instead of the model weaker
# Batch 256 (seq=225 too large for batch 512 on 24GB), 1050 epochs ≈ 10K steps

run_name="R1d-15x15-h128-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path

DISABLE_COMPILE=1 torchrun --nproc-per-node 1 pretrain.py \
  data_path=data/arc1concept-aug-1000-size-15 \
  arch=urm_r1_15x15 \
  epochs=1050 \
  eval_interval=210 \
  global_batch_size=256 \
  lr=3e-4 \
  lr_min_ratio=0.1 \
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
