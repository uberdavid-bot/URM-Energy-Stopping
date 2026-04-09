#!/bin/bash
# Experiment R1: Right-sized URM baseline (scale search)
# Goal: Find model scale where URM needs most of its 8-step budget to converge
# Config: depth=2, hidden=64, 4 heads, 8 recurrence steps, batch 512, 10x10 grids
# Expected: per-step accuracy should improve meaningfully between steps 4-8
# ~4000 epochs ≈ 10K steps at batch 512 on 10x10 data; 5 eval checkpoints

run_name="R1-scale-h64-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path

DISABLE_COMPILE=1 torchrun --nproc-per-node 1 pretrain.py \
  data_path=data/arc1concept-aug-1000-size-10 \
  arch=urm_energy_r1 \
  epochs=4000 \
  eval_interval=800 \
  global_batch_size=512 \
  lr=3e-4 \
  lr_min_ratio=0.1 \
  lr_warmup_steps=100 \
  beta1=0.9 \
  beta2=0.95 \
  puzzle_emb_lr=1e-2 \
  puzzle_emb_weight_decay=0.01 \
  weight_decay=0.1 \
  target_q_update_every=10 \
  '+loop_deltas=[0]' \
  +run_name=$run_name \
  +checkpoint_path=$checkpoint_path \
  +ema=True
