#!/bin/bash
# R1e: expansion=2 full 80K training with constant LR
# Ground truth for reduced MLP capacity — same training budget as R1-full (exp=4).
# Key difference from R1c: 80K steps instead of 10K, constant LR after warmup.
# Key difference from R1-full: expansion=2 instead of 4 (~330K vs ~530K params).
# eval_interval=2106 gives 15 checkpoints (must divide 31590 evenly).

run_name="R1e-exp2-full-h128-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path

DISABLE_COMPILE=1 torchrun --nproc-per-node 1 pretrain.py \
  data_path=data/arc1concept-aug-1000-size-10 \
  arch=urm_r1e_exp2_full \
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
