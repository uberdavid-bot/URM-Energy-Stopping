#!/bin/bash
# R1i: Dropout regularization experiment
# Same as R1h but with attn_dropout=0.1, mlp_dropout=0.1
# Goal: close the train/eval gap (23% vs 3% exact accuracy in R1h)
# If gap closes significantly, re-run R2 with dropout backbone.
# If learning collapses, try 0.05.

run_name="R1i-dropout-d1-h64-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path

DISABLE_COMPILE=1 torchrun --nproc-per-node 1 pretrain.py \
  data_path=data/arc1concept-aug-1000-size-10 \
  arch=urm_r1i_dropout \
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
