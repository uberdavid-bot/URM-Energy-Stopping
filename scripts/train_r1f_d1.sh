#!/bin/bash
# R1f: depth=1 full 80K training with constant LR
# Single transformer layer per recurrence step — forces multi-step computation.
# Key difference from R1e: depth=1 instead of 2 (~164K vs ~330K params).
# eval_interval=2106 gives 15 checkpoints (must divide 31590 evenly).

run_name="R1f-d1-h128-exp2-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path

DISABLE_COMPILE=1 torchrun --nproc-per-node 1 pretrain.py \
  data_path=data/arc1concept-aug-1000-size-10 \
  arch=urm_r1f_d1 \
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
