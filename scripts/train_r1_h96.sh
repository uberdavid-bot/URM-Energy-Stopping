#!/bin/bash
# Experiment R1 (h96): Scale search — hidden=96 after h64 was too small
# Goal: Find sweet spot where URM needs most of its 8-step budget
# ~300K params; between h64 (too small) and h128 (converges too fast)

run_name="R1-scale-h96-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path

DISABLE_COMPILE=1 torchrun --nproc-per-node 1 pretrain.py \
  data_path=data/arc1concept-aug-1000-size-10 \
  arch=urm_r1_h96 \
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
  +run_name=$run_name \
  +checkpoint_path=$checkpoint_path \
  +ema=True
