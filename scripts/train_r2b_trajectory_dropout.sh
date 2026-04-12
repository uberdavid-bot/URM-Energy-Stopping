#!/bin/bash
# R2b: Re-run trajectory ranking on dropout backbone (R1i)
# R2 failed due to backbone overfitting (eval Spearman -0.48 vs train -1.0).
# R1i closed gap from 6:1 to 3.7:1. Expecting eval Spearman to improve.
#
# Key metrics to compare vs R2:
#   eval Spearman: was -0.48, want -0.7+
#   energy pass@K: was dramatically worse than Q-halt, want competitive
#   URM eval exact acc: was 2.9%, R1i got 5.3%, should maintain
#   active_pairs: was 28/28, should stay high
#   cosine_sim: was -0.01 on eval, want more negative

run_name="R2b-traj-dropout-d1-h64-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path

DISABLE_COMPILE=1 torchrun --nproc-per-node 1 pretrain.py \
  data_path=data/arc1concept-aug-1000-size-10 \
  arch=urm_r2b_trajectory_dropout \
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
