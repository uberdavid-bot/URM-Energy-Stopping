#!/bin/bash
# R2: Energy verifier via trajectory ranking (first-order)
# Same backbone as R1h. Co-trains energy head with URM via trajectory ranking loss.
# energy_loss_weight=0.1, ranking_margin=0.1
# Monitor: energy_accuracy_spearman (should stay negative), active_pairs (should stay >30% of 28),
#          URM metrics should NOT regress vs R1h baseline.

run_name="R2-trajectory-d1-h64-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path

DISABLE_COMPILE=1 torchrun --nproc-per-node 1 pretrain.py \
  data_path=data/arc1concept-aug-1000-size-10 \
  arch=urm_r2_trajectory \
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
