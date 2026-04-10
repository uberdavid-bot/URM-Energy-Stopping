#!/bin/bash
# R1 grid size sweep — run one grid size at a time
# Usage: bash scripts/train_r1_grid_sweep.sh <grid_size> <epochs> <eval_interval>
# Example: bash scripts/train_r1_grid_sweep.sh 10 3945 789

GRID_SIZE=$1
EPOCHS=$2
EVAL_INTERVAL=$3
BATCH_SIZE=${4:-512}

if [ -z "$GRID_SIZE" ] || [ -z "$EPOCHS" ] || [ -z "$EVAL_INTERVAL" ]; then
  echo "Usage: $0 <grid_size> <epochs> <eval_interval> [batch_size]"
  exit 1
fi

run_name="R1-grid${GRID_SIZE}-h128-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path

DISABLE_COMPILE=1 torchrun --nproc-per-node 1 pretrain.py \
  data_path=data/arc1concept-aug-1000-size-${GRID_SIZE} \
  arch=urm_r1_15x15 \
  epochs=$EPOCHS \
  eval_interval=$EVAL_INTERVAL \
  global_batch_size=$BATCH_SIZE \
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
