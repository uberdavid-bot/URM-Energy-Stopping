#!/bin/bash
# R1 re-run with inner_loops=1 fix: 4 configs at 10x10, 80K steps each
# 1. depth=1, h=64,  exp=2 (urm_r1g_d1_h64)
# 2. depth=1, h=128, exp=2 (urm_r1f_d1)
# 3. depth=1, h=128, exp=4 (urm_r1h_d1_h128_exp4)
# 4. depth=2, h=128, exp=4 (urm_r1_15x15)

set -e

run_experiment() {
  local ARCH=$1
  local NAME=$2

  local run_name="${NAME}-$(date +%y%m%d)"
  local checkpoint_path="checkpoints/${run_name}"
  mkdir -p "$checkpoint_path"

  echo ""
  echo "=========================================="
  echo "Starting: $NAME (arch=$ARCH)"
  echo "=========================================="

  DISABLE_COMPILE=1 torchrun --nproc-per-node 1 pretrain.py \
    data_path=data/arc1concept-aug-1000-size-10 \
    arch=$ARCH \
    epochs=31586 \
    eval_interval=1858 \
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

  echo ""
  echo "Completed: $NAME"
  echo "=========================================="
}

run_experiment urm_r1g_d1_h64       "R1-rerun-d1-h64-exp2"
run_experiment urm_r1f_d1           "R1-rerun-d1-h128-exp2"
run_experiment urm_r1h_d1_h128_exp4 "R1-rerun-d1-h128-exp4"
run_experiment urm_r1_15x15         "R1-rerun-d2-h128-exp4"

echo ""
echo "All 4 R1 re-runs complete!"
