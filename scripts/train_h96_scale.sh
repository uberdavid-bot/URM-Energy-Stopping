#!/bin/bash
# Scaling experiments: h=64 -> h=96
#
# Experiment 1: R1-h96 baseline (no energy, ~3h)
#   Go/no-go gate: must show multi-step convergence (>5% eval exact variation step 1->6)
#   If flat per-step curves -> STOP, do not proceed to R2c-h96
#
# Experiment 2: R2c-h96 trajectory ranking (energy co-training, ~5h)
#   Only runs if R1-h96 shows multi-step convergence
#
# Baselines:
#   R1i  (h=64, no energy):  5.33% eval exact, 22.1% pass@1000
#   R2c  (h=64, energy):     6.95% eval exact, 29.2% pass@1000, energy pass@100 ~= 0
#   A1d  (h=64, elw=0.5):    4.67% eval exact, energy pass@100=12.3% (capacity trade-off)

set -e

COMMON_ARGS="data_path=data/arc1concept-aug-1000-size-10 \
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
  +ema=True"

# Experiment 1: R1-h96 baseline
run_name="R1-h96-baseline-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path
echo "=== Starting R1-h96 baseline (no energy, h=96, d=1, exp=2) ==="
DISABLE_COMPILE=1 torchrun --nproc-per-node 1 pretrain.py \
  arch=urm_r1_h96_baseline \
  $COMMON_ARGS \
  +run_name=$run_name \
  +checkpoint_path=$checkpoint_path

echo "=== R1-h96 complete. CHECK WANDB FOR MULTI-STEP CONVERGENCE ==="
echo "=== If per-step exact accuracy variation < 2% (flat), DO NOT proceed ==="
echo "=== If monotonic ramp confirmed, proceeding to R2c-h96 ==="

# Experiment 2: R2c-h96 trajectory ranking
run_name="R2c-h96-pos-mlp-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path
echo "=== Starting R2c-h96 (trajectory ranking, position_mlp energy head) ==="
DISABLE_COMPILE=1 torchrun --nproc-per-node 1 pretrain.py \
  arch=urm_r2c_h96_pos_mlp \
  $COMMON_ARGS \
  +run_name=$run_name \
  +checkpoint_path=$checkpoint_path

echo "=== Both h=96 experiments complete ==="
