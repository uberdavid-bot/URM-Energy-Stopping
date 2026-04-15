#!/bin/bash
# Energy head shortcut-breaking experiments at h=96
#
# R2g-h96: Ranking noise — noise hidden states before energy head scores them (~5h)
# R2i-h96: Cross-trajectory ranking — same-step pairs across augmentations (~5h)
#
# Both use elw=0.05 (half of R2c-h96's 0.1) to limit reconstruction impact.
#
# Baseline:   R1-h96 (no energy):       15.59% eval exact, 40.91% pass@1000
# Reference:  R2c-h96 (vanilla ranking): 11.78% eval exact (-3.8pp, inversion)
#
# Success criteria:
#   1. Eval exact >= 14%  (backbone not significantly hurt)
#   2. Eval Spearman < -0.2  (energy head generalizes -- vs R2c-h96's +0.007)
#   3. Energy pass@100 > 10% (first practical cross-input energy ranking)

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

# R2g-h96: Ranking noise
run_name="R2g-h96-ranking-noise-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path
echo "=== Starting R2g-h96: ranking noise (sigma ~ U[0,0.01], elw=0.05) ==="
DISABLE_COMPILE=1 torchrun --nproc-per-node 1 pretrain.py \
  arch=urm_r2g_h96_ranking_noise \
  $COMMON_ARGS \
  +run_name=$run_name \
  +checkpoint_path=$checkpoint_path

echo "=== R2g-h96 complete ==="

# R2i-h96: Cross-trajectory ranking
run_name="R2i-h96-cross-traj-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path
echo "=== Starting R2i-h96: cross-trajectory ranking (K=2, elw=0.05) ==="
DISABLE_COMPILE=1 torchrun --nproc-per-node 1 pretrain.py \
  arch=urm_r2i_h96_cross_traj \
  $COMMON_ARGS \
  +run_name=$run_name \
  +checkpoint_path=$checkpoint_path

echo "=== Both R2g and R2i experiments complete ==="
