#!/bin/bash
set -e

# R3b: Hybrid URM(5) + MCMC(3) at h=96, step_size=0.1 (10x R3)
# R3 proved the energy landscape was directionally correct but step_size=0.01
# was a negligible perturbation relative to URM's natural update magnitude.
# R3b tests whether the trained landscape can produce useful refinement when
# given enough step magnitude to flip decoded tokens.
#
# Baselines:
#   R1-h96 (pure URM):        15.59% eval exact, peak 16.09% @ step 7
#   R3   (step_size=0.01):    killed at 47%, mcmc_improvement ~= 0
#
# R3-diag warning: at step_size >= 0.3, Q-halt-gradient MCMC went adversarial.
# 0.1 is on the safe side of that but still large enough to matter.

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate urm

run_name="R3b-hybrid-h96-ss01-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path
mkdir -p logs

echo "=== R3b: Hybrid URM(5) + MCMC(3) at h=96, step_size=0.1 ==="
echo "=== mcmc_random_step_size samples from [0.033, 0.1] during training ==="
echo "=== Checkpoint: $checkpoint_path ==="
echo ""

DISABLE_COMPILE=1 torchrun --nproc-per-node 1 pretrain.py \
  data_path=data/arc1concept-aug-1000-size-10 \
  arch=urm_r3b_hybrid_h96 \
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

echo "=== R3b complete ==="
echo "=== KEY CHECKS (metrics now correctly aggregated after Part 1 fix): ==="
echo "===   1. MCMC delta_norm at steps 6-8: expect ~0.08-0.1 ==="
echo "===   2. mcmc_improvement: THE primary metric ==="
echo "===   3. energy_decrease: should be negative (landscape descent) ==="
echo "===   4. URM steps 1-5 accuracy: should track R1-h96 trajectory ==="
