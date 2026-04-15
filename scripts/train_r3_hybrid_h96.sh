#!/bin/bash
set -e

# R3: Hybrid URM(5) + MCMC(3) at h=96
# Energy head trained through MCMC reconstruction loss (create_graph=True).
# No trajectory ranking — energy learns from its own optimization landscape.
#
# Baselines to beat:
#   R1-h96 (pure URM):        15.59% eval exact, peak 16.09% @ step 7
#   R2g-h96 (URM + ranking):  13.59% eval exact
#
# At batch 512: epochs=31590, eval_interval=2106 (matches R1-h96 budget)
# If OOM (create_graph VRAM), fallback to batch 256: epochs=63180, eval_interval=4212

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate urm

run_name="R3-hybrid-h96-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path
mkdir -p logs

echo "=== R3: Hybrid URM(5) + MCMC(3) at h=96 ==="
echo "=== Energy head trained via create_graph MCMC reconstruction ==="
echo "=== Checkpoint: $checkpoint_path ==="
echo ""

DISABLE_COMPILE=1 torchrun --nproc-per-node 1 pretrain.py \
  data_path=data/arc1concept-aug-1000-size-10 \
  arch=urm_r3_hybrid_h96 \
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

echo "=== R3 complete ==="
echo "=== Check wandb for: ==="
echo "===   1. Per-step accuracy (do MCMC steps 6-8 improve over URM steps 6-8?) ==="
echo "===   2. mcmc_improvement (accuracy after MCMC - accuracy at mcmc_start_step) ==="
echo "===   3. Energy values decreasing across MCMC steps (landscape is being optimized) ==="
echo "===   4. No URM degradation (steps 1-5 accuracy should match R1-h96) ==="
