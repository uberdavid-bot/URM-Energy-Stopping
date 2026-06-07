#!/bin/bash
# R7a — GRAM stochastic latent transitions at h=128
#
# Baseline: R4d (d=1, h=128, exp=2, loops=8, dropout=0.1, 137K params, 21.25% eval exact).
# Adds GRAM prior/posterior VAE perturbation at every URM recurrence step.
# No energy head training — energy_loss_weight=0.
#
# Metrics to watch:
#   gram_kl         — mean per-step KL (should be positive, not collapsing to 0)
#   gram_sigma_mean — mean prior std (if →0, perturbation collapsed)
#   gram_majority_exact      — majority-vote exact accuracy over M=8 trajectories
#   gram_qhalt_bestof_exact  — Q-halt best-of-N exact accuracy
#
# Runtime estimate: ~2-3h on 3090 (h=128 baseline + GRAM MLP overhead).

set -e

CONDA_RUN="conda run -n urm --no-capture-output"

# Pre-flight: verify no leftover pretrain processes
if pgrep -f "pretrain.py" > /dev/null; then
    echo "ERROR: pretrain.py already running. Kill before launching R7a."
    exit 1
fi
mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$mem_used" -gt 2000 ]; then
    echo "WARNING: GPU has ${mem_used}MB allocated — previous run may not have cleaned up."
fi

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
  target_q_update_every=10 \
  grad_clip_backbone=5.0 \
  +ema=True"

run_name="R7a-gram-h128-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p "$checkpoint_path"

echo "=== R7a: GRAM stochastic latent transitions, h=128 ==="
DISABLE_COMPILE=1 $CONDA_RUN torchrun --nproc-per-node 1 pretrain.py \
  arch=urm_r7a_gram_h128 \
  weight_decay=0.1 \
  $COMMON_ARGS \
  +run_name=$run_name \
  +checkpoint_path=$checkpoint_path
echo "=== R7a complete ==="
