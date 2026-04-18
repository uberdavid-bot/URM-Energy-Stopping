#!/bin/bash
# R5a/R5b/R5c — scaling curve completion + single-run strengthening.
#
# Motivation: R4d (h=128, +5.66pp) and R4e (h=160, +8.13pp) both PASSED the
# >17% eval-exact bar. Width is the effective lever on ARC-10x10 at 80K steps.
# R5 completes the width curve at h=192, tests whether R4c's loops=16 pathology
# survives at h=128, and extends the h=160 run to probe training headroom.
#
# Sequence:
#   1. R5a: h=192, loops=8, 80K steps (~3-3.5h)
#   2. R5b: h=128, loops=16, 80K steps (~4h, 2x compute per step)
#   3. R5c: h=160, loops=8, 120K steps (~4h, 1.5x training)
#
# Baselines:
#   R1-h96:  15.59% eval exact, 40.91% pass@1000
#   R4d h=128: 21.25% eval exact, 43.51% pass@1000
#   R4e h=160: 23.72% eval exact, 48.05% pass@1000 (best backbone so far)
#   R4c loops=16 at h=96: 13.38% eval exact (the pathology R5b tests)
#
# Total runtime: ~11-12h. Safe overnight.

set -e

# conda run so this script can be launched non-interactively (nohup).
CONDA_RUN="conda run -n urm --no-capture-output"

# Pre-flight: refuse to start if pretrain.py is already running
if pgrep -f "pretrain.py" > /dev/null; then
    echo "ERROR: pretrain.py already running. Kill before launching R5 sweep."
    exit 1
fi
mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$mem_used" -gt 2000 ]; then
    echo "WARNING: GPU has ${mem_used}MB allocated — previous run may not have cleaned up."
fi

# Args shared by the 80K-step runs (R5a and R5b).
COMMON_ARGS_80K="data_path=data/arc1concept-aug-1000-size-10 \
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
  +ema=True"

# [1/3] R5a: h=192, loops=8, 80K steps
run_name="R5a-h192-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path
echo "=== [1/3] R5a: h=192, loops=8, 80K steps ==="
DISABLE_COMPILE=1 $CONDA_RUN torchrun --nproc-per-node 1 pretrain.py \
  arch=urm_r5a_h192 \
  $COMMON_ARGS_80K \
  +run_name=$run_name \
  +checkpoint_path=$checkpoint_path
echo "=== R5a complete ==="

# [2/3] R5b: h=128, loops=16, 80K steps
run_name="R5b-h128-loops16-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path
echo "=== [2/3] R5b: h=128, loops=16, 80K steps ==="
DISABLE_COMPILE=1 $CONDA_RUN torchrun --nproc-per-node 1 pretrain.py \
  arch=urm_r5b_h128_loops16 \
  $COMMON_ARGS_80K \
  +run_name=$run_name \
  +checkpoint_path=$checkpoint_path
echo "=== R5b complete ==="

# [3/3] R5c: h=160, loops=8, 120K steps (1.5x R4e)
# epochs=47385 (vs 31590 for 80K). eval_interval MUST divide epochs (pretrain.py
# enforces this) — 3159 = 47385/15, giving 15 evals, the same cadence as R4e.
run_name="R5c-h160-long-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path
echo "=== [3/3] R5c: h=160, loops=8, 120K steps ==="
DISABLE_COMPILE=1 $CONDA_RUN torchrun --nproc-per-node 1 pretrain.py \
  arch=urm_r5c_h160_long \
  data_path=data/arc1concept-aug-1000-size-10 \
  epochs=47385 \
  eval_interval=3159 \
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
  +ema=True \
  +run_name=$run_name \
  +checkpoint_path=$checkpoint_path
echo "=== R5c complete ==="

echo "=== All R5 experiments complete ==="
echo "Key metrics to compare on wandb:"
echo "  R5a: all.exact_accuracy vs R4e 23.72% (does width still climb at h=192?)"
echo "  R5b: all.exact_accuracy vs R4d 21.25% (does loops=16 hurt at h=128 too?)"
echo "  R5c: all.exact_accuracy vs R4e 23.72% (does training longer help at h=160?)"
echo "  R5c: train/eval ratio vs R4e 2.37 (overfitting under longer training?)"
