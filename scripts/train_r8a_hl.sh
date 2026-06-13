#!/bin/bash
# R8a — H/L hierarchical structure, paper-faithful deep supervision (deterministic).
#
# Restores HRM/GRAM hierarchy (separate fL/fH modules, one decode per supervision
# step, detach-between-steps carry) with the TRM training recipe on the R4d block
# config (h=128). See docs_hypotheses.md "R8a".
#
# Budget: same number of BATCHES as R4d's 80K-step run (epochs=31590) — this is
# 4x the optimizer steps (n_sup=4 per batch) and ~4x wall-clock (~8h on 3090).
#
# Decision rule (pre-committed): compare step-4 eval exact_accuracy vs R4d
# (~21.25%) and vs R4e h=160 (23.72%, approx param-matched). Beat both ->
# ablations (half-width modules, lower Nsup), then terminal noise (R8c).
# Lose to both by >2pp -> hierarchy negative at this scale; discuss first.

set -e

CONDA_RUN="conda run -n urm --no-capture-output"

if pgrep -f "pretrain.py" > /dev/null; then
    echo "ERROR: pretrain.py already running. Kill before launching R8a."
    exit 1
fi
mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$mem_used" -gt 2000 ]; then
    echo "WARNING: GPU has ${mem_used}MB allocated — previous run may not have cleaned up."
fi

run_name="R8a-hl-det-h128-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path
echo "=== R8a: H/L hierarchy, deterministic, h=128, n_sup=4 ==="
DISABLE_COMPILE=1 $CONDA_RUN torchrun --nproc-per-node 1 pretrain.py \
  arch=urm_r8a_hl_h128 \
  data_path=data/arc1concept-aug-1000-size-10 \
  epochs=31590 \
  eval_interval=2106 \
  global_batch_size=512 \
  lr=3e-4 \
  lr_min_ratio=1.0 \
  lr_warmup_steps=100 \
  beta1=0.9 \
  beta2=0.95 \
  weight_decay=0.1 \
  puzzle_emb_lr=1e-2 \
  puzzle_emb_weight_decay=0.01 \
  target_q_update_every=10 \
  grad_clip_backbone=5.0 \
  +ema=True \
  +run_name=$run_name \
  +checkpoint_path=$checkpoint_path
echo "=== R8a complete ==="
echo "Key metrics on wandb:"
echo "  - all.step_4_exact_accuracy (headline) vs R4d 21.25% and R4e 23.72%"
echo "  - all.step_{1..4}_exact_accuracy (per-supervision-step curve)"
echo "  - all.qhalt_stop_accuracy (secondary)"
