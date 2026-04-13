#!/bin/bash
# Resume ablation experiments after reboot.
# A1a, A1b, A1c completed. A1d was ~60% done. A2 and A3 not started.
# Restarts A1d from latest checkpoint, then runs A2 and A3.

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

# A1d: resume from checkpoint
run_name="A1d-elw050-260413"
checkpoint_path="checkpoints/${run_name}"
echo "=== Resuming A1d: energy_loss_weight=0.5 from checkpoint ==="
DISABLE_COMPILE=1 torchrun --nproc-per-node 1 pretrain.py \
  arch=ablation_a1_elw050 \
  $COMMON_ARGS \
  +run_name=$run_name \
  +checkpoint_path=$checkpoint_path \
  load_checkpoint=latest

# A2: random auxiliary head (shuffled quality)
run_name="A2-random-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path
echo "=== Starting A2: random auxiliary head ==="
DISABLE_COMPILE=1 torchrun --nproc-per-node 1 pretrain.py \
  arch=ablation_a2_random \
  $COMMON_ARGS \
  +run_name=$run_name \
  +checkpoint_path=$checkpoint_path

# A3: frozen energy head (detached hidden)
run_name="A3-detach-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path
echo "=== Starting A3: frozen energy head ==="
DISABLE_COMPILE=1 torchrun --nproc-per-node 1 pretrain.py \
  arch=ablation_a3_detach \
  $COMMON_ARGS \
  +run_name=$run_name \
  +checkpoint_path=$checkpoint_path

echo "=== All remaining ablation experiments complete ==="
