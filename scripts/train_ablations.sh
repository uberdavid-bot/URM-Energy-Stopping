#!/bin/bash
# Ablation experiments: why does energy co-training help the backbone?
#
# A1 (4 runs): energy_loss_weight sweep [0.01, 0.05, 0.2, 0.5]
#   - R2c used 0.1 → 6.95% eval exact. Sweet spot or flat?
# A2 (1 run): random quality shuffle — same gradients, noise supervision
#   - If backbone still improves → mechanism is gradient diversity, not trajectory signal
# A3 (1 run): detach hidden before energy head — energy head trains, backbone doesn't see it
#   - If backbone improvement disappears → mechanism is gradient flow through shared layers
#
# All configs use R2c architecture (position_mlp, dropout=0.1)
# Baseline comparisons: R1i (no energy, 5.33%), R2c (energy=0.1, 6.95%)
#
# Run sequentially — each is ~4 hours on 3090

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

# A1a: energy_loss_weight=0.01
run_name="A1a-elw001-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path
echo "=== Starting A1a: energy_loss_weight=0.01 ==="
DISABLE_COMPILE=1 torchrun --nproc-per-node 1 pretrain.py \
  arch=ablation_a1_elw001 \
  $COMMON_ARGS \
  +run_name=$run_name \
  +checkpoint_path=$checkpoint_path

# A1b: energy_loss_weight=0.05
run_name="A1b-elw005-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path
echo "=== Starting A1b: energy_loss_weight=0.05 ==="
DISABLE_COMPILE=1 torchrun --nproc-per-node 1 pretrain.py \
  arch=ablation_a1_elw005 \
  $COMMON_ARGS \
  +run_name=$run_name \
  +checkpoint_path=$checkpoint_path

# A1c: energy_loss_weight=0.2
run_name="A1c-elw020-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path
echo "=== Starting A1c: energy_loss_weight=0.2 ==="
DISABLE_COMPILE=1 torchrun --nproc-per-node 1 pretrain.py \
  arch=ablation_a1_elw020 \
  $COMMON_ARGS \
  +run_name=$run_name \
  +checkpoint_path=$checkpoint_path

# A1d: energy_loss_weight=0.5
run_name="A1d-elw050-$(date +%y%m%d)"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path
echo "=== Starting A1d: energy_loss_weight=0.5 ==="
DISABLE_COMPILE=1 torchrun --nproc-per-node 1 pretrain.py \
  arch=ablation_a1_elw050 \
  $COMMON_ARGS \
  +run_name=$run_name \
  +checkpoint_path=$checkpoint_path

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

echo "=== All ablation experiments complete ==="
