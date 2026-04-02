run_name="URM-energy-arcagi1-dsm-v1"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path

torchrun --nproc-per-node 1 pretrain.py \
data_path=data/arc1concept-aug-10 \
arch=urm_energy \
arch.loops=16 \
arch.H_cycles=4 \
arch.L_cycles=3 \
arch.num_layers=4 \
arch.energy_threshold=0.005 \
arch.min_steps=8 \
epochs=200000 \
eval_interval=50 \
puzzle_emb_lr=1e-2 \
lr=1e-4 \
weight_decay=0.1 \
global_batch_size=12 \
+run_name=$run_name \
+checkpoint_path=$checkpoint_path \
+ema=True
