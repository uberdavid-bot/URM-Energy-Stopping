"""R8a smoke test: 200 batches through the real train path (train_batch_hl).

Checks:
  1. Nonzero grad norms on BOTH fL and fH after one supervision step
  2. Loss decreases over 200 batches
  3. VRAM < 24GB
  4. Throughput (expect ~4x R4d wall-clock, i.e. ~1.5-2 it/s on 3090)

Run: DISABLE_COMPILE=1 conda run -n urm python scripts/smoke_r8a.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DISABLE_COMPILE", "1")

import torch
from omegaconf import OmegaConf

from pretrain import PretrainConfig, create_dataloader, init_train_state, train_batch

NUM_BATCHES = 200

arch = OmegaConf.to_container(
    OmegaConf.load("config/arch/urm_r8a_hl_h128.yaml"), resolve=True
)

config = PretrainConfig(
    arch=arch,
    data_path="data/arc1concept-aug-1000-size-10",
    global_batch_size=512,
    epochs=31590,
    eval_interval=2106,
    lr=3e-4,
    lr_min_ratio=1.0,
    lr_warmup_steps=100,
    weight_decay=0.1,
    beta1=0.9,
    beta2=0.95,
    target_q_update_every=10,
    puzzle_emb_lr=1e-2,
    puzzle_emb_weight_decay=0.01,
    grad_clip_backbone=5.0,
)

torch.random.manual_seed(config.seed)

train_loader, train_metadata = create_dataloader(
    config, "train", test_set_mode=False, epochs_per_iter=config.eval_interval,
    global_batch_size=config.global_batch_size, rank=0, world_size=1,
)
train_state = init_train_state(config, train_metadata, rank=0, world_size=1)
model = train_state.model
model.train()

print(f"total_steps: {train_state.total_steps}")
print(f"num_params: {sum(p.numel() for p in model.parameters())}")

# --- Check 1: grad norms on both modules after one supervision step ---
loader_iter = iter(train_loader)
_, batch, gbs = next(loader_iter)
batch_gpu = {k: v.cuda() for k, v in batch.items()}
carry, loss, _, _, _ = model(batch=batch_gpu, return_keys=[], carry=None)
loss.backward()
def grad_norm(params):
    sq = sum(p.grad.float().norm() ** 2 for p in params if p.grad is not None)
    return float(sq) ** 0.5

fl_gn = grad_norm(list(model.model.inner.layers.parameters()))
fh_gn = grad_norm(list(model.model.h_layers.parameters()))
print(f"fL grad norm: {fl_gn:.4f} | fH grad norm: {fh_gn:.4f}")
assert fl_gn > 0, "fL received no gradient"
assert fh_gn > 0, "fH received no gradient"
model.zero_grad(set_to_none=True)

# --- Checks 2-4: 200 batches through train_batch ---
torch.cuda.reset_peak_memory_stats()
losses = []
t0 = time.time()
n = 0
while n < NUM_BATCHES:
    try:
        _, batch, gbs = next(loader_iter)
    except StopIteration:
        loader_iter = iter(train_loader)
        continue
    metrics = train_batch(config, train_state, batch, gbs, rank=0, world_size=1)
    n += 1
    if metrics is not None:
        losses.append(metrics["train/reconstruction_loss"])
    if n % 20 == 0:
        elapsed = time.time() - t0
        print(f"batch {n}: recon_loss={losses[-1]:.4f}  "
              f"step4_exact={metrics['train/step_4_exact_accuracy']:.4f}  "
              f"{n / elapsed:.2f} it/s  "
              f"VRAM peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

elapsed = time.time() - t0
first, last = sum(losses[:10]) / 10, sum(losses[-10:]) / 10
peak_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"\nloss (mean first 10): {first:.4f} -> (mean last 10): {last:.4f}")
print(f"throughput: {NUM_BATCHES / elapsed:.2f} it/s ({elapsed:.1f}s total)")
print(f"VRAM peak: {peak_gb:.2f} GB")
assert last < first, "Loss did not decrease over 200 batches"
assert peak_gb < 24, f"VRAM peak {peak_gb:.2f} GB exceeds 24GB"
print("\nSMOKE TEST PASSED")
