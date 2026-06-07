"""R7d smoke test: ~500 steps, report per-step exact curve + sigma diagnostics."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import yaml
from pathlib import Path
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from data.common import PuzzleDatasetMetadata
from utils import load_model_class
from models.losses import EnergyLossHead

DEVICE = "cuda"
STEPS = 500
BATCH_SIZE = 64
LR = 3e-4
DATA_PATH = "data/arc1concept-aug-1000-size-10"
CONFIG_PATH = "config/arch/urm_r7d_gram_deltacal_h128.yaml"


def main():
    with open(CONFIG_PATH) as f:
        arch_config = yaml.safe_load(f)

    if isinstance(arch_config.get("puzzle_emb_ndim"), str):
        arch_config["puzzle_emb_ndim"] = arch_config["hidden_size"]

    loss_config = arch_config.pop("loss")
    loss_config.pop("name", None)
    arch_config.pop("ema", None)
    arch_config.pop("ema_rate", None)

    # Load metadata
    with open(os.path.join(DATA_PATH, "train", "dataset.json")) as f:
        meta = PuzzleDatasetMetadata(**json.load(f))

    # Dataset
    ds_config = PuzzleDatasetConfig(
        seed=42,
        dataset_path=DATA_PATH,
        global_batch_size=BATCH_SIZE,
        test_set_mode=False,
        epochs_per_iter=50,
        rank=0,
        num_replicas=1,
    )
    dataset = PuzzleDataset(ds_config, split="train")
    dataloader = iter(torch.utils.data.DataLoader(
        dataset, batch_size=None, num_workers=0
    ))

    # Model
    arch_config["batch_size"] = BATCH_SIZE
    arch_config["seq_len"] = meta.seq_len
    arch_config["num_puzzle_identifiers"] = meta.num_puzzle_identifiers
    arch_config["vocab_size"] = meta.vocab_size

    ModelClass = load_model_class("urm.urm_energy@ARCModel")
    with torch.device(DEVICE):
        model = ModelClass(arch_config)
        loss_head = EnergyLossHead(model, **loss_config)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model params: {param_count:,}")
    print(f"Config: alpha={model.config.gram_sigma_alpha}, k={model.config.gram_latent_dim}, "
          f"predecode={model.config.gram_predecode}, warmup={model.config.gram_kl_warmup_steps}")
    print(f"Running {STEPS} steps, batch={BATCH_SIZE}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)

    N = model.config.loops
    report_interval = 100
    step_exact_accum = [0.0] * N
    count_accum = 0
    kl_accum = 0.0
    sigma_eff_accum = 0.0
    sigma_eff_count = 0

    for step in range(1, STEPS + 1):
        _, batch, _ = next(dataloader)
        batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        model.train()
        loss_head.current_step = step
        _, loss, metrics, _, _ = loss_head(batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        count = metrics["count"].item()
        count_accum += count
        for t in range(N):
            step_exact_accum[t] += metrics[f"step_{t+1}_exact_accuracy"].item()
        if "gram_kl" in metrics:
            kl_accum += metrics["gram_kl"].item()
        if "gram_sigma_effective" in metrics:
            sigma_eff_accum += metrics["gram_sigma_effective"].item()
            sigma_eff_count += 1

        if step % report_interval == 0:
            print(f"--- Step {step} ---")
            print(f"  Per-step exact (train): ", end="")
            for t in range(N):
                pct = 100.0 * step_exact_accum[t] / count_accum
                print(f"s{t+1}={pct:.2f}%", end="  ")
            print()
            print(f"  gram_kl (mean): {kl_accum / count_accum:.4f}")
            if sigma_eff_count > 0:
                print(f"  gram_sigma_effective (mean): {sigma_eff_accum / count_accum:.6f}")

            # Per-step delta norms from last batch
            print(f"  Last-batch delta norms: ", end="")
            for t in range(1, N):
                key = f"step_{t+1}_delta_norm"
                if key in metrics:
                    dn = metrics[key].item() / count
                    print(f"d{t+1}={dn:.4f}", end="  ")
            print()

            s8_exact = 100.0 * step_exact_accum[-1] / count_accum
            print(f"  step_8_exact cumulative: {s8_exact:.2f}% (leak check: should NOT be ~100%)")
            print()

    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY (cumulative over all steps)")
    print("="*60)
    print(f"\nPer-step exact accuracy curve (train):")
    for t in range(N):
        pct = 100.0 * step_exact_accum[t] / count_accum
        print(f"  step {t+1}: {pct:.2f}%")

    print(f"\ngram_kl (mean): {kl_accum / count_accum:.4f}")
    if sigma_eff_count > 0:
        print(f"gram_sigma_effective (mean): {sigma_eff_accum / count_accum:.6f}")

    # Final detailed per-step diagnostic
    print(f"\n--- Detailed per-step sigma/delta table (last batch) ---")
    model.train()
    _, batch, _ = next(dataloader)
    batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    with torch.no_grad():
        all_logits, all_q_logits, all_hidden, input_emb, gram_kl_list, gram_eff_sigma = model.forward_trajectory(
            batch, labels=batch["labels"]
        )

    init_hidden = model.inner.init_hidden.expand_as(all_hidden[0])
    print(f"\n{'Step':<6} {'Delta_norm':<12} {'Sigma_eff':<12} {'Ratio(σ/Δ)':<12}")
    print("-" * 44)
    for t in range(N):
        if t == 0:
            delta = (all_hidden[0] - init_hidden).norm(dim=-1).mean().item()
        else:
            delta = (all_hidden[t] - all_hidden[t-1]).norm(dim=-1).mean().item()

        if gram_eff_sigma is not None and t < len(gram_eff_sigma):
            sigma = gram_eff_sigma[t].item()
            ratio = sigma / max(delta, 1e-8)
            print(f"  {t+1:<4} {delta:<12.4f} {sigma:<12.6f} {ratio:<12.4f}")
        else:
            print(f"  {t+1:<4} {delta:<12.4f} {'N/A':<12} {'N/A':<12}")


if __name__ == "__main__":
    main()
