"""R7c2 posterior-eval probe: compare prior-ε vs posterior-ε eval accuracy.

Loads a GRAM checkpoint and runs eval twice:
  1. Standard (prior-ε): eval as normal, ε sampled from prior
  2. Posterior-ε: force posterior on (feeding labels) at eval time

This disambiguates whether R7c2's eval collapse is due to:
  (A) prior/posterior distribution mismatch (fixable)
  (B) stochastic perturbation genuinely breaking refinement (clean negative)

Usage:
  DISABLE_COMPILE=1 python scripts/eval_posterior_probe.py \
    --checkpoint checkpoints/R7c2-gram-warmup-h128-260607/step_80011.pt \
    --config checkpoints/R7c2-gram-warmup-h128-260607/config.yaml \
    --data_path data/arc1concept-aug-1000-size-10
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from torch.utils.data import DataLoader
from models.losses import IGNORE_LABEL_ID


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--global_batch_size", type=int, default=512)
    args = parser.parse_args()

    # Load config to get arch settings
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    arch_cfg = cfg.get("arch", cfg)

    # Build dataset
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=0, dataset_path=args.data_path, rank=0, num_replicas=1,
            global_batch_size=args.global_batch_size,
            test_set_mode=True,
            epochs_per_iter=1,
        ),
        split="test",
    )
    loader = DataLoader(dataset, batch_size=None, num_workers=1, pin_memory=True)
    metadata = dataset.metadata

    # Build model
    from models.urm.urm_energy import ARCModel, ARCModelConfig

    model_cfg_dict = dict(
        **{k: v for k, v in arch_cfg.items() if k not in ("name", "loss")},
        batch_size=args.global_batch_size,
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
        causal=False,
    )

    with torch.device("cuda"):
        model = ARCModel(model_cfg_dict)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cuda")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    # The checkpoint was saved from EnergyLossHead wrapping ARCModel.
    # Strip the "model." prefix that EnergyLossHead adds.
    stripped = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            stripped[k[len("model."):]] = v
        # Skip loss head keys (energy_head, etc. at top level)

    result = model.load_state_dict(stripped, strict=False, assign=True)
    print(f"Loaded checkpoint: {args.checkpoint}")
    if result.missing_keys:
        print(f"  Missing: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  Unexpected: {result.unexpected_keys}")

    model.eval()
    N = model.config.loops

    # Run eval twice: prior and posterior
    for mode in ["prior", "posterior"]:
        force_posterior = (mode == "posterior")
        print(f"\n{'='*60}")
        print(f"  EVAL MODE: {mode.upper()} ε")
        print(f"{'='*60}")

        total_valid = 0
        per_step_correct = [0] * N
        per_step_cell_correct = [0] * N
        total_cells = 0

        with torch.no_grad():
            for batch_idx, (set_name, batch, gbs) in enumerate(loader):
                batch = {k: v.cuda() for k, v in batch.items()}
                labels = batch["labels"]
                mask = labels != IGNORE_LABEL_ID
                loss_counts = mask.sum(-1)
                valid = loss_counts > 0
                B = labels.shape[0]

                all_logits, all_q_logits, all_hidden, input_emb, gram_kl, _ = model.forward_trajectory(
                    batch, N,
                    labels=labels if force_posterior else None,
                    force_posterior=force_posterior,
                )

                n_valid = valid.sum().item()
                total_valid += n_valid
                total_cells += mask.sum().item()

                for t in range(N):
                    preds_t = torch.argmax(all_logits[t], dim=-1)
                    tok_correct = mask & (preds_t == labels)
                    seq_correct = tok_correct.sum(-1) == loss_counts
                    per_step_correct[t] += (valid & seq_correct).sum().item()
                    per_step_cell_correct[t] += tok_correct.sum().item()

                if (batch_idx + 1) % 200 == 0:
                    print(f"  Processed {batch_idx + 1} batches, {total_valid} samples")

        print(f"\nTotal valid samples: {total_valid}")
        print(f"Total cells: {total_cells}")
        print(f"\n{'Step':>4} {'Exact':>10} {'Cell Acc':>10}")
        for t in range(N):
            exact = per_step_correct[t] / total_valid if total_valid > 0 else 0
            cell = per_step_cell_correct[t] / total_cells if total_cells > 0 else 0
            print(f"  {t+1:>2}   {exact:>9.4f}   {cell:>9.4f}")

        final_exact = per_step_correct[-1] / total_valid if total_valid > 0 else 0
        final_cell = per_step_cell_correct[-1] / total_cells if total_cells > 0 else 0
        print(f"\nFinal (step {N}): exact={final_exact:.4f}, cell_acc={final_cell:.4f}")


if __name__ == "__main__":
    main()
