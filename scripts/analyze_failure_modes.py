"""R4-prep: Failure mode bucketing on R1-h96 eval set.

For each (puzzle, augmentation) item in the test split, run forward_trajectory,
compute per-step Q-halt + hidden delta signals, and assign to exactly one
mutually-exclusive bucket (checked in order):

  1. Correct           -- exact match at step 6
  2. Stuck             -- wrong + step6_delta_norm < 0.003 + qhalt_fired_step > 6
  3. Oscillating       -- wrong + step8_delta_norm > 0.003
  4. Halted early      -- wrong + qhalt_fired_step <= 6
  5. Other (wrong)     -- catch-all for items that fall through the above

Why this matters:
  - Bucket 2 dominance argues for registers (more scratchpad to escape minima).
  - Bucket 3 dominance argues for XSA (stop reacting to own predictions).
  - Bucket 4 dominance argues for multi-register Q-halt extension.

This is a READ-ONLY analysis: no model modifications, no training.

Usage:
    conda activate urm
    python scripts/analyze_failure_modes.py \\
        --checkpoint checkpoints/R1-h96-baseline-260414 \\
        --data_path data/arc1concept-aug-1000-size-10 \\
        --output logs/failure_modes_R1h96.txt
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pretrain import (  # noqa: E402
    PretrainConfig,
    create_dataloader,
    init_train_state,
    load_config_from_checkpoint_path,
)
from data.build_arc_dataset import inverse_aug  # noqa: E402
from models.losses import IGNORE_LABEL_ID  # noqa: E402


STUCK_DELTA_THRESHOLD = 0.003
OSCILLATING_DELTA_THRESHOLD = 0.003
EARLY_HALT_STEP = 6
EXACT_STEP = 6  # 1-indexed: use step-6 prediction for correctness (R1-h96 peak)
N_STEPS = 8


def _resolve_checkpoint_dir(path: str) -> str:
    if not os.path.isdir(path):
        return path
    pts = [
        (int(m.group(1)), os.path.join(path, f))
        for f in os.listdir(path)
        for m in [re.match(r"step_(\d+)\.pt$", f)]
        if m
    ]
    if not pts:
        raise FileNotFoundError(f"No step_*.pt files in {path}")
    pts.sort()
    return pts[-1][1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Checkpoint dir or .pt file")
    parser.add_argument("--data_path", required=True, help="Dataset directory")
    parser.add_argument("--output", default="logs/failure_modes_R1h96.txt")
    parser.add_argument("--max_batches", type=int, default=0,
                        help="Cap eval batches for a quick sanity run (0 = full set)")
    args = parser.parse_args()

    os.environ["DISABLE_COMPILE"] = "1"

    # --- Load config + checkpoint ---
    config = load_config_from_checkpoint_path(args.checkpoint)
    if config is None:
        import yaml
        from pathlib import Path
        cfg_path = Path(args.checkpoint) / "config.yaml"
        if not cfg_path.exists():
            raise ValueError(f"Could not load config from {args.checkpoint}")
        with open(cfg_path) as f:
            config = PretrainConfig(**yaml.safe_load(f))

    ckpt_path = _resolve_checkpoint_dir(args.checkpoint)
    print(f"Using checkpoint: {ckpt_path}")

    config.load_checkpoint = ckpt_path
    config.data_path = args.data_path
    config.load_strict = True
    config.load_optimizer_state = False

    eval_loader, eval_metadata = create_dataloader(
        config, "test", test_set_mode=True, epochs_per_iter=1,
        global_batch_size=config.global_batch_size, rank=0, world_size=1,
    )

    train_state = init_train_state(config, eval_metadata, rank=0, world_size=1)
    train_state.model.eval()

    loss_head = train_state.model
    arc_model = loss_head.model
    blank_id = eval_metadata.blank_identifier_id

    with open(os.path.join(args.data_path, "identifiers.json"), "r") as f:
        identifier_map = json.load(f)

    bucket_names = ["Correct", "Stuck", "Oscillating", "Halted early", "Other (wrong)"]
    bucket_counts = Counter()
    bucket_step6_delta_sum = defaultdict(float)
    bucket_qhalt_at6_sum = defaultdict(float)
    qhalt_fired_hist_incorrect = Counter()  # step (0..N) -> count, where N means never
    incorrect_per_puzzle = Counter()

    print(f"Running eval with N={N_STEPS} steps...")
    batch_count = 0
    total_items = 0
    with torch.no_grad():
        for set_name, batch, _ in eval_loader:
            batch_count += 1
            if args.max_batches and batch_count > args.max_batches:
                break
            batch = {k: v.cuda() for k, v in batch.items()}
            labels = batch["labels"]
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            valid = (loss_counts > 0) & (batch["puzzle_identifiers"] != blank_id)

            all_logits, all_q_logits, all_hidden, _ = arc_model.forward_trajectory(
                batch, N=N_STEPS
            )

            # Step-6 (index 5) predictions for correctness
            step6_preds = torch.argmax(all_logits[EXACT_STEP - 1], dim=-1)
            step6_correct = (mask & (step6_preds == labels)).sum(-1) == loss_counts

            # Per-sample delta norms: (hidden_t - hidden_{t-1}).norm(-1).mean(-1)
            # step6_delta = ||h6 - h5||_2 mean over positions
            d6 = (all_hidden[5] - all_hidden[4]).norm(dim=-1).mean(dim=-1).to(torch.float32)
            d8 = (all_hidden[7] - all_hidden[6]).norm(dim=-1).mean(dim=-1).to(torch.float32)

            # Per-step Q-halt logits: list of [B] tensors. Stack to [B, N].
            q_logits = torch.stack(all_q_logits, dim=1).to(torch.float32)  # [B, N]
            q_probs = torch.sigmoid(q_logits)
            qhalt_at_6 = q_probs[:, EXACT_STEP - 1]

            # qhalt_fired_step: first step (1-indexed) where sigmoid(q) > 0.5,
            # equivalently q_logit > 0. Default to N if never.
            B = labels.shape[0]
            fired_mask = q_logits > 0  # [B, N]
            any_fired = fired_mask.any(dim=1)
            first_fired = torch.where(
                any_fired,
                fired_mask.to(torch.float32).argmax(dim=1) + 1,
                torch.full((B,), N_STEPS, device=labels.device, dtype=torch.float32),
            ).long()

            puzzle_ids = batch["puzzle_identifiers"].cpu().tolist()
            valid_cpu = valid.cpu().tolist()
            correct_cpu = step6_correct.cpu().tolist()
            d6_cpu = d6.cpu().tolist()
            d8_cpu = d8.cpu().tolist()
            qhalt_at6_cpu = qhalt_at_6.cpu().tolist()
            first_fired_cpu = first_fired.cpu().tolist()

            for i in range(B):
                if not valid_cpu[i]:
                    continue
                total_items += 1
                is_correct = bool(correct_cpu[i])
                ff = int(first_fired_cpu[i])
                d6_i = float(d6_cpu[i])
                d8_i = float(d8_cpu[i])
                q6_i = float(qhalt_at6_cpu[i])

                if is_correct:
                    bucket = "Correct"
                elif d6_i < STUCK_DELTA_THRESHOLD and ff > EARLY_HALT_STEP:
                    bucket = "Stuck"
                elif d8_i > OSCILLATING_DELTA_THRESHOLD:
                    bucket = "Oscillating"
                elif ff <= EARLY_HALT_STEP:
                    bucket = "Halted early"
                else:
                    bucket = "Other (wrong)"

                bucket_counts[bucket] += 1
                bucket_step6_delta_sum[bucket] += d6_i
                bucket_qhalt_at6_sum[bucket] += q6_i

                if not is_correct:
                    qhalt_fired_hist_incorrect[ff] += 1
                    aug_name = identifier_map[int(puzzle_ids[i])]
                    orig_name, _ = inverse_aug(aug_name)
                    incorrect_per_puzzle[orig_name] += 1

            if batch_count % 20 == 0:
                print(f"  processed {batch_count} batches, {total_items} items")

    print(f"Done: {batch_count} batches, {total_items} items")

    # --- Build report ---
    lines = []
    lines.append("=" * 78)
    lines.append("R1-h96 failure mode analysis")
    lines.append("=" * 78)
    lines.append(f"checkpoint:        {ckpt_path}")
    lines.append(f"data_path:         {args.data_path}")
    lines.append(f"total items:       {total_items}")
    lines.append(f"correctness step:  {EXACT_STEP} (1-indexed)")
    lines.append(f"recurrence steps:  {N_STEPS}")
    lines.append(f"stuck delta thr:   < {STUCK_DELTA_THRESHOLD} (step6 delta) AND qhalt_fired > {EARLY_HALT_STEP}")
    lines.append(f"oscillating thr:   > {OSCILLATING_DELTA_THRESHOLD} (step8 delta)")
    lines.append(f"halted-early thr:  qhalt_fired_step <= {EARLY_HALT_STEP}")
    lines.append("")
    lines.append("Bucket counts")
    lines.append("-" * 78)
    lines.append(f"{'bucket':<18} {'count':>8} {'pct':>8}  {'mean d6':>10} {'mean q@6':>10}")
    for name in bucket_names:
        c = bucket_counts.get(name, 0)
        if c == 0:
            d6_mean = 0.0
            q6_mean = 0.0
        else:
            d6_mean = bucket_step6_delta_sum[name] / c
            q6_mean = bucket_qhalt_at6_sum[name] / c
        pct = (c / total_items * 100.0) if total_items else 0.0
        lines.append(f"{name:<18} {c:>8d} {pct:>7.2f}%  {d6_mean:>10.4f} {q6_mean:>10.4f}")
    lines.append("")

    incorrect_total = total_items - bucket_counts.get("Correct", 0)
    lines.append(f"Q-halt fired step histogram (incorrect predictions, n={incorrect_total})")
    lines.append("-" * 78)
    lines.append(f"{'fired_step':<12} {'count':>8} {'pct_of_wrong':>14}")
    for step in range(1, N_STEPS + 1):
        c = qhalt_fired_hist_incorrect.get(step, 0)
        pct = (c / incorrect_total * 100.0) if incorrect_total else 0.0
        label = f"{step}" + (" (never)" if step == N_STEPS else "")
        lines.append(f"{label:<12} {c:>8d} {pct:>13.2f}%")
    lines.append("")

    lines.append("Top 10 puzzles by incorrect-across-augmentations")
    lines.append("-" * 78)
    lines.append(f"{'puzzle':<48} {'wrong_aug_count':>18}")
    for puzzle_name, cnt in incorrect_per_puzzle.most_common(10):
        lines.append(f"{puzzle_name:<48} {cnt:>18d}")
    lines.append("=" * 78)

    report = "\n".join(lines)
    print()
    print(report)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report + "\n")
    print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
