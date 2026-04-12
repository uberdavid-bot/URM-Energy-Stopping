"""Eval-only comparison of energy ranking strategies.

Loads an R2b checkpoint and compares four ranking strategies for pass@K:
  1. Q-halt (baseline)
  2. -E_final (current energy ranking)
  3. Energy drop (E_step1 - E_final)
  4. -E_step6 (best-step energy)

Usage:
    conda activate urm
    python scripts/eval_energy_ranking.py \
        --checkpoint checkpoints/R2b-traj-dropout-d1-h64-260412 \
        --data_path data/arc1concept-aug-1000-size-10
"""
import os
import sys
import json
import argparse
import copy

import torch
import torch.nn.functional as F
import numpy as np
from numba import njit

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pretrain import (
    load_config_from_checkpoint_path,
    create_dataloader,
    init_train_state,
    PretrainConfig,
)
from data.build_arc_dataset import inverse_aug, grid_hash, arc_grid_to_np
from data.common import PuzzleDatasetMetadata


@njit
def _crop(grid: np.ndarray):
    """Find maximum-sized rectangle without any EOS token inside."""
    side = int(np.sqrt(grid.shape[0]))
    grid = grid.reshape(side, side)
    max_area = 0
    max_size = (0, 0)
    nr, nc = grid.shape
    num_c = nc
    for num_r in range(1, nr + 1):
        for c in range(1, num_c + 1):
            x = grid[num_r - 1, c - 1]
            if (x < 2) | (x > 11):
                num_c = c - 1
                break
        area = num_r * num_c
        if area > max_area:
            max_area = area
            max_size = (num_r, num_c)
    return (grid[:max_size[0], :max_size[1]] - 2).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory")
    parser.add_argument("--data_path", required=True, help="Data directory")
    args = parser.parse_args()

    os.environ["DISABLE_COMPILE"] = "1"

    # Load config from checkpoint
    config = load_config_from_checkpoint_path(args.checkpoint)
    if config is None:
        # Fallback: try config.yaml directly (save_code_and_config uses this name)
        import yaml
        from pathlib import Path
        cfg_path = Path(args.checkpoint) / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path) as f:
                config = PretrainConfig(**yaml.safe_load(f))
        else:
            raise ValueError(f"Could not load config from {args.checkpoint}")

    # Resolve checkpoint to latest .pt file if directory given
    ckpt_path = args.checkpoint
    if os.path.isdir(ckpt_path):
        import re
        pts = [(int(m.group(1)), os.path.join(ckpt_path, f))
               for f in os.listdir(ckpt_path)
               for m in [re.match(r"step_(\d+)\.pt$", f)] if m]
        if not pts:
            raise FileNotFoundError(f"No step_*.pt files in {ckpt_path}")
        pts.sort()
        ckpt_path = pts[-1][1]
        print(f"Using checkpoint: {ckpt_path}")

    config.load_checkpoint = ckpt_path
    config.data_path = args.data_path
    config.load_strict = True
    config.load_optimizer_state = False

    # Create eval dataloader
    eval_loader, eval_metadata = create_dataloader(
        config, "test", test_set_mode=True, epochs_per_iter=1,
        global_batch_size=config.global_batch_size, rank=0, world_size=1,
    )

    # Create train state (loads checkpoint)
    train_state = init_train_state(config, eval_metadata, rank=0, world_size=1)
    train_state.model.eval()

    # Access inner ARCModel
    loss_head = train_state.model
    arc_model = loss_head.model
    P = arc_model.inner.puzzle_emb_len
    blank_id = eval_metadata.blank_identifier_id

    # Load puzzle metadata
    with open(os.path.join(args.data_path, "identifiers.json"), "r") as f:
        identifier_map = json.load(f)
    with open(os.path.join(args.data_path, "test_puzzles.json"), "r") as f:
        test_puzzles = json.load(f)

    # Accumulate predictions with multiple energy signals
    # Structure: {puzzle_name: {input_hash: [(pred_hash, q, q_log_prob, e_final, e_drop, e_best), ...]}}
    local_preds = {}
    local_hmap = {}

    print("Running eval inference...")
    batch_count = 0
    with torch.no_grad():
        for set_name, batch, global_batch_size in eval_loader:
            batch_count += 1
            batch = {k: v.cuda() for k, v in batch.items()}

            # Run full trajectory
            all_logits, all_q_logits, all_hidden, input_embeddings = arc_model.forward_trajectory(batch, N=8)

            # Energy at steps 1, 6, and 8
            energy_step1 = arc_model.compute_joint_energy(input_embeddings, all_hidden[0][:, P:])
            energy_step6 = arc_model.compute_joint_energy(input_embeddings, all_hidden[5][:, P:])
            energy_final = arc_model.compute_joint_energy(input_embeddings, all_hidden[-1][:, P:])

            # Confidence signals
            q_values = torch.sigmoid(all_q_logits[-1].to(torch.float64)).cpu()
            q_log_probs = F.logsigmoid(all_q_logits[-1].to(torch.float64)).cpu()
            e_final = (-energy_final).to(torch.float64).cpu()           # -E_final (higher = better)
            e_drop = (energy_step1 - energy_final).to(torch.float64).cpu()  # E1 - E8 (higher = bigger drop)
            e_best = (-energy_step6).to(torch.float64).cpu()            # -E_step6

            # Final predictions
            final_preds = torch.argmax(all_logits[-1], dim=-1).cpu()
            puzzle_ids = batch["puzzle_identifiers"].cpu()
            inputs = batch["inputs"].cpu()

            # Mask padding
            mask = puzzle_ids != blank_id
            puzzle_ids = puzzle_ids[mask]
            inputs = inputs[mask]
            final_preds = final_preds[mask]
            q_values = q_values[mask]
            q_log_probs = q_log_probs[mask]
            e_final = e_final[mask]
            e_drop = e_drop[mask]
            e_best = e_best[mask]

            # Store predictions
            for idx in range(len(puzzle_ids)):
                name = identifier_map[int(puzzle_ids[idx])]
                orig_name, _inverse_fn = inverse_aug(name)
                input_hash = grid_hash(_inverse_fn(_crop(inputs[idx].numpy())))
                pred = _inverse_fn(_crop(final_preds[idx].numpy()))
                if not np.all((pred >= 0) & (pred <= 9)):
                    continue
                pred_hash = grid_hash(pred)
                local_hmap[pred_hash] = pred

                local_preds.setdefault(orig_name, {})
                local_preds[orig_name].setdefault(input_hash, [])
                local_preds[orig_name][input_hash].append((
                    pred_hash,
                    float(q_values[idx]),
                    float(q_log_probs[idx]),
                    float(e_final[idx]),
                    float(e_drop[idx]),
                    float(e_best[idx]),
                ))

            if batch_count % 50 == 0:
                print(f"  Processed {batch_count} batches...")

    print(f"  Done. {batch_count} batches total.")

    # Compute pass@K for each ranking strategy
    pass_Ks = [1, 2, 5, 10, 100, 1000]
    ranking_names = ["Q-halt", "-E_final (current)", "Energy drop (E1-E8)", "-E_step6 (best step)"]
    # signal_indices: which tuple index to use for each ranking strategy
    # tuple: (pred_hash, q, q_log_prob, e_final, e_drop, e_best)
    #         0         1   2           3        4       5

    results = {name: [0.0] * len(pass_Ks) for name in ranking_names}

    for puzzle_name, puzzle in test_puzzles.items():
        per_test = {name: [0] * len(pass_Ks) for name in ranking_names}

        for pair in puzzle["test"]:
            input_hash = grid_hash(arc_grid_to_np(pair["input"]))
            label_hash = grid_hash(arc_grid_to_np(pair["output"]))

            preds_list = local_preds.get(puzzle_name, {}).get(input_hash, [])
            if not preds_list:
                continue

            # Build p_map: {pred_hash: [count, sum_q, sum_e_final, sum_e_drop, sum_e_best]}
            p_map = {}
            for pred_hash, q, q_lp, ef, ed, eb in preds_list:
                p_map.setdefault(pred_hash, [0, 0.0, 0.0, 0.0, 0.0])
                p_map[pred_hash][0] += 1
                p_map[pred_hash][1] += q
                p_map[pred_hash][2] += ef
                p_map[pred_hash][3] += ed
                p_map[pred_hash][4] += eb

            # Average
            for h, stats in p_map.items():
                cnt = stats[0]
                stats[1] /= cnt  # avg q
                stats[2] /= cnt  # avg e_final
                stats[3] /= cnt  # avg e_drop
                stats[4] /= cnt  # avg e_best

            # Rank by each signal (descending — higher is better for all signals)
            # Q-halt: sort by [count, avg_q] (same as original evaluator)
            ranked_q = sorted(p_map.items(), key=lambda kv: (kv[1][0], kv[1][1]), reverse=True)
            ranked_ef = sorted(p_map.items(), key=lambda kv: kv[1][2], reverse=True)
            ranked_ed = sorted(p_map.items(), key=lambda kv: kv[1][3], reverse=True)
            ranked_eb = sorted(p_map.items(), key=lambda kv: kv[1][4], reverse=True)

            rankings = [ranked_q, ranked_ef, ranked_ed, ranked_eb]
            for ri, (rname, ranked) in enumerate(zip(ranking_names, rankings)):
                for ki, k in enumerate(pass_Ks):
                    found = any(h == label_hash for h, _ in ranked[:k])
                    per_test[rname][ki] += found

        for rname in ranking_names:
            for ki in range(len(pass_Ks)):
                results[rname][ki] += per_test[rname][ki] / len(puzzle["test"])

    # Normalize by number of puzzles
    n_puzzles = len(test_puzzles)
    for rname in ranking_names:
        for ki in range(len(pass_Ks)):
            results[rname][ki] /= n_puzzles

    # Print comparison table
    print("\n" + "=" * 80)
    print("Energy Ranking Strategy Comparison (R2b checkpoint)")
    print("=" * 80)
    header = f"{'Ranking Strategy':<25}"
    for k in pass_Ks:
        header += f" | pass@{k:<5}"
    print(header)
    print("-" * len(header))
    for rname in ranking_names:
        row = f"{rname:<25}"
        for ki in range(len(pass_Ks)):
            row += f" | {results[rname][ki]:>7.1%}"
        print(row)
    print("=" * 80)


if __name__ == "__main__":
    main()
