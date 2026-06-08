"""R7e parallel-decode eval: three-arm comparison of diversity sources.

Tests whether terminal-perturbation diversity (GRAM prior samples) is
orthogonal to augmentation diversity (input transforms) at matched total
decode budget N.

Arms:
  A — augmentation-only: N augmentations × 1 deterministic decode (eps=0)
  B — perturb-only:      1 augmentation  × N terminal prior samples
  C — cross:             M augmentations × K terminal samples (M·K = N)

All arms use the same inverse-augment → hash → dedup → Q-based pass@K
ranking code so metrics are comparable.

Usage:
    conda activate urm
    python scripts/eval_parallel_decode.py \
        --checkpoint checkpoints/R7e-gram-terminal-h128-260607 \
        --data_path data/arc1concept-aug-1000-size-10
"""
import os
import sys
import json
import time
import argparse
import math

import torch
import torch.nn.functional as F
import numpy as np
from numba import njit

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


def compute_pass_at_k(p_map, label_hash, K):
    """Q-based pass@K: sort unique grids by (count, avg_q, max_q_log_prob), check top K."""
    ranked = sorted(p_map.items(), key=lambda kv: kv[1], reverse=True)
    for h, _ in ranked[:K]:
        if h == label_hash:
            return True
    return False


def compute_metrics_for_arm(predictions, test_puzzles, hmap, pass_ks):
    """Compute pass@K and distinct-grid count from collected predictions.

    predictions: {puzzle_name: {input_hash: [(pred_hash, q, q_log_prob, energy_conf), ...]}}
    Returns: {pass@K: float, ...}, mean_distinct
    """
    correct = {k: 0.0 for k in pass_ks}
    total_distinct = 0.0
    num_puzzle_tests = 0

    for name, puzzle in test_puzzles.items():
        for pair in puzzle["test"]:
            input_hash = grid_hash(arc_grid_to_np(pair["input"]))
            label_hash = grid_hash(arc_grid_to_np(pair["output"]))

            raw = predictions.get(name, {}).get(input_hash, [])
            if not raw:
                continue

            # Dedup into p_map: {pred_hash: [count, avg_q, max_q_log_prob, avg_energy_conf]}
            p_map = {}
            for pred_hash, q, q_log_prob, energy_conf in raw:
                p_map.setdefault(pred_hash, [0, 0.0, -np.inf, 0.0])
                p_map[pred_hash][0] += 1
                p_map[pred_hash][1] += q
                p_map[pred_hash][2] = max(p_map[pred_hash][2], q_log_prob)
                p_map[pred_hash][3] += energy_conf

            for h, stats in p_map.items():
                stats[1] /= stats[0]
                stats[3] /= stats[0]

            total_distinct += len(p_map)
            num_puzzle_tests += 1

            for k in pass_ks:
                if compute_pass_at_k(p_map, label_hash, k):
                    correct[k] += 1.0 / len(puzzle["test"])

    num_puzzles = len(test_puzzles)
    result = {k: correct[k] / num_puzzles for k in pass_ks}
    mean_distinct = total_distinct / max(num_puzzle_tests, 1)
    return result, mean_distinct


def collect_predictions_from_samples(
    preds_np, q_values_np, q_log_probs_np, energy_conf_np,
    puzzle_ids_np, inputs_np, identifier_map, blank_id,
):
    """Convert raw model outputs into {puzzle_name: {input_hash: [(pred_hash, q, q_log_prob, e_conf), ...]}}."""
    local_preds = {}
    local_hmap = {}

    for i in range(len(puzzle_ids_np)):
        pid = puzzle_ids_np[i]
        if pid == blank_id:
            continue

        name = identifier_map[pid]
        orig_name, _inverse_fn = inverse_aug(name)

        input_hash = grid_hash(_inverse_fn(_crop(inputs_np[i])))
        pred = _inverse_fn(_crop(preds_np[i]))
        pred_hash = grid_hash(pred)
        local_hmap[pred_hash] = pred

        local_preds.setdefault(orig_name, {})
        local_preds[orig_name].setdefault(input_hash, [])
        local_preds[orig_name][input_hash].append(
            (pred_hash, float(q_values_np[i]), float(q_log_probs_np[i]), float(energy_conf_np[i]))
        )

    return local_preds, local_hmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[5, 10, 100, 1000])
    args = parser.parse_args()

    os.environ["DISABLE_COMPILE"] = "1"

    config = load_config_from_checkpoint_path(args.checkpoint)
    if config is None:
        import yaml
        from pathlib import Path
        cfg_path = Path(args.checkpoint) / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path) as f:
                config = PretrainConfig(**yaml.safe_load(f))
        else:
            raise ValueError(f"Could not load config from {args.checkpoint}")

    # Resolve checkpoint
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
    with open(os.path.join(args.data_path, "test_puzzles.json"), "r") as f:
        test_puzzles = json.load(f)

    max_budget = max(args.budgets)
    # ARM C factorizations: balanced M×K=N
    arm_c_factors = {}
    for N in args.budgets:
        M = int(math.isqrt(N))
        while N % M != 0:
            M -= 1
        K = N // M
        arm_c_factors[N] = (M, K)

    max_augs_needed = max(max_budget, max(m for m, _ in arm_c_factors.values()))
    max_arm_c_K = max(k for _, k in arm_c_factors.values())
    # ARM B needs max_budget samples for first augmentation only;
    # ARM C needs max_arm_c_K samples for each of M augmentations.
    # Per-batch: aug_idx==0 draws max_budget, aug_idx>0 draws max_arm_c_K.
    terminal_samples_first_aug = max_budget
    terminal_samples_other_aug = max_arm_c_K

    print(f"Budgets: {args.budgets}")
    print(f"ARM C factorizations: {arm_c_factors}")
    print(f"Max augmentations needed per puzzle: {max_augs_needed}")
    print(f"Terminal samples: {terminal_samples_first_aug} (first aug), {terminal_samples_other_aug} (other augs)")

    # ========================================================================
    # Phase 1: Single pass — collect deterministic + terminal samples per aug
    # ========================================================================
    # For each augmentation, store:
    #   - deterministic prediction (for ARM A)
    #   - up to max_terminal_samples stochastic predictions (for ARM B/C)
    # Indexed by (orig_puzzle_name, input_hash, aug_index).

    # Per-puzzle augmentation counter
    aug_counter = {}

    # ARM A predictions: {puzzle: {input_hash: [(pred_hash, q, qlp, ec), ...]}}
    arm_a_all = {}
    arm_a_hmap = {}

    # ARM B predictions: same structure, but from terminal samples of first aug only
    arm_b_all = {}
    arm_b_hmap = {}

    # ARM C: store per-aug terminal samples. {puzzle: {input_hash: {aug_idx: [(pred_hash, q, qlp, ec), ...]}}}
    arm_c_raw = {}

    print("\nPhase 1: Processing eval batches...")
    t0 = time.time()
    batch_count = 0
    total_forward_passes = 0
    total_terminal_draws = 0

    with torch.no_grad():
        for set_name, batch, global_batch_size in eval_loader:
            batch_count += 1
            batch = {k: v.cuda() for k, v in batch.items()}

            puzzle_ids = batch["puzzle_identifiers"].cpu().numpy()
            inputs_np = batch["inputs"].cpu().numpy()
            B = len(puzzle_ids)

            # Determine per-example aug index and which examples need processing
            per_example_aug_idx = []
            per_example_orig_name = []
            per_example_input_hash = []
            needs_processing = []

            for i in range(B):
                pid = puzzle_ids[i]
                if pid == blank_id:
                    per_example_aug_idx.append(-1)
                    per_example_orig_name.append(None)
                    per_example_input_hash.append(None)
                    needs_processing.append(False)
                    continue

                name = identifier_map[pid]
                orig_name, _inv_fn = inverse_aug(name)
                ih = grid_hash(_inv_fn(_crop(inputs_np[i])))

                key = (orig_name, ih)
                idx = aug_counter.get(key, 0)
                aug_counter[key] = idx + 1

                per_example_aug_idx.append(idx)
                per_example_orig_name.append(orig_name)
                per_example_input_hash.append(ih)
                needs_processing.append(idx < max_augs_needed)

            if not any(needs_processing):
                continue

            # Run deterministic trajectory once, get u_{N-1}
            u_N, _ = arc_model._run_deterministic_trajectory(batch)
            total_forward_passes += 1
            P = arc_model.inner.puzzle_emb_len
            R = arc_model.config.num_registers

            # Deterministic decode (ARM A: eps=0)
            det_logits = arc_model.inner.lm_head(u_N)[:, P + R:]
            det_preds = torch.argmax(det_logits, dim=-1).cpu().numpy()
            det_q_raw = arc_model.inner.q_head(u_N[:, 0]).to(torch.float32).squeeze(-1)
            det_q_values = torch.sigmoid(det_q_raw.to(torch.float64)).cpu().numpy()
            det_q_log_probs = F.logsigmoid(det_q_raw.to(torch.float64)).cpu().numpy()
            det_energy_conf = np.zeros(B, dtype=np.float64)

            # Helper: draw terminal eps samples from cached prior params
            def _draw_terminal_samples(u_N_tensor, n_samples, P, R):
                """Draw n_samples from prior on cached u_N. Returns lists of numpy arrays."""
                preds_list, qv_list, qlp_list = [], [], []
                if n_samples == 0 or not arc_model.config.gram_enabled:
                    return preds_list, qv_list, qlp_list
                pooled = u_N_tensor.mean(dim=1)
                mu_p, logvar_p = arc_model._gram_split(arc_model.gram_prior_mlp(pooled))
                for _ in range(n_samples):
                    z = arc_model._gram_reparam(mu_p, logvar_p)
                    eps_t = arc_model.gram_up_proj(z).unsqueeze(1).expand_as(u_N_tensor)
                    h_final = u_N_tensor + eps_t
                    s_logits = arc_model.inner.lm_head(h_final)[:, P + R:]
                    s_preds = torch.argmax(s_logits, dim=-1).cpu().numpy()
                    s_q_raw = arc_model.inner.q_head(h_final[:, 0]).to(torch.float32).squeeze(-1)
                    preds_list.append(s_preds)
                    qv_list.append(torch.sigmoid(s_q_raw.to(torch.float64)).cpu().numpy())
                    qlp_list.append(F.logsigmoid(s_q_raw.to(torch.float64)).cpu().numpy())
                return preds_list, qv_list, qlp_list

            # Determine how many terminal samples this batch needs:
            # aug_idx==0 examples need terminal_samples_first_aug (for ARM B)
            # aug_idx>0 examples need terminal_samples_other_aug (for ARM C)
            has_first_aug = any(per_example_aug_idx[i] == 0 and needs_processing[i] for i in range(B))
            n_draws_base = terminal_samples_other_aug
            n_draws_extra = (terminal_samples_first_aug - n_draws_base) if has_first_aug else 0

            # Draw base samples (for all examples — ARM C)
            base_preds, base_qv, base_qlp = _draw_terminal_samples(u_N, n_draws_base, P, R)
            total_terminal_draws += n_draws_base

            # Draw extra samples for first-aug examples only (ARM B)
            extra_preds, extra_qv, extra_qlp = [], [], []
            if n_draws_extra > 0:
                first_aug_mask = torch.tensor(
                    [per_example_aug_idx[i] == 0 and needs_processing[i] for i in range(B)],
                    device=u_N.device
                )
                if first_aug_mask.any():
                    u_N_first = u_N[first_aug_mask]
                    ep, eqv, eqlp = _draw_terminal_samples(u_N_first, n_draws_extra, P, R)
                    extra_preds, extra_qv, extra_qlp = ep, eqv, eqlp
                    total_terminal_draws += n_draws_extra

            # Build mapping from batch index to extra_preds index for first-aug examples
            first_aug_extra_idx = {}
            fa_count = 0
            for i in range(B):
                if per_example_aug_idx[i] == 0 and needs_processing[i]:
                    first_aug_extra_idx[i] = fa_count
                    fa_count += 1

            # Distribute predictions into arm structures
            for i in range(B):
                if not needs_processing[i]:
                    continue

                orig_name = per_example_orig_name[i]
                ih = per_example_input_hash[i]
                aug_idx = per_example_aug_idx[i]
                name = identifier_map[puzzle_ids[i]]
                _, _inv_fn = inverse_aug(name)

                # Deterministic prediction (ARM A)
                pred = _inv_fn(_crop(det_preds[i]))
                ph = grid_hash(pred)
                arm_a_hmap[ph] = pred
                arm_a_all.setdefault(orig_name, {})
                arm_a_all[orig_name].setdefault(ih, [])
                arm_a_all[orig_name][ih].append(
                    (ph, float(det_q_values[i]), float(det_q_log_probs[i]), float(det_energy_conf[i]))
                )

                # Terminal samples for ARM B: only first augmentation
                if aug_idx == 0 and arc_model.config.gram_enabled:
                    arm_b_all.setdefault(orig_name, {})
                    arm_b_all[orig_name].setdefault(ih, [])
                    # Base draws (indexed by batch position i)
                    for s_idx in range(len(base_preds)):
                        s_pred = _inv_fn(_crop(base_preds[s_idx][i]))
                        s_ph = grid_hash(s_pred)
                        arm_b_hmap[s_ph] = s_pred
                        arm_b_all[orig_name][ih].append(
                            (s_ph, float(base_qv[s_idx][i]), float(base_qlp[s_idx][i]), 0.0)
                        )
                    # Extra draws (indexed by position within first-aug subset)
                    fa_idx = first_aug_extra_idx[i]
                    for s_idx in range(len(extra_preds)):
                        s_pred = _inv_fn(_crop(extra_preds[s_idx][fa_idx]))
                        s_ph = grid_hash(s_pred)
                        arm_b_hmap[s_ph] = s_pred
                        arm_b_all[orig_name][ih].append(
                            (s_ph, float(extra_qv[s_idx][fa_idx]), float(extra_qlp[s_idx][fa_idx]), 0.0)
                        )

                # Terminal samples for ARM C: store per aug_idx (base draws only)
                if base_preds:
                    arm_c_raw.setdefault(orig_name, {})
                    arm_c_raw[orig_name].setdefault(ih, {})
                    arm_c_raw[orig_name][ih].setdefault(aug_idx, [])
                    for s_idx in range(len(base_preds)):
                        s_pred = _inv_fn(_crop(base_preds[s_idx][i]))
                        s_ph = grid_hash(s_pred)
                        arm_c_raw[orig_name][ih][aug_idx].append(
                            (s_ph, float(base_qv[s_idx][i]), float(base_qlp[s_idx][i]), 0.0)
                        )

            if batch_count % 100 == 0:
                print(f"  Batch {batch_count}: {total_forward_passes} forward passes, "
                      f"{total_terminal_draws} terminal draws")

    phase1_time = time.time() - t0
    print(f"Phase 1 complete: {batch_count} batches, {total_forward_passes} forward passes, "
          f"{total_terminal_draws} terminal draws, {phase1_time:.1f}s")

    # ========================================================================
    # Phase 2: Compute metrics per arm per budget
    # ========================================================================
    print("\n" + "=" * 80)
    print("PARALLEL DECODE EVAL — THREE-ARM COMPARISON")
    print("=" * 80)

    results_table = []

    for N in args.budgets:
        M_c, K_c = arm_c_factors[N]
        pass_ks = [1, N]

        # ARM A: first N augmentations, deterministic decode
        arm_a_subset = {}
        for pname, inputs in arm_a_all.items():
            for ih, preds in inputs.items():
                arm_a_subset.setdefault(pname, {})
                arm_a_subset[pname][ih] = preds[:N]

        a_metrics, a_distinct = compute_metrics_for_arm(
            arm_a_subset, test_puzzles, arm_a_hmap, pass_ks
        )

        # ARM B: 1 augmentation, first N terminal samples
        arm_b_subset = {}
        for pname, inputs in arm_b_all.items():
            for ih, preds in inputs.items():
                arm_b_subset.setdefault(pname, {})
                arm_b_subset[pname][ih] = preds[:N]

        b_metrics, b_distinct = compute_metrics_for_arm(
            arm_b_subset, test_puzzles, arm_b_hmap, pass_ks
        )

        # ARM C: M_c augmentations × K_c terminal samples each
        arm_c_subset = {}
        arm_c_hmap = {}
        for pname, inputs in arm_c_raw.items():
            for ih, aug_dict in inputs.items():
                arm_c_subset.setdefault(pname, {})
                arm_c_subset[pname].setdefault(ih, [])
                for aug_idx in range(M_c):
                    if aug_idx in aug_dict:
                        arm_c_subset[pname][ih].extend(aug_dict[aug_idx][:K_c])

        c_metrics, c_distinct = compute_metrics_for_arm(
            arm_c_subset, test_puzzles, arm_c_hmap, pass_ks
        )

        results_table.append({
            "N": N,
            "arm_a": {"pass@1": a_metrics[1], f"pass@{N}": a_metrics[N], "distinct": a_distinct},
            "arm_b": {"pass@1": b_metrics[1], f"pass@{N}": b_metrics[N], "distinct": b_distinct},
            "arm_c": {"pass@1": c_metrics[1], f"pass@{N}": c_metrics[N], "distinct": c_distinct,
                       "M": M_c, "K": K_c},
        })

    # ========================================================================
    # Phase 3: Print results table
    # ========================================================================
    print(f"\nCheckpoint: {ckpt_path}")
    print(f"GRAM enabled: {arc_model.config.gram_enabled}, latent_dim: {arc_model.config.gram_latent_dim}")
    print(f"Compute: {total_forward_passes} backbone fwd passes, {total_terminal_draws} terminal eps draws")
    print(f"Wall-clock: {phase1_time:.1f}s total\n")

    header = f"{'N':>6} | {'Arm':>5} | {'pass@1':>8} | {'pass@N':>8} | {'distinct':>8} | {'config':>16}"
    print(header)
    print("-" * len(header))

    for row in results_table:
        N = row["N"]
        for arm_name, arm_key in [("A-aug", "arm_a"), ("B-pert", "arm_b"), ("C-cross", "arm_c")]:
            d = row[arm_key]
            pass_1 = d["pass@1"]
            pass_n = d[f"pass@{N}"]
            distinct = d["distinct"]
            if arm_key == "arm_a":
                cfg = f"{N} augs × 1 det"
            elif arm_key == "arm_b":
                cfg = f"1 aug × {N} eps"
            else:
                cfg = f"{d['M']} augs × {d['K']} eps"
            print(f"{N:>6} | {arm_name:>5} | {pass_1:>7.2%} | {pass_n:>7.2%} | {distinct:>8.1f} | {cfg:>16}")
        print("-" * len(header))

    # ========================================================================
    # Phase 4: Orthogonality verdict
    # ========================================================================
    print("\n" + "=" * 80)
    print("ORTHOGONALITY VERDICT")
    print("=" * 80)

    for row in results_table:
        N = row["N"]
        a_passN = row["arm_a"][f"pass@{N}"]
        b_passN = row["arm_b"][f"pass@{N}"]
        c_passN = row["arm_c"][f"pass@{N}"]
        b_dist = row["arm_b"]["distinct"]
        best_ab = max(a_passN, b_passN)

        print(f"\nN={N}:")
        print(f"  ARM A (aug-only)   pass@{N} = {a_passN:.2%}")
        print(f"  ARM B (perturb)    pass@{N} = {b_passN:.2%}, distinct = {b_dist:.1f}")
        print(f"  ARM C (cross)      pass@{N} = {c_passN:.2%} ({row['arm_c']['M']}×{row['arm_c']['K']})")
        print(f"  Best of A,B = {best_ab:.2%}")

        if c_passN > best_ab + 0.005:
            print(f"  → ORTHOGONAL: C beats best(A,B) by {c_passN - best_ab:+.2%}")
            print(f"    Diversity sources compound — publishable finding.")
        elif c_passN >= best_ab - 0.005:
            print(f"  → NEUTRAL: C ≈ best(A,B) (delta {c_passN - best_ab:+.2%})")
            if b_dist < 2.0:
                print(f"    Terminal samples collapsed (distinct={b_dist:.1f}) — σ too small.")
            else:
                print(f"    Perturbation adds diversity but it's redundant with augmentation.")
        else:
            print(f"  → REDUNDANT: C < best(A,B) by {c_passN - best_ab:+.2%}")
            if b_passN < a_passN * 0.5:
                print(f"    Perturbation diversity far below augmentation — samples incoherent.")

        if b_dist < 1.5:
            print(f"  ARM B distinctness ~1: terminal samples collapsed to single grid")
            print(f"    (attractor pulls back / σ too small at convergence)")
        elif b_dist < 5.0:
            print(f"  ARM B distinctness {b_dist:.1f}: modest terminal diversity")
        else:
            print(f"  ARM B distinctness {b_dist:.1f}: substantial terminal diversity")


if __name__ == "__main__":
    main()
