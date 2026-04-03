from typing import Dict, Sequence, Optional
import os
import json
import copy

import torch
import torch.nn.functional as F
import numpy as np
from numba import njit
import torch.distributed as dist

from data.build_arc_dataset import inverse_aug, grid_hash, arc_grid_to_np
from data.common import PuzzleDatasetMetadata


@njit
def _crop(grid: np.ndarray):
    """Find maximum-sized rectangle without any EOS token inside. """
    side = int(np.sqrt(grid.shape[0]))
    grid = grid.reshape(side, side)

    max_area = 0
    max_size = (0, 0)
    nr, nc = grid.shape
    
    num_c = nc
    for num_r in range(1, nr + 1):
        # Scan for maximum c
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


class ARC:
    required_outputs = {"inputs", "puzzle_identifiers", "q_halt_logits", "preds", "current_energy"}
    
    def __init__(self, data_path: str, eval_metadata: PuzzleDatasetMetadata, submission_K: int = 2, pass_Ks: Sequence[int] = (1, 2, 5, 10, 100, 1000), aggregated_voting: bool = True):
        super().__init__()
        self.pass_Ks = pass_Ks
        self.submission_K = submission_K
        self.aggregated_voting = aggregated_voting
        self.blank_identifier_id = eval_metadata.blank_identifier_id

        # Majority vote evaluation settings
        self.maj_sample_sizes = (10, 100, 1000, 10000)

        # Load identifiers and test puzzles
        with open(os.path.join(data_path, "identifiers.json"), "r") as f:
            self.identifier_map = json.load(f)
        with open(os.path.join(data_path, "test_puzzles.json"), "r") as f:
            self.test_puzzles = json.load(f)
            
        # States
        self._local_hmap = {}
        self._local_preds = {}
        
    def begin_eval(self):
        if not self.aggregated_voting:
            # Clear previous predictions
            self._local_hmap = {}
            self._local_preds = {}
    
    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        # Collect required outputs to CPU with explicit cloning to avoid inference mode issues
        outputs = {}
        q_values = None
        q_log_probs = None
        energy_values = None

        with torch.no_grad():  # Explicit no_grad context
            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in self.required_outputs:
                        if k == "q_halt_logits":
                            v_cpu = v.detach().clone().cpu().to(torch.float64)
                            q_values = v_cpu.sigmoid()
                            q_log_probs = F.logsigmoid(v_cpu)
                        elif k == "current_energy":
                            energy_values = v.detach().clone().cpu().to(torch.float64)
                        else:
                            outputs[k] = v.detach().clone().cpu()

        assert q_values is not None and q_log_probs is not None

        # Remove padding from outputs
        mask = outputs["puzzle_identifiers"] != self.blank_identifier_id
        outputs = {k: v[mask].clone() for k, v in outputs.items()}
        q_values_masked = q_values[mask].clone()
        q_log_probs_masked = q_log_probs[mask].clone()
        energy_values_masked = energy_values[mask].clone() if energy_values is not None else None

        # Convert to numpy immediately (numpy arrays don't have inference mode issues)
        identifier_np = outputs["puzzle_identifiers"].numpy()
        inputs_np = outputs["inputs"].numpy()
        preds_np = outputs["preds"].numpy()
        q_values_np = q_values_masked.numpy()
        q_log_probs_np = q_log_probs_masked.numpy()
        # Lower energy = better prediction → use -energy as confidence
        energy_conf_np = -energy_values_masked.numpy() if energy_values_masked is not None else np.zeros_like(q_values_np)

        # Get predictions
        for identifier, input, pred, q, q_log_prob, energy_conf in zip(
            identifier_np, inputs_np, preds_np, q_values_np, q_log_probs_np, energy_conf_np
        ):
            name = self.identifier_map[identifier]
            orig_name, _inverse_fn = inverse_aug(name)

            input_hash = grid_hash(_inverse_fn(_crop(input)))

            pred = _inverse_fn(_crop(pred))
            assert np.all((pred >= 0) & (pred <= 9)), f"Puzzle {name}'s prediction out of 0-9 range."

            pred_hash = grid_hash(pred)

            self._local_hmap[pred_hash] = pred

            self._local_preds.setdefault(orig_name, {})
            self._local_preds[orig_name].setdefault(input_hash, [])
            self._local_preds[orig_name][input_hash].append((pred_hash, float(q), float(q_log_prob), float(energy_conf)))
    
    def result(self, save_path: Optional[str], rank: int, world_size: int, group: Optional[torch.distributed.ProcessGroup] = None) -> Optional[Dict[str, float]]:
        # Gather predictions to rank 0 for voting
        # global_hmap_preds = [None for _ in range(world_size)] if rank == 0 else None
        # dist.gather_object((self._local_hmap, self._local_preds), global_hmap_preds, dst=0, group=group)

        # if dist.is_initialized():
        #     global_hmap_preds = [None for _ in range(world_size)] if rank == 0 else None
        #     dist.gather_object((copy.deepcopy(self._local_hmap), copy.deepcopy(self._local_preds)), global_hmap_preds, dst=0, group=group)
        # else:
        # Single-GPU mode: use local data directly
        global_hmap_preds = [(self._local_hmap, self._local_preds)]

        # Rank 0 logic
        if rank != 0:
            return

        submission = {}
        correct = [0.0 for _ in range(len(self.pass_Ks))]
        energy_correct = [0.0 for _ in range(len(self.pass_Ks))]
        maj_correct = {size: 0.0 for size in self.maj_sample_sizes}

        for name, puzzle in self.test_puzzles.items():
            # Process test examples in this puzzle
            submission[name] = []
            num_test_correct = [0 for _ in range(len(self.pass_Ks))]
            num_test_correct_energy = [0 for _ in range(len(self.pass_Ks))]
            maj_test_correct = {size: 0 for size in self.maj_sample_sizes}
            for pair in puzzle["test"]:
                input_hash = grid_hash(arc_grid_to_np(pair["input"]))
                label_hash = grid_hash(arc_grid_to_np(pair["output"]))

                p_map = {}
                pred_samples = []
                for hmap, preds in global_hmap_preds:  # type: ignore
                    for h, q, q_log_prob, energy_conf in preds.get(name, {}).get(input_hash, {}):
                        # [count, sum_q, max_q_log_prob, sum_energy_conf]
                        p_map.setdefault(h, [0, 0.0, -np.inf, 0.0])
                        p_map[h][0] += 1
                        p_map[h][1] += q
                        p_map[h][2] = max(p_map[h][2], q_log_prob)
                        p_map[h][3] += energy_conf
                        pred_samples.append((h, q_log_prob))

                if not len(p_map):
                    print (f"Puzzle {name} has no predictions.")
                    continue

                for h, stats in p_map.items():
                    stats[1] /= stats[0]  # avg q
                    stats[3] /= stats[0]  # avg energy confidence

                # Q-based ranking (original)
                p_map_q = sorted(p_map.items(), key=lambda kv: kv[1], reverse=True)

                for i, k in enumerate(self.pass_Ks):
                    ok = False
                    for h, stats in p_map_q[:k]:
                        ok |= h == label_hash
                    num_test_correct[i] += ok

                # Energy-based ranking (sort by avg energy confidence, descending)
                p_map_energy = sorted(p_map.items(), key=lambda kv: kv[1][3], reverse=True)

                for i, k in enumerate(self.pass_Ks):
                    ok = False
                    for h, stats in p_map_energy[:k]:
                        ok |= h == label_hash
                    num_test_correct_energy[i] += ok

                # Query grids (use energy ranking for submission)
                pred_grids = []
                for h, stats in p_map_energy[:self.submission_K]:
                    for hmap, preds in global_hmap_preds:  # type: ignore
                        if h in hmap:
                            pred_grids.append(hmap[h])
                            break

                # Pad to K
                while len(pred_grids) < self.submission_K:
                    pred_grids.append(pred_grids[0])

                submission[name].append({f"attempt_{i + 1}": grid.tolist() for i, grid in enumerate(pred_grids)})

                # Majority voting metrics (best-of-N with log-probability ranking)
                if len(pred_samples):
                    logps = np.array([lp for _, lp in pred_samples], dtype=np.float64)
                    max_logp = logps.max()
                    probs = np.exp(logps - max_logp)
                    prob_sum = probs.sum()
                    if prob_sum > 0:
                        probs /= prob_sum
                    else:
                        probs = np.full_like(probs, 1.0 / len(probs))

                    for sample_size in self.maj_sample_sizes:
                        sampled_indices = np.random.choice(len(pred_samples), size=sample_size, replace=True, p=probs)
                        sampled_logps = logps[sampled_indices]
                        best_idx = sampled_indices[np.argmax(sampled_logps)]
                        maj_test_correct[sample_size] += pred_samples[best_idx][0] == label_hash

            # Total correctness
            for i in range(len(self.pass_Ks)):
                correct[i] += num_test_correct[i] / len(puzzle["test"])
                energy_correct[i] += num_test_correct_energy[i] / len(puzzle["test"])
            for sample_size in self.maj_sample_sizes:
                maj_correct[sample_size] += maj_test_correct[sample_size] / len(puzzle["test"])

        # Save submission
        if save_path is not None:
            with open(os.path.join(save_path, "submission.json"), "w") as f:
                json.dump(submission, f)

        # Final result
        result = {f"{self.__class__.__name__}/pass@{k}": correct[i] / len(self.test_puzzles) for i, k in enumerate(self.pass_Ks)}
        result.update({f"{self.__class__.__name__}/energy_pass@{k}": energy_correct[i] / len(self.test_puzzles) for i, k in enumerate(self.pass_Ks)})
        result.update({f"{self.__class__.__name__}/maj@{k}": maj_correct[k] / len(self.test_puzzles) for k in self.maj_sample_sizes})
        return result
