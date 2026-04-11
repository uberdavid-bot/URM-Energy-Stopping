from typing import Any, Tuple, Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn


IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    valid_mask = labels != ignore_index
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


class EnergyLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        return_keys: Sequence[str] = (),
        return_raw_outputs: bool = False,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        """Run full trajectory with deep supervision.

        Training and eval use the same path. Returns (None, total_loss, metrics,
        returned_outputs, True) — carry placeholder is None, all_finished is always True.
        """
        N = self.model.config.loops
        all_logits, all_q_logits, all_hidden, input_embeddings = self.model.forward_trajectory(batch, N)
        P = self.model.inner.puzzle_emb_len

        labels = batch["labels"]
        mask = labels != IGNORE_LABEL_ID
        loss_counts = mask.sum(-1)
        loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)
        valid = loss_counts > 0
        B = labels.shape[0]
        device = labels.device

        # --- Deep supervision: weighted reconstruction + Q-halt loss at every step ---
        total_recon_loss = torch.tensor(0.0, device=device)
        total_qhalt_loss = torch.tensor(0.0, device=device)
        weight_sum = 0.0

        for t in range(N):
            w = (t + 1) / N
            step_loss = (self.loss_fn(all_logits[t], labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum()
            total_recon_loss = total_recon_loss + w * step_loss

            # Q-halt: predict whether step t's output is correct
            with torch.no_grad():
                step_preds = torch.argmax(all_logits[t].detach(), dim=-1)
                step_correct = (mask & (step_preds == labels)).sum(-1) == loss_counts

            qhalt_loss = F.binary_cross_entropy_with_logits(
                all_q_logits[t][..., 0],
                step_correct.to(all_q_logits[t].dtype),
                reduction="sum",
            )
            total_qhalt_loss = total_qhalt_loss + w * qhalt_loss
            weight_sum += w

        lm_loss = total_recon_loss / weight_sum
        qhalt_loss_avg = total_qhalt_loss / weight_sum
        total_loss = lm_loss + 0.5 * qhalt_loss_avg

        # --- Metrics (no_grad) ---
        with torch.no_grad():
            # Final-step aggregate metrics
            final_logits = all_logits[-1]
            final_preds = torch.argmax(final_logits, dim=-1)
            final_correct = mask & (final_preds == labels)
            final_seq_correct = final_correct.sum(-1) == loss_counts

            metrics = {
                "count": valid.sum(),
                "accuracy": torch.where(valid, (final_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid & final_seq_correct).sum(),
                "reconstruction_loss": lm_loss.detach(),
                "q_halt_loss": qhalt_loss_avg.detach(),
            }

            # Q-halt accuracy at final step
            final_q_pred = all_q_logits[-1][..., 0] >= 0
            metrics["q_halt_accuracy"] = (valid & (final_q_pred == final_seq_correct)).sum()

            # Energy at final step
            final_energy = self.model.compute_joint_energy(
                input_embeddings, all_hidden[-1][:, P:].detach()
            )
            metrics["current_energy"] = final_energy.mean().detach()

            # --- Per-step metrics (all steps) ---
            for t in range(N):
                step_preds_t = torch.argmax(all_logits[t], dim=-1)
                step_tok_correct = mask & (step_preds_t == labels)
                step_seq_correct = step_tok_correct.sum(-1) == loss_counts

                metrics[f"step_{t+1}_accuracy"] = torch.where(
                    valid, (step_tok_correct.to(torch.float32) / loss_divisor).sum(-1), 0
                ).sum()
                metrics[f"step_{t+1}_exact_accuracy"] = (valid & step_seq_correct).sum()

                if t > 0:
                    delta = (all_hidden[t] - all_hidden[t - 1]).norm(dim=-1).mean()
                    metrics[f"step_{t+1}_delta_norm"] = delta

            # --- Eval-only stopping metrics ---
            if not self.training:
                # Q-halt stopping: first step where q_logits >= 0
                # shape: [B, N]
                q_preds = torch.stack([q[..., 0] for q in all_q_logits], dim=1)  # [B, N]
                q_halt_mask = q_preds >= 0  # [B, N]

                # Per-sample exact accuracy at each step: [B, N]
                step_exact = torch.stack([
                    (mask & (torch.argmax(all_logits[t], dim=-1) == labels)).sum(-1) == loss_counts
                    for t in range(N)
                ], dim=1)  # [B, N]

                # Q-halt stop step (1-indexed, default N if never halts)
                q_any_halt = q_halt_mask.any(dim=1)
                q_first_halt = torch.where(
                    q_any_halt,
                    q_halt_mask.to(torch.float32).argmax(dim=1) + 1,
                    torch.full((B,), N, device=device, dtype=torch.float32),
                )
                metrics["qhalt_stop_step"] = torch.where(valid, q_first_halt, 0).sum()

                # Q-halt stop accuracy: accuracy at each sample's halt step
                q_stop_idx = (q_first_halt - 1).long().clamp(0, N - 1)
                q_stop_correct = step_exact.gather(1, q_stop_idx.unsqueeze(1)).squeeze(1)
                metrics["qhalt_stop_accuracy"] = (valid & q_stop_correct).sum()

                # Energy stopping: |E_t - E_{t-1}| < threshold
                energy_threshold = self.model.config.energy_threshold
                step_energies = torch.stack([
                    self.model.compute_joint_energy(
                        input_embeddings, all_hidden[t][:, P:].detach()
                    )
                    for t in range(N)
                ], dim=1)  # [B, N]

                energy_converged = torch.zeros(B, N, dtype=torch.bool, device=device)
                for t in range(1, N):
                    energy_converged[:, t] = torch.abs(step_energies[:, t] - step_energies[:, t - 1]) < energy_threshold

                e_any_conv = energy_converged.any(dim=1)
                e_first_conv = torch.where(
                    e_any_conv,
                    energy_converged.to(torch.float32).argmax(dim=1) + 1,
                    torch.full((B,), N, device=device, dtype=torch.float32),
                )
                metrics["energy_stop_step"] = torch.where(valid, e_first_conv, 0).sum()

                e_stop_idx = (e_first_conv - 1).long().clamp(0, N - 1)
                e_stop_correct = step_exact.gather(1, e_stop_idx.unsqueeze(1)).squeeze(1)
                metrics["energy_stop_accuracy"] = (valid & e_stop_correct).sum()

        # --- Return outputs ---
        returned_outputs: Dict[str, torch.Tensor] = {}
        if return_raw_outputs:
            returned_outputs["raw_outputs"] = {
                "all_logits": all_logits,
                "all_q_logits": all_q_logits,
                "all_hidden": all_hidden,
            }

        # For evaluator compatibility: final step predictions and signals
        final_outputs = {
            "logits": final_logits.detach(),
            "preds": final_preds.detach(),
            "q_halt_logits": all_q_logits[-1][..., 0].detach(),
            "current_energy": final_energy.detach(),
        }
        for k in return_keys:
            if k in final_outputs:
                returned_outputs[k] = final_outputs[k]

        return (
            None,  # carry placeholder (no longer used)
            total_loss,
            metrics,
            returned_outputs,
            torch.tensor(True, device=device),  # all_finished always True
        )