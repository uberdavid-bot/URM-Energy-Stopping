from typing import Any, Tuple, Dict, Set, Optional, Sequence

import torch
import torch.nn.functional as F
import torch._dynamo as dynamo
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


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Set[str],
        # Model args
        return_raw_outputs: bool = False,
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        profile = outputs.get("profile")
        labels = new_carry.current_data["labels"]

        # Correctness
        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (outputs["preds"] == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts

            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),

                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses
        # FIXME: Assuming the batch is always full
        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")

        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })

        # Q continue (bootstrapping target loss)
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()

        aux_loss = outputs.get("moe_aux_loss")
        if aux_loss is not None:
            metrics["moe_aux_loss"] = aux_loss.detach()

        router_metrics = outputs.get("router_metrics")
        if router_metrics is not None:
            for k, v in router_metrics.items():
                metrics[f"router/{k}"] = v.detach()

        if profile is not None:
            for name, duration in profile.items():
                metrics[f"profile/{name}"] = torch.tensor(duration, device=labels.device)

        # Filter outputs for return
        returned_outputs: Dict[str, torch.Tensor] = {}
        if return_raw_outputs:
            returned_outputs["raw_outputs"] = outputs

        for k in return_keys:
            if k in outputs:
                returned_outputs[k] = outputs[k].detach()

        total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)
        if aux_loss is not None:
            total_loss = total_loss + aux_loss

        return (
            new_carry,
            total_loss,
            metrics,
            returned_outputs,
            new_carry.halted.all(),
        )


class EnergyLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        return_raw_outputs: bool = False,
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        new_carry, outputs = self.model(**model_kwargs)
        profile = outputs.get("profile")
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)

            is_correct = mask & (outputs["preds"] == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts

            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # MCMC improvement metrics (compare refined vs unrefined predictions)
        with torch.no_grad():
            if "unrefined_logits" in outputs:
                unrefined_preds = torch.argmax(outputs["unrefined_logits"], dim=-1)
                refined_preds = outputs["preds"]
                unrefined_correct = (mask & (unrefined_preds == labels)).sum(-1).float() / loss_divisor.squeeze(-1)
                refined_correct = (mask & (refined_preds == labels)).sum(-1).float() / loss_divisor.squeeze(-1)
                metrics["mcmc_improvement"] = (refined_correct - unrefined_correct).mean()
                metrics["unrefined_accuracy"] = torch.where(valid_metrics, unrefined_correct, 0).sum()

        # Reconstruction loss — dual loss when MCMC active
        # Unrefined loss: clean URM learning signal (always computed)
        # Refined loss: trains energy head through MCMC (only when MCMC active)
        unrefined_logits = outputs.get("unrefined_logits", outputs["logits"])
        unrefined_lm_loss = (self.loss_fn(unrefined_logits, labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum()

        if "unrefined_logits" in outputs:
            refined_lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum()
            lm_loss = 0.5 * unrefined_lm_loss + 0.5 * refined_lm_loss
            metrics["unrefined_lm_loss"] = unrefined_lm_loss.detach()
            metrics["refined_lm_loss"] = refined_lm_loss.detach()
        else:
            lm_loss = unrefined_lm_loss

        metrics.update({
            "reconstruction_loss": lm_loss.detach(),
            "current_energy": outputs["current_energy"].mean().detach(),
        })

        # Per-step accuracy and delta norms (eval only, for convergence analysis)
        with torch.no_grad():
            if "per_step_logits" in outputs:
                for step_idx, step_logits in enumerate(outputs["per_step_logits"]):
                    step_preds = torch.argmax(step_logits, dim=-1)
                    step_correct = mask & (step_preds == labels)
                    step_acc = torch.where(
                        valid_metrics,
                        (step_correct.to(torch.float32) / loss_divisor).sum(-1),
                        0,
                    ).sum()
                    metrics[f"step_{step_idx + 1}_accuracy"] = step_acc
            if "per_step_delta_norms" in outputs:
                for step_idx, delta_norm in enumerate(outputs["per_step_delta_norms"]):
                    metrics[f"step_{step_idx + 1}_delta_norm"] = delta_norm

        if profile is not None:
            for name, duration in profile.items():
                metrics[f"profile/{name}"] = torch.tensor(duration, device=labels.device)

        returned_outputs: Dict[str, torch.Tensor] = {}
        if return_raw_outputs:
            returned_outputs["raw_outputs"] = outputs
        for k in return_keys:
            if k in outputs:
                returned_outputs[k] = outputs[k].detach()

        return (
            new_carry,
            lm_loss,
            metrics,
            returned_outputs,
            new_carry.halted.all(),
        )