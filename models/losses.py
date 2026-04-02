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
        from models.dsm_loss import dsm_loss as dsm_loss_fn

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

        # Reconstruction loss (standard LM loss on URM output)
        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum()

        # DSM loss on energy head
        input_embeddings = self.model.inner._input_embeddings(
            new_carry.current_data["inputs"],
            new_carry.current_data["puzzle_identifiers"]
        )
        true_embeddings = self.model.inner.embed_tokens(labels.clamp(min=0))
        true_embeddings = true_embeddings * mask.unsqueeze(-1).float()
        true_embeddings = self.model.inner.embed_scale * true_embeddings

        dsm_loss_val, dsm_metrics = dsm_loss_fn(
            energy_fn=self.model.compute_joint_energy,
            clean_embeddings=true_embeddings.detach(),
            input_embeddings=input_embeddings.detach(),
            noise_scales=self.model.config.dsm_noise_scales,
        )

        # Contrastive loss: E(true) should be lower than E(predicted) by margin
        true_energy = self.model.compute_joint_energy(
            input_embeddings.detach(), true_embeddings.detach()
        )
        predicted_energy = outputs["current_energy"]
        margin = self.model.config.contrastive_margin
        contrastive_loss = F.relu(true_energy - predicted_energy + margin).mean()

        metrics.update({
            "reconstruction_loss": lm_loss.detach(),
            "dsm_loss": dsm_loss_val.detach(),
            "contrastive_loss": contrastive_loss.detach(),
            "current_energy": outputs["current_energy"].mean().detach(),
            "true_energy": true_energy.mean().detach(),
            "predicted_energy": predicted_energy.mean().detach(),
            "energy_gap": (predicted_energy - true_energy).mean().detach(),
        })
        metrics.update(dsm_metrics)

        if profile is not None:
            for name, duration in profile.items():
                metrics[f"profile/{name}"] = torch.tensor(duration, device=labels.device)

        returned_outputs: Dict[str, torch.Tensor] = {}
        if return_raw_outputs:
            returned_outputs["raw_outputs"] = outputs
        for k in return_keys:
            if k in outputs:
                returned_outputs[k] = outputs[k].detach()

        dsm_weight = self.model.config.dsm_weight
        contrastive_weight = self.model.config.contrastive_weight
        total_loss = lm_loss + dsm_weight * dsm_loss_val + contrastive_weight * contrastive_loss

        return (
            new_carry,
            total_loss,
            metrics,
            returned_outputs,
            new_carry.halted.all(),
        )