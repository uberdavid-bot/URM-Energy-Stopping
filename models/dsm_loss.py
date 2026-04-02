"""
Multi-scale Denoising Score Matching (DSM) loss for energy-based models.

Trains the energy function's gradients to point from corrupted data toward
clean data, without needing to backpropagate through sequential MCMC steps.
"""
from typing import Callable, Dict, List

import torch


def dsm_loss(
    energy_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    clean_embeddings: torch.Tensor,
    input_embeddings: torch.Tensor,
    noise_scales: List[float] = [0.1, 0.3, 0.5, 1.0],
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Multi-scale Denoising Score Matching loss.

    For each noise scale sigma:
      1. x_noisy = x_clean + sigma * eps,  where eps ~ N(0, I)
      2. Compute energy gradient: grad_x E(input, x_noisy)
      3. The score should satisfy: score = -grad_x E ~ (x_clean - x_noisy) / sigma^2
      4. Loss: ||sigma^2 * grad_x E + (x_noisy - x_clean)||^2

    create_graph=True is used here to backprop through the single gradient
    computation into the energy head parameters. This is O(1) cost — NOT the
    O(steps) cost of backpropping through sequential MCMC.

    Args:
        energy_fn: callable(input_emb, output_emb) -> [B] scalar energy per example
        clean_embeddings: [B, seq_len, hidden] - true output embeddings
        input_embeddings: [B, seq_len + puzzle_emb_len, hidden] - input embeddings
        noise_scales: list of sigma values for multi-scale corruption

    Returns:
        loss: scalar DSM loss (averaged across scales)
        metrics: dict with per-scale losses and gradient norms
    """
    total_loss = torch.tensor(0.0, device=clean_embeddings.device)
    metrics: Dict[str, torch.Tensor] = {}

    for sigma in noise_scales:
        noise = torch.randn_like(clean_embeddings) * sigma
        noisy_embeddings = (clean_embeddings + noise).detach().requires_grad_(True)

        energy = energy_fn(input_embeddings, noisy_embeddings)

        # Single gradient computation with create_graph=True so we can
        # backprop into energy_fn's parameters. This is O(1), not O(steps).
        grad = torch.autograd.grad(
            energy.sum(), noisy_embeddings, create_graph=True
        )[0]

        # DSM objective: sigma^2 * grad_x E should equal -(x_noisy - x_clean) = -noise
        # Equivalently: ||sigma^2 * grad + noise||^2
        scale_loss = ((sigma ** 2 * grad + noise) ** 2).mean()
        total_loss = total_loss + scale_loss

        metrics[f"dsm_loss_sigma_{sigma}"] = scale_loss.detach()
        metrics[f"grad_norm_sigma_{sigma}"] = grad.norm(dim=-1).mean().detach()

    total_loss = total_loss / len(noise_scales)
    return total_loss, metrics
