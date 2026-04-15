import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

IGNORE_LABEL_ID = -100


def trajectory_ranking_loss(model, all_hidden, all_logits, labels, input_embeddings, margin=0.1,
                            shuffle_quality=False, detach_hidden=False,
                            ranking_noise_sigma=0.0, cross_trajectory_k=1,
                            training=True):
    """All-pairs weighted margin ranking loss across URM trajectory steps.

    Within-trajectory: for each pair (i, j) where step i has higher batch-mean
    quality than step j, E(h_i) should be < E(h_j).

    Cross-trajectory (cross_trajectory_k > 1): for each recurrence step t, sample
    same-step pairs across augmentations in the batch. Because _sample_batch fills
    each training batch with augmentations of one puzzle, intra-batch pairs are
    naturally same-puzzle cross-augmentation pairs. Step index is held constant
    within each pair, so the energy head cannot use step depth as a shortcut.

    Args:
        model: ARCModel (has compute_joint_energy and inner.puzzle_emb_len)
        all_hidden: list of N hidden states [B, seq_len + P, hidden_dim]
        all_logits: list of N logits [B, seq_len, vocab_size]
        labels: [B, seq_len] with IGNORE_LABEL_ID=-100 for padding
        input_embeddings: [B, seq_len + P, hidden_dim]
        margin: margin for ranking loss
        shuffle_quality: A2 ablation — randomly permute quality scores
        detach_hidden: A3 ablation — detach hidden/input_emb before energy head
        ranking_noise_sigma: R2g — max Gaussian σ applied to hidden states before
            the energy head scores them (training only). sigma is sampled per-step
            uniformly from [0, ranking_noise_sigma]. Reconstruction and Q-halt are
            unaffected — noise lives only in the ranking branch.
        cross_trajectory_k: R2i — enables cross-trajectory same-step pairs when > 1.
            Does not change the number of forward passes; reuses per-example energies.
        training: controls ranking_noise_sigma (noise only applied in training).

    Returns:
        loss: scalar trajectory ranking loss
        metrics: dict with trajectory_quality_first, trajectory_quality_last,
                 active_pairs, cross_traj_active_pairs, energy_accuracy_spearman,
                 total_pairs, energy_gradient_cosine_sim
    """
    N = len(all_hidden)
    P = model.inner.puzzle_emb_len
    device = labels.device
    B = labels.shape[0]

    mask = labels != IGNORE_LABEL_ID
    loss_counts = mask.sum(-1)  # [B]

    # --- Per-example per-step quality (detached labels, not differentiable) ---
    per_example_quality = []
    with torch.no_grad():
        for t in range(N):
            preds = all_logits[t].detach().argmax(dim=-1)
            correct = (mask & (preds == labels)).float().sum(-1)  # [B]
            per_example_quality.append(correct / loss_counts.clamp(min=1))  # [B]

    batch_quality = [q.mean().item() for q in per_example_quality]

    # A2 ablation: shuffle batch-averaged quality (breaks within-trajectory signal)
    if shuffle_quality:
        perm = torch.randperm(len(batch_quality))
        batch_quality = [batch_quality[i] for i in perm]

    # A3 ablation: detach hidden states / input embeddings before energy head
    input_emb_for_energy = input_embeddings.detach() if detach_hidden else input_embeddings

    # --- Per-step hidden states for energy head (detach / ranking noise applied here) ---
    hidden_for_energy = []
    for t in range(N):
        h = all_hidden[t][:, P:]
        if detach_hidden:
            h = h.detach()
        if ranking_noise_sigma > 0 and training:
            # R2g: σ ~ Uniform(0, ranking_noise_sigma) per step, fresh Gaussian per step
            sigma_t = torch.rand((), device=h.device) * ranking_noise_sigma
            h = h + sigma_t * torch.randn_like(h)
        hidden_for_energy.append(h)

    # --- Per-example and batch-averaged energies ---
    per_example_energies = []  # list of [B] tensors, used by cross-trajectory pairs
    energies_avg = []           # scalar per step, used by within-trajectory ranking
    for t in range(N):
        per_ex_e = model.compute_joint_energy(input_emb_for_energy, hidden_for_energy[t])  # [B]
        per_example_energies.append(per_ex_e)
        energies_avg.append(per_ex_e.mean())

    energies_tensor = torch.stack(energies_avg)  # [N]

    # --- Within-trajectory ranking (step i vs step j on batch-mean quality/energy) ---
    total_loss = torch.tensor(0.0, device=device)
    active_pairs = 0
    total_pairs = N * (N - 1) // 2

    for i in range(N):
        for j in range(i + 1, N):
            if batch_quality[i] != batch_quality[j]:
                active_pairs += 1
                if batch_quality[i] > batch_quality[j]:
                    gap = batch_quality[i] - batch_quality[j]
                    pair_loss = gap * F.relu(energies_tensor[i] - energies_tensor[j] + margin)
                else:
                    gap = batch_quality[j] - batch_quality[i]
                    pair_loss = gap * F.relu(energies_tensor[j] - energies_tensor[i] + margin)
                total_loss = total_loss + pair_loss

    # --- Cross-trajectory ranking: same-step pairs across augmentations ---
    # Pairs use random stride halving (B/2 pairs per step). Because the training
    # dataloader packs each batch with augmentations of a single puzzle, these are
    # naturally same-puzzle cross-augmentation pairs with matched step index.
    cross_traj_active_pairs = 0
    if cross_trajectory_k > 1 and B > 1:
        half = B // 2
        for t in range(N):
            q = per_example_quality[t]  # [B]
            e = per_example_energies[t]  # [B]

            perm = torch.randperm(B, device=device)
            idx_a = perm[:half]
            idx_b = perm[half: 2 * half]

            q_a = q[idx_a]
            q_b = q[idx_b]
            e_a = e[idx_a]
            e_b = e[idx_b]

            q_gap = q_a - q_b  # positive → a is better, negative → b is better
            pair_mask = q_gap != 0
            if not pair_mask.any():
                continue

            # Ranking loss: if q_a > q_b, want e_a < e_b (violation: e_a - e_b + margin > 0)
            # Combine both directions via signed gap.
            violation = F.relu(torch.sign(q_gap) * (e_a - e_b) + margin)
            pair_losses = q_gap.abs() * violation

            # Mean over active pairs at this step (keeps scale comparable to within-traj term)
            active_count = pair_mask.sum().clamp_min(1)
            total_loss = total_loss + (pair_losses.sum() / active_count)
            cross_traj_active_pairs += int(pair_mask.sum().item())

    # --- Spearman correlation (within-trajectory diagnostic, unchanged) ---
    with torch.no_grad():
        energies_np = energies_tensor.detach().float().cpu().numpy()
        if len(set(batch_quality)) > 1 and len(set(energies_np.tolist())) > 1:
            spearman_corr = spearmanr(energies_np, batch_quality)[0]
        else:
            spearman_corr = 0.0

    # --- Cosine similarity diagnostic (R3 readiness) ---
    cosine_sim = torch.tensor(0.0, device=device)
    worst_idx = batch_quality.index(min(batch_quality))
    best_idx = batch_quality.index(max(batch_quality))

    if worst_idx != best_idx:
        hidden_worst = all_hidden[worst_idx][:, P:].detach().requires_grad_(True)
        hidden_best = all_hidden[best_idx][:, P:].detach()
        input_emb_detached = input_embeddings.detach()

        with torch.enable_grad():
            energy_worst = model.compute_joint_energy(input_emb_detached, hidden_worst)
            grad_E = torch.autograd.grad(energy_worst.sum(), hidden_worst, create_graph=False)[0]

        direction = hidden_best - hidden_worst.detach()
        cosine_sim = F.cosine_similarity(grad_E.flatten(), direction.flatten(), dim=0)

    metrics = {
        "trajectory_quality_first": torch.tensor(batch_quality[0], device=device),
        "trajectory_quality_last": torch.tensor(batch_quality[-1], device=device),
        "active_pairs": torch.tensor(float(active_pairs), device=device),
        "total_pairs": torch.tensor(float(total_pairs), device=device),
        "cross_traj_active_pairs": torch.tensor(float(cross_traj_active_pairs), device=device),
        "energy_accuracy_spearman": torch.tensor(float(spearman_corr), device=device),
        "energy_gradient_cosine_sim": cosine_sim.detach(),
    }

    return total_loss, metrics
