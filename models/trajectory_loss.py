import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

IGNORE_LABEL_ID = -100


def trajectory_ranking_loss(model, all_hidden, all_logits, labels, input_embeddings, margin=0.1,
                            shuffle_quality=False, detach_hidden=False,
                            ranking_noise_sigma=0.0, cross_trajectory=False,
                            training=True):
    """Margin ranking loss over URM trajectory steps.

    Two ranking terms:
      1. Within-trajectory (default): for each pair of steps (i, j) where step i
         has higher batch-mean quality than step j, require E(mean h_i) < E(mean h_j).
         This learns temporal ordering of the refinement trajectory.
      2. Cross-trajectory (optional, `cross_trajectory=True`): at each step t, take
         the [B, B] all-pairs ranking over per-example quality and per-example
         energy. Because `_sample_batch` fills every training batch with
         augmentations of a single puzzle, intra-batch pairs at the same step are
         naturally same-puzzle cross-augmentation pairs — step index is held
         constant by construction, so the energy head cannot use step depth as a
         shortcut for these pairs.

    Both terms are mean-normalized by their active-pair count and summed so that
    `energy_loss_weight` controls the combined strength without one term drowning
    out the other.

    Args:
        model: ARCModel (has compute_joint_energy and inner.puzzle_emb_len)
        all_hidden: list of N hidden states [B, seq_len + P, hidden_dim]
        all_logits: list of N logits [B, seq_len, vocab_size]
        labels: [B, seq_len] with IGNORE_LABEL_ID=-100 for padding
        input_embeddings: [B, seq_len + P, hidden_dim]
        margin: margin for ranking loss
        shuffle_quality: A2 ablation — randomly permute batch-mean quality ordering
        detach_hidden: A3 ablation — detach hidden/input_emb before energy head
        ranking_noise_sigma: R2g — max Gaussian σ applied to hidden states before
            the energy head scores them (training only). sigma is sampled per-step
            uniformly from [0, ranking_noise_sigma]. Reconstruction and Q-halt are
            unaffected — noise lives only in the ranking branch.
        cross_trajectory: R2i — enable same-step cross-trajectory ranking pairs.
        training: controls ranking_noise_sigma (noise only applied in training).

    Returns:
        loss: scalar trajectory ranking loss (mean-normalized)
        metrics: dict with trajectory_quality_first, trajectory_quality_last,
                 active_pairs, cross_traj_active_pairs, cross_traj_quality_std,
                 energy_accuracy_spearman, total_pairs, energy_gradient_cosine_sim
    """
    N = len(all_hidden)
    P = model.inner.puzzle_emb_len
    device = labels.device
    B = labels.shape[0]

    mask = labels != IGNORE_LABEL_ID
    loss_counts = mask.sum(-1)  # [B]

    # --- Per-example per-step quality (detached labels, not differentiable) ---
    per_example_quality = []  # list of [B] tensors
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
    within_active = 0
    within_loss_sum = torch.tensor(0.0, device=device)
    total_pairs = N * (N - 1) // 2

    for i in range(N):
        for j in range(i + 1, N):
            if batch_quality[i] != batch_quality[j]:
                within_active += 1
                if batch_quality[i] > batch_quality[j]:
                    gap = batch_quality[i] - batch_quality[j]
                    pair_loss = gap * F.relu(energies_tensor[i] - energies_tensor[j] + margin)
                else:
                    gap = batch_quality[j] - batch_quality[i]
                    pair_loss = gap * F.relu(energies_tensor[j] - energies_tensor[i] + margin)
                within_loss_sum = within_loss_sum + pair_loss

    if within_active > 0:
        within_loss = within_loss_sum / within_active
    else:
        within_loss = within_loss_sum  # 0.0 tensor

    total_loss = within_loss

    # --- Cross-trajectory ranking: [B, B] all-pairs at each step ---
    cross_traj_active_pairs = 0
    cross_traj_quality_std_sum = 0.0
    if cross_trajectory and B > 1:
        cross_loss_sum = torch.tensor(0.0, device=device)
        active_step_count = 0
        for t in range(N):
            q = per_example_quality[t]  # [B]
            e = per_example_energies[t]  # [B]

            cross_traj_quality_std_sum += q.std().item() if B > 1 else 0.0

            # quality_diff[i,j] = q_i - q_j; positive -> i is the better prediction
            quality_diff = q.unsqueeze(1) - q.unsqueeze(0)           # [B, B]
            energy_diff = e.unsqueeze(1) - e.unsqueeze(0)            # [B, B]

            # Only pairs where i strictly beats j (upper triangle equivalent via asymmetry).
            valid = quality_diff > 0                                 # [B, B]
            n_valid = int(valid.sum().item())
            if n_valid == 0:
                continue

            # When q_i > q_j, we want E_i < E_j, i.e. penalize relu(E_i - E_j + margin).
            # quality_diff acts as the gap weight so small gaps contribute little.
            pair_loss = quality_diff * F.relu(energy_diff + margin)  # [B, B]
            step_loss = pair_loss[valid].mean()                      # scalar

            cross_loss_sum = cross_loss_sum + step_loss
            cross_traj_active_pairs += n_valid
            active_step_count += 1

        if active_step_count > 0:
            cross_loss = cross_loss_sum / active_step_count
        else:
            cross_loss = cross_loss_sum  # 0.0 tensor
        total_loss = total_loss + cross_loss

    cross_traj_quality_std_mean = cross_traj_quality_std_sum / max(N, 1)

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
        "active_pairs": torch.tensor(float(within_active), device=device),
        "total_pairs": torch.tensor(float(total_pairs), device=device),
        "cross_traj_active_pairs": torch.tensor(float(cross_traj_active_pairs), device=device),
        "cross_traj_quality_std": torch.tensor(float(cross_traj_quality_std_mean), device=device),
        "energy_accuracy_spearman": torch.tensor(float(spearman_corr), device=device),
        "energy_gradient_cosine_sim": cosine_sim.detach(),
    }

    return total_loss, metrics
