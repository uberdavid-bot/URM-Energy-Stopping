import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

IGNORE_LABEL_ID = -100


def trajectory_ranking_loss(model, all_hidden, all_logits, labels, input_embeddings, margin=0.1):
    """All-pairs weighted margin ranking loss across URM trajectory steps.

    For each pair (i, j) where step i has higher quality than step j:
        loss += quality_gap * F.relu(E(h_i) - E(h_j) + margin)

    Lower energy should correspond to better (higher accuracy) hidden states.

    Args:
        model: ARCModel (has compute_joint_energy and inner.puzzle_emb_len)
        all_hidden: list of N hidden states [B, seq_len + P, hidden_dim]
        all_logits: list of N logits [B, seq_len, vocab_size]
        labels: [B, seq_len] with IGNORE_LABEL_ID=-100 for padding
        input_embeddings: [B, seq_len + P, hidden_dim]
        margin: margin for ranking loss

    Returns:
        loss: scalar trajectory ranking loss
        metrics: dict with trajectory_quality_first, trajectory_quality_last,
                 active_pairs, energy_accuracy_spearman, total_pairs,
                 energy_gradient_cosine_sim
    """
    N = len(all_hidden)
    P = model.inner.puzzle_emb_len
    device = labels.device

    mask = labels != IGNORE_LABEL_ID
    loss_counts = mask.sum(-1)  # [B]

    # --- Per-step quality (detached — these are labels, not differentiable targets) ---
    qualities = []
    with torch.no_grad():
        for t in range(N):
            preds = all_logits[t].detach().argmax(dim=-1)
            correct = (mask & (preds == labels)).float().sum(-1)  # [B]
            quality = (correct / loss_counts.clamp(min=1)).mean()  # scalar
            qualities.append(quality.item())

    # --- Per-step energy (NOT detached — co-training) ---
    energies = []
    for t in range(N):
        energy = model.compute_joint_energy(input_embeddings, all_hidden[t][:, P:])
        energies.append(energy.mean())

    energies_tensor = torch.stack(energies)  # [N]

    # --- All-pairs weighted margin ranking loss ---
    total_loss = torch.tensor(0.0, device=device)
    active_pairs = 0
    total_pairs = N * (N - 1) // 2

    for i in range(N):
        for j in range(i + 1, N):
            if qualities[i] != qualities[j]:
                active_pairs += 1
                if qualities[i] > qualities[j]:
                    # Step i is better — should have lower energy
                    gap = qualities[i] - qualities[j]
                    pair_loss = gap * F.relu(energies_tensor[i] - energies_tensor[j] + margin)
                else:
                    # Step j is better — should have lower energy
                    gap = qualities[j] - qualities[i]
                    pair_loss = gap * F.relu(energies_tensor[j] - energies_tensor[i] + margin)
                total_loss = total_loss + pair_loss

    # --- Spearman correlation (diagnostic) ---
    with torch.no_grad():
        energies_np = energies_tensor.detach().float().cpu().numpy()
        if len(set(qualities)) > 1 and len(set(energies_np.tolist())) > 1:
            spearman_corr = spearmanr(energies_np, qualities)[0]
        else:
            spearman_corr = 0.0

    # --- Cosine similarity diagnostic (R3 readiness) ---
    # Detach everything — this is a diagnostic, not a training signal.
    cosine_sim = torch.tensor(0.0, device=device)
    worst_idx = qualities.index(min(qualities))
    best_idx = qualities.index(max(qualities))

    if worst_idx != best_idx:
        hidden_worst = all_hidden[worst_idx][:, P:].detach().requires_grad_(True)
        hidden_best = all_hidden[best_idx][:, P:].detach()
        input_emb_detached = input_embeddings.detach()

        with torch.enable_grad():
            energy_worst = model.compute_joint_energy(input_emb_detached, hidden_worst)
            grad_E = torch.autograd.grad(energy_worst.sum(), hidden_worst, create_graph=False)[0]

        direction = hidden_best - hidden_worst.detach()
        cosine_sim = F.cosine_similarity(grad_E.flatten(), direction.flatten(), dim=0)

    # --- Metrics ---
    metrics = {
        "trajectory_quality_first": torch.tensor(qualities[0], device=device),
        "trajectory_quality_last": torch.tensor(qualities[-1], device=device),
        "active_pairs": torch.tensor(float(active_pairs), device=device),
        "total_pairs": torch.tensor(float(total_pairs), device=device),
        "energy_accuracy_spearman": torch.tensor(float(spearman_corr), device=device),
        "energy_gradient_cosine_sim": cosine_sim.detach(),
    }

    return total_loss, metrics
