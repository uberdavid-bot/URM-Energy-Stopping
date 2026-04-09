import torch
import torch.nn.functional as F

IGNORE_LABEL_ID = -100


def trajectory_ranking_loss(
    energy_fn,
    trajectory,  # List of hidden state tensors [B, seq_len+puzzle_emb_len, hidden]
    input_embeddings,  # [B, seq_len+puzzle_emb_len, hidden]
    labels,  # [B, seq_len]
    lm_head,  # nn.Module to project hidden to logits
    puzzle_emb_len,  # int, number of puzzle embedding positions to skip
    margin=0.1,
    max_steps=4,
):
    """
    All-pairs weighted ranking loss on trajectory steps.

    For each step, compute quality (accuracy against labels).
    For each pair (i, j) where quality_j > quality_i,
    enforce E(step_j) < E(step_i) with margin, weighted by quality gap.

    Subsamples to at most max_steps evenly-spaced steps to limit memory usage.

    Returns:
        loss: scalar ranking loss
        metrics: dict with trajectory stats
    """
    if len(trajectory) < 2:
        return torch.tensor(0.0, device=labels.device, requires_grad=True), {}

    # Subsample to limit memory: pick evenly-spaced steps including first and last
    if len(trajectory) > max_steps:
        indices = torch.linspace(0, len(trajectory) - 1, max_steps).long().tolist()
        trajectory = [trajectory[i] for i in indices]

    mask = labels != IGNORE_LABEL_ID
    loss_counts = mask.sum(-1).clamp_min(1).float()  # [B]

    # Compute quality and energy at each trajectory step
    qualities = []  # per-step accuracy [num_steps, B]
    energies = []   # per-step energy [num_steps, B]

    for hidden in trajectory:
        # Quality: accuracy of this step's predictions
        with torch.no_grad():
            step_logits = lm_head(hidden)[:, puzzle_emb_len:]  # [B, seq_len, vocab]
            step_preds = step_logits.argmax(dim=-1)  # [B, seq_len]
            step_correct = (mask & (step_preds == labels)).sum(-1).float()  # [B]
            step_quality = step_correct / loss_counts  # [B], 0-1 accuracy
        qualities.append(step_quality)

        # Energy: score this hidden state (WITH gradients for energy head)
        output_hidden = hidden[:, puzzle_emb_len:]  # Remove puzzle positions
        step_energy = energy_fn(input_embeddings, output_hidden)  # [B]
        energies.append(step_energy)

    # Stack: [num_steps, B]
    qualities = torch.stack(qualities)  # [num_steps, B]
    energies = torch.stack(energies)    # [num_steps, B]

    # All-pairs weighted ranking loss
    num_steps = len(trajectory)
    total_loss = torch.tensor(0.0, device=labels.device)
    num_pairs = 0
    num_active_pairs = 0

    for i in range(num_steps):
        for j in range(num_steps):
            if i == j:
                continue
            # For pairs where j has higher quality than i
            quality_gap = qualities[j] - qualities[i]  # [B]
            valid = quality_gap > 0.01  # Only pairs with meaningful quality difference

            if not valid.any():
                continue

            num_pairs += 1

            # Energy of better step should be lower by margin
            # Loss: weight * relu(E(better) - E(worse) + margin)
            pair_loss = quality_gap * F.relu(energies[j] - energies[i] + margin)  # [B]
            pair_loss = pair_loss[valid].mean()

            if pair_loss.item() > 0:
                num_active_pairs += 1

            total_loss = total_loss + pair_loss

    if num_pairs > 0:
        total_loss = total_loss / num_pairs

    metrics = {
        "trajectory_loss": total_loss.detach(),
        "trajectory_steps": torch.tensor(float(num_steps), device=labels.device),
        "trajectory_pairs": torch.tensor(float(num_pairs), device=labels.device),
        "trajectory_active_pairs": torch.tensor(float(num_active_pairs), device=labels.device),
        "trajectory_quality_first": qualities[0].mean().detach(),
        "trajectory_quality_last": qualities[-1].mean().detach(),
        "trajectory_energy_first": energies[0].mean().detach(),
        "trajectory_energy_last": energies[-1].mean().detach(),
    }

    return total_loss, metrics
