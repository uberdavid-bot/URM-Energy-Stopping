"""Tests for trajectory ranking loss (R2: energy head co-training)."""
import torch
import pytest
from models.urm.urm_energy import ARCModel
from models.trajectory_loss import trajectory_ranking_loss

DEVICE = "cuda"


def make_config(**overrides):
    config = dict(
        batch_size=2,
        seq_len=100,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=100,
        vocab_size=14,
        num_layers=1,
        hidden_size=64,
        expansion=2.0,
        num_heads=4,
        pos_encodings="rope",
        loops=4,
        forward_dtype="bfloat16",
        energy_threshold=1e-6,
        min_steps=4,
    )
    config.update(overrides)
    return config


def make_batch(config):
    B = config["batch_size"]
    S = config["seq_len"]
    return {
        "inputs": torch.randint(0, config["vocab_size"], (B, S), device=DEVICE),
        "labels": torch.randint(0, config["vocab_size"], (B, S), device=DEVICE),
        "puzzle_identifiers": torch.zeros(B, dtype=torch.long, device=DEVICE),
    }


@pytest.fixture(autouse=True)
def skip_without_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


class TestTrajectoryRankingLoss:
    def test_loss_is_finite_with_expected_keys(self):
        """trajectory_ranking_loss returns finite loss with expected metric keys."""
        config = make_config()
        model = ARCModel(config).to(DEVICE).train()
        batch = make_batch(config)

        all_logits, all_q_logits, all_hidden, input_emb = model.forward_trajectory(batch)

        loss, metrics = trajectory_ranking_loss(
            model, all_hidden, all_logits, batch["labels"], input_emb
        )

        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        assert "trajectory_quality_first" in metrics
        assert "trajectory_quality_last" in metrics
        assert "active_pairs" in metrics
        assert "total_pairs" in metrics
        assert "energy_accuracy_spearman" in metrics
        assert "energy_gradient_cosine_sim" in metrics

    def test_total_pairs_count(self):
        """N=4 steps should give C(4,2)=6 total pairs."""
        config = make_config(loops=4)
        model = ARCModel(config).to(DEVICE).train()
        batch = make_batch(config)

        all_logits, _, all_hidden, input_emb = model.forward_trajectory(batch)

        _, metrics = trajectory_ranking_loss(
            model, all_hidden, all_logits, batch["labels"], input_emb
        )

        assert metrics["total_pairs"].item() == 6.0

    def test_gradients_flow_to_energy_head(self):
        """Energy head should receive gradients from trajectory ranking loss."""
        config = make_config()
        model = ARCModel(config).to(DEVICE).train()
        batch = make_batch(config)

        all_logits, _, all_hidden, input_emb = model.forward_trajectory(batch)

        loss, _ = trajectory_ranking_loss(
            model, all_hidden, all_logits, batch["labels"], input_emb
        )

        loss.backward()

        assert model.energy_head.weight.grad is not None, "energy_head.weight has no gradient"
        assert model.energy_head.weight.grad.abs().sum() > 0, "energy_head gradients are zero"

    def test_gradients_flow_to_backbone(self):
        """Backbone should receive gradients (co-training via shared layers)."""
        config = make_config()
        model = ARCModel(config).to(DEVICE).train()
        batch = make_batch(config)

        all_logits, _, all_hidden, input_emb = model.forward_trajectory(batch)

        loss, _ = trajectory_ranking_loss(
            model, all_hidden, all_logits, batch["labels"], input_emb
        )

        loss.backward()

        layer_param = next(model.inner.layers[0].parameters())
        assert layer_param.grad is not None, "Backbone layer has no gradient"
        assert layer_param.grad.abs().sum() > 0, "Backbone layer gradients are zero"

    def test_cosine_sim_is_scalar(self):
        """Cosine similarity diagnostic should be a scalar."""
        config = make_config()
        model = ARCModel(config).to(DEVICE).train()
        batch = make_batch(config)

        all_logits, _, all_hidden, input_emb = model.forward_trajectory(batch)

        _, metrics = trajectory_ranking_loss(
            model, all_hidden, all_logits, batch["labels"], input_emb
        )

        sim = metrics["energy_gradient_cosine_sim"]
        assert sim.dim() == 0, f"Expected scalar, got shape {sim.shape}"
        assert torch.isfinite(sim), f"Cosine sim not finite: {sim.item()}"


class TestEnergyLossHeadWithRanking:
    def test_loss_head_with_energy_loss_weight(self):
        """EnergyLossHead with energy_loss_weight > 0 should include ranking metrics."""
        from models.losses import EnergyLossHead

        config = make_config(loops=4)
        model = ARCModel(config).to(DEVICE).train()
        loss_head = EnergyLossHead(
            model, "stablemax_cross_entropy",
            energy_loss_weight=0.1, ranking_margin=0.1
        ).to(DEVICE)

        batch = make_batch(config)
        _, loss, metrics, _, _ = loss_head(batch=batch)

        assert torch.isfinite(loss)
        assert loss.grad_fn is not None
        assert "trajectory_quality_first" in metrics
        assert "active_pairs" in metrics
        assert "energy_accuracy_spearman" in metrics

        loss.backward()

        # Energy head should get gradients from ranking loss
        assert model.energy_head.weight.grad is not None
        assert model.energy_head.weight.grad.abs().sum() > 0

    def test_loss_head_without_energy_loss_weight(self):
        """EnergyLossHead with energy_loss_weight=0 should NOT include ranking metrics."""
        from models.losses import EnergyLossHead

        config = make_config(loops=4)
        model = ARCModel(config).to(DEVICE).train()
        loss_head = EnergyLossHead(
            model, "stablemax_cross_entropy",
            energy_loss_weight=0.0
        ).to(DEVICE)

        batch = make_batch(config)
        _, loss, metrics, _, _ = loss_head(batch=batch)

        assert "trajectory_quality_first" not in metrics
        assert "active_pairs" not in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
