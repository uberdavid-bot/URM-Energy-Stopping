"""
Tests for ARCModel architecture: URM, EBT, and hybrid modes.

All modes do one step per forward() call. The outer loop handles iteration and halting.
Requires CUDA — runs real flash_attn and model code on GPU.
"""
import torch
import pytest
from models.urm.urm_energy import ARCModel

DEVICE = "cuda"


def make_config(**overrides):
    config = dict(
        batch_size=2,
        seq_len=100,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=100,
        vocab_size=14,
        num_layers=2,
        hidden_size=64,
        expansion=2.0,
        num_heads=4,
        pos_encodings="rope",
        loops=12,
        forward_dtype="bfloat16",
        energy_threshold=1e-6,
        min_steps=8,
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


def make_carry(model, batch):
    with torch.device(DEVICE):
        return model.initial_carry(batch)


@pytest.fixture(autouse=True)
def skip_without_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


class TestURMForward:
    def test_forward_single_step(self):
        """URM forward() does one transformer pass and produces meaningful logits."""
        config = make_config()
        model = ARCModel(config).to(DEVICE).eval()

        batch = make_batch(config)
        carry = make_carry(model, batch)

        with torch.no_grad():
            carry_out, outputs = model(carry, batch)

        logits = outputs["logits"]
        assert logits.shape == (config["batch_size"], config["seq_len"], config["vocab_size"])
        assert not torch.allclose(logits, torch.zeros_like(logits), atol=1e-5)

    def test_forward_computes_energy(self):
        """forward() should compute current_energy on the output."""
        config = make_config()
        model = ARCModel(config).to(DEVICE).eval()

        batch = make_batch(config)
        carry = make_carry(model, batch)

        with torch.no_grad():
            _, outputs = model(carry, batch)

        assert "current_energy" in outputs
        energy = outputs["current_energy"]
        assert energy.shape == (config["batch_size"],)
        assert not torch.allclose(energy, torch.zeros_like(energy), atol=1e-8)

    def test_forward_returns_output_hidden(self):
        """forward() should return output_hidden without puzzle embedding positions."""
        config = make_config()
        model = ARCModel(config).to(DEVICE).eval()

        batch = make_batch(config)
        carry = make_carry(model, batch)

        with torch.no_grad():
            _, outputs = model(carry, batch)

        output_hidden = outputs["output_hidden"]
        assert output_hidden.shape == (config["batch_size"], config["seq_len"], config["hidden_size"])

    def test_no_unrefined_logits_in_urm(self):
        """URM mode should not produce unrefined_logits."""
        config = make_config()
        model = ARCModel(config).to(DEVICE).eval()

        batch = make_batch(config)
        carry = make_carry(model, batch)

        with torch.no_grad():
            _, outputs = model(carry, batch)

        assert "unrefined_logits" not in outputs


class TestHalting:
    def test_halting_after_max_loops(self):
        """Should halt after reaching max loops."""
        config = make_config(loops=3)
        model = ARCModel(config).to(DEVICE).eval()

        batch = make_batch(config)
        carry = make_carry(model, batch)

        with torch.no_grad():
            steps = 0
            while True:
                carry, outputs = model(carry, batch)
                steps += 1
                if carry.halted.all():
                    break
                assert steps < 10, "Did not halt within expected iterations"

        assert steps <= config["loops"]

    def test_halting_respects_min_steps(self):
        """Should not halt before min_steps even if energy converges."""
        config = make_config(loops=20, min_steps=5, energy_threshold=1e10)
        model = ARCModel(config).to(DEVICE).eval()

        batch = make_batch(config)
        carry = make_carry(model, batch)

        with torch.no_grad():
            carry, _ = model(carry, batch)
            steps = 1

            while not carry.halted.all():
                carry, outputs = model(carry, batch)
                steps += 1
                if steps > 25:
                    break

        assert steps >= config["min_steps"], (
            f"Halted after {steps} steps, expected at least {config['min_steps']}"
        )

    def test_halting_uniform_across_modes(self):
        """All three modes should use the same halting logic (steps >= loops)."""
        for mode in ["urm", "ebt", "hybrid"]:
            config = make_config(loops=3, refinement=mode, mcmc_start_step=1, mcmc_step_size=0.01)
            model = ARCModel(config).to(DEVICE).eval()
            batch = make_batch(config)
            carry = make_carry(model, batch)

            with torch.no_grad():
                steps = 0
                while True:
                    carry, _ = model(carry, batch)
                    steps += 1
                    if carry.halted.all() or steps > 10:
                        break

            assert steps == config["loops"], (
                f"Mode '{mode}' halted after {steps} steps, expected {config['loops']}"
            )

    def test_prev_energy_propagation(self):
        """prev_energy should be set after first forward pass."""
        config = make_config()
        model = ARCModel(config).to(DEVICE).eval()

        batch = make_batch(config)
        carry = make_carry(model, batch)
        assert carry.prev_energy is None

        with torch.no_grad():
            carry, _ = model(carry, batch)

        assert carry.prev_energy is not None
        assert carry.prev_energy.shape == (config["batch_size"],)


class TestEnergyHead:
    def test_energy_head_gets_gradients(self):
        """After backward, energy_head parameters should have gradients."""
        config = make_config()
        model = ARCModel(config).to(DEVICE).train()

        batch = make_batch(config)
        carry = make_carry(model, batch)
        _, outputs = model(carry, batch)

        input_emb = model.inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        labels = batch["labels"]
        true_emb = model.inner.embed_tokens(labels.clamp(min=0))
        true_emb = model.inner.embed_scale * true_emb

        energy = model.compute_joint_energy(input_emb.detach(), true_emb.detach())
        energy.sum().backward()

        assert model.energy_head.weight.grad is not None
        assert model.energy_head.weight.grad.abs().sum() > 0


class TestRotaryEmbeddingLength:
    def test_rotary_buffer_covers_energy_concat(self):
        """Rotary embedding buffer must be >= 2*seq_len + puzzle_emb_len."""
        config = make_config(puzzle_emb_ndim=64)
        model = ARCModel(config).to(DEVICE)

        seq_len = config["seq_len"]
        puzzle_emb_len = model.inner.puzzle_emb_len
        expected_len = 2 * seq_len + puzzle_emb_len

        rotary_len = model.inner.rotary_emb.cos_cached.shape[0]
        assert rotary_len == expected_len

    def test_rotary_without_puzzle_emb(self):
        """Rotary buffer correct when puzzle_emb_ndim=0."""
        config = make_config()
        model = ARCModel(config).to(DEVICE)

        expected_len = 2 * config["seq_len"]
        rotary_len = model.inner.rotary_emb.cos_cached.shape[0]
        assert rotary_len == expected_len


class TestEnergyConfidenceRanking:
    def test_sorting_by_negative_energy_ranks_correctly(self):
        """Lower energy should rank higher when sorted by -energy (confidence)."""
        predictions = {
            "hash_a": [1, 0.0, 0.0, 5.0],
            "hash_b": [1, 0.0, 0.0, -3.0],
        }
        ranked = sorted(predictions.items(), key=lambda kv: kv[1][3], reverse=True)
        assert ranked[0][0] == "hash_a"
        assert ranked[1][0] == "hash_b"


class TestEBTMode:
    def test_ebt_single_step(self):
        """EBT forward() does one MCMC gradient step and produces valid logits."""
        config = make_config(refinement="ebt", loops=4, mcmc_step_size=0.01)
        model = ARCModel(config).to(DEVICE).eval()

        batch = make_batch(config)
        carry = make_carry(model, batch)

        with torch.no_grad():
            carry_out, outputs = model(carry, batch)

        logits = outputs["logits"]
        assert logits.shape == (config["batch_size"], config["seq_len"], config["vocab_size"])
        assert not torch.allclose(logits, torch.zeros_like(logits), atol=1e-5)
        assert "current_energy" in outputs
        assert "output_hidden" in outputs
        # Should NOT halt after one call (outer loop handles this)
        assert not carry_out.halted.all(), "EBT should not halt after one step"

    def test_ebt_changes_hidden_each_step(self):
        """Each EBT step should change the hidden state (MCMC makes progress)."""
        config = make_config(refinement="ebt", loops=4, mcmc_step_size=0.01)
        model = ARCModel(config).to(DEVICE).eval()

        batch = make_batch(config)
        carry = make_carry(model, batch)

        with torch.no_grad():
            carry, out1 = model(carry, batch)
            hidden_after_1 = carry.current_hidden.clone()
            carry, out2 = model(carry, batch)
            hidden_after_2 = carry.current_hidden.clone()

        assert not torch.allclose(hidden_after_1, hidden_after_2, atol=1e-5), (
            "EBT should change hidden states between steps"
        )

    def test_ebt_gradients_flow_to_energy_head(self):
        """EBT training step should give gradients to energy_head via MCMC."""
        config = make_config(refinement="ebt", loops=4, mcmc_step_size=0.01)
        model = ARCModel(config).to(DEVICE).train()

        batch = make_batch(config)
        carry = make_carry(model, batch)

        _, outputs = model(carry, batch)

        labels = batch["labels"]
        loss = torch.nn.functional.cross_entropy(
            outputs["logits"].view(-1, config["vocab_size"]),
            labels.view(-1), ignore_index=-100,
        )
        loss.backward()

        assert model.energy_head.weight.grad is not None, "energy_head has no gradients"
        assert model.energy_head.weight.grad.abs().sum() > 0, "energy_head gradients are zero"


class TestHybridMode:
    def test_hybrid_urm_then_mcmc(self):
        """Hybrid: first steps are URM, later steps are MCMC."""
        config = make_config(refinement="hybrid", loops=6, mcmc_start_step=3, mcmc_step_size=0.01)
        model = ARCModel(config).to(DEVICE).eval()

        batch = make_batch(config)
        carry = make_carry(model, batch)

        with torch.no_grad():
            steps = 0
            while True:
                carry, outputs = model(carry, batch)
                steps += 1
                if carry.halted.all() or steps > 10:
                    break

        assert steps == config["loops"]

    def test_hybrid_gradients_both_paths(self):
        """Hybrid should give gradients to URM layers (URM phase) and energy_head (MCMC phase)."""
        config = make_config(refinement="hybrid", loops=6, mcmc_start_step=3, mcmc_step_size=0.01)
        model = ARCModel(config).to(DEVICE).train()

        batch = make_batch(config)
        carry = make_carry(model, batch)

        # Run through all steps, accumulating loss
        total_loss = torch.tensor(0.0, device=DEVICE)
        labels = batch["labels"]
        V = config["vocab_size"]
        for _ in range(config["loops"]):
            carry, outputs = model(carry, batch)
            step_loss = torch.nn.functional.cross_entropy(
                outputs["logits"].view(-1, V), labels.view(-1), ignore_index=-100,
            )
            total_loss = total_loss + step_loss

        total_loss.backward()

        # Energy head gets gradients through MCMC phase
        assert model.energy_head.weight.grad is not None, "energy_head has no gradients"
        assert model.energy_head.weight.grad.abs().sum() > 0, "energy_head gradients are zero"
        # URM layers get gradients through URM phase
        layer_param = next(model.inner.layers[0].parameters())
        assert layer_param.grad is not None, "URM layer has no gradients"
        assert layer_param.grad.abs().sum() > 0, "URM layer gradients are zero"


class TestLossHeadIntegration:
    def test_loss_head_produces_valid_loss(self):
        """EnergyLossHead should produce positive, differentiable reconstruction loss."""
        from models.losses import EnergyLossHead

        config = make_config(loops=4)
        model = ARCModel(config).to(DEVICE).train()
        loss_head = EnergyLossHead(model, "stablemax_cross_entropy").to(DEVICE)

        batch = make_batch(config)
        with torch.device(DEVICE):
            carry = loss_head.initial_carry(batch)

        carry, loss, metrics, _, _ = loss_head(
            return_keys=[], carry=carry, batch=batch
        )

        assert "reconstruction_loss" in metrics
        assert loss.item() > 0
        assert loss.grad_fn is not None

        loss.backward()

    def test_no_per_step_metrics_in_loss_head(self):
        """Per-step metrics are computed in the outer eval loop, not EnergyLossHead."""
        from models.losses import EnergyLossHead

        config = make_config(loops=4)
        model = ARCModel(config).to(DEVICE).eval()
        loss_head = EnergyLossHead(model, "stablemax_cross_entropy").to(DEVICE)

        batch = make_batch(config)
        with torch.device(DEVICE):
            carry = loss_head.initial_carry(batch)

        with torch.no_grad():
            carry, loss, metrics, _, _ = loss_head(
                return_keys=[], carry=carry, batch=batch
            )

        step_keys = [k for k in metrics if k.startswith("step_")]
        assert len(step_keys) == 0, f"Unexpected per-step metrics in loss_head: {step_keys}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
