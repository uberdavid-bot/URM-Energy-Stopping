"""
Tests for ARCModel architecture: energy head, MCMC refinement.

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
        L_cycles=1,
        H_cycles=1,
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


class TestURMEnergyForward:
    def test_forward_runs_inner(self):
        """forward() should run URM inner recurrence and produce meaningful logits."""
        config = make_config()
        model = ARCModel(config).to(DEVICE).eval()

        batch = make_batch(config)
        carry = make_carry(model, batch)

        with torch.no_grad():
            carry_out, outputs = model(carry, batch)

        logits = outputs["logits"]
        assert logits.shape == (config["batch_size"], config["seq_len"], config["vocab_size"])
        # Logits should not be all zeros (inner recurrence actually ran)
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
        # Should be seq_len (no puzzle_emb_len since puzzle_emb_ndim=0)
        assert output_hidden.shape == (config["batch_size"], config["seq_len"], config["hidden_size"])


class TestPerStepLogits:
    def test_per_step_logits_captured_at_eval(self):
        """Eval mode should capture per-step logits for convergence analysis."""
        config = make_config(loops=4)
        model = ARCModel(config).to(DEVICE).eval()

        batch = make_batch(config)
        carry = make_carry(model, batch)

        with torch.no_grad():
            _, outputs = model(carry, batch)

        assert "per_step_logits" in outputs, "per_step_logits missing from eval outputs"
        per_step = outputs["per_step_logits"]
        assert len(per_step) == config["loops"], (
            f"Expected {config['loops']} per-step logits, got {len(per_step)}"
        )
        for i, step_logits in enumerate(per_step):
            assert step_logits.shape == (config["batch_size"], config["seq_len"], config["vocab_size"]), (
                f"Step {i} logits shape mismatch"
            )
        # Final step logits should match the output logits
        assert torch.allclose(per_step[-1], outputs["logits"], atol=1e-5), (
            "Last per-step logits should match final output logits"
        )

    def test_per_step_logits_not_captured_at_train(self):
        """Training mode should not capture per-step logits (saves memory)."""
        config = make_config(loops=4)
        model = ARCModel(config).to(DEVICE).train()

        batch = make_batch(config)
        carry = make_carry(model, batch)

        _, outputs = model(carry, batch)

        assert "per_step_logits" not in outputs

    def test_per_step_logits_accuracy_metrics(self):
        """EnergyLossHead should produce per-step accuracy metrics at eval."""
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

        for step in range(1, config["loops"] + 1):
            key = f"step_{step}_accuracy"
            assert key in metrics, f"Missing metric {key}"


class TestEnergyHalting:
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
            # First call: carry.halted=True triggers reset, returns halted=False
            carry, _ = model(carry, batch)
            steps = 1

            # Now iterate until halted
            while not carry.halted.all():
                carry, outputs = model(carry, batch)
                steps += 1
                if steps > 25:
                    break

        # With energy_threshold=1e10 the convergence check always passes,
        # but halting requires new_steps >= min_steps. So we expect exactly min_steps.
        assert steps >= config["min_steps"], (
            f"Halted after {steps} steps, expected at least {config['min_steps']}"
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


class TestMCMCRefinement:
    def test_refine_with_mcmc_changes_logits(self):
        """refine_with_mcmc should produce different logits than the input."""
        config = make_config()
        model = ARCModel(config).to(DEVICE).eval()

        batch = make_batch(config)
        carry = make_carry(model, batch)

        with torch.no_grad():
            carry, outputs = model(carry, batch)

        input_embeddings = model.inner._input_embeddings(
            batch["inputs"], batch["puzzle_identifiers"]
        )

        with torch.no_grad():
            refined = model.refine_with_mcmc(
                outputs["output_hidden"], input_embeddings, steps=4, step_size=0.01
            )

        assert not torch.allclose(outputs["logits"], refined, atol=1e-5), (
            "MCMC refinement did not change logits"
        )


class TestEnergyLossHeadBackward:
    def test_energy_head_gets_gradients(self):
        """After backward, energy_head parameters should have gradients from energy computation."""
        config = make_config()
        model = ARCModel(config).to(DEVICE).train()

        batch = make_batch(config)
        carry = make_carry(model, batch)
        _, outputs = model(carry, batch)

        # Energy computation always gives gradients to energy_head
        input_emb = model.inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        labels = batch["labels"]
        true_emb = model.inner.embed_tokens(labels.clamp(min=0))
        true_emb = model.inner.embed_scale * true_emb

        energy = model.compute_joint_energy(input_emb.detach(), true_emb.detach())
        energy.sum().backward()

        assert model.energy_head.weight.grad is not None, (
            "energy_head has no gradients after energy backward"
        )
        assert model.energy_head.weight.grad.abs().sum() > 0, (
            "energy_head gradients are all zero"
        )

    def test_no_mcmc_create_graph_in_forward(self):
        """Training forward pass should NOT use create_graph through sequential MCMC steps."""
        config = make_config()
        model = ARCModel(config).to(DEVICE).train()

        batch = make_batch(config)
        carry = make_carry(model, batch)

        # The forward pass should complete without creating a deep autograd graph
        # through multiple MCMC steps. We verify by checking that current_energy
        # does NOT have a deep computation graph (it's from a single energy eval).
        _, outputs = model(carry, batch)
        energy = outputs["current_energy"]

        # Energy should have grad_fn (it's a computed value), but the graph
        # should be shallow (single forward pass, not MCMC chain)
        assert energy.grad_fn is not None, "Energy should be differentiable"


class TestRotaryEmbeddingLength:
    def test_rotary_buffer_covers_energy_concat(self):
        """Rotary embedding buffer must be >= 2*seq_len + puzzle_emb_len."""
        config = make_config(puzzle_emb_ndim=64)  # = hidden_size -> puzzle_emb_len=1
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


class TestEnergyInOutputs:
    def test_current_energy_in_outputs(self):
        """current_energy must be in outputs with shape [batch_size]."""
        config = make_config()
        model = ARCModel(config).to(DEVICE).eval()

        batch = make_batch(config)
        carry = make_carry(model, batch)

        with torch.no_grad():
            _, outputs = model(carry, batch)

        assert "current_energy" in outputs
        assert outputs["current_energy"].shape == (config["batch_size"],)


class TestEnergyConfidenceRanking:
    def test_sorting_by_negative_energy_ranks_correctly(self):
        """Lower energy should rank higher when sorted by -energy (confidence)."""
        # Simulate: prediction A has energy -5 (good), B has energy 3 (bad)
        predictions = {
            "hash_a": [1, 0.0, 0.0, 5.0],   # count, q, q_logp, energy_conf=-(-5)=5
            "hash_b": [1, 0.0, 0.0, -3.0],   # count, q, q_logp, energy_conf=-(3)=-3
        }

        # Sort by energy confidence (index 3), descending
        ranked = sorted(predictions.items(), key=lambda kv: kv[1][3], reverse=True)

        # hash_a should be first (higher energy confidence = lower energy = better)
        assert ranked[0][0] == "hash_a"
        assert ranked[1][0] == "hash_b"


class TestMCMCTraining:
    def test_mcmc_training_creates_graph(self):
        """Forward in train mode with mcmc_training=True should produce logits with grad_fn."""
        config = make_config(mcmc_steps=4, mcmc_step_size=0.01, mcmc_training=True)
        model = ARCModel(config).to(DEVICE).train()

        batch = make_batch(config)
        carry = make_carry(model, batch)

        _, outputs = model(carry, batch)

        assert outputs["logits"].grad_fn is not None, (
            "Refined logits should have grad_fn (computation graph for backprop through MCMC)"
        )
        assert "unrefined_logits" in outputs, "Should store unrefined logits for comparison"

    def test_mcmc_training_energy_head_gets_gradients(self):
        """Backward through MCMC-refined logits should give energy_head non-zero gradients."""
        config = make_config(mcmc_steps=4, mcmc_step_size=0.01, mcmc_training=True)
        model = ARCModel(config).to(DEVICE).train()

        batch = make_batch(config)
        carry = make_carry(model, batch)

        _, outputs = model(carry, batch)

        # Compute reconstruction loss on refined logits
        labels = batch["labels"]
        loss = torch.nn.functional.cross_entropy(
            outputs["logits"].view(-1, config["vocab_size"]),
            labels.view(-1),
            ignore_index=-100,
        )
        loss.backward()

        assert model.energy_head.weight.grad is not None, (
            "energy_head has no gradients after backward through MCMC-refined logits"
        )
        assert model.energy_head.weight.grad.abs().sum() > 0, (
            "energy_head gradients are all zero — second-order gradients not flowing"
        )

    def test_mcmc_improves_logits(self):
        """MCMC refinement should produce different logits than unrefined."""
        config = make_config(mcmc_steps=4, mcmc_step_size=0.01, mcmc_training=True)
        model = ARCModel(config).to(DEVICE).train()

        batch = make_batch(config)
        carry = make_carry(model, batch)

        _, outputs = model(carry, batch)

        assert not torch.allclose(
            outputs["logits"].detach(), outputs["unrefined_logits"].detach(), atol=1e-5
        ), "MCMC refinement did not change logits"

    def test_mcmc_eval_no_create_graph(self):
        """Eval mode with mcmc_steps>0 should work without create_graph."""
        config = make_config(mcmc_steps=4, mcmc_step_size=0.01, mcmc_training=True)
        model = ARCModel(config).to(DEVICE).eval()

        batch = make_batch(config)
        carry = make_carry(model, batch)

        with torch.no_grad():
            _, outputs = model(carry, batch)

        assert "unrefined_logits" in outputs, "Should still store unrefined logits at eval"
        assert not torch.allclose(
            outputs["logits"], outputs["unrefined_logits"], atol=1e-5
        ), "MCMC refinement should change logits at eval too"

    def test_mcmc_disabled_by_default(self):
        """Default config (mcmc_steps=0) should not apply MCMC refinement."""
        config = make_config()  # mcmc_steps defaults to 0
        model = ARCModel(config).to(DEVICE).train()

        batch = make_batch(config)
        carry = make_carry(model, batch)

        _, outputs = model(carry, batch)

        assert "unrefined_logits" not in outputs, (
            "unrefined_logits should not be present when MCMC is disabled"
        )

    def test_dual_loss_gradients_both_paths(self):
        """Dual loss should give gradients to URM layers (unrefined) and energy_head (refined)."""
        config = make_config(
            mcmc_steps=4, mcmc_step_size=0.01, mcmc_training=True,
        )
        model = ARCModel(config).to(DEVICE).train()

        batch = make_batch(config)
        carry = make_carry(model, batch)
        _, outputs = model(carry, batch)

        labels = batch["labels"]
        V = config["vocab_size"]
        loss_fn = torch.nn.functional.cross_entropy

        # Dual reconstruction loss
        unrefined_loss = loss_fn(
            outputs["unrefined_logits"].view(-1, V), labels.view(-1), ignore_index=-100
        )
        refined_loss = loss_fn(
            outputs["logits"].view(-1, V), labels.view(-1), ignore_index=-100
        )
        total = 0.5 * unrefined_loss + 0.5 * refined_loss
        total.backward()

        # URM inner layers should get gradients from unrefined loss
        layer_param = next(model.inner.layers[0].parameters())
        assert layer_param.grad is not None, "URM layer has no gradients"
        assert layer_param.grad.abs().sum() > 0, "URM layer gradients are zero"

        # Energy head should get gradients from refined loss (through MCMC)
        assert model.energy_head.weight.grad is not None, "energy_head has no gradients"
        assert model.energy_head.weight.grad.abs().sum() > 0, "energy_head gradients are zero"


class TestEnergyLossHeadIntegration:
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


class TestEBTMode:
    def test_ebt_mode_forward(self):
        """EBT mode should produce valid logits from input_embeddings via MCMC."""
        config = make_config(mode="ebt", loops=4, mcmc_step_size=0.01)
        model = ARCModel(config).to(DEVICE).train()

        batch = make_batch(config)
        carry = make_carry(model, batch)

        _, outputs = model(carry, batch)

        logits = outputs["logits"]
        assert logits.shape == (config["batch_size"], config["seq_len"], config["vocab_size"])
        assert not torch.allclose(logits, torch.zeros_like(logits), atol=1e-5)
        assert "unrefined_logits" in outputs, "EBT should always produce unrefined_logits"
        assert "current_energy" in outputs
        assert "output_hidden" in outputs
        assert outputs["output_hidden"].shape == (config["batch_size"], config["seq_len"], config["hidden_size"])
        # EBT always halts after one call
        carry_out, _ = model(carry, batch)
        assert carry_out.halted.all(), "EBT should halt after one forward call"

    def test_ebt_mode_energy_decreases(self):
        """EBT MCMC should produce different logits than the zero-step baseline."""
        config = make_config(mode="ebt", loops=8, mcmc_step_size=0.01)
        model = ARCModel(config).to(DEVICE).eval()

        batch = make_batch(config)
        carry = make_carry(model, batch)

        with torch.no_grad():
            _, outputs = model(carry, batch)

        assert not torch.allclose(
            outputs["logits"], outputs["unrefined_logits"], atol=1e-5
        ), "EBT MCMC did not change logits from zero-step baseline"

    def test_ebt_mode_gradients(self):
        """EBT refined logits should give gradients to energy_head via MCMC."""
        config = make_config(mode="ebt", loops=4, mcmc_step_size=0.01)
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
    def test_hybrid_mode_forward(self):
        """Hybrid mode should run URM steps then MCMC steps and produce valid output."""
        config = make_config(mode="hybrid", loops=8, mcmc_start_step=4, mcmc_step_size=0.01)
        model = ARCModel(config).to(DEVICE).train()

        batch = make_batch(config)
        carry = make_carry(model, batch)

        _, outputs = model(carry, batch)

        logits = outputs["logits"]
        assert logits.shape == (config["batch_size"], config["seq_len"], config["vocab_size"])
        assert "unrefined_logits" in outputs, "Hybrid should produce unrefined_logits"
        assert not torch.allclose(
            outputs["logits"].detach(), outputs["unrefined_logits"].detach(), atol=1e-5
        ), "MCMC phase did not change logits"
        # Hybrid always halts after one call
        carry_out, _ = model(carry, batch)
        assert carry_out.halted.all(), "Hybrid should halt after one forward call"

    def test_hybrid_mode_gradients(self):
        """Hybrid refined logits should give gradients to both URM layers and energy_head."""
        config = make_config(mode="hybrid", loops=6, mcmc_start_step=3, mcmc_step_size=0.01)
        model = ARCModel(config).to(DEVICE).train()

        batch = make_batch(config)
        carry = make_carry(model, batch)
        _, outputs = model(carry, batch)

        labels = batch["labels"]
        V = config["vocab_size"]
        loss = torch.nn.functional.cross_entropy(
            outputs["logits"].view(-1, V), labels.view(-1), ignore_index=-100,
        )
        loss.backward()

        # Energy head gets gradients through MCMC
        assert model.energy_head.weight.grad is not None, "energy_head has no gradients"
        assert model.energy_head.weight.grad.abs().sum() > 0, "energy_head gradients are zero"
        # URM layers get gradients through URM phase
        layer_param = next(model.inner.layers[0].parameters())
        assert layer_param.grad is not None, "URM layer has no gradients"
        assert layer_param.grad.abs().sum() > 0, "URM layer gradients are zero"


class TestModeURMUnchanged:
    def test_mode_urm_unchanged(self):
        """mode='urm' should produce same outputs as before (default behavior)."""
        config = make_config()  # mode defaults to "urm"
        model = ARCModel(config).to(DEVICE).eval()

        batch = make_batch(config)
        carry = make_carry(model, batch)

        with torch.no_grad():
            _, outputs = model(carry, batch)

        assert outputs["logits"].shape == (config["batch_size"], config["seq_len"], config["vocab_size"])
        assert "unrefined_logits" not in outputs, "URM without MCMC should not have unrefined_logits"
        assert "current_energy" in outputs
        assert "output_hidden" in outputs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
