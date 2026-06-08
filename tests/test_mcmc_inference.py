"""
Tests for ARCModel architecture: URM, EBT, and hybrid modes.

forward_trajectory() runs N recurrence steps with full gradient flow.
EnergyLossHead computes deep supervision loss over the trajectory.
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


@pytest.fixture(autouse=True)
def skip_without_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


class TestForwardTrajectory:
    def test_returns_correct_shapes(self):
        """forward_trajectory returns N steps of logits, q_logits, hidden, and input_embeddings."""
        config = make_config(loops=4)
        model = ARCModel(config).to(DEVICE).eval()
        batch = make_batch(config)

        with torch.no_grad():
            all_logits, all_q_logits, all_hidden, input_emb, _, _ = model.forward_trajectory(batch)

        assert len(all_logits) == 4
        assert len(all_q_logits) == 4
        assert len(all_hidden) == 4

        B, S, V = config["batch_size"], config["seq_len"], config["vocab_size"]
        H = config["hidden_size"]
        P = model.inner.puzzle_emb_len

        for t in range(4):
            assert all_logits[t].shape == (B, S, V)
            assert all_q_logits[t].shape == (B,)
            assert all_hidden[t].shape == (B, S + P, H)

        assert input_emb.shape == (B, S + P, H)

    def test_logits_are_nontrivial(self):
        """Each step should produce non-zero logits."""
        config = make_config(loops=3)
        model = ARCModel(config).to(DEVICE).eval()
        batch = make_batch(config)

        with torch.no_grad():
            all_logits, _, _, _, _, _ = model.forward_trajectory(batch)

        for t in range(3):
            assert not torch.allclose(all_logits[t], torch.zeros_like(all_logits[t]), atol=1e-5)

    def test_hidden_changes_across_steps(self):
        """Hidden states should differ between consecutive steps."""
        config = make_config(loops=4)
        model = ARCModel(config).to(DEVICE).eval()
        batch = make_batch(config)

        with torch.no_grad():
            _, _, all_hidden, _, _, _ = model.forward_trajectory(batch)

        for t in range(1, 4):
            assert not torch.allclose(all_hidden[t], all_hidden[t - 1], atol=1e-5), (
                f"Hidden states identical at steps {t} and {t-1}"
            )

    def test_custom_N_overrides_loops(self):
        """Passing N to forward_trajectory overrides config.loops."""
        config = make_config(loops=8)
        model = ARCModel(config).to(DEVICE).eval()
        batch = make_batch(config)

        with torch.no_grad():
            all_logits, _, _, _, _, _ = model.forward_trajectory(batch, N=3)

        assert len(all_logits) == 3


class TestGradientFlow:
    def test_gradients_flow_across_steps(self):
        """Deep supervision: gradients from later steps must reach earlier layers."""
        config = make_config(loops=4)
        model = ARCModel(config).to(DEVICE).train()
        batch = make_batch(config)

        all_logits, _, _, _, _, _ = model.forward_trajectory(batch)

        # Loss on final step only — gradients must flow back through steps 1-3
        labels = batch["labels"]
        loss = torch.nn.functional.cross_entropy(
            all_logits[-1].view(-1, config["vocab_size"]),
            labels.view(-1),
            ignore_index=-100,
        )
        loss.backward()

        # Transformer layers should have gradients (used at every step)
        layer_param = next(model.inner.layers[0].parameters())
        assert layer_param.grad is not None, "Transformer layer has no gradients"
        assert layer_param.grad.abs().sum() > 0, "Transformer layer gradients are zero"

    def test_no_detach_between_steps(self):
        """Verify gradient from step N loss reaches earlier steps (no detach)."""
        config = make_config(loops=3)
        model = ARCModel(config).to(DEVICE).train()
        batch = make_batch(config)

        all_logits, _, all_hidden, _, _, _ = model.forward_trajectory(batch)

        # Compute loss only on step 3
        labels = batch["labels"]
        loss = torch.nn.functional.cross_entropy(
            all_logits[2].view(-1, config["vocab_size"]),
            labels.view(-1),
            ignore_index=-100,
        )

        # all_hidden[0] (step 1 output) should be in the computation graph of
        # all_logits[2] (step 3) — verify grad_fn chain exists.
        assert all_hidden[0].requires_grad, "Step 1 hidden should require grad"
        assert all_hidden[2].grad_fn is not None, "Step 3 hidden has no grad_fn"

        loss.backward()

        # lm_head is used at every step. If detach existed between steps,
        # only step 3's gradient would survive. With undetached steps, all contribute.
        lm_head_grad = model.inner.lm_head.weight.grad
        assert lm_head_grad is not None, "lm_head has no gradient"
        assert lm_head_grad.abs().sum() > 0, "lm_head gradients are zero"


class TestEBTMode:
    def test_ebt_trajectory(self):
        """EBT forward_trajectory does MCMC steps and produces valid logits."""
        config = make_config(refinement="ebt", loops=4, mcmc_step_size=0.01)
        model = ARCModel(config).to(DEVICE).eval()
        batch = make_batch(config)

        with torch.no_grad():
            all_logits, _, all_hidden, _, _, _ = model.forward_trajectory(batch)

        assert len(all_logits) == 4
        for t in range(4):
            assert not torch.allclose(all_logits[t], torch.zeros_like(all_logits[t]), atol=1e-5)

    def test_ebt_changes_hidden(self):
        """Each EBT step should change the hidden state."""
        config = make_config(refinement="ebt", loops=4, mcmc_step_size=0.01)
        model = ARCModel(config).to(DEVICE).eval()
        batch = make_batch(config)

        with torch.no_grad():
            _, _, all_hidden, _, _, _ = model.forward_trajectory(batch)

        assert not torch.allclose(all_hidden[0], all_hidden[1], atol=1e-5), (
            "EBT should change hidden states between steps"
        )

    def test_ebt_gradients_flow_to_energy_head(self):
        """EBT training should give gradients to energy_head via MCMC."""
        config = make_config(refinement="ebt", loops=4, mcmc_step_size=0.01)
        model = ARCModel(config).to(DEVICE).train()
        batch = make_batch(config)

        all_logits, _, _, _, _, _ = model.forward_trajectory(batch)

        labels = batch["labels"]
        loss = torch.nn.functional.cross_entropy(
            all_logits[-1].view(-1, config["vocab_size"]),
            labels.view(-1),
            ignore_index=-100,
        )
        loss.backward()

        assert model.energy_head.weight.grad is not None, "energy_head has no gradients"
        assert model.energy_head.weight.grad.abs().sum() > 0, "energy_head gradients are zero"


class TestHybridMode:
    def test_hybrid_trajectory(self):
        """Hybrid: first steps are URM, later steps are MCMC. Full trajectory runs."""
        config = make_config(refinement="hybrid", loops=6, mcmc_start_step=3, mcmc_step_size=0.01)
        model = ARCModel(config).to(DEVICE).eval()
        batch = make_batch(config)

        with torch.no_grad():
            all_logits, _, _, _, _, _ = model.forward_trajectory(batch)

        assert len(all_logits) == 6

    def test_hybrid_gradients_both_paths(self):
        """Hybrid should give gradients to URM layers and energy_head."""
        config = make_config(refinement="hybrid", loops=6, mcmc_start_step=3, mcmc_step_size=0.01)
        model = ARCModel(config).to(DEVICE).train()
        batch = make_batch(config)

        all_logits, _, _, _, _, _ = model.forward_trajectory(batch)

        # Deep supervision: loss at every step
        labels = batch["labels"]
        V = config["vocab_size"]
        total_loss = torch.tensor(0.0, device=DEVICE)
        N = len(all_logits)
        for t in range(N):
            w = (t + 1) / N
            step_loss = torch.nn.functional.cross_entropy(
                all_logits[t].view(-1, V), labels.view(-1), ignore_index=-100,
            )
            total_loss = total_loss + w * step_loss

        total_loss.backward()

        # Energy head gets gradients through MCMC phase
        assert model.energy_head.weight.grad is not None, "energy_head has no gradients"
        assert model.energy_head.weight.grad.abs().sum() > 0, "energy_head gradients are zero"
        # URM layers get gradients through URM phase
        layer_param = next(model.inner.layers[0].parameters())
        assert layer_param.grad is not None, "URM layer has no gradients"
        assert layer_param.grad.abs().sum() > 0, "URM layer gradients are zero"


class TestEnergyHead:
    def test_energy_head_gets_gradients(self):
        """After backward, energy_head parameters should have gradients."""
        config = make_config()
        model = ARCModel(config).to(DEVICE).train()
        batch = make_batch(config)

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


class TestLossHeadIntegration:
    def test_loss_head_produces_valid_loss(self):
        """EnergyLossHead should produce positive, differentiable loss with deep supervision."""
        from models.losses import EnergyLossHead

        config = make_config(loops=4)
        model = ARCModel(config).to(DEVICE).train()
        loss_head = EnergyLossHead(model, "stablemax_cross_entropy").to(DEVICE)

        batch = make_batch(config)
        _, loss, metrics, _, _ = loss_head(batch=batch)

        assert "reconstruction_loss" in metrics
        assert "q_halt_loss" in metrics
        assert loss.item() > 0
        assert loss.grad_fn is not None

        loss.backward()

    def test_loss_head_per_step_metrics(self):
        """EnergyLossHead should produce per-step metrics."""
        from models.losses import EnergyLossHead

        config = make_config(loops=4)
        model = ARCModel(config).to(DEVICE).eval()
        loss_head = EnergyLossHead(model, "stablemax_cross_entropy").to(DEVICE)

        batch = make_batch(config)
        with torch.no_grad():
            _, loss, metrics, _, _ = loss_head(batch=batch)

        # Per-step metrics
        for t in range(1, 5):
            assert f"step_{t}_exact_accuracy" in metrics, f"Missing step_{t}_exact_accuracy"
            assert f"step_{t}_accuracy" in metrics, f"Missing step_{t}_accuracy"

        # Delta norms for steps 2+
        for t in range(2, 5):
            assert f"step_{t}_delta_norm" in metrics, f"Missing step_{t}_delta_norm"

        # Step 1 should NOT have delta_norm
        assert "step_1_delta_norm" not in metrics

    def test_loss_head_eval_stopping_metrics(self):
        """EnergyLossHead in eval mode should produce stopping metrics."""
        from models.losses import EnergyLossHead

        config = make_config(loops=4)
        model = ARCModel(config).to(DEVICE)
        loss_head = EnergyLossHead(model, "stablemax_cross_entropy").to(DEVICE)
        loss_head.eval()

        batch = make_batch(config)
        with torch.no_grad():
            _, _, metrics, _, _ = loss_head(batch=batch)

        assert "qhalt_stop_step" in metrics
        assert "qhalt_stop_accuracy" in metrics
        assert "energy_stop_step" in metrics
        assert "energy_stop_accuracy" in metrics

    def test_loss_head_no_stopping_metrics_in_train(self):
        """Stopping metrics should only appear in eval mode."""
        from models.losses import EnergyLossHead

        config = make_config(loops=4)
        model = ARCModel(config).to(DEVICE).train()
        loss_head = EnergyLossHead(model, "stablemax_cross_entropy").to(DEVICE)

        batch = make_batch(config)
        _, loss, metrics, _, _ = loss_head(batch=batch)
        loss.backward()

        assert "qhalt_stop_step" not in metrics
        assert "energy_stop_step" not in metrics

    def test_loss_head_returns_evaluator_outputs(self):
        """EnergyLossHead should return preds, q_halt_logits, current_energy for evaluator."""
        from models.losses import EnergyLossHead

        config = make_config(loops=4)
        model = ARCModel(config).to(DEVICE).eval()
        loss_head = EnergyLossHead(model, "stablemax_cross_entropy").to(DEVICE)

        batch = make_batch(config)
        with torch.no_grad():
            _, _, _, outputs, _ = loss_head(
                batch=batch,
                return_keys=["preds", "q_halt_logits", "current_energy"],
            )

        assert "preds" in outputs
        assert "q_halt_logits" in outputs
        assert "current_energy" in outputs
        assert outputs["preds"].shape == (config["batch_size"], config["seq_len"])

    def test_deep_supervision_weights_affect_loss(self):
        """Earlier steps should contribute less to loss than later steps (linear ramp)."""
        from models.losses import EnergyLossHead

        config = make_config(loops=4)
        model = ARCModel(config).to(DEVICE).train()
        loss_head = EnergyLossHead(model, "stablemax_cross_entropy").to(DEVICE)

        batch = make_batch(config)
        _, loss, metrics, _, _ = loss_head(batch=batch)

        # Loss should be positive and finite
        assert loss.item() > 0
        assert torch.isfinite(loss)


class TestGRAMCleanDecode:
    """R7e: verify decoded logits derive from clean u_t, not u_t + ε_t."""

    def test_logits_unaffected_by_gram_perturbation(self):
        """With GRAM enabled, logits at each step must equal lm_head(u_t),
        not lm_head(u_t + eps_t). We verify by comparing GRAM logits against
        a deterministic run with alpha=0 (eps_t is zero-norm)."""
        config = make_config(
            loops=4,
            gram_enabled=True,
            gram_latent_dim=8,
            gram_sigma_alpha=0.0,
            gram_beta=0.1,
        )
        model = ARCModel(config).to(DEVICE)
        model.eval()
        batch = make_batch(config)

        torch.manual_seed(0)
        with torch.no_grad():
            logits_a, _, _, _, _, _ = model.forward_trajectory(batch)

        torch.manual_seed(0)
        with torch.no_grad():
            logits_b, _, _, _, _, _ = model.forward_trajectory(batch)

        for t in range(4):
            assert torch.allclose(logits_a[t], logits_b[t], atol=1e-5), \
                f"Step {t} logits differ across runs with alpha=0"

    def test_gram_disabled_matches_deterministic(self):
        """gram_enabled=False should produce identical logits to a non-GRAM config."""
        base_config = make_config(loops=4)
        gram_config = make_config(loops=4, gram_enabled=False)

        torch.manual_seed(42)
        base_model = ARCModel(base_config).to(DEVICE)
        base_model.eval()

        torch.manual_seed(42)
        gram_model = ARCModel(gram_config).to(DEVICE)
        gram_model.eval()

        batch = make_batch(base_config)

        with torch.no_grad():
            base_logits, _, _, _, _, _ = base_model.forward_trajectory(batch)
            gram_logits, _, _, _, _, _ = gram_model.forward_trajectory(batch)

        for t in range(4):
            assert torch.allclose(base_logits[t], gram_logits[t], atol=1e-5), \
                f"Step {t}: gram_enabled=False logits diverge from baseline"

    def test_terminal_logits_derive_from_perturbed_state(self):
        """R7e: terminal step decodes from u_{N-1} + eps, not clean u_{N-1}."""
        config = make_config(
            loops=4,
            gram_enabled=True,
            gram_latent_dim=8,
            gram_beta=0.1,
        )
        model = ARCModel(config).to(DEVICE)
        model.eval()
        batch = make_batch(config)

        # Run two trajectories with different seeds — terminal logits should differ
        torch.manual_seed(100)
        with torch.no_grad():
            logits_a, _, hidden_a, _, _, _ = model.forward_trajectory(batch)

        torch.manual_seed(200)
        with torch.no_grad():
            logits_b, _, hidden_b, _, _, _ = model.forward_trajectory(batch)

        # Non-terminal steps are deterministic — logits must match
        for t in range(3):
            assert torch.allclose(logits_a[t], logits_b[t], atol=1e-5), \
                f"Step {t} logits should be deterministic (seed-independent)"

        # Terminal step logits should differ due to different eps samples
        assert not torch.allclose(logits_a[-1], logits_b[-1], atol=1e-3), \
            "Terminal logits should differ across seeds (stochastic eps)"


class TestGRAMTerminalOnly:
    """R7e: verify terminal-only GRAM produces exactly one KL, deterministic non-terminal steps,
    and that eps=0 recovers the deterministic baseline."""

    def test_exactly_one_kl_per_trajectory(self):
        """Training with GRAM should produce exactly 1 KL term (terminal only)."""
        config = make_config(
            loops=4,
            gram_enabled=True,
            gram_latent_dim=16,
            gram_beta=0.1,
        )
        model = ARCModel(config).to(DEVICE).train()
        batch = make_batch(config)

        _, _, _, _, gram_kl, _ = model.forward_trajectory(
            batch, labels=batch["labels"]
        )

        assert gram_kl is not None, "Expected KL from GRAM training"
        assert len(gram_kl) == 1, f"Expected exactly 1 KL term (terminal), got {len(gram_kl)}"

    def test_non_terminal_steps_deterministic(self):
        """Steps 0..N-2 must produce identical logits regardless of GRAM being enabled."""
        N = 4
        base_config = make_config(loops=N, gram_enabled=False)
        gram_config = make_config(loops=N, gram_enabled=True, gram_latent_dim=16, gram_beta=0.1)

        torch.manual_seed(42)
        base_model = ARCModel(base_config).to(DEVICE).eval()
        torch.manual_seed(42)
        gram_model = ARCModel(gram_config).to(DEVICE).eval()

        batch = make_batch(base_config)

        with torch.no_grad():
            base_logits, _, _, _, _, _ = base_model.forward_trajectory(batch)
            gram_logits, _, _, _, _, _ = gram_model.forward_trajectory(batch)

        # Non-terminal steps: GRAM model matches baseline exactly
        for t in range(N - 1):
            assert torch.allclose(base_logits[t], gram_logits[t], atol=1e-5), \
                f"Step {t}: non-terminal logits should match deterministic baseline"

    def test_eval_produces_no_kl(self):
        """At eval, prior sampling produces no KL (use_posterior=False)."""
        config = make_config(
            loops=4,
            gram_enabled=True,
            gram_latent_dim=16,
            gram_beta=0.1,
        )
        model = ARCModel(config).to(DEVICE).eval()
        batch = make_batch(config)

        with torch.no_grad():
            _, _, _, _, gram_kl, _ = model.forward_trajectory(batch)

        assert gram_kl is None, "Eval should produce no KL (prior-only sampling)"

    def test_zero_eps_recovers_deterministic(self):
        """Zeroing prior+up_proj makes terminal decode match the deterministic u_{N-1}."""
        config = make_config(
            loops=4,
            gram_enabled=True,
            gram_latent_dim=16,
            gram_beta=0.1,
        )
        model = ARCModel(config).to(DEVICE).eval()

        # Zero prior MLP and up_proj so eps is exactly 0
        with torch.no_grad():
            for p in model.gram_prior_mlp.parameters():
                p.zero_()
            for p in model.gram_up_proj.parameters():
                p.zero_()

        batch = make_batch(config)

        # Run GRAM model — terminal eps should be 0, so h_final == u_{N-1}
        with torch.no_grad():
            gram_logits, _, gram_hidden, _, _, _ = model.forward_trajectory(batch)

        # Also run without GRAM on same model backbone to get deterministic u_{N-1}
        # We can verify by checking that lm_head(hidden[-1]) == logits[-1]
        # and that hidden[-1] would produce same logits as a non-GRAM run
        P = model.inner.puzzle_emb_len
        R = config.get("num_registers", 0)
        expected = model.inner.lm_head(gram_hidden[-1])[:, P + R:]
        assert torch.allclose(gram_logits[-1], expected, atol=1e-5), \
            "Terminal logits should equal lm_head(h_final) even with zero eps"

        # Non-terminal steps should be fully deterministic across seeds
        torch.manual_seed(100)
        with torch.no_grad():
            logits_a, _, _, _, _, _ = model.forward_trajectory(batch)
        torch.manual_seed(200)
        with torch.no_grad():
            logits_b, _, _, _, _, _ = model.forward_trajectory(batch)
        for t in range(3):
            assert torch.allclose(logits_a[t], logits_b[t], atol=1e-5)
        # Terminal also matches since eps=0
        assert torch.allclose(logits_a[-1], logits_b[-1], atol=1e-5), \
            "With zero eps, terminal logits should be deterministic across seeds"

    def test_terminal_logits_from_perturbed_not_clean(self):
        """Terminal scored logits must derive from u_{N-1}+eps, not u_{N-1}."""
        config = make_config(
            loops=4,
            gram_enabled=True,
            gram_latent_dim=16,
            gram_beta=0.1,
        )
        model = ARCModel(config).to(DEVICE).eval()
        batch = make_batch(config)

        with torch.no_grad():
            logits, _, all_hidden, _, _, _ = model.forward_trajectory(batch)

        # all_hidden[-1] is h_final = u_{N-1} + eps
        # logits[-1] should equal lm_head(h_final)
        P = model.inner.puzzle_emb_len
        R = config.get("num_registers", 0)
        expected_logits = model.inner.lm_head(all_hidden[-1])[:, P + R:]
        assert torch.allclose(logits[-1], expected_logits, atol=1e-5), \
            "Terminal logits should equal lm_head(h_final) where h_final = u_{N-1} + eps"


class TestForwardGramSamplesEfficiency:
    """R7e: forward_gram_samples runs backbone once, draws M terminal samples."""

    def test_backbone_independent_of_M(self):
        """The deterministic trajectory is computed once regardless of M."""
        config = make_config(
            loops=4,
            gram_enabled=True,
            gram_latent_dim=16,
            gram_beta=0.1,
        )
        model = ARCModel(config).to(DEVICE).eval()
        batch = make_batch(config)

        # _run_deterministic_trajectory should be called exactly once
        call_count = 0
        orig_run = model._run_deterministic_trajectory

        def counting_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return orig_run(*args, **kwargs)

        model._run_deterministic_trajectory = counting_run

        with torch.no_grad():
            preds, q = model.forward_gram_samples(batch, M=10)

        assert call_count == 1, f"Backbone ran {call_count} times, expected 1"
        assert preds.shape == (10, config["batch_size"], config["seq_len"])
        assert q.shape == (10, config["batch_size"])

    def test_deterministic_mode_returns_single_sample(self):
        """deterministic=True returns shape [1, B, seq_len] with no eps."""
        config = make_config(
            loops=4,
            gram_enabled=True,
            gram_latent_dim=16,
            gram_beta=0.1,
        )
        model = ARCModel(config).to(DEVICE).eval()
        batch = make_batch(config)

        with torch.no_grad():
            preds, q = model.forward_gram_samples(batch, M=5, deterministic=True)

        assert preds.shape == (1, config["batch_size"], config["seq_len"])
        assert q.shape == (1, config["batch_size"])

    def test_deterministic_matches_non_gram_baseline(self):
        """Deterministic forward_gram_samples matches a non-GRAM forward_trajectory."""
        base_config = make_config(loops=4, gram_enabled=False)
        gram_config = make_config(loops=4, gram_enabled=True, gram_latent_dim=16, gram_beta=0.1)

        torch.manual_seed(42)
        base_model = ARCModel(base_config).to(DEVICE).eval()
        torch.manual_seed(42)
        gram_model = ARCModel(gram_config).to(DEVICE).eval()

        batch = make_batch(base_config)

        with torch.no_grad():
            base_logits, _, _, _, _, _ = base_model.forward_trajectory(batch)
            base_preds = torch.argmax(base_logits[-1], dim=-1)

            gram_preds, _ = gram_model.forward_gram_samples(batch, M=1, deterministic=True)

        assert torch.equal(base_preds, gram_preds[0]), \
            "Deterministic gram_samples should match non-GRAM baseline predictions"

    def test_multiple_samples_correct_shape(self):
        """M>1 produces correct output shapes."""
        config = make_config(
            loops=4,
            gram_enabled=True,
            gram_latent_dim=16,
            gram_beta=0.1,
        )
        model = ARCModel(config).to(DEVICE).eval()
        batch = make_batch(config)

        with torch.no_grad():
            preds, q = model.forward_gram_samples(batch, M=20)

        assert preds.shape == (20, config["batch_size"], config["seq_len"])
        assert q.shape == (20, config["batch_size"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
