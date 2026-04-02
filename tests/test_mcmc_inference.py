"""
Test that MCMC refinement produces real gradients and updates embeddings
at inference time (model.eval() + torch.no_grad()).

Requires CUDA — runs real flash_attn and model code on GPU.
"""
import torch
import pytest
from models.urm.urm_energy import URM_Energy

DEVICE = "cuda"


def make_config():
    return dict(
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
        energy_threshold=1e-6,  # Very small so we hit min_steps
        langevin_noise_std=0.01,
    )


def make_batch(config):
    B = config["batch_size"]
    S = config["seq_len"]
    return {
        "inputs": torch.randint(0, config["vocab_size"], (B, S), device=DEVICE),
        "labels": torch.randint(0, config["vocab_size"], (B, S), device=DEVICE),
        "puzzle_identifiers": torch.zeros(B, dtype=torch.long, device=DEVICE),
    }


def make_carry(model, batch):
    """Create initial carry with torch.device context, matching pretrain.py pattern."""
    with torch.device(DEVICE):
        return model.initial_carry(batch)


@pytest.fixture(autouse=True)
def skip_without_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


class TestMCMCInference:
    def test_eval_mode_embeddings_change(self):
        """MCMC must actually update predicted_embeddings when model is in eval mode."""
        config = make_config()
        model = URM_Energy(config).to(DEVICE).eval()

        batch = make_batch(config)
        carry = make_carry(model, batch)
        initial_embeddings = carry.predicted_embeddings.clone()

        with torch.no_grad():
            carry_out, outputs = model(carry, batch)

        final_embeddings = carry_out.predicted_embeddings
        assert not torch.allclose(initial_embeddings, final_embeddings, atol=1e-5), (
            "Predicted embeddings did not change during eval-mode MCMC — "
            "gradients are likely still zeroed out"
        )

    def test_eval_mode_energies_nonzero(self):
        """Energy values must be computed (not all zeros) during eval."""
        config = make_config()
        model = URM_Energy(config).to(DEVICE).eval()

        batch = make_batch(config)
        carry = make_carry(model, batch)

        with torch.no_grad():
            _, outputs = model(carry, batch)

        energies = outputs["energies_trajectory"]
        assert len(energies) > 0, "No energies recorded"
        all_energies = torch.stack(energies)
        assert not torch.allclose(all_energies, torch.zeros_like(all_energies), atol=1e-8), (
            "All energy values are zero — energy head is not computing"
        )

    def test_eval_mode_min_steps(self):
        """At least min_steps (8) MCMC iterations must occur."""
        config = make_config()
        model = URM_Energy(config).to(DEVICE).eval()

        batch = make_batch(config)
        carry = make_carry(model, batch)

        with torch.no_grad():
            _, outputs = model(carry, batch)

        num_steps = len(outputs["energies_trajectory"])
        # min_steps = 8, so convergence check can only trigger after step index 8
        # meaning we need at least 9 entries (steps 0..8) before early stop is possible
        assert num_steps >= 9, (
            f"Only {num_steps} MCMC steps recorded, expected at least 9 "
            f"(min_steps=8 means convergence check first possible at step 9)"
        )

    def test_train_mode_gradients_flow(self):
        """In train mode, energy tensors must have grad_fn (autograd graph attached)."""
        config = make_config()
        model = URM_Energy(config).to(DEVICE).train()

        batch = make_batch(config)
        carry = make_carry(model, batch)
        _, outputs = model(carry, batch)

        final_energy = outputs["final_energy"]
        assert final_energy.grad_fn is not None, (
            "Final energy has no grad_fn in train mode — "
            "create_graph=True may not be working"
        )

    def test_eval_logits_differ_from_initial(self):
        """The final logits should differ from what the initial random embeddings produce."""
        config = make_config()
        model = URM_Energy(config).to(DEVICE).eval()

        batch = make_batch(config)
        carry = make_carry(model, batch)

        initial_logits = model.embeddings_to_logits(carry.predicted_embeddings)

        with torch.no_grad():
            _, outputs = model(carry, batch)

        final_logits = outputs["logits"]
        assert not torch.allclose(initial_logits, final_logits, atol=1e-5), (
            "Final logits are identical to initial — MCMC did not refine predictions"
        )


class TestAlphaStepSize:
    def _single_step_update_norm(self, model, carry, batch, alpha_val):
        """Run one MCMC step with a given alpha and return the embedding update norm."""
        model.alpha.data.fill_(alpha_val)
        initial_emb = carry.predicted_embeddings.clone()
        new_carry, _ = model.mcmc_update(carry, batch, training=True)
        delta = new_carry.predicted_embeddings.float() - initial_emb.float()
        return delta.norm().item()

    def test_alpha_controls_step_size(self):
        """Update norm must scale proportionally with alpha (no hidden hard-coded scaling)."""
        config = make_config()
        # Disable Langevin noise so it doesn't obscure the alpha signal
        config["langevin_noise_std"] = 0.0
        model = URM_Energy(config).to(DEVICE).train()

        torch.manual_seed(42)
        batch = make_batch(config)
        carry = make_carry(model, batch)

        norm_small = self._single_step_update_norm(model, carry, batch, alpha_val=0.1)
        norm_large = self._single_step_update_norm(model, carry, batch, alpha_val=0.5)

        # With unit-normalized gradients, update = alpha * unit_grad, so norms
        # should be proportional to alpha. Allow 20% tolerance for numerical noise.
        ratio = norm_large / max(norm_small, 1e-12)
        expected_ratio = 0.5 / 0.1  # = 5.0
        assert abs(ratio - expected_ratio) / expected_ratio < 0.2, (
            f"Alpha ratio test failed: norm(alpha=0.5)/norm(alpha=0.1) = {ratio:.3f}, "
            f"expected ~{expected_ratio:.1f}. Step size is not proportional to alpha."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
