"""
Test that MCMC refinement produces real gradients and updates embeddings
at inference time (model.eval() + torch.no_grad()).
"""
import sys
import types
import torch
import torch.nn.functional as F


# Provide a CPU-compatible flash_attn mock before importing the model.
# flash_attn expects (q, k, v) in [B, S, H, D] layout and returns the same.
def _mock_flash_attn_func(q, k, v, causal=False, **kwargs):
    # Transpose to [B, H, S, D] for scaled_dot_product_attention
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return out.transpose(1, 2)  # Back to [B, S, H, D]


for mod_name in ("flash_attn_interface", "flash_attn"):
    mod = types.ModuleType(mod_name)
    mod.flash_attn_func = _mock_flash_attn_func
    sys.modules[mod_name] = mod

import pytest
from models.urm.urm_energy import URM_Energy


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
        forward_dtype="float32",
        energy_threshold=1e-6,  # Very small so we hit min_steps
        langevin_noise_std=0.01,
    )


def make_batch(config, device="cpu"):
    B = config["batch_size"]
    S = config["seq_len"]
    return {
        "inputs": torch.randint(0, config["vocab_size"], (B, S), device=device),
        "labels": torch.randint(0, config["vocab_size"], (B, S), device=device),
        "puzzle_identifiers": torch.zeros(B, dtype=torch.long, device=device),
    }


class TestMCMCInference:
    def test_eval_mode_embeddings_change(self):
        """MCMC must actually update predicted_embeddings when model is in eval mode."""
        config = make_config()
        model = URM_Energy(config)
        model.eval()

        batch = make_batch(config)
        carry = model.initial_carry(batch)
        initial_embeddings = carry.predicted_embeddings.clone()

        with torch.no_grad():
            carry_out, outputs = model(carry, batch)

        final_embeddings = carry_out.predicted_embeddings
        assert not torch.allclose(initial_embeddings, final_embeddings, atol=1e-7), (
            "Predicted embeddings did not change during eval-mode MCMC — "
            "gradients are likely still zeroed out"
        )

    def test_eval_mode_energies_nonzero(self):
        """Energy values must be computed (not all zeros) during eval."""
        config = make_config()
        model = URM_Energy(config)
        model.eval()

        batch = make_batch(config)
        carry = model.initial_carry(batch)

        with torch.no_grad():
            _, outputs = model(carry, batch)

        energies = outputs["energies_trajectory"]
        assert len(energies) > 0, "No energies recorded"
        all_energies = torch.stack(energies)
        assert not torch.allclose(all_energies, torch.zeros_like(all_energies), atol=1e-10), (
            "All energy values are zero — energy head is not computing"
        )

    def test_eval_mode_min_steps(self):
        """At least min_steps (8) MCMC iterations must occur."""
        config = make_config()
        model = URM_Energy(config)
        model.eval()

        batch = make_batch(config)
        carry = model.initial_carry(batch)

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
        model = URM_Energy(config)
        model.train()

        batch = make_batch(config)
        carry = model.initial_carry(batch)
        _, outputs = model(carry, batch)

        final_energy = outputs["final_energy"]
        assert final_energy.grad_fn is not None, (
            "Final energy has no grad_fn in train mode — "
            "create_graph=True may not be working"
        )

    def test_eval_logits_differ_from_initial(self):
        """The final logits should differ from what the initial random embeddings produce."""
        config = make_config()
        model = URM_Energy(config)
        model.eval()

        batch = make_batch(config)
        carry = model.initial_carry(batch)

        initial_logits = model.embeddings_to_logits(carry.predicted_embeddings)

        with torch.no_grad():
            _, outputs = model(carry, batch)

        final_logits = outputs["logits"]
        assert not torch.allclose(initial_logits, final_logits, atol=1e-7), (
            "Final logits are identical to initial — MCMC did not refine predictions"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
