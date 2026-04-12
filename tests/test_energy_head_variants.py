"""Tests for position-aware energy head variants (R2c/d/e)."""
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


class TestEnergyHeadVariants:
    @pytest.mark.parametrize("head_type", [
        "linear", "position_mlp", "position_conv_mlp", "position_attn_mlp",
    ])
    def test_energy_output_shape(self, head_type):
        """All energy head types produce [B] shaped output."""
        config = make_config(energy_head_type=head_type, energy_head_hidden=16)
        model = ARCModel(config).to(DEVICE).eval()
        batch = make_batch(config)

        with torch.no_grad():
            _, _, all_hidden, input_emb = model.forward_trajectory(batch)
            P = model.inner.puzzle_emb_len
            energy = model.compute_joint_energy(input_emb, all_hidden[-1][:, P:])

        assert energy.shape == (config["batch_size"],)
        assert torch.isfinite(energy).all()

    @pytest.mark.parametrize("head_type", [
        "linear", "position_mlp", "position_conv_mlp", "position_attn_mlp",
    ])
    def test_gradient_flow(self, head_type):
        """Gradients flow through all energy head variants."""
        config = make_config(energy_head_type=head_type, energy_head_hidden=16)
        model = ARCModel(config).to(DEVICE).train()
        batch = make_batch(config)

        _, _, all_hidden, input_emb = model.forward_trajectory(batch)
        P = model.inner.puzzle_emb_len
        energy = model.compute_joint_energy(input_emb, all_hidden[-1][:, P:])
        energy.sum().backward()

        # Energy head params should have gradients
        has_grad = False
        for n, p in model.named_parameters():
            if "energy_head" in n and p.grad is not None:
                if p.grad.abs().sum() > 0:
                    has_grad = True
                    break
        assert has_grad, f"No gradients in energy_head for type={head_type}"

    def test_linear_backward_compat(self):
        """'linear' energy head produces same architecture as old code."""
        config = make_config(energy_head_type="linear")
        model = ARCModel(config).to(DEVICE)

        # Should be a plain nn.Linear, not PositionEnergyHead
        assert isinstance(model.energy_head, torch.nn.Linear)
        assert model.energy_head.in_features == 64
        assert model.energy_head.out_features == 1

    def test_position_mlp_is_position_energy_head(self):
        """Non-linear types use PositionEnergyHead."""
        from models.urm.urm_energy import PositionEnergyHead

        config = make_config(energy_head_type="position_mlp", energy_head_hidden=16)
        model = ARCModel(config).to(DEVICE)
        assert isinstance(model.energy_head, PositionEnergyHead)

    def test_spatial_sensitivity(self):
        """Position-aware head should produce different energy when specific positions change."""
        config = make_config(energy_head_type="position_mlp", energy_head_hidden=16)
        model = ARCModel(config).to(DEVICE).eval()

        B, S, H = 2, 100, 64
        P = model.inner.puzzle_emb_len

        with torch.no_grad():
            input_emb = torch.randn(B, S + P, H, device=DEVICE, dtype=torch.bfloat16)
            hidden_a = torch.randn(B, S, H, device=DEVICE, dtype=torch.bfloat16)
            hidden_b = hidden_a.clone()
            # Change only a few positions
            hidden_b[:, 10:15, :] += 1.0

            energy_a = model.compute_joint_energy(input_emb, hidden_a)
            energy_b = model.compute_joint_energy(input_emb, hidden_b)

        # Energies should differ when positions change
        assert not torch.allclose(energy_a, energy_b, atol=1e-3), \
            "Position-aware head should be sensitive to position-level changes"


class TestEnergyHeadParamCount:
    def test_param_counts_reasonable(self):
        """Verify approximate parameter counts for each variant."""
        counts = {}
        for head_type in ["linear", "position_mlp", "position_conv_mlp", "position_attn_mlp"]:
            config = make_config(energy_head_type=head_type, energy_head_hidden=32)
            model = ARCModel(config).to(DEVICE)
            eh_params = sum(p.numel() for n, p in model.named_parameters() if "energy_head" in n)
            counts[head_type] = eh_params

        # linear: 64*1 + 1 = 65
        assert counts["linear"] == 65
        # position_mlp: 64*32 + 32 + 32*1 + 1 = 2081
        assert counts["position_mlp"] > 2000
        # conv adds depthwise conv params
        assert counts["position_conv_mlp"] > counts["position_mlp"]
        # attn adds full attention params
        assert counts["position_attn_mlp"] > counts["position_conv_mlp"]

        print(f"Energy head param counts: {counts}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
