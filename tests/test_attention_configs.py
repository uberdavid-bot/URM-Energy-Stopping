"""Smoke tests for R6 attention architecture experiments.

Each test instantiates the model with an R6 config, runs forward_trajectory
with N=2 steps, checks output shapes, and verifies gradients flow.
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
        num_layers=1,
        hidden_size=128,
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


def _run_smoke(config_overrides, N=2):
    config = make_config(**config_overrides)
    model = ARCModel(config).to(DEVICE)

    model.train()
    batch = make_batch(config)
    all_logits, all_q_logits, all_hidden, input_emb = model.forward_trajectory(batch, N=N)

    B = config["batch_size"]
    S = config["seq_len"]
    V = config["vocab_size"]
    R = config.get("num_registers", 0)

    assert len(all_logits) == N
    assert all_logits[0].shape == (B, S, V)
    assert all_q_logits[0].shape == (B,)
    assert all_hidden[0].shape == (B, S + model.inner.puzzle_emb_len + R, config["hidden_size"])

    loss = all_logits[-1].sum()
    loss.backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, "No gradients flowed"


class TestR6Configs:
    def test_r6a_sink(self):
        _run_smoke(dict(attention_sink=True))

    def test_r6b_temperature(self):
        _run_smoke(dict(attention_temperature=True))

    def test_r6c_registers(self):
        _run_smoke(dict(num_registers=4))

    def test_r6d_heads8(self):
        _run_smoke(dict(num_heads=8))

    def test_r6e_heads16(self):
        _run_smoke(dict(num_heads=16))

    def test_r6f_heads2(self):
        _run_smoke(dict(num_heads=2))

    def test_r6g_partial_rope(self):
        _run_smoke(dict(rope_fraction=0.5))

    def test_r6h_gqa(self):
        _run_smoke(dict(num_heads=8, num_kv_heads=2))

    def test_sink_and_temperature_mutually_exclusive(self):
        with pytest.raises(AssertionError):
            _run_smoke(dict(attention_sink=True, attention_temperature=True))

    def test_gqa_divisibility(self):
        with pytest.raises(AssertionError):
            _run_smoke(dict(num_heads=8, num_kv_heads=3))

    def test_default_config_unchanged(self):
        _run_smoke({})
