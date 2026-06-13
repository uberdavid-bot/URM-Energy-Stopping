"""Smoke tests for R8a H/L hierarchical model (urm_hl@HLModel + losses@HLLossHead).

Checks: supervision-step shapes, gradient flow into BOTH fL and fH, separate
parameters, detached carry between supervision steps, eval metrics, and
deterministic eval forward.
"""
import torch
import pytest

from models.urm.urm_hl import HLModel
from models.losses import HLLossHead

DEVICE = "cuda"


def make_config(**overrides):
    config = dict(
        batch_size=2,
        seq_len=100,
        puzzle_emb_ndim=128,
        num_puzzle_identifiers=100,
        vocab_size=14,
        num_layers=1,
        hidden_size=128,
        expansion=2.0,
        num_heads=4,
        pos_encodings="rope",
        forward_dtype="bfloat16",
        hl_T=2,
        hl_K=3,
        n_sup=4,
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


class TestHLModel:
    def test_supervision_step_shapes(self):
        config = make_config()
        model = HLModel(config).to(DEVICE)
        model.train()
        batch = make_batch(config)

        B, S, V, H = config["batch_size"], config["seq_len"], config["vocab_size"], config["hidden_size"]
        P = model.inner.puzzle_emb_len

        emb = model.input_embeddings(batch)
        h, l = model.initial_state(B)
        assert h.shape == (B, P + S, H)
        assert l.shape == (B, P + S, H)

        h, l, logits, q_logits = model.forward_supervision_step(h, l, emb)
        assert logits.shape == (B, S, V)
        assert q_logits.shape == (B,)
        assert h.shape == (B, P + S, H)

    def test_gradients_flow_to_both_modules(self):
        config = make_config()
        model = HLModel(config).to(DEVICE)
        model.train()
        batch = make_batch(config)

        h, l = model.initial_state(config["batch_size"])
        _, _, logits, _ = model.forward_supervision_step(h, l, model.input_embeddings(batch))
        logits.float().sum().backward()

        fl_grad = sum(p.grad.abs().sum().item() for p in model.inner.layers.parameters() if p.grad is not None)
        fh_grad = sum(p.grad.abs().sum().item() for p in model.h_layers.parameters() if p.grad is not None)
        assert fl_grad > 0, "No gradient reached fL (should arrive via fH)"
        assert fh_grad > 0, "No gradient reached fH"

    def test_separate_parameters(self):
        model = HLModel(make_config()).to(DEVICE)
        fl_ids = {id(p) for p in model.inner.layers.parameters()}
        fh_ids = {id(p) for p in model.h_layers.parameters()}
        assert fl_ids.isdisjoint(fh_ids), "fL and fH share parameters"
        assert len(fh_ids) == len(fl_ids), "fL and fH should have matching block structure"

    def test_init_buffers_persistent(self):
        model = HLModel(make_config()).to(DEVICE)
        state = model.state_dict()
        assert "init_h" in state, "h0 must be saved with the checkpoint"
        assert "inner.init_hidden" in state, "l0 must be saved with the checkpoint"


class TestHLLossHead:
    def test_train_step_returns_detached_carry(self):
        config = make_config()
        head = HLLossHead(HLModel(config), loss_type="stablemax_cross_entropy").to(DEVICE)
        head.train()
        batch = make_batch(config)

        carry, loss, metrics, _, _ = head(batch=batch, carry=None)
        assert torch.isfinite(loss)
        assert not carry[0].requires_grad and not carry[1].requires_grad
        assert "exact_accuracy" in metrics and "q_halt_loss" in metrics

        # Second supervision step consumes the detached carry
        carry2, loss2, _, _, _ = head(batch=batch, carry=carry)
        assert torch.isfinite(loss2)
        loss2.backward()  # graph must be self-contained within the step

    def test_n_sup_attribute(self):
        head = HLLossHead(HLModel(make_config()), loss_type="stablemax_cross_entropy")
        assert head.is_hl is True
        assert head.n_sup == 4

    def test_eval_metrics(self):
        config = make_config()
        head = HLLossHead(HLModel(config), loss_type="stablemax_cross_entropy").to(DEVICE)
        head.eval()
        batch = make_batch(config)

        with torch.no_grad():
            carry, loss, metrics, outputs, _ = head(
                batch=batch, return_keys=["preds", "q_halt_logits", "current_energy"]
            )

        assert carry is None
        for k in range(1, 5):
            assert f"step_{k}_exact_accuracy" in metrics
        assert "qhalt_stop_accuracy" in metrics and "qhalt_stop_step" in metrics
        assert outputs["preds"].shape == (config["batch_size"], config["seq_len"])
        assert outputs["current_energy"].shape == (config["batch_size"],)

    def test_eval_deterministic(self):
        config = make_config()
        head = HLLossHead(HLModel(config), loss_type="stablemax_cross_entropy").to(DEVICE)
        head.eval()
        batch = make_batch(config)

        with torch.no_grad():
            _, _, m1, o1, _ = head(batch=batch, return_keys=["logits"])
            _, _, m2, o2, _ = head(batch=batch, return_keys=["logits"])
        assert torch.equal(o1["logits"], o2["logits"]), "Eval forward must be deterministic"
