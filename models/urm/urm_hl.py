"""R8a: H/L hierarchical model (HRM/GRAM structure, TRM training recipe).

Two modules with SEPARATE parameters, each a full URM block stack:
  fL (low-level):  l = fL(l + h + e_x)   -- input injected at every L eval
  fH (high-level): h = fH(h + l)         -- input never injected into fH

One supervision step (one forward) =
    for t in 1..T:
        for k in 1..K: l = fL(l + h + e_x)
        h = fH(h + l)
Decode ONCE per supervision step from the final h: logits = lm_head(h).
fL is never decoded or directly supervised -- its gradient arrives via fH.

Deterministic: no noise injection anywhere. The deep-supervision outer loop
(carry detached between supervision steps, backward + optimizer step per
supervision step) lives in pretrain.train_batch_hl / losses.HLLossHead.
"""
from typing import Tuple

import torch
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.urm.urm_energy import ARCModelConfig, ARCBackbone, ARCBlock


class HLModelConfig(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int
    num_layers: int
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str
    attn_dropout: float = 0.0
    mlp_dropout: float = 0.0
    forward_dtype: str = "bfloat16"
    # Hierarchy schedule: one supervision step = hl_T cycles of (hl_K fL evals + 1 fH eval)
    hl_T: int = 2
    hl_K: int = 3
    # Deep supervision: supervision steps per batch (carry detached between steps)
    n_sup: int = 4


class HLModel(nn.Module):
    """H/L hierarchical model. fL = self.inner.layers, fH = self.h_layers."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HLModelConfig(**config_dict)

        # Reuse ARCBackbone for embeddings / lm_head / q_head / rotary.
        # Its `layers` ModuleList serves as fL.
        backbone_cfg = dict(config_dict)
        backbone_cfg.setdefault("loops", 1)  # required by ARCModelConfig, unused here
        self.inner = ARCBackbone(ARCModelConfig(**backbone_cfg))

        # fH: separate parameters, same block config as fL
        self.h_layers = nn.ModuleList(
            [ARCBlock(self.inner.config) for _ in range(self.config.num_layers)]
        )

        # h0: fixed random draw saved with the checkpoint (GRAM convention).
        # l0 reuses self.inner.init_hidden (same convention, also persistent).
        self.init_h = nn.Buffer(
            trunc_normal_init_(
                torch.empty(self.config.hidden_size, dtype=self.inner.forward_dtype), std=1
            ),
            persistent=True,
        )

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fresh (h, l) carry, each [B, P+R+seq, H], broadcast from h0/l0."""
        total_len = (
            self.inner.config.seq_len
            + self.inner.puzzle_emb_len
            + self.inner.config.num_registers
        )
        h = self.init_h.expand(batch_size, total_len, -1).clone()
        l = self.inner.init_hidden.expand(batch_size, total_len, -1).clone()
        return h, l

    def input_embeddings(self, batch) -> torch.Tensor:
        return self.inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

    def forward_supervision_step(
        self,
        h: torch.Tensor,
        l: torch.Tensor,
        input_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One supervision step: hl_T cycles of (hl_K fL evals + 1 fH eval), one decode.

        Full gradient flow through all fL/fH evals within the step; detaching
        between supervision steps is the caller's responsibility.

        Returns (h, l, logits, q_logits).
        """
        cos_sin = self.inner.rotary_emb()
        P = self.inner.puzzle_emb_len
        R = self.inner.config.num_registers

        for _t in range(self.config.hl_T):
            for _k in range(self.config.hl_K):
                x = l + h + input_embeddings
                for layer in self.inner.layers:
                    x = layer(cos_sin=cos_sin, hidden_states=x)
                l = x
            x = h + l
            for layer in self.h_layers:
                x = layer(cos_sin=cos_sin, hidden_states=x)
            h = x

        logits = self.inner.lm_head(h)[:, P + R:]
        q_logits = self.inner.q_head(h[:, 0]).to(torch.float32).squeeze(-1)
        return h, l, logits, q_logits
