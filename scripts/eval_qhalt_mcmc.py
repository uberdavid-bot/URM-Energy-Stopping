"""R3-diag: Eval-only MCMC diagnostic using Q-halt as the energy function.

Takes the gradient of Q-halt's pre-sigmoid logit w.r.t. hidden states and ascends
it to push hidden states toward higher-confidence regions. Compares the resulting
accuracy against pure URM at matched compute (M URM + K MCMC vs M+K URM).

**Important implementation note on the gradient signal.** The q_head operates on
`hidden[:, 0]`, the first puzzle-embedding position. Computing the gradient of
`q_head(hidden[:, 0])` directly with respect to `hidden` gives a gradient that is
nonzero only at position 0 — a pure slice has no cross-position dependencies.
Since `lm_head(hidden)[:, P:]` discards the puzzle-embedding positions before
decoding, a position-0-only gradient update cannot change any decoded token.
A naive implementation of the spec therefore produces zero change in token
predictions regardless of step size (verified empirically).

To make the Q-halt signal actually influence token positions, we route it
through a transformer probe: at each MCMC step, compute
`q_head(transformer(hidden + input_emb)[:, 0])` and take the gradient with
respect to `hidden`. Autograd propagates that gradient through attention back
to every position, so the update touches the token positions that lm_head
decodes. This costs one transformer forward per MCMC step, which preserves the
matched-compute comparison with (M+K) pure URM steps. After K MCMC steps,
decode via `lm_head(hidden_final)[:, P:]` directly (the refined hidden is
treated as "what step M's output should have been").

No retraining. Uses the R1-h96 checkpoint (best backbone) by default. All MCMC
happens in eval mode with no gradients flowing into model parameters — only
into the hidden state tensor at each step.

Usage:
    conda activate urm
    python scripts/eval_qhalt_mcmc.py \\
        --checkpoint checkpoints/R1-h96-baseline-260414 \\
        --data_path data/arc1concept-aug-1000-size-10

Outputs:
    - stdout table
    - results/qhalt_mcmc_eval.tsv
"""
import argparse
import csv
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pretrain import (  # noqa: E402
    PretrainConfig,
    create_dataloader,
    init_train_state,
    load_config_from_checkpoint_path,
)
from models.losses import IGNORE_LABEL_ID  # noqa: E402


M_VALUES = (4, 5, 6)
K_VALUES = (1, 2, 4)
# Extended from the original spec grid {1e-4, 1e-3, 1e-2} after a sanity run
# showed those step sizes only flip ~0.05% of tokens even at the largest value
# — far too small to affect exact accuracy. The gradient magnitude is dominated
# by position 0 (puzzle-emb) and only leaks weakly through attention to token
# positions, so meaningful updates need larger per-position steps. 1e-4..1e-2
# kept for reference + no-op sanity check.
STEP_SIZES = (1e-4, 1e-3, 1e-2, 1e-1, 3e-1, 1.0)
NORMALIZED_VARIANTS = (False, True)


def _resolve_checkpoint_dir(path: str) -> str:
    if not os.path.isdir(path):
        return path
    pts = [
        (int(m.group(1)), os.path.join(path, f))
        for f in os.listdir(path)
        for m in [re.match(r"step_(\d+)\.pt$", f)]
        if m
    ]
    if not pts:
        raise FileNotFoundError(f"No step_*.pt files in {path}")
    pts.sort()
    return pts[-1][1]


def _exact_and_token_accuracy(preds, labels, mask, valid):
    """Return (token_correct_sum, exact_correct_sum, token_count, valid_count).

    preds:  [B, S] integer predictions
    labels: [B, S] integer labels (IGNORE_LABEL_ID where invalid)
    mask:   [B, S] bool — True where labels are valid
    valid:  [B] bool — True where the whole example has any valid tokens
    """
    per_token_correct = mask & (preds == labels)  # [B, S]
    loss_counts = mask.sum(-1).clamp_min(1)        # [B]
    per_example_token_acc = per_token_correct.sum(-1).to(torch.float32) / loss_counts.to(torch.float32)
    per_example_exact = per_token_correct.sum(-1) == mask.sum(-1)

    token_acc_sum = (per_example_token_acc * valid.to(torch.float32)).sum().item()
    exact_sum = (per_example_exact & valid).sum().item()
    valid_count = valid.sum().item()
    return token_acc_sum, exact_sum, valid_count


def _qhalt_logit_from_state(arc_model, hidden):
    """Pre-sigmoid Q-halt scalar from an already-computed transformer output state.

    Mirrors how losses.py reads Q-halt: q_head on the first (puzzle-emb) position.
    """
    return arc_model.inner.q_head(hidden[:, 0]).to(torch.float32).squeeze(-1)


def _qhalt_via_transformer_probe(arc_model, hidden, input_embeddings, cos_sin):
    """Compute Q-halt by running one URM-style transformer pass on `hidden`.

    Routes the Q-halt signal through attention so that gradients w.r.t. `hidden`
    have support on every position, not just position 0.
    """
    probe = hidden + input_embeddings
    for layer in arc_model.inner.layers:
        probe = layer(cos_sin=cos_sin, hidden_states=probe)
    return _qhalt_logit_from_state(arc_model, probe)


def _mcmc_ascend_qhalt(arc_model, hidden_start, input_embeddings, cos_sin,
                       K, step_size, normalize):
    """Run K MCMC ascent steps on Q-halt. Returns (hidden_final, qhalt_before, qhalt_after).

    Each MCMC step takes the gradient of `q_head(transformer(hidden + input_emb)[:, 0])`
    with respect to `hidden`, and ascends. The gradient flows through attention so
    the update modifies every position of `hidden`, which then gets decoded directly
    by `lm_head[:, P:]`.
    """
    hidden = hidden_start.detach().clone()

    with torch.no_grad():
        q_before = _qhalt_via_transformer_probe(
            arc_model, hidden, input_embeddings, cos_sin
        ).detach()

    for _ in range(K):
        hidden = hidden.detach().requires_grad_(True)
        with torch.enable_grad():
            q = _qhalt_via_transformer_probe(arc_model, hidden, input_embeddings, cos_sin)
            grad = torch.autograd.grad(q.sum(), hidden, create_graph=False)[0]

        if normalize:
            grad = grad / (grad.norm(dim=-1, keepdim=True) + 1e-8)

        # Ascend: maximize Q-halt confidence
        hidden = (hidden + step_size * grad).detach()

    with torch.no_grad():
        q_after = _qhalt_via_transformer_probe(
            arc_model, hidden, input_embeddings, cos_sin
        ).detach()

    return hidden, q_before, q_after


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/R1-h96-baseline-260414")
    parser.add_argument("--data_path", default="data/arc1concept-aug-1000-size-10")
    parser.add_argument("--output_tsv", default="results/qhalt_mcmc_eval.tsv")
    parser.add_argument("--max_batches", type=int, default=0,
                        help="Cap eval batches for a quick sanity run (0 = full eval set)")
    args = parser.parse_args()

    os.environ["DISABLE_COMPILE"] = "1"

    # --- Load config + checkpoint ---
    config = load_config_from_checkpoint_path(args.checkpoint)
    if config is None:
        import yaml
        from pathlib import Path
        cfg_path = Path(args.checkpoint) / "config.yaml"
        if not cfg_path.exists():
            raise ValueError(f"Could not load config from {args.checkpoint}")
        with open(cfg_path) as f:
            config = PretrainConfig(**yaml.safe_load(f))

    ckpt_path = _resolve_checkpoint_dir(args.checkpoint)
    print(f"Using checkpoint: {ckpt_path}")

    config.load_checkpoint = ckpt_path
    config.data_path = args.data_path
    config.load_strict = True
    config.load_optimizer_state = False

    eval_loader, eval_metadata = create_dataloader(
        config, "test", test_set_mode=True, epochs_per_iter=1,
        global_batch_size=config.global_batch_size, rank=0, world_size=1,
    )

    # Checkpoints are saved with EMA weights already applied (see pretrain.py:900),
    # so loading the checkpoint gives us the EMA'd model directly.
    train_state = init_train_state(config, eval_metadata, rank=0, world_size=1)
    train_state.model.eval()

    loss_head = train_state.model
    arc_model = loss_head.model
    P = arc_model.inner.puzzle_emb_len
    blank_id = eval_metadata.blank_identifier_id

    # --- Accumulators ---
    # For URM M-step and M+K-step (pure reference), keyed by (M,) and (M+K,)
    urm_acc = {}  # step_count -> [tok_sum, exact_sum, valid_sum]
    # For MCMC, keyed by (M, K, step_size, normalized)
    mcmc_acc = {}  # cond_key -> [tok_sum, exact_sum, valid_sum, q_before_sum, q_after_sum]

    pure_urm_steps = sorted({m for m in M_VALUES} | {m + k for m in M_VALUES for k in K_VALUES})
    for s in pure_urm_steps:
        urm_acc[s] = [0.0, 0, 0]
    for M in M_VALUES:
        for K in K_VALUES:
            for ss in STEP_SIZES:
                for norm in NORMALIZED_VARIANTS:
                    mcmc_acc[(M, K, ss, norm)] = [0.0, 0, 0, 0.0, 0.0]

    max_step = max(pure_urm_steps)
    print(f"Running URM up to step {max_step}, MCMC from M in {M_VALUES} with K in {K_VALUES}")

    batch_count = 0
    for set_name, batch, _ in eval_loader:
        batch_count += 1
        if args.max_batches and batch_count > args.max_batches:
            break

        batch = {k: v.cuda() for k, v in batch.items()}
        labels = batch["labels"]
        mask = labels != IGNORE_LABEL_ID
        loss_counts = mask.sum(-1)
        valid = (loss_counts > 0) & (batch["puzzle_identifiers"] != blank_id)

        # --- Pure URM up to max_step (no grad) ---
        # We need hidden_at_M for M in M_VALUES (for MCMC) AND logits at every
        # `pure_urm_steps` index (for the URM baselines).
        with torch.no_grad():
            all_logits, all_q_logits, all_hidden, input_embeddings = arc_model.forward_trajectory(
                batch, N=max_step
            )

            for s in pure_urm_steps:
                # all_logits[s-1] is already logits[:, P:] (lm_head on hidden with puzzle tokens trimmed)
                preds = torch.argmax(all_logits[s - 1], dim=-1)
                tok, exact, vc = _exact_and_token_accuracy(preds, labels, mask, valid)
                urm_acc[s][0] += tok
                urm_acc[s][1] += exact
                urm_acc[s][2] += vc

        # --- MCMC conditions ---
        # For each M, take hidden_M (including the puzzle-emb prefix) and run K MCMC
        # steps of Q-halt ascent. Gradient is routed through one transformer pass so
        # every position receives an update. Decode via lm_head on the refined hidden.
        cos_sin = arc_model.inner.rotary_emb()
        for M in M_VALUES:
            hidden_M = all_hidden[M - 1]  # [B, P+seq_len, H] — no_grad detached

            for K in K_VALUES:
                for ss in STEP_SIZES:
                    for norm in NORMALIZED_VARIANTS:
                        hidden_final, q_before, q_after = _mcmc_ascend_qhalt(
                            arc_model, hidden_M, input_embeddings, cos_sin, K, ss, norm
                        )

                        with torch.no_grad():
                            logits = arc_model.inner.lm_head(hidden_final)[:, P:]
                            preds = torch.argmax(logits, dim=-1)
                            tok, exact, vc = _exact_and_token_accuracy(preds, labels, mask, valid)

                        key = (M, K, ss, norm)
                        mcmc_acc[key][0] += tok
                        mcmc_acc[key][1] += exact
                        mcmc_acc[key][2] += vc
                        mcmc_acc[key][3] += (q_before * valid.to(q_before.dtype)).sum().item()
                        mcmc_acc[key][4] += (q_after * valid.to(q_after.dtype)).sum().item()

        if batch_count % 20 == 0:
            print(f"  processed {batch_count} batches")

    print(f"Done: {batch_count} batches")

    # --- Normalize ---
    def _avg(acc_entry, num_valid):
        return acc_entry / num_valid if num_valid > 0 else 0.0

    urm_norm = {}
    for s, (tok, ex, vc) in urm_acc.items():
        urm_norm[s] = {
            "token_acc": _avg(tok, vc),
            "exact_acc": _avg(ex, vc),
            "valid": vc,
        }

    rows = []
    for (M, K, ss, norm), (tok, ex, vc, qb, qa) in mcmc_acc.items():
        mcmc_token = _avg(tok, vc)
        mcmc_exact = _avg(ex, vc)
        urm_M_token = urm_norm[M]["token_acc"]
        urm_M_exact = urm_norm[M]["exact_acc"]
        urm_MK_token = urm_norm[M + K]["token_acc"]
        urm_MK_exact = urm_norm[M + K]["exact_acc"]
        rows.append({
            "M": M,
            "K": K,
            "step_size": ss,
            "normalized": int(norm),
            "mcmc_token_acc": mcmc_token,
            "mcmc_exact_acc": mcmc_exact,
            "urm_M_token_acc": urm_M_token,
            "urm_M_exact_acc": urm_M_exact,
            "urm_M+K_token_acc": urm_MK_token,
            "urm_M+K_exact_acc": urm_MK_exact,
            "delta_exact_vs_urm_M": mcmc_exact - urm_M_exact,
            "delta_exact_vs_urm_M+K": mcmc_exact - urm_MK_exact,
            "qhalt_conf_before": _avg(qb, vc),
            "qhalt_conf_after": _avg(qa, vc),
        })

    rows.sort(key=lambda r: (r["M"], r["K"], r["step_size"], r["normalized"]))

    # --- Save TSV ---
    os.makedirs(os.path.dirname(args.output_tsv), exist_ok=True)
    fieldnames = [
        "M", "K", "step_size", "normalized",
        "mcmc_token_acc", "mcmc_exact_acc",
        "urm_M_token_acc", "urm_M_exact_acc",
        "urm_M+K_token_acc", "urm_M+K_exact_acc",
        "delta_exact_vs_urm_M", "delta_exact_vs_urm_M+K",
        "qhalt_conf_before", "qhalt_conf_after",
    ]
    with open(args.output_tsv, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved results to {args.output_tsv}")

    # --- Print table ---
    print()
    print("=" * 128)
    print("R3-diag: Q-halt MCMC vs pure URM (exact accuracy)")
    print("=" * 128)
    print(f"{'M':>3} {'K':>3} {'step_size':>10} {'norm':>5}  "
          f"{'mcmc_tok':>9} {'mcmc_ex':>8}  "
          f"{'urm_M_tok':>10} {'urm_M_ex':>9}  "
          f"{'urm_MK_tok':>11} {'urm_MK_ex':>10}  "
          f"{'Δex_M':>8} {'Δex_MK':>8}  "
          f"{'q_before':>9} {'q_after':>8}")
    print("-" * 128)
    for r in rows:
        print(
            f"{r['M']:>3} {r['K']:>3} {r['step_size']:>10.4g} {r['normalized']:>5}  "
            f"{r['mcmc_token_acc']:>9.4f} {r['mcmc_exact_acc']:>8.4f}  "
            f"{r['urm_M_token_acc']:>10.4f} {r['urm_M_exact_acc']:>9.4f}  "
            f"{r['urm_M+K_token_acc']:>11.4f} {r['urm_M+K_exact_acc']:>10.4f}  "
            f"{r['delta_exact_vs_urm_M']:>+8.4f} {r['delta_exact_vs_urm_M+K']:>+8.4f}  "
            f"{r['qhalt_conf_before']:>+9.3f} {r['qhalt_conf_after']:>+8.3f}"
        )
    print("=" * 128)

    # --- Quick summary ---
    best_vs_M = max(rows, key=lambda r: r["delta_exact_vs_urm_M"])
    best_vs_MK = max(rows, key=lambda r: r["delta_exact_vs_urm_M+K"])
    print()
    print("Best delta vs URM-M (did MCMC help at all?):")
    print(f"  M={best_vs_M['M']}, K={best_vs_M['K']}, step_size={best_vs_M['step_size']:.4g}, "
          f"norm={best_vs_M['normalized']}: "
          f"{best_vs_M['delta_exact_vs_urm_M']:+.4f} exact "
          f"(mcmc={best_vs_M['mcmc_exact_acc']:.4f} vs urm_M={best_vs_M['urm_M_exact_acc']:.4f})")
    print()
    print("Best delta vs URM-(M+K) (core research question):")
    print(f"  M={best_vs_MK['M']}, K={best_vs_MK['K']}, step_size={best_vs_MK['step_size']:.4g}, "
          f"norm={best_vs_MK['normalized']}: "
          f"{best_vs_MK['delta_exact_vs_urm_M+K']:+.4f} exact "
          f"(mcmc={best_vs_MK['mcmc_exact_acc']:.4f} vs urm_MK={best_vs_MK['urm_M+K_exact_acc']:.4f})")


if __name__ == "__main__":
    main()
