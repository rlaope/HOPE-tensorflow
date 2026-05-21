"""HOPE vs MiniTransformer benchmark — 3 scenarios at equal parameter budget.

Run:
    python scripts/benchmark.py --scenario all \
        --steps 50 --seq-len 64 --batch-size 4

Scenarios (each writes a PNG into assets/):
  longctx   — synthetic (key, value) needle-in-a-haystack recall
  continual — train on TinyShakespeare, then on random alphabet
              sequences, then re-measure cross-entropy on TinyShakespeare
  incontext — few-shot character substitution: show k examples in the
              prompt and ask the model to apply the same substitution

The benchmarks are intentionally tiny and CPU-friendly. They show
direction, not headline numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from hope.baseline import MiniTransformer  # noqa: E402
from hope.model import HOPE  # noqa: E402
from scripts.download_data import get_tinyshakespeare  # noqa: E402

ASSETS_DIR = os.path.join(_REPO_ROOT, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


def _set_seed(seed: int) -> None:
    import random
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)


def _build_hope(vocab: int, seq_len: int, d_model: int, n_heads: int) -> HOPE:
    return HOPE(
        vocab_size=vocab,
        d_model=d_model,
        n_self_mod_layers=1,
        cms_banks=(1, 4),
        cms_decays=(0.01, 0.005),
        n_heads=n_heads,
        max_seq_len=seq_len,
    )


def _build_matched_baseline(hope_model: HOPE) -> MiniTransformer:
    return MiniTransformer.matched_to(hope_model, tolerance=0.10)


def _ce_loss(logits: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
    )


def _train_on_batches(model, batches, n_steps: int, lr: float) -> list[float]:
    opt = tf.keras.optimizers.Adam(lr)
    it = iter(batches.repeat())
    losses = []
    for _ in range(n_steps):
        x, y = next(it)
        with tf.GradientTape() as tape:
            logits = model(x)
            loss = _ce_loss(logits, y)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        losses.append(float(loss))
    return losses


def _eval_mean_loss(model, batches, max_batches: int = 5) -> float:
    losses = []
    for i, (x, y) in enumerate(batches):
        logits = model(x)
        losses.append(float(_ce_loss(logits, y)))
        if i >= max_batches - 1:
            break
    return float(np.mean(losses)) if losses else float("nan")


def scenario_longctx(args) -> dict:
    """Plant a (key, value) pair at the start, query it near the end.

    Metric: mean probability the model assigns to ``val_id`` at the
    recall position. Softer than argmax accuracy so the comparison is
    visible on a tiny CPU-friendly training budget.
    """
    vocab = 8
    n_heads = 2
    d_model = 32

    # prob_correct_per_seed[name][seq_len] = [seed0_result, seed1_result, ...]
    prob_correct_per_seed: dict[str, dict[int, list[float]]] = {
        "hope": {sl: [] for sl in args.longctx_seq_lens},
        "transformer": {sl: [] for sl in args.longctx_seq_lens},
    }

    for seed in range(args.n_seeds):
        _set_seed(seed)
        rng = np.random.default_rng(seed)
        for seq_len in args.longctx_seq_lens:
            B = args.batch_size
            key_id = int(rng.integers(1, 4))
            val_id = int(rng.integers(4, vocab))
            x = rng.integers(0, 4, size=(B, seq_len), dtype=np.int32)
            x[:, 0] = key_id
            x[:, 1] = val_id
            x[:, -2] = key_id
            x[:, -1] = val_id
            x_t = tf.constant(x, dtype=tf.int32)
            y_t = x_t

            for name in ("hope", "transformer"):
                if name == "hope":
                    model = _build_hope(vocab, seq_len, d_model, n_heads)
                else:
                    ref = _build_hope(vocab, seq_len, d_model, n_heads)
                    model = _build_matched_baseline(ref)
                _ = model(tf.zeros((1, seq_len), dtype=tf.int32))
                opt = tf.keras.optimizers.Adam(args.lr)
                for _ in range(args.longctx_train_steps):
                    with tf.GradientTape() as tape:
                        logits = model(x_t)
                        loss = _ce_loss(logits, y_t)
                    grads = tape.gradient(loss, model.trainable_variables)
                    opt.apply_gradients(zip(grads, model.trainable_variables))
                logits = model(x_t)
                probs = tf.nn.softmax(logits[:, -2, :], axis=-1).numpy()
                prob_correct = float(np.mean(probs[:, val_id]))
                prob_correct_per_seed[name][seq_len].append(prob_correct)
                print(
                    f"[longctx] seed={seed} seq_len={seq_len} {name}: P(val_id)={prob_correct:.3f}"
                )

    fig, ax = plt.subplots(figsize=(7, 4))
    sls = sorted(args.longctx_seq_lens)
    for name in ("hope", "transformer"):
        means = np.array([np.mean(prob_correct_per_seed[name][sl]) for sl in sls])
        stds = np.array([np.std(prob_correct_per_seed[name][sl]) for sl in sls])
        ax.errorbar(sls, means, yerr=stds, marker="o", label=name, capsize=3)
    ax.axhline(1.0 / vocab, linestyle="--", color="gray", alpha=0.6,
               label=f"random (1/vocab={1.0/vocab:.3f})")
    ax.set_xlabel("sequence length")
    ax.set_ylabel("P(val_id | context) at recall position")
    ax.set_title("Long-context retrieval (synthetic key/value)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(ASSETS_DIR, "bench_longctx.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)

    payload = {
        "vocab": vocab,
        "n_seeds": args.n_seeds,
        "results": {
            name: {str(sl): prob_correct_per_seed[name][sl] for sl in args.longctx_seq_lens}
            for name in ("hope", "transformer")
        },
        "random_baseline": 1.0 / vocab,
    }
    json_out = os.path.join(ASSETS_DIR, "bench_longctx.json")
    with open(json_out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[longctx] saved bench_longctx.png + bench_longctx.json (n_seeds={args.n_seeds})")
    return prob_correct_per_seed


def scenario_continual(args) -> dict:
    """Train on domain A (TinyShakespeare), then B (random), re-eval A."""
    ds_a, vocab_a = get_tinyshakespeare(seq_len=args.seq_len, batch_size=args.batch_size)
    n_heads = 2
    d_model = 32

    # loss_pairs_per_seed[name] = [(before_seed0, after_seed0), ...]
    loss_pairs_per_seed: dict[str, list[tuple[float, float]]] = {
        "hope": [],
        "transformer": [],
    }

    for seed in range(args.n_seeds):
        _set_seed(seed)
        rng = np.random.default_rng(seed)
        n_samples = args.steps * args.batch_size
        b_data = rng.integers(0, vocab_a, size=(n_samples, args.seq_len + 1)).astype(np.int32)
        ds_b = tf.data.Dataset.from_tensor_slices(tf.constant(b_data))
        ds_b = ds_b.map(lambda c: (c[:-1], c[1:])).batch(args.batch_size, drop_remainder=True)

        for name in ("hope", "transformer"):
            if name == "hope":
                model = _build_hope(vocab_a, args.seq_len, d_model, n_heads)
            else:
                ref = _build_hope(vocab_a, args.seq_len, d_model, n_heads)
                model = _build_matched_baseline(ref)
            _ = model(tf.zeros((1, args.seq_len), dtype=tf.int32))
            _train_on_batches(model, ds_a, args.steps, args.lr)
            loss_a_before_b = _eval_mean_loss(model, ds_a, max_batches=5)
            _train_on_batches(model, ds_b, args.steps, args.lr)
            loss_a_after_b = _eval_mean_loss(model, ds_a, max_batches=5)
            loss_pairs_per_seed[name].append((loss_a_before_b, loss_a_after_b))
            print(
                f"[continual] seed={seed} {name}: lossA before={loss_a_before_b:.3f} after={loss_a_after_b:.3f}"
            )

    fig, ax = plt.subplots(figsize=(7, 4))
    x_axis = np.arange(2)
    width = 0.35
    for name, offset in (("hope", -width / 2), ("transformer", width / 2)):
        pairs = loss_pairs_per_seed[name]
        befores = [p[0] for p in pairs]
        afters = [p[1] for p in pairs]
        mean_before = float(np.mean(befores))
        mean_after = float(np.mean(afters))
        std_before = float(np.std(befores))
        std_after = float(np.std(afters))
        ax.bar(
            x_axis + offset,
            [mean_before, mean_after],
            width,
            yerr=[std_before, std_after],
            capsize=3,
            label=name,
        )
    ax.set_xticks(x_axis)
    ax.set_xticklabels(["A loss (before B)", "A loss (after B)"])
    ax.set_ylabel("cross-entropy on domain A")
    ax.set_title("Continual LM: domain-A loss before / after training on B")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(ASSETS_DIR, "bench_continual.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)

    payload = {
        "n_seeds": args.n_seeds,
        "results": {
            name: [
                {"before": p[0], "after": p[1]}
                for p in loss_pairs_per_seed[name]
            ]
            for name in ("hope", "transformer")
        },
    }
    json_out = os.path.join(ASSETS_DIR, "bench_continual.json")
    with open(json_out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[continual] saved bench_continual.png + bench_continual.json (n_seeds={args.n_seeds})")
    return loss_pairs_per_seed


def scenario_incontext(args) -> dict:
    """Show k (src, perm[src]) pairs in the prompt; ask for perm[query].

    Averages over many random permutations per k so the plot reports a
    probability instead of a single binary flip.
    """
    rng = np.random.default_rng(0)
    vocab = 8
    n_heads = 2
    d_model = 32
    shots = (1, 2, 4)
    n_trials = int(args.incontext_trials)

    results: dict[str, dict[int, float]] = {"hope": {}, "transformer": {}}
    per_trial_raw: dict[str, dict[int, list[float]]] = {
        "hope": {k: [] for k in shots},
        "transformer": {k: [] for k in shots},
    }
    for k in shots:
        per_trial: dict[str, list[float]] = {"hope": [], "transformer": []}
        for trial in range(n_trials):
            perm = rng.permutation(vocab).astype(np.int32)
            src_examples = rng.integers(0, vocab, size=k)
            prompt: list[int] = []
            for s in src_examples:
                prompt.extend([int(s), int(perm[s])])
            q = int(rng.integers(0, vocab))
            prompt.append(q)
            prompt.append(int(perm[q]))
            x_arr = np.array(prompt, dtype=np.int32)[None, :]
            T = x_arr.shape[1]
            seq_len = max(T, 8)
            if seq_len > T:
                pad = rng.integers(0, vocab, size=(1, seq_len - T)).astype(np.int32)
                x_arr = np.concatenate([x_arr, pad], axis=1)
            x_t = tf.constant(x_arr, dtype=tf.int32)
            y_t = x_t

            for name in ("hope", "transformer"):
                if name == "hope":
                    model = _build_hope(vocab, seq_len, d_model, n_heads)
                else:
                    ref = _build_hope(vocab, seq_len, d_model, n_heads)
                    model = _build_matched_baseline(ref)
                _ = model(tf.zeros((1, seq_len), dtype=tf.int32))
                opt = tf.keras.optimizers.Adam(args.lr)
                for _ in range(args.incontext_train_steps):
                    with tf.GradientTape() as tape:
                        logits = model(x_t)
                        loss = _ce_loss(logits, y_t)
                    grads = tape.gradient(loss, model.trainable_variables)
                    opt.apply_gradients(zip(grads, model.trainable_variables))
                logits = model(x_t)
                pred_at_q = int(tf.argmax(logits[0, T - 2, :]).numpy())
                result = 1.0 if pred_at_q == int(perm[q]) else 0.0
                per_trial[name].append(result)
                per_trial_raw[name][k].append(result)

        for name in ("hope", "transformer"):
            mean_acc = float(np.mean(per_trial[name]))
            results[name][k] = mean_acc
            print(
                f"[incontext] k={k} {name}: acc={mean_acc:.3f} over {n_trials} trials"
            )

    fig, ax = plt.subplots(figsize=(7, 4))
    for name in ("hope", "transformer"):
        ax.plot(shots, [results[name][k] for k in shots], marker="o", label=name)
    ax.axhline(1.0 / vocab, linestyle="--", color="gray", alpha=0.6,
               label=f"random (1/vocab={1.0/vocab:.3f})")
    ax.set_xlabel("number of shots k")
    ax.set_ylabel("next-token correct (1 = yes, 0 = no)")
    ax.set_title("In-context adaptation (random char substitution)")
    ax.set_ylim(-0.1, 1.1)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(ASSETS_DIR, "bench_incontext.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)

    payload = {
        "vocab": vocab,
        "n_trials": n_trials,
        "results": {
            name: {str(k): per_trial_raw[name][k] for k in shots}
            for name in ("hope", "transformer")
        },
        "random_baseline": 1.0 / vocab,
    }
    json_out = os.path.join(ASSETS_DIR, "bench_incontext.json")
    with open(json_out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[incontext] saved bench_incontext.png + bench_incontext.json (n_trials={n_trials})")
    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--scenario",
        choices=["longctx", "continual", "incontext", "all"],
        default="all",
    )
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--longctx-seq-lens", type=int, nargs="+", default=[64, 256])
    p.add_argument("--longctx-train-steps", type=int, default=200)
    p.add_argument("--incontext-train-steps", type=int, default=20)
    p.add_argument("--incontext-trials", type=int, default=8)
    p.add_argument("--n-seeds", type=int, default=3,
                   help="Seeds to average longctx and continual over (incontext already uses trials).")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.scenario in ("longctx", "all"):
        scenario_longctx(args)
    if args.scenario in ("continual", "all"):
        scenario_continual(args)
    if args.scenario in ("incontext", "all"):
        scenario_incontext(args)


if __name__ == "__main__":
    main()
