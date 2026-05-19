"""Training CLI for HOPE and MiniTransformer on a text dataset.

Examples:
    python scripts/train.py --model hope --dataset tinyshakespeare \
        --steps 200 --seq-len 64 --batch-size 8 --d-model 64 --n-layers 1
    python scripts/train.py --model transformer --dataset tinyshakespeare \
        --steps 200 --seq-len 64 --batch-size 8 --d-model 64 --n-layers 2

Both branches share the same Adam loop and report parameter counts so
the two models can be compared at equal compute / equal parameter
budget.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Allow `from hope...` and `from scripts...` regardless of cwd.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import tensorflow as tf  # noqa: E402

from hope.baseline import MiniTransformer  # noqa: E402
from hope.model import HOPE  # noqa: E402
from scripts.download_data import get_tinyshakespeare  # noqa: E402


def build_model(
    name: str,
    vocab_size: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    seq_len: int,
) -> tf.keras.Model:
    if name == "hope":
        return HOPE(
            vocab_size=vocab_size,
            d_model=d_model,
            n_self_mod_layers=n_layers,
            cms_banks=(1, 4),
            cms_decays=(0.01, 0.005),
            n_heads=n_heads,
            max_seq_len=seq_len,
        )
    if name == "transformer":
        return MiniTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            max_seq_len=seq_len,
        )
    raise ValueError(f"unknown model {name!r}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["hope", "transformer"], required=True)
    p.add_argument("--dataset", choices=["tinyshakespeare"], default="tinyshakespeare")
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    if args.dataset == "tinyshakespeare":
        ds, vocab_size = get_tinyshakespeare(
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
        )
    else:  # pragma: no cover - only one dataset is implemented today
        raise ValueError(f"unknown dataset {args.dataset!r}")

    model = build_model(
        name=args.model,
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        seq_len=args.seq_len,
    )
    _ = model(tf.zeros((1, args.seq_len), dtype=tf.int32))
    n_params = model.count_params()
    print(
        f"[init] model={args.model} vocab={vocab_size} params={n_params:,} "
        f"seq_len={args.seq_len} batch_size={args.batch_size}"
    )

    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
    it = iter(ds.repeat())

    t0 = time.time()
    loss_value: tf.Tensor | None = None
    for step in range(1, args.steps + 1):
        x, y = next(it)
        with tf.GradientTape() as tape:
            logits = model(x)
            loss_value = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
            )
        grads = tape.gradient(loss_value, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        if step == 1 or step % args.log_every == 0:
            dt = time.time() - t0
            print(f"[step {step:5d}] loss={float(loss_value):.4f}  elapsed={dt:.1f}s")

    final_loss = float(loss_value) if loss_value is not None else float("nan")
    print(f"[done] model={args.model} steps={args.steps} final_loss={final_loss:.4f}")


if __name__ == "__main__":
    main()
