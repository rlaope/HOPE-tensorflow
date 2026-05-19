"""Dataset utilities for HOPE-tensorflow training.

Currently supports TinyShakespeare (char-level), pulled from
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

Usage:
    python scripts/download_data.py            # just download
    from scripts.download_data import get_tinyshakespeare
    ds, vocab_size = get_tinyshakespeare(seq_len=64, batch_size=8)
"""

from __future__ import annotations

import argparse
import os
import urllib.request
from typing import Optional

import tensorflow as tf

TINYSHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_REPO_ROOT, "data", "tinyshakespeare")
TINYSHAKESPEARE_PATH = os.path.join(DATA_DIR, "input.txt")


def download_tinyshakespeare(force: bool = False) -> str:
    """Fetch TinyShakespeare into ``data/tinyshakespeare/input.txt``.

    Returns the absolute path. No-op if the file already exists unless
    ``force=True``.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    if force or not os.path.exists(TINYSHAKESPEARE_PATH):
        print(f"[download_data] {TINYSHAKESPEARE_URL} -> {TINYSHAKESPEARE_PATH}")
        urllib.request.urlretrieve(TINYSHAKESPEARE_URL, TINYSHAKESPEARE_PATH)
    return TINYSHAKESPEARE_PATH


def build_char_vocab(text: str) -> tuple[list[str], dict[str, int], dict[int, str]]:
    chars = sorted(set(text))
    char_to_id = {c: i for i, c in enumerate(chars)}
    id_to_char = {i: c for i, c in enumerate(chars)}
    return chars, char_to_id, id_to_char


def get_tinyshakespeare(
    seq_len: int,
    batch_size: int,
    max_samples: Optional[int] = None,
    shuffle: bool = True,
) -> tuple[tf.data.Dataset, int]:
    """Return a ``(dataset, vocab_size)`` pair of char-level next-token batches.

    Each element yielded by ``dataset`` is a tuple of two int32 tensors
    of shape ``(batch_size, seq_len)``: the inputs and the targets
    (inputs shifted left by one).
    """
    path = download_tinyshakespeare()
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    _, char_to_id, _ = build_char_vocab(text)
    ids = [char_to_id[c] for c in text]

    ds = tf.data.Dataset.from_tensor_slices(tf.constant(ids, dtype=tf.int32))
    ds = ds.batch(seq_len + 1, drop_remainder=True)
    ds = ds.map(lambda chunk: (chunk[:-1], chunk[1:]))
    if max_samples is not None:
        ds = ds.take(max_samples)
    if shuffle:
        ds = ds.shuffle(buffer_size=1024, seed=0)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, len(set(text))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()
    path = download_tinyshakespeare(force=args.force)
    print(f"ready: {path}")


if __name__ == "__main__":
    main()
