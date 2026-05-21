"""nanoGPT-style mini-Transformer baseline for the HOPE comparisons.

The whole point of this baseline is to make the Phase 5 benchmarks a
fair head-to-head: HOPE versus a transformer of *the same parameter
budget*. :meth:`MiniTransformer.matched_to` sweeps ``(n_layers, d_model)``
and returns the candidate whose total parameter count lands closest to
the reference HOPE.

The transformer here is intentionally vanilla: token + positional
embedding -> N (LayerNorm + causal MHA + residual + LayerNorm + MLP
+ residual) blocks -> final LayerNorm -> Dense LM head. The MHA reuses
:class:`hope.layers.HopeAttention` since the paper's "Hope-Attention"
variant is just standard softmax attention anyway.
"""

from __future__ import annotations

import warnings

import tensorflow as tf
from tensorflow import keras

from hope.layers import HopeAttention


def _count_mini_transformer_params(
    vocab_size: int,
    d_model: int,
    n_layers: int,
    max_seq_len: int,
    mlp_ratio: int,
) -> int:
    """Analytical parameter count for :class:`MiniTransformer`.

    Mirrors the actual ``count_params()`` of a fully-built model exactly,
    given the current block layout (no biases on attention projections,
    biases on MLP Dense layers, gamma+beta on every LayerNorm).
    """
    # Embeddings (no biases).
    embed = vocab_size * d_model + max_seq_len * d_model
    # Per-block:
    #   Attn = 4 * d^2 (Wq, Wk, Wv, Wo, no bias)
    #   2 LayerNorms = 2 * 2 * d = 4d  (gamma + beta each)
    #   MLP = d * (mlp_ratio * d) + (mlp_ratio * d) + (mlp_ratio * d) * d + d
    #       = 2 * mlp_ratio * d^2 + (mlp_ratio + 1) * d
    per_block_attn = 4 * d_model * d_model
    per_block_norms = 4 * d_model
    per_block_mlp = 2 * mlp_ratio * d_model * d_model + (mlp_ratio + 1) * d_model
    per_block = per_block_attn + per_block_norms + per_block_mlp
    # Final LayerNorm + LM head (no bias on LM head).
    final = 2 * d_model + d_model * vocab_size
    return embed + n_layers * per_block + final


class TransformerBlock(keras.layers.Layer):
    """Pre-norm decoder block: ``x + Attn(LN(x))`` then ``x + MLP(LN(x))``."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: int = 4,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.mlp_ratio = int(mlp_ratio)
        self.norm1 = keras.layers.LayerNormalization()
        self.norm2 = keras.layers.LayerNormalization()
        self.attn = HopeAttention(self.d_model, n_heads=self.n_heads)
        self.mlp = keras.Sequential(
            [
                keras.layers.Dense(self.d_model * self.mlp_ratio, activation="gelu"),
                keras.layers.Dense(self.d_model),
            ]
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MiniTransformer(keras.Model):
    """Decoder-only mini-Transformer (nanoGPT-class)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        max_seq_len: int = 128,
        mlp_ratio: int = 4,
        name: str = "mini_transformer",
    ) -> None:
        super().__init__(name=name)
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.max_seq_len = int(max_seq_len)
        self.mlp_ratio = int(mlp_ratio)

        self.tok_embed = keras.layers.Embedding(self.vocab_size, self.d_model, name="tok_embed")
        self.pos_embed = keras.layers.Embedding(self.max_seq_len, self.d_model, name="pos_embed")
        self.blocks = [
            TransformerBlock(self.d_model, self.n_heads, self.mlp_ratio, name=f"block_{i}")
            for i in range(self.n_layers)
        ]
        self.norm = keras.layers.LayerNormalization(name="final_norm")
        self.lm_head = keras.layers.Dense(self.vocab_size, use_bias=False, name="lm_head")

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        if x.shape.rank != 2:
            raise ValueError(f"MiniTransformer expects rank-2 int input (B, T), got {x.shape}")
        T = tf.shape(x)[1]
        positions = tf.range(T)
        h = self.tok_embed(x) + self.pos_embed(positions)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return self.lm_head(h)

    @classmethod
    def matched_to(
        cls,
        hope_model,
        tolerance: float = 0.05,
        n_heads: int | None = None,
        max_seq_len: int | None = None,
    ) -> "MiniTransformer":
        """Return a MiniTransformer with param count within ``+/- tolerance``.

        Sweeps ``(mlp_ratio, n_layers, d_model)`` over a bounded grid
        centred on the HOPE model's own dimensions and returns the first
        candidate whose relative parameter-count error drops below
        ``tolerance``. If no candidate clears the bar, returns the closest
        one. Param counts are computed analytically (no model construction
        inside the sweep) so the search is fast even on a wide grid.

        The returned model has a ``matched_rel_error`` attribute giving the
        achieved relative parameter-count error.
        """
        warm_T = min(int(hope_model.max_seq_len), 2)
        _ = hope_model(tf.zeros((1, warm_T), dtype=tf.int32))
        target = int(hope_model.count_params())
        vocab = int(hope_model.vocab_size)
        n_heads = int(n_heads if n_heads is not None else hope_model.n_heads)
        max_seq_len = int(max_seq_len if max_seq_len is not None else hope_model.max_seq_len)

        hope_d = int(hope_model.d_model)
        d_lo = max(n_heads, hope_d // 4)
        d_hi = max(d_lo + n_heads, hope_d * 6)
        d_candidates = [d for d in range(d_lo, d_hi + 1, n_heads)]

        best_cfg: tuple[int, int, int] | None = None
        best_rel = float("inf")
        for mlp_ratio in (1, 2, 3, 4):
            for n_layers in (1, 2, 3, 4, 6):
                for d_model in d_candidates:
                    p = _count_mini_transformer_params(
                        vocab_size=vocab,
                        d_model=d_model,
                        n_layers=n_layers,
                        max_seq_len=max_seq_len,
                        mlp_ratio=mlp_ratio,
                    )
                    rel = abs(p - target) / target
                    if rel < best_rel:
                        best_rel = rel
                        best_cfg = (n_layers, d_model, mlp_ratio)
                    if rel <= tolerance:
                        n_l, d_m, mr = n_layers, d_model, mlp_ratio
                        m = cls(
                            vocab_size=vocab,
                            d_model=d_m,
                            n_layers=n_l,
                            n_heads=n_heads,
                            max_seq_len=max_seq_len,
                            mlp_ratio=mr,
                        )
                        _ = m(tf.zeros((1, min(max_seq_len, 2)), dtype=tf.int32))
                        final_rel = abs(m.count_params() - target) / target
                        m.matched_rel_error = float(final_rel)
                        return m
        if best_cfg is None:  # pragma: no cover - sweep grid is non-empty
            raise RuntimeError("matched_to: empty sweep grid")
        n_l, d_m, mr = best_cfg
        m = cls(
            vocab_size=vocab,
            d_model=d_m,
            n_layers=n_l,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            mlp_ratio=mr,
        )
        _ = m(tf.zeros((1, min(max_seq_len, 2)), dtype=tf.int32))
        final_rel = abs(m.count_params() - target) / target
        if final_rel > tolerance:
            warnings.warn(
                f"MiniTransformer.matched_to: could not meet tolerance={tolerance:.3f}; "
                f"achieved rel_error={final_rel:.4f} "
                f"(target params={target}, achieved={m.count_params()})",
                stacklevel=2,
            )
        m.matched_rel_error = float(final_rel)
        return m

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "max_seq_len": self.max_seq_len,
            "mlp_ratio": self.mlp_ratio,
        }
