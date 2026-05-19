"""HOPE model assembly.

References the paper:

    Behrouz, Razaviyayn, Zhong, Mirrokni.
    "Nested Learning: The Illusion of Deep Learning Architectures."
    NeurIPS 2025. arXiv:2512.24695.

  * §8.1 (Eq. 76 - 91): Deep Self-Referential Titans with a self-modifying
    associative memory.
  * §7.1 (Eq. 70 - 71): Continuum Memory System (CMS).
  * §8.3, Figure 5: HOPE combines self-modifying Titans with CMS.
  * Eq. 94 - 97: HOPE forward pass.

This module assembles a minimal didactic HOPE language model:

    token embedding + positional embedding
      -> SelfModifyingLayer (the self-referential Titans block)
      -> ContinuumMemorySystem (the CMS chain of memory banks)
      -> HopeAttention (softmax attention for long-range fusion)
      -> LayerNorm + LM head

It is not a faithful reproduction of the paper's large-scale Hope but it
keeps every component of Figure 5 visible end to end.
"""

from __future__ import annotations

from typing import Sequence

import tensorflow as tf
from tensorflow import keras

from hope.layers import HopeAttention, SelfModifyingLayer
from hope.memory import ContinuumMemorySystem


class HOPE(keras.Model):
    """HOPE language model — Figure 5 of the paper, in TF / Keras.

    Args:
        vocab_size: vocabulary size for the LM head.
        d_model: hidden dimension (must be divisible by ``n_heads``).
        n_self_mod_layers: number of stacked :class:`SelfModifyingLayer`
            blocks before the CMS. The paper's full Hope uses several,
            but a single layer is plenty for a didactic toy.
        cms_banks: per-bank update frequencies for the CMS chain
            (Eq. 71). Smaller numbers = faster bank.
        cms_decays: per-bank multiplicative decay applied every step.
            Must match the length of ``cms_banks``.
        n_heads: number of attention heads in :class:`HopeAttention`.
        max_seq_len: maximum sequence length for the positional embedding.
        name: Keras model name.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_self_mod_layers: int = 1,
        cms_banks: Sequence[int] = (1, 4),
        cms_decays: Sequence[float] = (0.01, 0.005),
        n_heads: int = 4,
        max_seq_len: int = 128,
        name: str = "hope",
    ) -> None:
        super().__init__(name=name)
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        if len(cms_banks) != len(cms_decays):
            raise ValueError("cms_banks and cms_decays must be the same length")
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.max_seq_len = int(max_seq_len)
        self.n_heads = int(n_heads)

        self.token_embed = keras.layers.Embedding(self.vocab_size, self.d_model, name="tok_embed")
        self.pos_embed = keras.layers.Embedding(self.max_seq_len, self.d_model, name="pos_embed")
        self.self_mod_layers = [
            SelfModifyingLayer(self.d_model, name=f"self_mod_{i}")
            for i in range(int(n_self_mod_layers))
        ]
        self.cms = ContinuumMemorySystem(
            dim=self.d_model,
            update_every=tuple(int(u) for u in cms_banks),
            decays=tuple(float(d) for d in cms_decays),
            rule="hebbian",
            learning_rate=0.1,
            name="cms",
        )
        self.attn = HopeAttention(self.d_model, n_heads=self.n_heads, name="hope_attn")
        self.norm = keras.layers.LayerNormalization(name="norm")
        self.lm_head = keras.layers.Dense(self.vocab_size, use_bias=False, name="lm_head")

    def _apply_cms_per_sample(self, h: tf.Tensor) -> tf.Tensor:
        """Run the CMS chain once per batch element.

        CMS is stateful per call (each bank holds a tf.Variable); to avoid
        cross-sample contamination we reset it before each sample. This
        keeps the implementation simple and eager-friendly at the cost of
        a Python-side batch loop.
        """
        batch_size = int(h.shape[0]) if h.shape[0] is not None else int(tf.shape(h)[0])
        outs = []
        for b in range(batch_size):
            self.cms.reset()
            outs.append(self.cms.forward_sequence(h[b]))
        return tf.stack(outs, axis=0)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        if x.shape.rank != 2:
            raise ValueError(f"HOPE expects rank-2 int input (B, T), got shape {x.shape}")
        T = tf.shape(x)[1]
        positions = tf.range(T)
        h = self.token_embed(x) + self.pos_embed(positions)  # (B, T, d)

        for sm in self.self_mod_layers:
            h = h + sm(h)  # residual around the self-modifying block

        h = self._apply_cms_per_sample(h)  # (B, T, d)
        h = h + self.attn(h)               # residual around attention
        h = self.norm(h)
        logits = self.lm_head(h)           # (B, T, vocab_size)
        return logits

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_self_mod_layers": len(self.self_mod_layers),
            "cms_banks": [b.update_every for b in self.cms.banks],
            "cms_decays": [b.decay for b in self.cms.banks],
            "n_heads": self.n_heads,
            "max_seq_len": self.max_seq_len,
        }
