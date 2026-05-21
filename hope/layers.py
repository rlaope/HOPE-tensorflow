"""HOPE Keras layers.

References the paper:

    Behrouz, Razaviyayn, Zhong, Mirrokni.
    "Nested Learning: The Illusion of Deep Learning Architectures."
    NeurIPS 2025. arXiv:2512.24695.

Sections / equations:

  * Eq. 18: outer-product / Hebbian fast-weight update used by
    :class:`SelfModifyingLayer`.
  * Eq. 31: a linear layer's slow weights are themselves an associative
    memory mapping inputs to a local surprise signal.
  * Eq. 76, 86 - 88: projection layers W_k, W_v, W_q feeding a
    self-modifying associative memory used in Self-Referential Titans
    (§8.1).
  * §8 "Hope-Attention" paragraph: a Hope variant replaces self-modifying
    Titans with global softmax attention (Vaswani et al. 2017). In the
    minimal stack of this repo we keep both blocks; :class:`HopeAttention`
    is the post-CMS softmax-attention fusion block.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras


class SelfModifyingLayer(keras.layers.Layer):
    """Minimal self-modifying layer with a per-token Hebbian fast weight.

    For each token ``x_t`` in an input of shape ``(B, T, d_in)``::

        k_t = x_t W_k,  v_t = x_t W_v,  q_t = x_t W_q
        y_t = W_fast q_t                                   # retrieval (Eq. 78)
        W_fast_{t} = alpha * W_fast_{t-1} + eta * v_t k_t^T  # Hebbian (Eq. 18)

    The slow weights ``W_k``, ``W_v``, ``W_q`` are standard Keras trainable
    variables; the fast weight ``W_fast`` is *not* a trainable parameter and
    is reset to zeros at the start of every forward pass. That is the
    "self-modifying" behaviour described in §8.1: the operator changes
    itself while processing a single sequence.

    Exposed on ``self.last_fast`` after each *eager* forward pass for
    inspection (e.g. unit tests). In graph mode this is set to ``None``
    because the value captured inside ``tf.while_loop`` is not safely
    accessible. Do not depend on ``last_fast`` inside ``@tf.function`` code.
    """

    def __init__(
        self,
        units: int,
        eta: float = 0.5,
        alpha: float = 0.9,
        name: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.units = int(units)
        self.eta = float(eta)
        self.alpha = float(alpha)
        self.last_fast: tf.Tensor | None = None

    def build(self, input_shape) -> None:
        d_in = int(input_shape[-1])
        self.W_k = self.add_weight(
            shape=(d_in, self.units), initializer="glorot_uniform", name="W_k"
        )
        self.W_v = self.add_weight(
            shape=(d_in, self.units), initializer="glorot_uniform", name="W_v"
        )
        self.W_q = self.add_weight(
            shape=(d_in, self.units), initializer="glorot_uniform", name="W_q"
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        if not tf.executing_eagerly():
            # last_fast is only well-defined in eager mode — make the limitation loud.
            self.last_fast = None
        if x.shape.rank != 3:
            raise ValueError(f"SelfModifyingLayer expects rank-3 input, got shape {x.shape}")
        k = tf.matmul(x, self.W_k)
        v = tf.matmul(x, self.W_v)
        q = tf.matmul(x, self.W_q)

        B = tf.shape(x)[0]
        T_dyn = tf.shape(x)[1]

        fast0 = tf.zeros((B, self.units, self.units), dtype=x.dtype)
        outputs_ta = tf.TensorArray(dtype=x.dtype, size=T_dyn)

        def step(t, fast_state, ta):
            k_t = k[:, t, :]
            v_t = v[:, t, :]
            q_t = q[:, t, :]
            y_t = tf.einsum("buv,bv->bu", fast_state, q_t)
            outer = tf.einsum("bv,bk->bvk", v_t, k_t)
            new_fast = self.alpha * fast_state + self.eta * outer
            ta = ta.write(t, y_t)
            return t + 1, new_fast, ta

        _, final_fast, outputs_ta = tf.while_loop(
            cond=lambda t, *_: t < T_dyn,
            body=step,
            loop_vars=(tf.constant(0), fast0, outputs_ta),
        )
        out = outputs_ta.stack()
        out = tf.transpose(out, [1, 0, 2])
        out.set_shape([x.shape[0], x.shape[1], self.units])
        self.last_fast = final_fast
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units, "eta": self.eta, "alpha": self.alpha})
        return cfg


class HopeAttention(keras.layers.Layer):
    """Causal multi-head softmax attention used as HOPE's attention block.

    Reference: §8 "Hope-Attention" paragraph — the paper's Hope-Attention
    variant replaces the self-modifying Titans block with global softmax
    attention (Vaswani et al. 2017). In our minimal stack we keep both
    blocks; this layer sits *after* the CMS to fuse long-range context.

    The implementation is plain scaled dot-product multi-head attention
    with a causal mask, written from scratch (no external transformer
    library), exactly as Eq. 62-style attention is interpreted in §8 of
    the paper.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        name: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_head = self.d_model // self.n_heads

    def build(self, input_shape) -> None:
        d = self.d_model
        self.Wq = self.add_weight(shape=(d, d), initializer="glorot_uniform", name="Wq")
        self.Wk = self.add_weight(shape=(d, d), initializer="glorot_uniform", name="Wk")
        self.Wv = self.add_weight(shape=(d, d), initializer="glorot_uniform", name="Wv")
        self.Wo = self.add_weight(shape=(d, d), initializer="glorot_uniform", name="Wo")
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        if x.shape.rank != 3:
            raise ValueError(f"HopeAttention expects rank-3 input, got {x.shape}")
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]

        q = tf.matmul(x, self.Wq)
        k = tf.matmul(x, self.Wk)
        v = tf.matmul(x, self.Wv)

        def split_heads(t):
            t = tf.reshape(t, (B, T, self.n_heads, self.d_head))
            return tf.transpose(t, [0, 2, 1, 3])  # (B, h, T, d_head)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        scale = tf.cast(self.d_head, x.dtype) ** 0.5
        scores = tf.matmul(q, k, transpose_b=True) / scale  # (B, h, T, T)

        LARGE_NEG = scores.dtype.min / 2.0
        mask = tf.linalg.band_part(tf.ones((T, T), dtype=tf.bool), -1, 0)
        scores = tf.where(mask, scores, tf.cast(LARGE_NEG, scores.dtype))

        attn = tf.nn.softmax(scores, axis=-1)
        out = tf.matmul(attn, v)  # (B, h, T, d_head)
        out = tf.transpose(out, [0, 2, 1, 3])  # (B, T, h, d_head)
        out = tf.reshape(out, (B, T, self.d_model))
        return tf.matmul(out, self.Wo)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "n_heads": self.n_heads})
        return cfg
