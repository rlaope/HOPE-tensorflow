"""HOPE Keras layers.

References the paper:

    Behrouz, Razaviyayn, Zhong, Mirrokni.
    "Nested Learning: The Illusion of Deep Learning Architectures."
    NeurIPS 2025. arXiv:2512.24695.

  * Eq. 18: outer-product / Hebbian fast-weight update used here.
  * Eq. 31: a linear layer's slow weights are themselves an associative
    memory mapping inputs to a local surprise signal.
  * Eq. 76, 86 - 88: projection layers W_k, W_v, W_q feeding a
    self-modifying associative memory used in Self-Referential Titans
    (§8.1).
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
    variables; the fast weight ``W_fast`` is *not* a trainable parameter and is
    reset to zeros at the start of every forward pass. That is the
    "self-modifying" behaviour described in §8.1: the operator changes itself
    while processing a single sequence.

    The most recent fast-weight tensor is exposed as ``self.last_fast`` for
    inspection in tests.

    Args:
        units: output / fast-weight dimension.
        eta: write step size ``eta`` applied to ``v_t k_t^T``.
        alpha: retention factor ``alpha`` applied to the previous fast weight
            (analogue of the gate ``alpha`` in Eq. 88).
        name: optional Keras layer name.
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
        if x.shape.rank != 3:
            raise ValueError(f"SelfModifyingLayer expects rank-3 input, got shape {x.shape}")
        k = tf.matmul(x, self.W_k)  # (B, T, units)
        v = tf.matmul(x, self.W_v)
        q = tf.matmul(x, self.W_q)

        B = tf.shape(x)[0]
        T_dyn = tf.shape(x)[1]

        fast0 = tf.zeros((B, self.units, self.units), dtype=x.dtype)
        outputs_ta = tf.TensorArray(dtype=x.dtype, size=T_dyn)

        def step(t, fast_state, ta):
            k_t = k[:, t, :]  # (B, units)
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
        out = outputs_ta.stack()  # (T, B, units)
        out = tf.transpose(out, [1, 0, 2])  # (B, T, units)
        static_t = x.shape[1]
        out.set_shape([x.shape[0], static_t, self.units])
        self.last_fast = final_fast
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units, "eta": self.eta, "alpha": self.alpha})
        return cfg
