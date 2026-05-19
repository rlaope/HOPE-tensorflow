"""Associative-memory primitives for HOPE.

This module is one of the basic building blocks for the HOPE architecture
described in:

    Behrouz, Razaviyayn, Zhong, Mirrokni.
    "Nested Learning: The Illusion of Deep Learning Architectures."
    NeurIPS 2025. arXiv:2512.24695.

Key references inside the paper:

  * Definition 1 (Eq. 6): an associative memory M(.) is the operator that
    maps a set of keys K to values V by minimising a mapping objective
    L_tilde.
  * Eq. 17 - 18: linear-attention / outer-product (Hebbian) update,
    M_{t+1} = M_t + v_{t+1} k_{t+1}^T.
  * Eq. 88 / 92 / 93: general self-referential Titans update rule with a
    Hebbian decay term k k^T M.

The classes here are intentionally small and side-effectful; they keep their
state in a single `tf.Variable` so that a `SelfModifyingLayer` can write to
them token by token inside a forward pass.
"""

from __future__ import annotations

from typing import Literal

import tensorflow as tf

UpdateRule = Literal["hebbian", "delta", "oja"]
_VALID_RULES = ("hebbian", "delta", "oja")


class AssociativeMemory(tf.Module):
    """Matrix-valued associative memory with selectable update rules.

    The memory state is a single matrix ``M`` of shape ``(value_dim, key_dim)``.
    Each ``write(k, v)`` call mutates it under one of three rules:

    * ``hebbian`` -- Eq. 18::

          M <- M + lr * v k^T

      The classic linear-attention outer-product update; equivalent to the
      gradient of the dot-product similarity loss ``-<M k, v>``.

    * ``delta`` -- gradient of the L2-regression objective
      ``L(M; k, v) = ||M k - v||^2`` (the loss used by self-referential
      Titans in Eq. 93)::

          M <- M - lr * (M k - v) k^T

    * ``oja`` -- Eq. 88-style update with ``alpha = 1`` and a Hebbian
      decay term ``k k^T`` regularising the memory norm::

          M <- M - lr * (M k - v) k^T - lr * (k k^T) M

      Unlike the plain delta rule, the extra decay term keeps ``||M||``
      bounded for arbitrary key streams.

    Args:
        key_dim: dimension of the key vector ``k``.
        value_dim: dimension of the value vector ``v``.
        rule: one of ``"hebbian"``, ``"delta"``, ``"oja"``.
        learning_rate: scalar step size ``lr`` applied to every write.
        name: optional ``tf.Module`` name.
    """

    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        rule: UpdateRule = "hebbian",
        learning_rate: float = 1.0,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if rule not in _VALID_RULES:
            raise ValueError(f"Unknown rule {rule!r}; expected one of {_VALID_RULES}.")
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.rule = rule
        self.learning_rate = float(learning_rate)
        self.memory = tf.Variable(
            tf.zeros((self.value_dim, self.key_dim), dtype=tf.float32),
            trainable=False,
            name="memory",
        )

    def reset(self) -> None:
        """Zero the memory state."""
        self.memory.assign(tf.zeros_like(self.memory))

    def retrieve(self, k: tf.Tensor) -> tf.Tensor:
        """Return ``M @ k``. Accepts ``k`` of shape ``(..., key_dim)``."""
        k = tf.cast(k, tf.float32)
        return tf.einsum("vk,...k->...v", self.memory, k)

    def write(self, k: tf.Tensor, v: tf.Tensor) -> tf.Tensor:
        """Apply one (key, value) write and return the updated memory."""
        k = tf.cast(k, tf.float32)
        v = tf.cast(v, tf.float32)
        if int(k.shape[-1]) != self.key_dim:
            raise ValueError(f"key last dim {int(k.shape[-1])} != key_dim {self.key_dim}")
        if int(v.shape[-1]) != self.value_dim:
            raise ValueError(f"value last dim {int(v.shape[-1])} != value_dim {self.value_dim}")
        outer_vk = tf.einsum("v,k->vk", v, k)  # (value_dim, key_dim)

        if self.rule == "hebbian":
            delta = self.learning_rate * outer_vk
        else:
            pred = tf.linalg.matvec(self.memory, k)  # M k, shape (value_dim,)
            err = pred - v  # (value_dim,)
            err_outer = tf.einsum("v,k->vk", err, k)  # (M k - v) k^T
            if self.rule == "delta":
                delta = -self.learning_rate * err_outer
            else:  # oja
                kk = tf.einsum("a,b->ab", k, k)  # k k^T, shape (key_dim, key_dim)
                decay = tf.matmul(self.memory, kk)  # M (k k^T), shape (value_dim, key_dim)
                delta = -self.learning_rate * err_outer - self.learning_rate * decay
        self.memory.assign_add(delta)
        return self.memory

    def write_batch(self, keys: tf.Tensor, values: tf.Tensor) -> tf.Tensor:
        """Apply a sequence of writes in chronological order.

        Args:
            keys: shape ``(T, key_dim)``.
            values: shape ``(T, value_dim)``.

        Returns:
            The memory state after the last write.
        """
        keys = tf.cast(keys, tf.float32)
        values = tf.cast(values, tf.float32)
        if int(keys.shape[0]) != int(values.shape[0]):
            raise ValueError("keys and values must share the leading T dimension")
        for t in range(int(keys.shape[0])):
            self.write(keys[t], values[t])
        return self.memory
