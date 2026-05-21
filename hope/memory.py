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
  * Eq. 70 - 71 (§7.1): Continuum Memory System (CMS) — a chain of memory
    blocks with per-bank update frequencies C^(l).

The classes here are intentionally small and side-effectful; they keep
their state in a single ``tf.Variable`` so a SelfModifyingLayer can write
to them token by token inside a forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import tensorflow as tf

UpdateRule = Literal["hebbian", "delta", "oja"]
_VALID_RULES = ("hebbian", "delta", "oja")


class AssociativeMemory(tf.Module):
    """Matrix-valued associative memory with selectable update rules.

    The memory state is a single matrix ``M`` of shape
    ``(value_dim, key_dim)``. Each ``write(k, v)`` call mutates it under
    one of three rules:

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

      The extra decay term has per-step gain ``lr * ||k||^2`` applied to
      ``M``. For unit-scale keys (``||k|| ~ 1``) and ``lr <= 0.1`` this
      keeps ``||M||`` bounded; for keys with large norm the step can
      overshoot and the iteration diverges. Callers feeding wide-dynamic-
      range keys should either normalise keys before ``write`` or scale
      ``learning_rate`` down by roughly ``1 / max(||k||^2, 1)``. The
      ``test_oja_handles_large_norm_keys`` regression test pins this
      limitation as a known boundary of the implementation.
    """

    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        rule: UpdateRule = "hebbian",
        learning_rate: float = 1.0,
        dtype: tf.DType = tf.float32,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if rule not in _VALID_RULES:
            raise ValueError(f"Unknown rule {rule!r}; expected one of {_VALID_RULES}.")
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.rule = rule
        self.learning_rate = float(learning_rate)
        self.dtype = dtype
        self.memory = tf.Variable(
            tf.zeros((self.value_dim, self.key_dim), dtype=self.dtype),
            trainable=False,
            name="memory",
        )

    def reset(self) -> None:
        """Zero the memory state."""
        self.memory.assign(tf.zeros_like(self.memory))

    def retrieve(self, k: tf.Tensor) -> tf.Tensor:
        """Return ``M @ k``. Accepts ``k`` of shape ``(..., key_dim)``."""
        k = tf.cast(k, self.dtype)
        return tf.einsum("vk,...k->...v", self.memory, k)

    def write(self, k: tf.Tensor, v: tf.Tensor) -> tf.Tensor:
        """Apply one (key, value) write and return the updated memory."""
        # explicit cast to memory dtype — callers asking for fp16/bf16 should pass dtype=... at construction
        k = tf.cast(k, self.dtype)
        v = tf.cast(v, self.dtype)
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
                decay = tf.matmul(self.memory, kk)  # M (k k^T)
                delta = -self.learning_rate * err_outer - self.learning_rate * decay
        self.memory.assign_add(delta)
        return self.memory

    def write_batch(self, keys: tf.Tensor, values: tf.Tensor) -> tf.Tensor:
        """Apply a sequence of writes in chronological order."""
        keys = tf.cast(keys, self.dtype)
        values = tf.cast(values, self.dtype)
        if int(keys.shape[0]) != int(values.shape[0]):
            raise ValueError("keys and values must share the leading T dimension")
        for t in range(int(keys.shape[0])):
            self.write(keys[t], values[t])
        return self.memory


@dataclass
class CMSBank:
    """A single bank inside a :class:`ContinuumMemorySystem`.

    Attributes:
        memory: backing :class:`AssociativeMemory` state.
        update_every: chunk size ``C^(l)`` from Eq. 71 — this bank is
            written to only on steps where ``step_index % update_every == 0``.
        decay: per-step multiplicative shrink applied to the memory
            (analogue of the gate ``alpha`` in Eq. 88; here used at the
            CMS-bank granularity rather than per individual key).
    """

    memory: AssociativeMemory
    update_every: int
    decay: float


class ContinuumMemorySystem(tf.Module):
    """Continuum Memory System (CMS).

    Reference: §7.1 of the paper.

    Implements Eq. 70 (sequential composition of MLP / memory blocks)
    and Eq. 71 (per-bank update frequency ``C^(l)``). Banks are run in
    order: bank 0 sees the input, bank ``i+1`` sees bank ``i``'s
    retrieval. On every step we:

      1. apply each bank's multiplicative decay (per-step shrink),
      2. write the current token-shaped vector into the bank if the
         current step index is divisible by its ``update_every``,
      3. retrieve from the bank to produce the input for the next bank.

    The faster banks (small ``update_every``) react to every token and
    accumulate short-term context; slower banks accumulate persistent
    structure across many steps and barely change. This is the
    catastrophic-forgetting protection described in §7.1.
    """

    def __init__(
        self,
        dim: int,
        update_every: Sequence[int] = (1, 4, 16),
        decays: Sequence[float] = (0.01, 0.005, 0.001),
        rule: UpdateRule = "hebbian",
        learning_rate: float = 1.0,
        dtype: tf.DType = tf.float32,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if len(update_every) != len(decays):
            raise ValueError("update_every and decays must be the same length")
        if len(update_every) == 0:
            raise ValueError("CMS requires at least one bank")
        self.dim = int(dim)
        self.rule = rule
        self.dtype = dtype
        self.banks: list[CMSBank] = [
            CMSBank(
                memory=AssociativeMemory(
                    key_dim=self.dim,
                    value_dim=self.dim,
                    rule=rule,
                    learning_rate=learning_rate,
                    dtype=dtype,
                    name=f"bank_{i}",
                ),
                update_every=int(ue),
                decay=float(d),
            )
            for i, (ue, d) in enumerate(zip(update_every, decays))
        ]

    def reset(self) -> None:
        """Zero every bank's memory."""
        for b in self.banks:
            b.memory.reset()

    def forward_step(self, token: tf.Tensor, step_index: int) -> tf.Tensor:
        """Push one token through the bank chain; return the final retrieval."""
        x = tf.cast(token, self.dtype)
        if int(x.shape[-1]) != self.dim:
            raise ValueError(f"token last dim {int(x.shape[-1])} != dim {self.dim}")
        for b in self.banks:
            if b.decay > 0:
                b.memory.memory.assign(b.memory.memory * (1.0 - b.decay))
            if step_index % b.update_every == 0:
                b.memory.write(x, x)
            x = b.memory.retrieve(x)
        return x

    def forward_sequence(self, tokens: tf.Tensor) -> tf.Tensor:
        """Apply :meth:`forward_step` to a sequence of shape ``(T, dim)``."""
        tokens = tf.cast(tokens, self.dtype)
        outs = []
        for t in range(int(tokens.shape[0])):
            outs.append(self.forward_step(tokens[t], t))
        return tf.stack(outs, axis=0)
