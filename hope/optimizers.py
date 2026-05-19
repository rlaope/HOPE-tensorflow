"""Optimizers viewed as associative-memory modules.

References the paper:

    Behrouz, Razaviyayn, Zhong, Mirrokni.
    "Nested Learning: The Illusion of Deep Learning Architectures."
    NeurIPS 2025. arXiv:2512.24695.

  * §4.1 (Eq. 31): a linear layer's slow weights are themselves an
    associative memory mapping inputs to a local surprise signal.
  * §4.2 (Eq. 33 - 34): momentum-based gradient descent reinterpreted
    as a nested associative memory (the momentum buffer is the inner
    memory).
  * §4.5: "Going beyond simple gradient descent and momentum" -- the
    DGD family adds a data-dependent decay term that compresses recent
    inputs into the update rule.

The two classes here are intentionally framework-light: plain Python
objects with a single ``apply_gradients`` method, so they slot into a
``tf.GradientTape`` loop without depending on Keras 3's Optimizer base
class.
"""

from __future__ import annotations

from typing import Iterable

import tensorflow as tf


class DGD:
    """SGD with explicit weight decay, framed as a didactic stand-in
    for the paper's Delta Gradient Descent (DGD, §4.5).

    Update rule (echoes Eq. 31 plus the ``(k k^T) M`` decay term that
    appears in the self-referential Titans update of Eq. 88)::

        w <- w - lr * grad - lr * decay * w

    Equivalent to vanilla SGD with explicit per-step weight decay. The
    paper's full DGD has a *data-dependent* decay term that compresses
    recent inputs into the update rule; here ``decay`` is held constant
    per call so the API stays a tiny drop-in replacement for SGD inside
    a ``tf.GradientTape`` loop. To recover the paper's variant, swap
    ``self.decay`` for a function of recent gradients or inputs.
    """

    def __init__(self, learning_rate: float = 1e-3, decay: float = 0.0) -> None:
        self.lr = float(learning_rate)
        self.decay = float(decay)

    def apply_gradients(self, grads_and_vars: Iterable[tuple[tf.Tensor, tf.Variable]]) -> None:
        for g, v in grads_and_vars:
            if g is None:
                continue
            v.assign_sub(self.lr * (g + self.decay * v))


class DeepOptimizer:
    """Momentum optimizer treating the momentum buffer as an associative memory.

    Reference: §4.2, Eq. 33 - 34. The update rule is the standard
    Polyak / Nesterov form, but the momentum buffer ``m`` is explicitly
    materialised per variable -- the "deep" naming follows the paper's
    framing of momentum as a nested associative memory.

        m_{t+1} = beta * m_t + grad
        w_{t+1} = w_t - lr * (m_{t+1} + decay * w_t)

    A non-zero ``decay`` recovers the Eq. 88 ``(k k^T) M`` decay style
    at the optimizer level.
    """

    def __init__(
        self,
        variables: Iterable[tf.Variable],
        learning_rate: float = 1e-3,
        beta: float = 0.9,
        decay: float = 0.0,
    ) -> None:
        self.lr = float(learning_rate)
        self.beta = float(beta)
        self.decay = float(decay)
        # one momentum buffer per variable, keyed by tf.Variable identity
        self._momentums: dict[int, tf.Variable] = {}
        for v in variables:
            self._momentums[id(v)] = tf.Variable(
                tf.zeros_like(v), trainable=False, name=f"momentum/{v.name}"
            )

    def apply_gradients(self, grads_and_vars: Iterable[tuple[tf.Tensor, tf.Variable]]) -> None:
        for g, v in grads_and_vars:
            if g is None:
                continue
            m = self._momentums.get(id(v))
            if m is None:
                m = tf.Variable(tf.zeros_like(v), trainable=False, name=f"momentum/{v.name}")
                self._momentums[id(v)] = m
            new_m = self.beta * m + g
            m.assign(new_m)
            v.assign_sub(self.lr * (new_m + self.decay * v))
