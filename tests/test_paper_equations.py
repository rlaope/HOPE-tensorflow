"""Equation-level tests: docstring math == what the code actually computes."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from hope.layers import HopeAttention, SelfModifyingLayer
from hope.memory import AssociativeMemory


def test_hebbian_closed_form_repeated_writes():
    """N identical (k, v) writes must equal eta * N * (k.k) * v analytically."""
    eta, N = 0.25, 7
    mem = AssociativeMemory(rule="hebbian", learning_rate=eta, key_dim=4, value_dim=3)
    k = tf.constant([1.0, -0.5, 0.2, 0.7])
    v = tf.constant([2.0, -1.0, 0.5])
    for _ in range(N):
        mem.write(k, v)
    expected = eta * N * float(tf.tensordot(k, k, 1)) * v.numpy()
    np.testing.assert_allclose(mem.retrieve(k).numpy(), expected, atol=1e-5)


def test_delta_rule_equals_l2_gradient_step():
    """Delta update must equal -lr * dL/dM where L = 0.5 * ||M k - v||^2."""
    lr = 0.3
    mem = AssociativeMemory(rule="delta", learning_rate=lr, key_dim=4, value_dim=3)
    tf.random.set_seed(42)
    mem.memory.assign(tf.random.normal((3, 4), seed=42))
    M_before = mem.memory.numpy().copy()

    k_t = tf.constant([1.0, -0.5, 0.2, 0.7])
    v_t = tf.constant([2.0, -1.0, 0.5])

    M_var = tf.Variable(M_before)
    with tf.GradientTape() as tape:
        L = 0.5 * tf.reduce_sum(tf.square(tf.linalg.matvec(M_var, k_t) - v_t))
    grad = tape.gradient(L, M_var)
    expected_M_after = M_before - lr * grad.numpy()

    mem.write(k_t, v_t)
    np.testing.assert_allclose(mem.memory.numpy(), expected_M_after, atol=1e-6)


def test_attention_is_causal_invariance():
    """HopeAttention output at positions 0..3 must be unaffected by positions 4+."""
    layer = HopeAttention(d_model=16, n_heads=2)
    tf.random.set_seed(0)
    x1 = tf.random.normal((1, 8, 16))
    y1 = layer(x1).numpy()

    x2 = tf.concat([x1[:, :4, :], tf.random.normal((1, 4, 16), seed=99)], axis=1)
    y2 = layer(x2).numpy()

    np.testing.assert_allclose(y1[:, :4, :], y2[:, :4, :], atol=1e-5)


def test_self_modifying_layer_grad_matches_finite_diff():
    """SelfModifyingLayer gradient through tf.while_loop must agree with finite diff."""
    tf.random.set_seed(0)
    layer = SelfModifyingLayer(units=4)
    x = tf.random.normal((1, 3, 4))

    # Build the layer.
    _ = layer(x)

    def loss_fn():
        return tf.reduce_sum(layer(x))

    with tf.GradientTape() as tape:
        L = loss_fn()
    analytic_grad = tape.gradient(L, layer.W_k)

    # Central difference for W_k[0, 0] only.
    eps = 1e-3
    W0 = layer.W_k.numpy().copy()
    W_plus = W0.copy(); W_plus[0, 0] += eps
    layer.W_k.assign(W_plus)
    L_plus = float(loss_fn())
    W_minus = W0.copy(); W_minus[0, 0] -= eps
    layer.W_k.assign(W_minus)
    L_minus = float(loss_fn())
    layer.W_k.assign(W0)

    central_diff = (L_plus - L_minus) / (2 * eps)
    analytic = float(analytic_grad[0, 0])
    rel_err = abs(analytic - central_diff) / (abs(central_diff) + 1e-6)
    assert rel_err < 5e-2, f"rel_err={rel_err:.4f}, analytic={analytic:.6f}, fd={central_diff:.6f}"


@pytest.mark.xfail(
    reason="oja rule diverges for large-norm keys (lr * ||k||^2 gain); see docstring in hope/memory.py",
    strict=True,
)
def test_oja_handles_large_norm_keys():
    """Oja rule must stay bounded under adversarially large-norm keys."""
    tf.random.set_seed(0)
    mem = AssociativeMemory(key_dim=6, value_dim=4, rule="oja", learning_rate=0.05)
    max_norm = 0.0
    for _ in range(300):
        k = tf.random.normal((6,)) * 50.0
        v = tf.random.normal((4,))
        mem.write(k, v)
        max_norm = max(max_norm, float(tf.norm(mem.memory)))
    assert max_norm < 1e6, f"oja norm blew up: max_norm={max_norm}"
    assert not np.isnan(mem.memory.numpy()).any(), "oja memory contains NaN"
