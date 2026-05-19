"""Unit tests for hope.baseline.MiniTransformer and hope.optimizers."""

from __future__ import annotations

import tensorflow as tf

from hope.baseline import MiniTransformer
from hope.model import HOPE
from hope.optimizers import DGD, DeepOptimizer


def test_mini_transformer_forward_shape():
    m = MiniTransformer(vocab_size=10, d_model=16, n_layers=2, n_heads=2, max_seq_len=8)
    x = tf.constant([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=tf.int32)
    logits = m(x)
    assert tuple(logits.shape) == (1, 8, 10)


def test_matched_to_param_budget_within_tolerance():
    """matched_to must return a MiniTransformer whose param count is within
    +/- 5% of the reference HOPE."""
    hope = HOPE(
        vocab_size=32,
        d_model=32,
        n_self_mod_layers=1,
        cms_banks=(1, 4),
        cms_decays=(0.01, 0.005),
        n_heads=4,
        max_seq_len=8,
    )
    _ = hope(tf.zeros((1, 4), dtype=tf.int32))
    target = hope.count_params()
    m = MiniTransformer.matched_to(hope, tolerance=0.05)
    p = m.count_params()
    rel = abs(p - target) / target
    assert rel <= 0.05, f"matched_to landed at {p} vs target {target} (rel {rel:.3f})"


def test_dgd_decreases_quadratic_loss():
    w = tf.Variable(tf.ones(8))
    opt = DGD(learning_rate=0.1, decay=0.0)
    for _ in range(50):
        with tf.GradientTape() as tape:
            loss = 0.5 * tf.reduce_sum(w * w)
        grads = tape.gradient(loss, [w])
        opt.apply_gradients(zip(grads, [w]))
    assert float(tf.norm(w)) < 0.1


def test_deep_optimizer_decreases_quadratic_loss():
    w = tf.Variable(tf.ones(8))
    opt = DeepOptimizer(variables=[w], learning_rate=0.05, beta=0.9, decay=0.0)
    for _ in range(80):
        with tf.GradientTape() as tape:
            loss = 0.5 * tf.reduce_sum(w * w)
        grads = tape.gradient(loss, [w])
        opt.apply_gradients(zip(grads, [w]))
    assert float(tf.norm(w)) < 0.1
