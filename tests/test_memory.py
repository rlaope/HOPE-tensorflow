"""Unit tests for hope.memory."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from hope.memory import AssociativeMemory, ContinuumMemorySystem


def test_hebbian_matches_outer_product():
    """Eq. 18: a single Hebbian write must equal the outer product v k^T."""
    mem = AssociativeMemory(key_dim=3, value_dim=2, rule="hebbian", learning_rate=1.0)
    k = tf.constant([1.0, 0.0, -1.0])
    v = tf.constant([2.0, 0.5])
    mem.write(k, v)
    expected = np.outer([2.0, 0.5], [1.0, 0.0, -1.0])
    np.testing.assert_allclose(mem.memory.numpy(), expected, atol=1e-6)


def test_delta_reduces_reconstruction_error():
    """Delta rule (gradient of ||Mk - v||^2) must shrink ||Mk - v||."""
    mem = AssociativeMemory(key_dim=4, value_dim=3, rule="delta", learning_rate=0.5)
    k = tf.constant([1.0, 0.5, -0.2, 0.7])
    v = tf.constant([1.0, 2.0, -1.0])
    err_before = float(tf.norm(mem.retrieve(k) - v))
    mem.write(k, v)
    err_after = float(tf.norm(mem.retrieve(k) - v))
    assert err_after < err_before


def test_oja_keeps_norm_bounded():
    """Oja-style decay must keep ||M||_F from blowing up over many writes."""
    tf.random.set_seed(0)
    mem = AssociativeMemory(key_dim=6, value_dim=4, rule="oja", learning_rate=0.1)
    norms = []
    for _ in range(500):
        k = tf.random.normal((6,))
        v = tf.random.normal((4,))
        mem.write(k, v)
        norms.append(float(tf.norm(mem.memory)))
    assert max(norms) < 50.0, f"oja norm blew up: max={max(norms)}"
    assert not np.isnan(norms[-1])


def test_reset_zeros_memory():
    mem = AssociativeMemory(key_dim=3, value_dim=3, rule="hebbian")
    mem.write(tf.ones(3), tf.ones(3))
    assert float(tf.norm(mem.memory)) > 0
    mem.reset()
    assert float(tf.norm(mem.memory)) == 0


def test_cms_high_freq_bank_changes_more_than_low_freq():
    """Bank with update_every=1 must accumulate more change than update_every=16
    under identical input — verifies the Eq. 71 frequency gating."""
    tf.random.set_seed(0)
    cms = ContinuumMemorySystem(
        dim=4,
        update_every=(1, 16),
        decays=(0.0, 0.0),
        rule="hebbian",
        learning_rate=0.1,
    )
    tokens = tf.random.normal((50, 4))
    for t in range(50):
        cms.forward_step(tokens[t], t)
    fast_norm = float(tf.norm(cms.banks[0].memory.memory))
    slow_norm = float(tf.norm(cms.banks[1].memory.memory))
    assert fast_norm > slow_norm, (
        f"fast bank should accumulate more than slow bank: {fast_norm} vs {slow_norm}"
    )


def test_cms_forward_sequence_shape():
    cms = ContinuumMemorySystem(dim=3, update_every=(1, 4), decays=(0.0, 0.0))
    out = cms.forward_sequence(tf.random.normal((10, 3)))
    assert tuple(out.shape) == (10, 3)
