"""Unit tests for hope.layers.SelfModifyingLayer."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from hope.layers import SelfModifyingLayer


def test_output_shape_matches_request():
    layer = SelfModifyingLayer(units=8)
    x = tf.random.normal((2, 5, 4))
    y = layer(x)
    assert tuple(y.shape) == (2, 5, 8)


def test_fast_state_changes_between_inputs():
    """Two different input sequences must yield two different final fast weights."""
    tf.random.set_seed(0)
    layer = SelfModifyingLayer(units=6)
    x1 = tf.random.normal((1, 4, 3))
    x2 = tf.random.normal((1, 4, 3))
    _ = layer(x1)
    assert layer.last_fast is not None
    state1 = layer.last_fast.numpy().copy()
    _ = layer(x2)
    state2 = layer.last_fast.numpy().copy()
    assert not np.allclose(state1, state2)
