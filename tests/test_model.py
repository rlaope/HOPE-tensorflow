"""Unit tests for hope.model.HOPE."""

from __future__ import annotations

import tensorflow as tf

from hope.model import HOPE


def _small_hope(vocab_size: int = 8, max_seq_len: int = 4) -> HOPE:
    return HOPE(
        vocab_size=vocab_size,
        d_model=16,
        n_self_mod_layers=1,
        cms_banks=(1, 4),
        cms_decays=(0.01, 0.005),
        n_heads=2,
        max_seq_len=max_seq_len,
    )


def test_forward_shape_b_t_vocab():
    model = _small_hope(vocab_size=10, max_seq_len=8)
    x = tf.constant([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=tf.int32)
    logits = model(x)
    assert tuple(logits.shape) == (1, 8, 10)


def test_loss_decreases_on_synthetic_copy_task():
    """Train HOPE to copy the input — loss must drop over 10 steps."""
    tf.random.set_seed(0)
    model = _small_hope(vocab_size=8, max_seq_len=4)
    opt = tf.keras.optimizers.Adam(1e-2)
    x = tf.constant([[1, 2, 3, 4]], dtype=tf.int32)
    y = x

    losses = []
    for _ in range(10):
        with tf.GradientTape() as tape:
            logits = model(x)
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
            )
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        losses.append(float(loss))

    assert losses[-1] < losses[0], f"loss did not decrease over 10 steps: {losses}"


def test_runs_on_cpu():
    with tf.device("/CPU:0"):
        model = HOPE(
            vocab_size=5,
            d_model=8,
            n_self_mod_layers=1,
            cms_banks=(1,),
            cms_decays=(0.0,),
            n_heads=2,
            max_seq_len=4,
        )
        x = tf.constant([[0, 1, 2, 3]], dtype=tf.int32)
        logits = model(x)
        assert tuple(logits.shape) == (1, 4, 5)
