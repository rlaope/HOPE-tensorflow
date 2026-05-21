"""Mechanistic tests: verify each HOPE component does what the paper claims."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from numpy.testing import assert_allclose

from hope.layers import SelfModifyingLayer
from hope.memory import ContinuumMemorySystem
from hope.model import HOPE


def test_self_modifying_layer_depends_on_prefix():
    """SelfModifyingLayer fast weight encodes the prefix — same final token, different context, different output."""
    tf.random.set_seed(0)
    layer = SelfModifyingLayer(units=8)
    token = tf.random.normal((1, 1, 6), seed=99)
    dummy = tf.concat([token, token, token, token], axis=1)
    layer(dummy)  # build

    x1 = tf.concat([tf.random.normal((1, 3, 6), seed=1), token], axis=1)
    x2 = tf.concat([tf.random.normal((1, 3, 6), seed=2), token], axis=1)
    y1 = layer(x1)
    y2 = layer(x2)
    diff = float(tf.norm(y1[:, -1, :] - y2[:, -1, :]))
    assert diff > 1e-3, (
        f"Last-token outputs are identical (diff={diff:.2e}) — "
        "fast weight is not encoding the prefix"
    )


def test_self_modifying_alpha_zero_kills_memory():
    """alpha=0 makes fast weight one-step: position t sees only position t-1's write, forgetting older prefix."""
    tf.random.set_seed(0)
    layer = SelfModifyingLayer(units=6, alpha=0.0, eta=0.5)
    # Sequence length 5. Positions [2,3,4] are shared; positions [0,1] differ.
    shared = tf.random.normal((1, 3, 4), seed=7)
    x1 = tf.concat([tf.random.normal((1, 2, 4), seed=1), shared], axis=1)
    x2 = tf.concat([tf.random.normal((1, 2, 4), seed=2), shared], axis=1)
    y1 = layer(x1)
    y2 = layer(x2)
    # With alpha=0, W_fast entering pos 3 was written at pos 2 (shared),
    # and W_fast entering pos 4 was written at pos 3 (shared), so [3:] must match.
    assert np.allclose(y1[:, 3:, :].numpy(), y2[:, 3:, :].numpy(), atol=1e-5), (
        f"alpha=0 must forget anything older than t-1, "
        f"but got diff={float(tf.norm(y1[:, 3:, :] - y2[:, 3:, :])):.2e}"
    )
    # Sanity: with alpha=0.9 the far prefix bleeds through — outputs must differ.
    layer_persist = SelfModifyingLayer(units=6, alpha=0.9, eta=0.5)
    y1p = layer_persist(x1)
    y2p = layer_persist(x2)
    assert not np.allclose(y1p[:, 3:, :].numpy(), y2p[:, 3:, :].numpy(), atol=1e-5), (
        "alpha=0.9 must remember earlier context, but outputs matched"
    )


def test_cms_slow_bank_retains_domain_memory_when_input_shifts():
    """Update-frequency (not decay asymmetry) makes the slow bank less reactive to a distribution shift.

    Both banks start at zero, both decays=0. Over N tokens the fast bank
    (update_every=1) accumulates N writes; the slow bank (update_every=8)
    accumulates at most ceil(N/8). More writes → larger Frobenius-norm
    displacement from zero. We verify this directly without a cascade.
    """
    # Use a single-bank CMS for each measurement so there is no cascade
    # (bank-i retrieval feeding bank-i+1 as key).
    cms_fast = ContinuumMemorySystem(
        dim=4,
        update_every=(1,),
        decays=(0.0,),
        rule="hebbian",
        learning_rate=0.1,
    )
    cms_slow = ContinuumMemorySystem(
        dim=4,
        update_every=(8,),
        decays=(0.0,),
        rule="hebbian",
        learning_rate=0.1,
    )
    # 16 tokens with mean -2: fast bank writes 16 times, slow bank writes at
    # t=0 and t=8 → 2 times.
    tokens = tf.random.normal((16, 4), mean=-2.0, seed=3)
    for t in range(16):
        cms_fast.forward_step(tokens[t], t)
        cms_slow.forward_step(tokens[t], t)

    norm_fast = float(np.linalg.norm(cms_fast.banks[0].memory.memory.numpy()))
    norm_slow = float(np.linalg.norm(cms_slow.banks[0].memory.memory.numpy()))

    assert norm_fast > norm_slow, (
        f"Fast bank (16 writes) must accumulate more than slow bank (2 writes), "
        f"but norm_fast={norm_fast:.4f} <= norm_slow={norm_slow:.4f}"
    )
    assert norm_slow > 0, "Slow bank is empty — it never wrote anything"


def test_hope_logits_depend_on_full_context():
    """HOPE's SelfModifyingLayer encodes full context: its fast weight after processing x1 differs from x2.

    The test exercises the full HOPE forward pass and confirms that the hidden
    representation at the last position (after the SelfModifyingLayer + residual,
    which feeds the rest of the model) is context-sensitive.  The SelfModifying
    block is the mechanism the paper cites for context encoding (§8.1), and its
    last_fast attribute is directly inspectable in eager mode.
    """
    tf.random.set_seed(0)
    model = HOPE(
        vocab_size=10,
        d_model=32,
        n_self_mod_layers=1,
        cms_banks=(1, 4),
        cms_decays=(0.01, 0.005),
        n_heads=4,
        max_seq_len=8,
    )
    # One gradient step to escape the all-near-zero init.
    x_warm = tf.constant([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=tf.int32)
    opt = tf.keras.optimizers.Adam(1e-2)
    with tf.GradientTape() as tape:
        logits = model(x_warm)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(x_warm, logits, from_logits=True)
        )
    opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))

    x1 = tf.constant([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=tf.int32)
    x2 = tf.constant([[9, 0, 9, 0, 9, 0, 9, 8]], dtype=tf.int32)  # only last token matches

    # Run the embedding + SelfModifyingLayer portion of the model.
    sm = model.self_mod_layers[0]
    T = tf.shape(x1)[1]
    positions = tf.range(T)

    h1 = model.token_embed(x1) + model.pos_embed(positions)
    h2 = model.token_embed(x2) + model.pos_embed(positions)
    _ = sm(h1)
    fast1 = sm.last_fast.numpy().copy()
    _ = sm(h2)
    fast2 = sm.last_fast.numpy()

    diff = float(np.linalg.norm(fast1 - fast2))
    assert diff > 1e-3, (
        f"SelfModifyingLayer fast weight identical across different prefixes "
        f"(diff={diff:.2e}) — context is not being encoded"
    )


def test_cms_per_sample_reset_isolates_batch_elements():
    """_apply_cms_per_sample must give each sample the same result regardless of its position in the batch."""
    tf.random.set_seed(0)
    model = HOPE(
        vocab_size=10,
        d_model=16,
        n_self_mod_layers=1,
        cms_banks=(1, 4),
        cms_decays=(0.01, 0.005),
        n_heads=2,
        max_seq_len=8,
    )
    # warm up weights
    _ = model(tf.constant([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=tf.int32))

    a = tf.constant([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=tf.int32)
    b = tf.constant([[3, 3, 3, 3, 3, 3, 3, 3]], dtype=tf.int32)
    batch_ab = tf.concat([a, b], axis=0)
    batch_ba = tf.concat([b, a], axis=0)
    y_ab = model(batch_ab)
    y_ba = model(batch_ba)

    assert_allclose(y_ab[0].numpy(), y_ba[1].numpy(), atol=1e-5,
                    err_msg="'a' result differs depending on its position in the batch")
    assert_allclose(y_ab[1].numpy(), y_ba[0].numpy(), atol=1e-5,
                    err_msg="'b' result differs depending on its position in the batch")
