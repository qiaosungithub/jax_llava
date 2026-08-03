import jax
import jax.numpy as jnp
import numpy as np

from models.paligemma import token_xent_loss, token_xent_loss_from_hidden


def test_token_xent_from_hidden_matches_full_logits():
    rng = np.random.RandomState(0)
    hidden = jnp.asarray(rng.randn(2, 5, 7).astype(np.float32))
    embedding = jnp.asarray(rng.randn(23, 7).astype(np.float32))
    labels = jnp.asarray(
        [
            [0, 3, 22, -100, 5],
            [7, -100, 1, 9, 12],
        ],
        dtype=jnp.int32,
    )

    centered_embedding = embedding - jax.lax.stop_gradient(
        embedding.mean(axis=0, keepdims=True)
    )
    full_logits = jnp.einsum("...d,vd->...v", hidden, centered_embedding)
    full_loss = token_xent_loss(full_logits, labels)
    raw_full_loss = token_xent_loss(
        jnp.einsum("...d,vd->...v", hidden, embedding),
        labels,
    )
    chunked_loss, pred_ids, _ = token_xent_loss_from_hidden(
        hidden,
        embedding,
        labels,
        chunk_size=5,
    )

    np.testing.assert_allclose(
        np.asarray(chunked_loss),
        np.asarray(full_loss),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(full_loss),
        np.asarray(raw_full_loss),
        rtol=1e-3,
        atol=1e-3,
    )
    assert pred_ids.shape == labels.shape


def test_token_xent_from_hidden_matches_full_logits_with_cfg_subtract():
    rng = np.random.RandomState(1)
    hidden = jnp.asarray(rng.randn(2, 4, 6).astype(np.float32))
    subtract_hidden = jnp.asarray(rng.randn(2, 4, 6).astype(np.float32))
    embedding = jnp.asarray(rng.randn(19, 6).astype(np.float32))
    labels = jnp.asarray(
        [
            [2, 18, -100, 4],
            [0, 3, 6, -100],
        ],
        dtype=jnp.int32,
    )
    alpha = 0.35
    softcap = 3.0

    # token_xent_loss_from_hidden centers the table unconditionally, so the
    # dense reference has to decode through the same centered table.
    centered = embedding - jax.lax.stop_gradient(embedding.mean(axis=0, keepdims=True))
    cond_logits = jnp.einsum("...d,vd->...v", hidden, centered)
    cond_logits = jnp.tanh(cond_logits / softcap) * softcap
    subtract_logits = jnp.einsum("...d,vd->...v", subtract_hidden, centered)
    subtract_logits = jnp.tanh(subtract_logits / softcap) * softcap
    full_logits = cond_logits - alpha * jax.lax.stop_gradient(subtract_logits)
    full_loss = token_xent_loss(full_logits, labels)

    chunked_loss, _, _ = token_xent_loss_from_hidden(
        hidden,
        embedding,
        labels,
        final_logit_softcap=softcap,
        chunk_size=7,
        subtract_hidden=subtract_hidden,
        subtract_alpha=alpha,
    )

    np.testing.assert_allclose(
        np.asarray(chunked_loss),
        np.asarray(full_loss),
        rtol=1e-5,
        atol=1e-5,
    )


def test_token_xent_from_hidden_is_nonnegative_for_large_logits():
    rng = np.random.RandomState(2)
    base_embedding = jnp.asarray(rng.randn(256, 32).astype(np.float32))
    labels = jnp.asarray(rng.randint(0, 256, size=(2, 8)), dtype=jnp.int32)
    embedding = base_embedding * 64.0
    hidden = embedding[labels] * 16.0

    loss, _, _ = token_xent_loss_from_hidden(
        hidden,
        embedding,
        labels,
        chunk_size=31,
    )

    assert np.asarray(loss) >= -1e-6


def test_token_xent_from_hidden_invariant_to_table_common_mode():
    """CE is invariant to adding one shared row-offset to the table."""
    rng = np.random.RandomState(2)
    hidden = jnp.asarray(rng.randn(2, 5, 7).astype(np.float32) * 10)
    embedding = jnp.asarray(rng.randn(23, 7).astype(np.float32))
    shift = jnp.asarray(rng.randn(1, 7).astype(np.float32) * 5)
    labels = jnp.asarray(
        [[0, 3, 22, -100, 5], [7, -100, 1, 9, 12]],
        dtype=jnp.int32,
    )

    base_loss, base_pred, _ = token_xent_loss_from_hidden(
        hidden, embedding, labels, chunk_size=5
    )
    shifted_loss, shifted_pred, _ = token_xent_loss_from_hidden(
        hidden, embedding + shift, labels, chunk_size=5
    )

    np.testing.assert_allclose(
        np.asarray(shifted_loss),
        np.asarray(base_loss),
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_array_equal(np.asarray(shifted_pred), np.asarray(base_pred))
    assert float(base_loss) >= 0.0


def test_token_xent_common_mode_preserves_output_gradients():
    """A shared table-row shift must not alter CE output gradients."""
    rng = np.random.RandomState(3)
    hidden = jnp.asarray(rng.randn(2, 4, 6).astype(np.float32))
    embedding = jnp.asarray(rng.randn(29, 6).astype(np.float32))
    shift = jnp.asarray(rng.randn(1, 6).astype(np.float32) * 3)
    labels = jnp.asarray(
        [[0, 5, -100, 28], [7, 2, 11, -100]],
        dtype=jnp.int32,
    )

    def loss_fn(h, table):
        loss, _, _ = token_xent_loss_from_hidden(
            h,
            table,
            labels,
            chunk_size=7,
        )
        return loss

    base_grads = jax.grad(loss_fn, argnums=(0, 1))(hidden, embedding)
    shifted_grads = jax.grad(loss_fn, argnums=(0, 1))(
        hidden,
        embedding + shift,
    )

    np.testing.assert_allclose(
        np.asarray(shifted_grads[0]),
        np.asarray(base_grads[0]),
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(shifted_grads[1]),
        np.asarray(base_grads[1]),
        rtol=1e-4,
        atol=1e-4,
    )


def test_detached_center_matches_raw_full_ce_gradients():
    """Detached centering preserves the standard CE value and gradients."""
    rng = np.random.RandomState(4)
    hidden = jnp.asarray(rng.randn(2, 3, 5).astype(np.float32))
    embedding = jnp.asarray(rng.randn(17, 5).astype(np.float32))
    labels = jnp.asarray([[0, 16, -100], [3, 8, 5]], dtype=jnp.int32)

    def chunked_loss(h, table):
        return token_xent_loss_from_hidden(
            h, table, labels, chunk_size=6
        )[0]

    def centered_full_loss(h, table):
        centered_table = table - jax.lax.stop_gradient(
            table.mean(axis=0, keepdims=True)
        )
        logits = jnp.einsum("...d,vd->...v", h, centered_table)
        return token_xent_loss(logits, labels)

    def raw_full_loss(h, table):
        return token_xent_loss(
            jnp.einsum("...d,vd->...v", h, table), labels
        )

    chunked_value, chunked_grads = jax.value_and_grad(
        chunked_loss, argnums=(0, 1)
    )(hidden, embedding)
    centered_value, centered_grads = jax.value_and_grad(
        centered_full_loss, argnums=(0, 1)
    )(hidden, embedding)
    raw_value, raw_grads = jax.value_and_grad(
        raw_full_loss, argnums=(0, 1)
    )(hidden, embedding)

    np.testing.assert_allclose(
        chunked_value, centered_value, rtol=1e-4, atol=1e-4
    )
    for chunked_grad, centered_grad in zip(chunked_grads, centered_grads):
        np.testing.assert_allclose(
            np.asarray(chunked_grad),
            np.asarray(centered_grad),
            rtol=1e-4,
            atol=1e-4,
        )
    np.testing.assert_allclose(
        centered_value, raw_value, rtol=5e-3, atol=5e-3
    )
    for centered_grad, raw_grad in zip(centered_grads, raw_grads):
        np.testing.assert_allclose(
            np.asarray(centered_grad),
            np.asarray(raw_grad),
            rtol=5e-3,
            atol=5e-3,
        )


def test_detached_center_matches_raw_cfg_gradients_without_softcap():
    """The zero-softcap CFG path keeps standard forward/backward semantics."""
    rng = np.random.RandomState(5)
    hidden = jnp.asarray(rng.randn(2, 3, 5).astype(np.float32))
    subtract_hidden = jnp.asarray(rng.randn(2, 3, 5).astype(np.float32))
    embedding = jnp.asarray(rng.randn(19, 5).astype(np.float32))
    labels = jnp.asarray([[1, -100, 18], [4, 7, 2]], dtype=jnp.int32)
    alpha = 0.4

    def chunked_loss(h, subtract_h, table):
        return token_xent_loss_from_hidden(
            h,
            table,
            labels,
            chunk_size=7,
            subtract_hidden=subtract_h,
            subtract_alpha=alpha,
        )[0]

    def centered_full_loss(h, subtract_h, table):
        centered_table = table - jax.lax.stop_gradient(
            table.mean(axis=0, keepdims=True)
        )
        logits = jnp.einsum("...d,vd->...v", h, centered_table)
        subtract_logits = jnp.einsum(
            "...d,vd->...v", subtract_h, centered_table
        )
        logits = logits - alpha * jax.lax.stop_gradient(subtract_logits)
        return token_xent_loss(logits, labels)

    def raw_full_loss(h, subtract_h, table):
        logits = jnp.einsum("...d,vd->...v", h, table)
        subtract_logits = jnp.einsum("...d,vd->...v", subtract_h, table)
        logits = logits - alpha * jax.lax.stop_gradient(subtract_logits)
        return token_xent_loss(logits, labels)

    chunked_value, chunked_grads = jax.value_and_grad(
        chunked_loss, argnums=(0, 1, 2)
    )(hidden, subtract_hidden, embedding)
    centered_value, centered_grads = jax.value_and_grad(
        centered_full_loss, argnums=(0, 1, 2)
    )(hidden, subtract_hidden, embedding)
    raw_value, raw_grads = jax.value_and_grad(
        raw_full_loss, argnums=(0, 1, 2)
    )(hidden, subtract_hidden, embedding)

    np.testing.assert_allclose(
        chunked_value, centered_value, rtol=1e-4, atol=1e-4
    )
    for chunked_grad, centered_grad in zip(chunked_grads, centered_grads):
        np.testing.assert_allclose(
            np.asarray(chunked_grad),
            np.asarray(centered_grad),
            rtol=1e-4,
            atol=1e-4,
        )
    np.testing.assert_allclose(
        centered_value, raw_value, rtol=5e-3, atol=1e-2
    )
    for centered_grad, raw_grad in zip(centered_grads, raw_grads):
        np.testing.assert_allclose(
            np.asarray(centered_grad),
            np.asarray(raw_grad),
            rtol=5e-3,
            atol=5e-3,
        )


def test_log_z_aux_matches_full_logsumexp_and_carries_gradient():
    rng = np.random.RandomState(11)
    hidden = jnp.asarray(rng.randn(2, 3, 5).astype(np.float32))
    embedding = jnp.asarray(rng.randn(17, 5).astype(np.float32))
    labels = jnp.asarray([[0, 16, -100], [3, 8, 5]], dtype=jnp.int32)
    valid = np.asarray(labels) != -100

    _, _, aux = token_xent_loss_from_hidden(
        hidden, embedding, labels, chunk_size=6
    )

    # Reference: full-logits logsumexp on the centered table (softcap==0 path).
    centered = embedding - embedding.mean(axis=0, keepdims=True)
    logits = jnp.einsum("...d,vd->...v", hidden, centered)
    log_z = np.asarray(jax.nn.logsumexp(logits, axis=-1))
    denom = valid.sum()
    np.testing.assert_allclose(
        aux["log_z_mean"], (log_z * valid).sum() / denom, rtol=1e-5, atol=1e-5
    )

    # log_z_mean must carry gradient to hidden AND table (z-loss-style usage).
    def z_loss(h, table):
        _, _, a = token_xent_loss_from_hidden(h, table, labels, chunk_size=6)
        return 1e-4 * a["log_z_mean"]

    def z_loss_ref(h, table):
        c = table - jax.lax.stop_gradient(table.mean(axis=0, keepdims=True))
        lz = jax.nn.logsumexp(jnp.einsum("...d,vd->...v", h, c), axis=-1)
        return 1e-4 * (lz * valid).sum() / denom

    gh, gt = jax.grad(z_loss, argnums=(0, 1))(hidden, embedding)
    gh_ref, gt_ref = jax.grad(z_loss_ref, argnums=(0, 1))(hidden, embedding)
    assert float(jnp.abs(gh).max()) > 0.0
    np.testing.assert_allclose(gh, gh_ref, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(gt, gt_ref, rtol=1e-4, atol=1e-6)


def test_log_z_aux_respects_softcap_path():
    rng = np.random.RandomState(12)
    hidden = jnp.asarray(rng.randn(2, 3, 5).astype(np.float32))
    embedding = jnp.asarray(rng.randn(17, 5).astype(np.float32))
    labels = jnp.asarray([[1, -100, 4], [2, 9, 16]], dtype=jnp.int32)
    valid = np.asarray(labels) != -100
    softcap = 2.0

    _, _, aux = token_xent_loss_from_hidden(
        hidden, embedding, labels, chunk_size=6, final_logit_softcap=softcap
    )

    centered = embedding - jax.lax.stop_gradient(embedding.mean(axis=0, keepdims=True))
    logits = jnp.einsum("...d,vd->...v", hidden, centered)
    logits = jnp.tanh(logits / softcap) * softcap
    log_z = np.asarray(jax.nn.logsumexp(logits, axis=-1))
    denom = valid.sum()
    np.testing.assert_allclose(
        aux["log_z_mean"], (log_z * valid).sum() / denom, rtol=1e-5, atol=1e-5
    )
