"""image_drop_prob: dropped samples must equal the images=None text-only forward.

generate_cfg's uncond stream runs the images=None forward at eval; a training
sample whose image is dropped must compute exactly that forward, otherwise the
trained uncond condition and the eval-time uncond condition diverge.
"""
import jax
import jax.numpy as jnp
import numpy as np

from models.llava import LlavaGemma

_TINY = dict(
    lm_backbone_str="gemma3_270M",
    txt_feature_layer=0,
)

B, T = 4, 10


def _data():
    input_ids = jax.random.randint(
        jax.random.PRNGKey(0), (B, T), 3, 1000, dtype=jnp.int32
    )
    prefix_len = jnp.full((B,), 4, dtype=jnp.int32)
    images = jax.random.uniform(jax.random.PRNGKey(1), (B, 336, 336, 3))
    labels = jnp.where(
        jnp.arange(T)[None, :] >= 4, input_ids, jnp.full_like(input_ids, -100)
    )
    attention_mask = jnp.ones((B, T), dtype=jnp.int32)
    return input_ids, prefix_len, images, labels, attention_mask


def _make_params():
    base = LlavaGemma(image_drop_prob=0.0, **_TINY)
    input_ids, prefix_len, images, labels, attention_mask = _data()
    params = base.init(
        {"params": jax.random.PRNGKey(2), "gen": jax.random.PRNGKey(3)},
        input_ids,
        images,
        prefix_len,
        attention_mask=attention_mask,
        labels=labels,
    )["params"]
    return base, params


def _run(model, params, drop_rng, images):
    input_ids, prefix_len, _, labels, attention_mask = _data()
    loss, log_dict, debug = model.apply(
        {"params": params},
        input_ids,
        images,
        prefix_len,
        attention_mask=attention_mask,
        labels=labels,
        rngs={"gen": drop_rng},
    )
    return float(loss), log_dict, np.asarray(debug["preds"])


def test_drop_prob_one_equals_images_none():
    base, params = _make_params()
    dropped = LlavaGemma(image_drop_prob=1.0, **_TINY)
    _, _, images, _, _ = _data()

    rng = jax.random.PRNGKey(7)
    loss_p1, log_p1, preds_p1 = _run(dropped, params, rng, images)
    loss_ref, _, preds_ref = _run(base, params, rng, None)
    loss_base, _, preds_base = _run(base, params, rng, images)

    np.testing.assert_allclose(loss_p1, loss_ref, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(preds_p1, preds_ref)
    assert float(log_p1["image_drop_fraction"]) == 1.0
    assert abs(loss_p1 - loss_base) > 1e-6  # dropping must change the loss
    assert np.any(preds_p1 != preds_base)


def test_drop_prob_half_routes_per_sample():
    base, params = _make_params()
    half = LlavaGemma(image_drop_prob=0.5, **_TINY)
    _, _, images, _, _ = _data()

    _, _, preds_none = _run(base, params, jax.random.PRNGKey(0), None)
    _, _, preds_full = _run(base, params, jax.random.PRNGKey(0), images)
    assert np.all(np.any(preds_none != preds_full, axis=-1)), (
        "refs must differ per sample for the routing check to be meaningful"
    )

    for seed in range(64):
        _, log_h, preds_h = _run(half, params, jax.random.PRNGKey(seed), images)
        frac = float(log_h["image_drop_fraction"])
        if 0.0 < frac < 1.0:
            break
    else:
        raise AssertionError("no mixed drop mask found in 64 seeds")

    n_dropped = 0
    for b in range(B):
        if np.array_equal(preds_h[b], preds_none[b]):
            n_dropped += 1
        else:
            np.testing.assert_array_equal(preds_h[b], preds_full[b])
    assert n_dropped == round(frac * B)


def test_eval_paths_never_drop():
    base, params = _make_params()
    half = LlavaGemma(image_drop_prob=0.5, **_TINY)
    input_ids, prefix_len, images, _, _ = _data()

    out_a = base.apply(
        {"params": params}, input_ids, prefix_len, images,
        max_new_tokens=3, method=base.generate,
    )
    out_b = half.apply(
        {"params": params}, input_ids, prefix_len, images,
        max_new_tokens=3, method=half.generate,
    )
    np.testing.assert_array_equal(np.asarray(out_a), np.asarray(out_b))
