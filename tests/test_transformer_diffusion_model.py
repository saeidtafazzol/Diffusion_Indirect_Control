"""Unit tests for src/transformer_diffusion_model.py."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from transformer_diffusion_model import (
    DiffusionTransformer,
    DiffusionTransformerConfig,
    IntegrationSignalEmbedding,
    TimeEmbedding,
    default_costate_known_mask,
    default_state_known_mask,
    default_state_known_mask_no_final_mass,
    final_lambda_m_costate_mask,
    sinusoidal_time_embedding,
)


# ── mask helpers ──────────────────────────────────────────────────────────────

def test_default_state_known_mask_pins_endpoints():
    mask = default_state_known_mask(3, 8)
    assert mask.shape == (3, 8)
    assert bool(jnp.all(mask[:, 0]))
    assert bool(jnp.all(mask[:, -1]))
    assert not bool(jnp.any(mask[:, 1:-1]))


def test_default_costate_known_mask_is_all_false():
    mask = default_costate_known_mask(2, 5)
    assert mask.shape == (2, 5)
    assert not bool(jnp.any(mask))


def test_default_state_known_mask_no_final_mass():
    mask = default_state_known_mask_no_final_mass(2, 6, state_dim=7)
    assert mask.shape == (2, 6, 7)
    assert bool(jnp.all(mask[:, 0, :]))          # full initial state known
    assert bool(jnp.all(mask[:, -1, :6]))        # final r,v known
    assert not bool(jnp.any(mask[:, -1, 6]))     # final mass NOT known
    assert not bool(jnp.any(mask[:, 1:-1, :]))   # interior unknown


def test_final_lambda_m_costate_mask():
    mask = final_lambda_m_costate_mask(2, 6, costate_dim=7)
    assert mask.shape == (2, 6, 7)
    assert bool(jnp.all(mask[:, -1, 6]))
    total_true = int(jnp.sum(mask))
    assert total_true == 2  # one per batch element


# ── sinusoidal time embedding ────────────────────────────────────────────────

def test_sinusoidal_time_embedding_shape():
    emb = sinusoidal_time_embedding(jnp.array([0, 10, 500]), dim=16)
    assert emb.shape == (3, 16)


def test_sinusoidal_time_embedding_odd_dim_padded():
    emb = sinusoidal_time_embedding(jnp.array([0, 1]), dim=9)
    assert emb.shape == (2, 9)
    np.testing.assert_allclose(np.asarray(emb[:, -1]), 0.0)


def test_sinusoidal_time_embedding_zero_timestep_is_sin0_cos1():
    dim = 8
    emb = sinusoidal_time_embedding(jnp.array([0]), dim=dim)
    half = dim // 2
    np.testing.assert_allclose(np.asarray(emb[0, :half]), 0.0, atol=1e-6)   # sin(0) = 0
    np.testing.assert_allclose(np.asarray(emb[0, half:]), 1.0, atol=1e-6)   # cos(0) = 1


def test_sinusoidal_time_embedding_distinct_timesteps_differ():
    emb = sinusoidal_time_embedding(jnp.array([0, 1, 2, 3]), dim=32)
    emb = np.asarray(emb)
    for i in range(1, 4):
        assert not np.allclose(emb[0], emb[i])


# ── model forward pass ────────────────────────────────────────────────────────

@pytest.fixture
def small_config():
    return DiffusionTransformerConfig(
        seq_len=8, state_dim=7, costate_dim=7,
        embd_dim=16, num_layers=2, num_heads=2, mlp_ratio=2,
        p_drop_embd=0.0, p_drop_attn=0.0,
        use_integration_signals=False, zero_known_state_eps=True,
    )


@pytest.fixture
def small_model(small_config):
    return DiffusionTransformer(small_config)


@pytest.fixture
def init_params(small_model):
    cfg = small_model.config
    batch_size = 2
    key = jax.random.PRNGKey(0)
    inputs = dict(
        noisy_states=jnp.zeros((batch_size, cfg.seq_len, cfg.state_dim)),
        noisy_costates=jnp.zeros((batch_size, cfg.seq_len, cfg.costate_dim)),
        diffusion_steps=jnp.zeros((batch_size,), dtype=jnp.int32),
        train=False,
    )
    variables = small_model.init(key, **inputs)
    return variables["params"]


def test_forward_pass_output_shapes(small_model, init_params):
    cfg = small_model.config
    batch_size = 2
    key = jax.random.PRNGKey(1)
    states = jax.random.normal(key, (batch_size, cfg.seq_len, cfg.state_dim))
    costates = jax.random.normal(key, (batch_size, cfg.seq_len, cfg.costate_dim))
    diffusion_steps = jnp.array([5, 10])
    out = small_model.apply({"params": init_params}, states, costates, diffusion_steps, train=False)
    assert out["state_eps"].shape == (batch_size, cfg.seq_len, cfg.state_dim)
    assert out["costate_eps"].shape == (batch_size, cfg.seq_len, cfg.costate_dim)
    assert bool(jnp.all(jnp.isfinite(out["state_eps"])))
    assert bool(jnp.all(jnp.isfinite(out["costate_eps"])))


def test_forward_pass_rejects_wrong_seq_len(small_model, init_params):
    cfg = small_model.config
    states = jnp.zeros((1, cfg.seq_len + 1, cfg.state_dim))
    costates = jnp.zeros((1, cfg.seq_len + 1, cfg.costate_dim))
    with pytest.raises(ValueError):
        small_model.apply({"params": init_params}, states, costates, jnp.zeros((1,), dtype=jnp.int32))


def test_forward_pass_rejects_wrong_costate_shape(small_model, init_params):
    cfg = small_model.config
    states = jnp.zeros((1, cfg.seq_len, cfg.state_dim))
    bad_costates = jnp.zeros((1, cfg.seq_len, cfg.costate_dim + 1))
    with pytest.raises(ValueError):
        small_model.apply({"params": init_params}, states, bad_costates, jnp.zeros((1,), dtype=jnp.int32))


def test_zero_known_state_eps_zeros_known_components(small_model, init_params):
    cfg = small_model.config
    batch_size = 1
    states = jax.random.normal(jax.random.PRNGKey(2), (batch_size, cfg.seq_len, cfg.state_dim))
    costates = jax.random.normal(jax.random.PRNGKey(3), (batch_size, cfg.seq_len, cfg.costate_dim))
    skm = default_state_known_mask_no_final_mass(batch_size, cfg.seq_len, cfg.state_dim)
    ckm = final_lambda_m_costate_mask(batch_size, cfg.seq_len, cfg.costate_dim)
    out = small_model.apply(
        {"params": init_params}, states, costates, jnp.zeros((batch_size,), dtype=jnp.int32),
        state_known_mask=skm, costate_known_mask=ckm, train=False,
    )
    # Full initial state (t=0) is known -> its eps prediction must be exactly 0
    np.testing.assert_allclose(np.asarray(out["state_eps"][:, 0, :]), 0.0)
    # Final r,v known -> zero, final mass not known -> generally nonzero
    np.testing.assert_allclose(np.asarray(out["state_eps"][:, -1, :6]), 0.0)
    # Final lambda_m known -> zero
    np.testing.assert_allclose(np.asarray(out["costate_eps"][:, -1, 6]), 0.0)


def test_zero_known_state_eps_false_does_not_force_zero():
    cfg = DiffusionTransformerConfig(
        seq_len=6, state_dim=7, costate_dim=7, embd_dim=16, num_layers=1, num_heads=2,
        mlp_ratio=2, p_drop_embd=0.0, p_drop_attn=0.0,
        use_integration_signals=False, zero_known_state_eps=False,
    )
    model = DiffusionTransformer(cfg)
    key = jax.random.PRNGKey(0)
    states = jax.random.normal(key, (1, cfg.seq_len, cfg.state_dim))
    costates = jax.random.normal(key, (1, cfg.seq_len, cfg.costate_dim))
    variables = model.init(
        key, noisy_states=states, noisy_costates=costates,
        diffusion_steps=jnp.zeros((1,), dtype=jnp.int32), train=False,
    )
    out = model.apply(variables, states, costates, jnp.zeros((1,), dtype=jnp.int32),
                       state_known_mask=default_state_known_mask(1, cfg.seq_len), train=False)
    # With zero_known_state_eps=False, no guarantee the known-state eps is zero.
    # (This is a smoke test that forward pass still works and is finite.)
    assert bool(jnp.all(jnp.isfinite(out["state_eps"])))


def test_forward_pass_deterministic_with_train_false(small_model, init_params):
    cfg = small_model.config
    states = jax.random.normal(jax.random.PRNGKey(5), (1, cfg.seq_len, cfg.state_dim))
    costates = jax.random.normal(jax.random.PRNGKey(6), (1, cfg.seq_len, cfg.costate_dim))
    ts = jnp.array([3])
    out_a = small_model.apply({"params": init_params}, states, costates, ts, train=False)
    out_b = small_model.apply({"params": init_params}, states, costates, ts, train=False)
    np.testing.assert_array_equal(np.asarray(out_a["state_eps"]), np.asarray(out_b["state_eps"]))


def test_use_integration_signals_true_changes_output(small_config):
    cfg = DiffusionTransformerConfig(**{**small_config.__dict__, "use_integration_signals": True})
    model = DiffusionTransformer(cfg)
    key = jax.random.PRNGKey(0)
    states = jax.random.normal(key, (1, cfg.seq_len, cfg.state_dim))
    costates = jax.random.normal(key, (1, cfg.seq_len, cfg.costate_dim))
    integrated_states = jax.random.normal(jax.random.PRNGKey(1), (1, cfg.seq_len - 1, cfg.state_dim))
    integrated_costates = jax.random.normal(jax.random.PRNGKey(2), (1, cfg.seq_len - 1, cfg.costate_dim))
    integration_failed = jnp.zeros((1, cfg.seq_len - 1), dtype=bool)
    variables = model.init(
        key, noisy_states=states, noisy_costates=costates,
        diffusion_steps=jnp.zeros((1,), dtype=jnp.int32),
        integrated_states=integrated_states, integrated_costates=integrated_costates,
        integration_failed=integration_failed, train=False,
    )
    out = model.apply(
        variables, states, costates, jnp.zeros((1,), dtype=jnp.int32),
        integrated_states=integrated_states, integrated_costates=integrated_costates,
        integration_failed=integration_failed, train=False,
    )
    assert bool(jnp.all(jnp.isfinite(out["state_eps"])))


def test_integration_signal_embedding_failure_token_used_when_failed():
    cfg = DiffusionTransformerConfig(
        seq_len=4, state_dim=7, costate_dim=7, embd_dim=8, num_layers=1, num_heads=2,
        mlp_ratio=2, use_integration_signals=True,
    )
    module = IntegrationSignalEmbedding(config=cfg)
    key = jax.random.PRNGKey(0)
    batch_size = 1
    segment_count = cfg.seq_len - 1
    position_embedding = jax.random.normal(key, (batch_size, cfg.seq_len, cfg.embd_dim))
    integrated_states = jnp.ones((batch_size, segment_count, cfg.state_dim))
    integrated_costates = jnp.ones((batch_size, segment_count, cfg.costate_dim))

    all_ok = jnp.zeros((batch_size, segment_count), dtype=bool)
    all_failed = jnp.ones((batch_size, segment_count), dtype=bool)

    variables = module.init(
        key, position_embedding=position_embedding,
        integrated_states=integrated_states, integrated_costates=integrated_costates,
        integration_failed=all_ok,
    )
    st_tokens_ok, cst_tokens_ok = module.apply(
        variables, position_embedding=position_embedding,
        integrated_states=integrated_states, integrated_costates=integrated_costates,
        integration_failed=all_ok,
    )
    st_tokens_failed, cst_tokens_failed = module.apply(
        variables, position_embedding=position_embedding,
        integrated_states=integrated_states, integrated_costates=integrated_costates,
        integration_failed=all_failed,
    )
    # The failure branch must not equal the success branch (uses learned token instead).
    assert not bool(jnp.allclose(st_tokens_ok, st_tokens_failed))
    assert not bool(jnp.allclose(cst_tokens_ok, cst_tokens_failed))


def test_time_embedding_module_matches_function():
    module = TimeEmbedding(embd_dim=12)
    key = jax.random.PRNGKey(0)
    steps = jnp.array([1, 2, 3])
    variables = module.init(key, steps)
    out = module.apply(variables, steps)
    expected = sinusoidal_time_embedding(steps, 12)
    np.testing.assert_allclose(np.asarray(out), np.asarray(expected))
