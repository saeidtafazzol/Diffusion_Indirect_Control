"""Unit tests for src/policy.py: IndiffCtrlPolicy.

Uses a small randomly-initialized model/dynamics (not the trained checkpoint)
so these run fast; the full checkpoint-driven convergence tests live in
test_known_trajectories.py.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core import STATE_DIM, COSTATE_DIM
from jax_ddpm_scheduler import JaxDDPMScheduler
from policy import IndiffCtrlPolicy, PreparedTrainingBatch
from transformer_diffusion_model import (
    DiffusionTransformer,
    DiffusionTransformerConfig,
    default_state_known_mask_no_final_mass,
    final_lambda_m_costate_mask,
)


SEQ_LEN = 6


@pytest.fixture(scope="module")
def small_time_grid():
    return np.linspace(0.0, 1.0, SEQ_LEN, dtype=np.float32)


@pytest.fixture(scope="module")
def small_policy(small_time_grid):
    model = DiffusionTransformer(DiffusionTransformerConfig(
        seq_len=SEQ_LEN, state_dim=STATE_DIM, costate_dim=COSTATE_DIM,
        embd_dim=16, num_layers=1, num_heads=2, mlp_ratio=2,
        p_drop_embd=0.0, p_drop_attn=0.0,
        use_integration_signals=False, zero_known_state_eps=True,
    ))
    scheduler = JaxDDPMScheduler(num_train_timesteps=50, beta_schedule="squaredcos_cap_v2")
    return IndiffCtrlPolicy(
        model=model, noise_scheduler=scheduler, time_grid=small_time_grid,
        segment_rtol=1e-6, segment_atol=1e-8, eps=1e-4,
    )


@pytest.fixture(scope="module")
def small_params(small_policy):
    key = jax.random.PRNGKey(0)
    return small_policy.init_params(key, batch_size=2)


# ── construction ──────────────────────────────────────────────────────────────

def test_time_grid_validation(small_policy):
    model = small_policy.model
    scheduler = small_policy.noise_scheduler
    with pytest.raises(ValueError):
        IndiffCtrlPolicy(model=model, noise_scheduler=scheduler, time_grid=np.array([1.0]))
    with pytest.raises(ValueError):
        IndiffCtrlPolicy(model=model, noise_scheduler=scheduler, time_grid=np.zeros((2, 2)))


def test_init_params_shapes(small_policy, small_params):
    # Just checking init doesn't crash and returns a pytree with leaves.
    leaves = jax.tree_util.tree_leaves(small_params)
    assert len(leaves) > 0
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)


# ── mask resolution ───────────────────────────────────────────────────────────

def test_resolve_known_masks_defaults(small_policy):
    skm, ckm = small_policy._resolve_known_masks(batch_size=2, seq_len=SEQ_LEN)
    assert skm.shape == (2, SEQ_LEN)
    assert ckm.shape == (2, SEQ_LEN)
    assert bool(jnp.all(skm[:, 0])) and bool(jnp.all(skm[:, -1]))
    assert not bool(jnp.any(ckm))


def test_resolve_known_masks_custom(small_policy):
    skm_in = default_state_known_mask_no_final_mass(2, SEQ_LEN)
    ckm_in = final_lambda_m_costate_mask(2, SEQ_LEN)
    skm, ckm = small_policy._resolve_known_masks(
        batch_size=2, seq_len=SEQ_LEN, state_known_mask=skm_in, costate_known_mask=ckm_in,
    )
    np.testing.assert_array_equal(np.asarray(skm), np.asarray(skm_in))
    np.testing.assert_array_equal(np.asarray(ckm), np.asarray(ckm_in))


def test_validate_mask_rejects_bad_shape(small_policy):
    bad_mask = jnp.zeros((2, SEQ_LEN + 1), dtype=bool)
    with pytest.raises(ValueError):
        small_policy._validate_mask(bad_mask, batch_size=2, seq_len=SEQ_LEN, name="state_known_mask")


def test_build_condition_mask_broadcasts_2d_and_keeps_3d(small_policy):
    skm_2d = jnp.zeros((2, SEQ_LEN), dtype=bool).at[:, 0].set(True)
    ckm_3d = jnp.zeros((2, SEQ_LEN, COSTATE_DIM), dtype=bool).at[:, -1, -1].set(True)
    cond = small_policy._build_condition_mask(skm_2d, ckm_3d)
    assert cond.shape == (2, SEQ_LEN, STATE_DIM + COSTATE_DIM)
    # state part: all STATE_DIM components true at t=0 (broadcast from 2D)
    assert bool(jnp.all(cond[:, 0, :STATE_DIM]))
    # costate part: only last component true at final t (kept per-component from 3D)
    assert bool(jnp.all(cond[:, -1, STATE_DIM:STATE_DIM + COSTATE_DIM - 1] == False))
    assert bool(jnp.all(cond[:, -1, -1]))


# ── combine / split trajectory ────────────────────────────────────────────────

def test_combine_split_trajectory_roundtrip(small_policy):
    states = jax.random.normal(jax.random.PRNGKey(0), (2, SEQ_LEN, STATE_DIM))
    costates = jax.random.normal(jax.random.PRNGKey(1), (2, SEQ_LEN, COSTATE_DIM))
    combined = small_policy._combine_trajectory(states, costates)
    assert combined.shape == (2, SEQ_LEN, STATE_DIM + COSTATE_DIM)
    s2, c2 = small_policy._split_trajectory(combined)
    np.testing.assert_array_equal(np.asarray(s2), np.asarray(states))
    np.testing.assert_array_equal(np.asarray(c2), np.asarray(costates))


# ── integration signals ───────────────────────────────────────────────────────

def _safe_states(batch_size):
    """Nonzero, non-singular position/velocity (r=0 is a 1/r^3 dynamics
    singularity) so segment integration behaves normally."""
    state = jnp.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0])
    return jnp.broadcast_to(state, (batch_size, SEQ_LEN, STATE_DIM))


def test_compute_integration_signals_shapes(small_policy):
    batch_size = 2
    states = _safe_states(batch_size)
    costates = jnp.zeros((batch_size, SEQ_LEN, COSTATE_DIM))
    int_states, int_costates, failed = small_policy.compute_integration_signals(states, costates)
    assert int_states.shape == (batch_size, SEQ_LEN - 1, STATE_DIM)
    assert int_costates.shape == (batch_size, SEQ_LEN - 1, COSTATE_DIM)
    assert failed.shape == (batch_size, SEQ_LEN - 1)


def test_compute_integration_signals_rejects_wrong_seq_len(small_policy):
    states = jnp.zeros((1, SEQ_LEN + 1, STATE_DIM))
    costates = jnp.zeros((1, SEQ_LEN + 1, COSTATE_DIM))
    with pytest.raises(ValueError):
        small_policy.compute_integration_signals(states, costates)


def test_compute_integration_signals_rejects_wrong_costate_shape(small_policy):
    states = jnp.zeros((1, SEQ_LEN, STATE_DIM))
    bad_costates = jnp.zeros((1, SEQ_LEN, COSTATE_DIM + 1))
    with pytest.raises(ValueError):
        small_policy.compute_integration_signals(states, bad_costates)


def test_compute_integration_signals_failed_segments_are_zeroed(small_policy):
    """Failed integrations must be masked to exactly 0, not leak NaN/Inf —
    verified by directly overriding the segment integrator's `success` output."""
    batch_size = 1
    states = _safe_states(batch_size)
    costates = jnp.zeros((batch_size, SEQ_LEN, COSTATE_DIM))

    original_integrator = small_policy.segment_integrator
    try:
        def _always_fails(y0, costate, dt):
            int_state, int_costate, _ = original_integrator(y0, costate, dt)
            success = jnp.zeros((y0.shape[0],), dtype=bool)
            return int_state, int_costate, success

        small_policy.segment_integrator = _always_fails
        int_states, int_costates, failed = small_policy.compute_integration_signals(states, costates)
    finally:
        small_policy.segment_integrator = original_integrator

    assert bool(jnp.all(failed))
    np.testing.assert_array_equal(np.asarray(int_states), 0.0)
    np.testing.assert_array_equal(np.asarray(int_costates), 0.0)


# ── prepare_training_batch / compute_loss ─────────────────────────────────────

def _dummy_batch(batch_size=2):
    rng = np.random.default_rng(0)
    states = rng.normal(size=(batch_size, SEQ_LEN, STATE_DIM)).astype(np.float32)
    states[:, :, 6] = 1.0
    costates = rng.normal(size=(batch_size, SEQ_LEN, COSTATE_DIM)).astype(np.float32)
    return {"states": states, "costates": costates}


def test_prepare_training_batch_shapes_and_masking(small_policy):
    batch = _dummy_batch(batch_size=2)
    prepared = small_policy.prepare_training_batch(batch, jax.random.PRNGKey(0))
    assert isinstance(prepared, PreparedTrainingBatch)
    assert prepared.noisy_states.shape == (2, SEQ_LEN, STATE_DIM)
    assert prepared.noisy_costates.shape == (2, SEQ_LEN, COSTATE_DIM)
    assert prepared.timesteps.shape == (2,)
    # default state mask pins t=0 and t=-1 exactly (no noise added there)
    np.testing.assert_allclose(
        np.asarray(prepared.noisy_states[:, 0, :]), np.asarray(batch["states"])[:, 0, :], atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(prepared.noisy_states[:, -1, :]), np.asarray(batch["states"])[:, -1, :], atol=1e-5,
    )


def test_prepare_training_batch_unsupported_prediction_type_raises(small_time_grid):
    model = DiffusionTransformer(DiffusionTransformerConfig(
        seq_len=SEQ_LEN, state_dim=STATE_DIM, costate_dim=COSTATE_DIM,
        embd_dim=8, num_layers=1, num_heads=2, mlp_ratio=2,
    ))

    scheduler = JaxDDPMScheduler(num_train_timesteps=10)
    # monkeypatch config after the fact to an unsupported value
    # (SimpleNamespace is mutable, so this is a legitimate way to reach the
    # `else: raise ValueError` branch in prepare_training_batch)
    scheduler.config.prediction_type = "bogus"
    policy = IndiffCtrlPolicy(model=model, noise_scheduler=scheduler, time_grid=small_time_grid)
    batch = _dummy_batch(batch_size=1)
    with pytest.raises(ValueError):
        policy.prepare_training_batch(batch, jax.random.PRNGKey(0))


def test_compute_loss_is_finite_scalar_and_nonnegative(small_policy, small_params):
    batch = _dummy_batch(batch_size=2)
    loss = small_policy.compute_loss(
        small_params, batch, rng_key=jax.random.PRNGKey(0), dropout_key=jax.random.PRNGKey(1),
    )
    assert loss.shape == ()
    assert float(loss) >= 0.0
    assert np.isfinite(float(loss))


def test_compute_loss_gradients_are_finite(small_policy, small_params):
    batch = _dummy_batch(batch_size=2)

    def loss_fn(params):
        return small_policy.compute_loss(
            params, batch, rng_key=jax.random.PRNGKey(0), dropout_key=jax.random.PRNGKey(1),
        )

    loss, grads = jax.value_and_grad(loss_fn)(small_params)
    leaves = jax.tree_util.tree_leaves(grads)
    assert len(leaves) > 0
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)


# ── predict_trajectory ────────────────────────────────────────────────────────

def test_predict_trajectory_shapes_and_endpoint_conditioning(small_policy, small_params):
    batch_size = 2
    initial_states = np.zeros((batch_size, STATE_DIM), dtype=np.float32)
    initial_states[:, 6] = 1.0
    final_states = np.ones((batch_size, STATE_DIM), dtype=np.float32)
    final_states[:, 6] = 1.0

    out = small_policy.predict_trajectory(
        params=small_params, rng_key=jax.random.PRNGKey(0),
        initial_states=initial_states, final_states=final_states,
        num_inference_steps=3,
    )
    assert out["states"].shape == (batch_size, SEQ_LEN, STATE_DIM)
    assert out["costates"].shape == (batch_size, SEQ_LEN, COSTATE_DIM)
    np.testing.assert_allclose(np.asarray(out["states"][:, 0, :]), initial_states, atol=1e-5)
    np.testing.assert_allclose(np.asarray(out["states"][:, -1, :]), final_states, atol=1e-5)
    assert bool(jnp.all(jnp.isfinite(out["states"])))
    assert bool(jnp.all(jnp.isfinite(out["costates"])))


def test_predict_trajectory_rejects_mismatched_shapes(small_policy, small_params):
    initial_states = np.zeros((2, STATE_DIM), dtype=np.float32)
    final_states = np.zeros((3, STATE_DIM), dtype=np.float32)
    with pytest.raises(ValueError):
        small_policy.predict_trajectory(
            params=small_params, rng_key=jax.random.PRNGKey(0),
            initial_states=initial_states, final_states=final_states,
        )


def test_predict_trajectory_rejects_wrong_state_dim(small_policy, small_params):
    initial_states = np.zeros((1, STATE_DIM + 1), dtype=np.float32)
    final_states = np.zeros((1, STATE_DIM + 1), dtype=np.float32)
    with pytest.raises(ValueError):
        small_policy.predict_trajectory(
            params=small_params, rng_key=jax.random.PRNGKey(0),
            initial_states=initial_states, final_states=final_states,
        )


def test_predict_trajectory_with_known_costates(small_policy, small_params):
    batch_size = 1
    initial_states = np.zeros((batch_size, STATE_DIM), dtype=np.float32)
    initial_states[:, 6] = 1.0
    final_states = np.ones((batch_size, STATE_DIM), dtype=np.float32)
    final_states[:, 6] = 1.0
    known_costates = np.zeros((batch_size, SEQ_LEN, COSTATE_DIM), dtype=np.float32)
    known_costates[:, -1, -1] = 0.0
    ckm = final_lambda_m_costate_mask(batch_size, SEQ_LEN)

    out = small_policy.predict_trajectory(
        params=small_params, rng_key=jax.random.PRNGKey(0),
        initial_states=initial_states, final_states=final_states,
        costate_known_mask=ckm, known_costates=known_costates,
        num_inference_steps=3,
    )
    np.testing.assert_allclose(np.asarray(out["costates"][:, -1, -1]), 0.0, atol=1e-6)
