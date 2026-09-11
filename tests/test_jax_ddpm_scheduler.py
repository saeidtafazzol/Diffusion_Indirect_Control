"""Unit tests for src/jax_ddpm_scheduler.py: JaxDDPMScheduler."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_ddpm_scheduler import JaxDDPMScheduler, _betas_for_alpha_bar


# ── construction / validation ────────────────────────────────────────────────

def test_invalid_prediction_type_raises():
    with pytest.raises(ValueError):
        JaxDDPMScheduler(prediction_type="bogus")


def test_invalid_variance_type_raises():
    with pytest.raises(ValueError):
        JaxDDPMScheduler(variance_type="learned")


def test_invalid_noise_scale_raises():
    with pytest.raises(ValueError):
        JaxDDPMScheduler(noise_scale=0.0)
    with pytest.raises(ValueError):
        JaxDDPMScheduler(noise_scale=-1.0)


def test_invalid_beta_schedule_raises():
    with pytest.raises(ValueError):
        JaxDDPMScheduler(beta_schedule="not_a_schedule")


@pytest.mark.parametrize("schedule", ["linear", "scaled_linear", "squaredcos_cap_v2"])
def test_beta_schedules_produce_valid_betas(schedule):
    T = 50
    sched = JaxDDPMScheduler(num_train_timesteps=T, beta_schedule=schedule)
    betas = np.asarray(sched.betas)
    assert betas.shape == (T,)
    assert np.all(betas > 0.0)
    assert np.all(betas <= 0.999 + 1e-6)
    alphas_cumprod = np.asarray(sched.alphas_cumprod)
    # alphas_cumprod should be monotonically decreasing
    assert np.all(np.diff(alphas_cumprod) < 0)
    assert alphas_cumprod[0] <= 1.0
    assert alphas_cumprod[-1] > 0.0


def test_betas_for_alpha_bar_respects_max_beta():
    betas = _betas_for_alpha_bar(1000, max_beta=0.5)
    assert np.all(betas <= 0.5 + 1e-9)


# ── sample_noise ──────────────────────────────────────────────────────────────

def test_sample_noise_shape_and_scale():
    sched = JaxDDPMScheduler(noise_scale=2.0)
    key = jax.random.PRNGKey(0)
    noise = sched.sample_noise(key, (4, 8, 14))
    assert noise.shape == (4, 8, 14)
    # std should scale with noise_scale (loose statistical check, large sample)
    key2 = jax.random.PRNGKey(1)
    big_noise = sched.sample_noise(key2, (10000,))
    assert float(jnp.std(big_noise)) == pytest.approx(2.0, rel=0.1)


# ── set_timesteps ─────────────────────────────────────────────────────────────

def test_set_timesteps_uniform_spacing_descending():
    sched = JaxDDPMScheduler(num_train_timesteps=1000)
    sched.set_timesteps(10)
    ts = sched.timesteps
    assert len(ts) <= 10
    assert np.all(np.diff(ts) < 0)  # strictly descending
    assert ts[0] <= 999


def test_set_timesteps_out_of_range_raises():
    sched = JaxDDPMScheduler(num_train_timesteps=100)
    with pytest.raises(ValueError):
        sched.set_timesteps(0)
    with pytest.raises(ValueError):
        sched.set_timesteps(101)


def test_set_timesteps_invalid_power_raises():
    sched = JaxDDPMScheduler(num_train_timesteps=100)
    with pytest.raises(ValueError):
        sched.set_timesteps(10, power=0.0)


def test_set_timesteps_power_concentrates_near_zero():
    """Higher power should push more of the (deduplicated) steps toward t=0."""
    sched = JaxDDPMScheduler(num_train_timesteps=1000)
    sched.set_timesteps(20, power=1.0)
    ts_uniform = sched.timesteps
    sched.set_timesteps(20, power=2.7)
    ts_power = sched.timesteps
    # median timestep should be smaller (closer to 0) under higher power
    assert np.median(ts_power) < np.median(ts_uniform)


def test_set_timesteps_updates_prev_timestep_map():
    sched = JaxDDPMScheduler(num_train_timesteps=1000)
    sched.set_timesteps(5)
    ts = sched.timesteps
    for i, t in enumerate(ts[:-1]):
        assert sched._prev_timestep[int(t)] == int(ts[i + 1])
    assert sched._prev_timestep[int(ts[-1])] == -1


# ── add_noise ─────────────────────────────────────────────────────────────────

def test_add_noise_at_timestep_zero_is_almost_original():
    sched = JaxDDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    x0 = jnp.ones((2, 3, 4), dtype=jnp.float32)
    noise = jnp.zeros_like(x0)
    timesteps = jnp.array([0, 0], dtype=jnp.int32)
    out = sched.add_noise(x0, noise, timesteps)
    # sqrt(alphas_cumprod[0]) should be close to 1 (beta_start is tiny)
    np.testing.assert_allclose(np.asarray(out), np.asarray(x0), atol=1e-2)


def test_add_noise_matches_closed_form():
    sched = JaxDDPMScheduler(num_train_timesteps=100, beta_schedule="linear")
    key = jax.random.PRNGKey(0)
    x0 = jax.random.normal(key, (3, 5, 7))
    noise = jax.random.normal(jax.random.PRNGKey(1), (3, 5, 7))
    timesteps = jnp.array([0, 50, 99], dtype=jnp.int32)
    out = np.asarray(sched.add_noise(x0, noise, timesteps))

    alphas_cumprod = np.asarray(sched.alphas_cumprod)
    for i, t in enumerate([0, 50, 99]):
        expected = np.sqrt(alphas_cumprod[t]) * np.asarray(x0)[i] + np.sqrt(1 - alphas_cumprod[t]) * np.asarray(noise)[i]
        np.testing.assert_allclose(out[i], expected, atol=1e-5)


# ── step (reverse process) ────────────────────────────────────────────────────

def test_step_requires_rng_or_variance_noise_when_not_final():
    sched = JaxDDPMScheduler(num_train_timesteps=100)
    sample = jnp.zeros((1, 4))
    model_output = jnp.zeros((1, 4))
    with pytest.raises(ValueError):
        sched.step(model_output=model_output, timestep=50, sample=sample, rng_key=None)


def test_step_at_final_timestep_has_no_stochastic_noise():
    sched = JaxDDPMScheduler(num_train_timesteps=100)
    sched.set_timesteps(10)
    sample = jax.random.normal(jax.random.PRNGKey(0), (2, 6))
    model_output = jax.random.normal(jax.random.PRNGKey(1), (2, 6))
    final_t = int(sched.timesteps[-1])
    assert sched._prev_timestep[final_t] == -1

    out_a = sched.step(model_output=model_output, timestep=final_t, sample=sample, rng_key=None)
    out_b = sched.step(model_output=model_output, timestep=final_t, sample=sample,
                        rng_key=jax.random.PRNGKey(999))
    # deterministic at t=0 regardless of rng_key/variance
    np.testing.assert_allclose(np.asarray(out_a.prev_sample), np.asarray(out_b.prev_sample), atol=1e-6)


def test_step_pred_original_sample_epsilon_type_matches_formula():
    sched = JaxDDPMScheduler(num_train_timesteps=100, prediction_type="epsilon")
    t = 30
    sample = jax.random.normal(jax.random.PRNGKey(0), (2, 5))
    model_output = jax.random.normal(jax.random.PRNGKey(1), (2, 5))
    out = sched.step(model_output=model_output, timestep=t, sample=sample,
                      rng_key=jax.random.PRNGKey(2))
    alpha_prod_t = float(sched.alphas_cumprod[t])
    beta_prod_t = 1.0 - alpha_prod_t
    expected_pred = (np.asarray(sample) - np.sqrt(beta_prod_t) * np.asarray(model_output)) / np.sqrt(alpha_prod_t)
    np.testing.assert_allclose(np.asarray(out.pred_original_sample), expected_pred, atol=1e-4)


def test_step_sample_prediction_type_uses_model_output_directly():
    sched = JaxDDPMScheduler(num_train_timesteps=100, prediction_type="sample")
    t = 30
    sample = jax.random.normal(jax.random.PRNGKey(0), (2, 5))
    model_output = jax.random.normal(jax.random.PRNGKey(1), (2, 5))
    out = sched.step(model_output=model_output, timestep=t, sample=sample,
                      rng_key=jax.random.PRNGKey(2))
    np.testing.assert_allclose(np.asarray(out.pred_original_sample), np.asarray(model_output), atol=1e-6)


def test_step_clip_sample_clips_pred_original():
    sched = JaxDDPMScheduler(num_train_timesteps=100, prediction_type="sample",
                              clip_sample=True, clip_sample_range=0.5)
    t = 30
    sample = jnp.zeros((1, 3))
    model_output = jnp.array([[10.0, -10.0, 0.1]])
    out = sched.step(model_output=model_output, timestep=t, sample=sample,
                      rng_key=jax.random.PRNGKey(0))
    pred = np.asarray(out.pred_original_sample)
    assert np.all(pred <= 0.5 + 1e-6)
    assert np.all(pred >= -0.5 - 1e-6)


def test_full_reverse_loop_runs_without_nans():
    """Sanity check: driving `step` across a full timestep schedule with an
    identity (zero) model output should stay finite throughout."""
    sched = JaxDDPMScheduler(num_train_timesteps=200, beta_schedule="squaredcos_cap_v2")
    sched.set_timesteps(20, power=2.0)
    key = jax.random.PRNGKey(0)
    sample = sched.sample_noise(key, (1, 4, 3))
    for t in sched.timesteps:
        key, step_key = jax.random.split(key)
        model_output = jnp.zeros_like(sample)
        out = sched.step(model_output=model_output, timestep=int(t), sample=sample, rng_key=step_key)
        sample = out.prev_sample
        assert bool(jnp.all(jnp.isfinite(sample)))
