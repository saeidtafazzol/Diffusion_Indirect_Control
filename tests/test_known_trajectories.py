"""Regression guardrail: known Earth->Mars trajectories the diffusion + BVP
refinement pipeline is known to converge on.

These six (shift, seed=0) cases were pulled from
shift_eps_study_50_with_json/ — a 50-trial-per-shift convergence study run
against the checkpoint at checkpoints/indiff_ctrl_latest.msgpack — where
Method A (diffusion -> one-shot CasADi multiple-shooting BVP refine)
converged with max continuity residual well under the study's 1e-8 tolerance
(see tests/fixtures/known_convergent_trajectories.json).

The pipeline here is assembled directly from src/ building blocks (policy's
model/scheduler/condition-mask machinery + casadi_bvp_refine.BVPRefiner) using
the exact settings (30 diffusion steps, power=2.7 timestep spacing, the
"no final mass" state mask + "final lambda_m=0" costate mask) that produced
that convergence — mirroring experiments/shift_eps_comparison.py's
run_diffusion()/solve_bvp_oneshot(), but calling only src/ code, not
experiments code.

Intent: this file must stay green across the planned src/ refactor. A break
here means the refactor changed externally-visible numerical behavior.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core import STATE_DIM, COSTATE_DIM, AUGMENTED_DIM
from casadi_bvp_refine import BVPRefiner
from transformer_diffusion_model import (
    default_state_known_mask_no_final_mass,
    final_lambda_m_costate_mask,
)

pytestmark = pytest.mark.slow

NUM_DIFFUSION_STEPS = 30
DIFFUSION_POWER = 2.7
RES_TOL = 1e-8
N_RK4_STEPS = 8


def _run_diffusion(policy, params, initial_state, final_state, num_steps, rng_seed):
    """Reverse-diffusion rollout using only policy/model/scheduler pieces from
    src/policy.py — the same conditioning used to train and validate the model."""
    seq_len = len(policy.time_grid)
    cfg = policy.model.config
    key = jax.random.PRNGKey(rng_seed)

    skm = default_state_known_mask_no_final_mass(1, seq_len)
    ckm = final_lambda_m_costate_mask(1, seq_len)
    skm, ckm = policy._resolve_known_masks(1, seq_len, state_known_mask=skm, costate_known_mask=ckm)

    cond_st = np.zeros((1, seq_len, STATE_DIM), np.float32)
    cond_cs = np.zeros((1, seq_len, COSTATE_DIM), np.float32)
    cond_st[0, 0, :] = initial_state
    cond_st[0, -1, :] = final_state

    cond_traj = jnp.asarray(np.concatenate([cond_st, cond_cs], axis=-1), dtype=jnp.float32)
    cond_mask = policy._build_condition_mask(skm, ckm)

    key, sk = jax.random.split(key)
    traj = policy.noise_scheduler.sample_noise(sk, cond_traj.shape, dtype=jnp.float32)
    policy.noise_scheduler.set_timesteps(num_steps, power=DIFFUSION_POWER)

    for ts in policy.noise_scheduler.timesteps:
        traj = jnp.where(cond_mask, cond_traj, traj)
        st = traj[..., :STATE_DIM]
        cst = traj[..., STATE_DIM:STATE_DIM + COSTATE_DIM]

        int_st = int_cst = int_fail = None
        if cfg.use_integration_signals:
            int_st, int_cst, int_fail = policy.compute_integration_signals(st, cst)

        outputs = policy.model.apply(
            {"params": params},
            noisy_states=st, noisy_costates=cst,
            diffusion_steps=jnp.full((1,), int(ts), dtype=jnp.int32),
            state_known_mask=skm, costate_known_mask=ckm,
            integrated_states=int_st, integrated_costates=int_cst,
            integration_failed=int_fail, train=False,
        )
        model_output = policy._combine_trajectory(outputs["state_eps"], outputs["costate_eps"])
        key, step_key = jax.random.split(key)
        traj = policy.noise_scheduler.step(
            model_output=model_output, timestep=int(ts), sample=traj, rng_key=step_key,
        ).prev_sample

    traj = jnp.where(cond_mask, cond_traj, traj)
    return np.asarray(traj[0], dtype=np.float32)  # (seq_len, 14)


@pytest.fixture(scope="module")
def refiner(checkpoint_bundle):
    _, policy, time_grid, _ = checkpoint_bundle
    n_points = len(time_grid)
    return BVPRefiner(eps=1e-4, n_points=n_points, n_rk4_steps=N_RK4_STEPS, ipopt_verbosity=0)


@pytest.fixture(
    params=list(range(6)),
    ids=lambda i: f"traj{i}",
)
def known_trajectory_case(request, known_trajectories):
    return known_trajectories["trajectories"][request.param]


def test_diffusion_plus_bvp_converges_on_known_trajectory(
    checkpoint_bundle, refiner, known_trajectory_case,
):
    params, policy, time_grid, _ = checkpoint_bundle
    seed = 0  # matches known_convergent_trajectories.json's fixed seed

    initial_state = known_trajectory_case["initial_state"].astype(np.float32)
    final_state = known_trajectory_case["final_state"].astype(np.float32)
    shift_days = known_trajectory_case["shift_days"]

    diff_out = _run_diffusion(policy, params, initial_state, final_state, NUM_DIFFUSION_STEPS, seed)
    assert np.all(np.isfinite(diff_out)), f"diffusion output non-finite for shift={shift_days:+d}d"

    z_opt, warm_start = refiner.solve(
        diff_out, initial_state, final_state, verbose=False,
    )
    residuals = refiner.continuity_residuals(z_opt, n_rk4_eval=8)
    max_residual = float(np.max(residuals))

    assert np.all(np.isfinite(z_opt))
    np.testing.assert_allclose(z_opt[0, :STATE_DIM], initial_state, atol=1e-6)
    np.testing.assert_allclose(z_opt[-1, :6], final_state[:6], atol=1e-6)
    assert abs(z_opt[-1, AUGMENTED_DIM - 1]) < 1e-6  # lambda_m(t_f) = 0 transversality

    assert max_residual < RES_TOL, (
        f"shift={shift_days:+d}d seed={seed}: max continuity residual {max_residual:.3e} "
        f"did not beat {RES_TOL:.0e} (reference run achieved "
        f"{known_trajectory_case['reference_max_continuity_residual']:.3e})"
    )
