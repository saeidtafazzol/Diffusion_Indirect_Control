"""Unit tests for src/casadi_bvp_refine.py.

Full-convergence, real-checkpoint IPOPT solves are exercised in
test_known_trajectories.py; the tests here target the individual building
blocks (bounds, residual diagnostics, switching function, iterate capture)
plus a small/fast end-to-end BVPRefiner smoke test.
"""

import numpy as np
import casadi as ca
import pytest

from core import AUGMENTED_DIM, STATE_DIM, COSTATE_DIM, make_dynamics
from casadi_bvp_refine import (
    BVPRefiner,
    IterCapture,
    continuity_residuals,
    make_bounds,
    switching_function,
)


# ── make_bounds ───────────────────────────────────────────────────────────────

def test_make_bounds_pins_initial_state_exactly():
    initial_state = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 1.0])
    final_state = np.array([4.0, 5.0, 6.0, 0.4, 0.5, 0.6, 0.9])
    n_points = 5
    lbw, ubw = make_bounds(initial_state, final_state, n_points=n_points)
    np.testing.assert_allclose(lbw[:STATE_DIM], initial_state)
    np.testing.assert_allclose(ubw[:STATE_DIM], initial_state)


def test_make_bounds_pins_final_rv_but_frees_mass_by_default():
    initial_state = np.zeros(7)
    final_state = np.array([4.0, 5.0, 6.0, 0.4, 0.5, 0.6, 0.9])
    n_points = 4
    lbw, ubw = make_bounds(initial_state, final_state, n_points=n_points)
    D = AUGMENTED_DIM
    start_last = (n_points - 1) * D
    np.testing.assert_allclose(lbw[start_last:start_last + 6], final_state[:6])
    np.testing.assert_allclose(ubw[start_last:start_last + 6], final_state[:6])
    # mass (index 6) is NOT pinned -> wide open bounds
    assert lbw[start_last + 6] < -1e6
    assert ubw[start_last + 6] > 1e6


def test_make_bounds_fix_final_mass_pins_mass_too():
    initial_state = np.zeros(7)
    final_state = np.array([4.0, 5.0, 6.0, 0.4, 0.5, 0.6, 0.9])
    n_points = 4
    lbw, ubw = make_bounds(initial_state, final_state, n_points=n_points, fix_final_mass=True)
    D = AUGMENTED_DIM
    start_last = (n_points - 1) * D
    assert lbw[start_last + 6] == pytest.approx(0.9)
    assert ubw[start_last + 6] == pytest.approx(0.9)


def test_make_bounds_pins_final_lambda_m_to_zero():
    initial_state = np.zeros(7)
    final_state = np.zeros(7)
    n_points = 6
    lbw, ubw = make_bounds(initial_state, final_state, n_points=n_points)
    D = AUGMENTED_DIM
    idx = (n_points - 1) * D + D - 1
    assert lbw[idx] == 0.0
    assert ubw[idx] == 0.0


def test_make_bounds_interior_points_unbounded():
    initial_state = np.zeros(7)
    final_state = np.zeros(7)
    n_points = 5
    lbw, ubw = make_bounds(initial_state, final_state, n_points=n_points, large=1e9)
    D = AUGMENTED_DIM
    interior_start = D  # point index 1
    interior_end = (n_points - 1) * D  # up to (exclusive) last point block
    assert np.all(lbw[interior_start:interior_end] == -1e9)
    assert np.all(ubw[interior_start:interior_end] == 1e9)


# ── continuity_residuals (module-level diagnostic) ───────────────────────────

def test_continuity_residuals_near_zero_for_self_consistent_rk4_trajectory(dynamics_bundle):
    """Build a trajectory by RK4-propagating the real dynamics, then check the
    module-level continuity_residuals sees it as (near) dynamically consistent."""
    dy, jdy, norm = dynamics_bundle
    n_points = 6
    t_f = float(norm["t_f"])
    dts = np.diff(np.linspace(0.0, t_f, n_points))

    f_np = ca.Function("f_test", [dy.states, dy.costates], [dy.augmented_dot_sub])

    def f(Z):
        return np.asarray(f_np(Z[:7], Z[7:])).flatten()

    def rk4(Z, h):
        k1 = f(Z)
        k2 = f(Z + 0.5 * h * k1)
        k3 = f(Z + 0.5 * h * k2)
        k4 = f(Z + h * k3)
        return Z + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # lambda_v = 0 is a genuine 0/0 singularity in the optimal thrust direction
    # (alpha = -lambda_v / |lambda_v|); keep it small-but-nonzero.
    nonsingular_costate = np.array([0, 0, 0, 1e-3, 1e-3, 1e-3, 0])
    z0 = np.concatenate([norm["r_i"], norm["v_i"], [norm["m0"]], nonsingular_costate])
    traj = [z0]
    z = z0.copy()
    for dt in dts:
        h = dt / 16
        for _ in range(16):
            z = rk4(z, h)
        traj.append(z.copy())
    traj = np.asarray(traj)

    residuals = continuity_residuals(traj, dy, norm, n_points=n_points, n_rk4_eval=16)
    assert residuals.shape == (n_points - 1,)
    assert np.all(residuals < 1e-6)


def test_continuity_residuals_large_for_inconsistent_trajectory(dynamics_bundle):
    dy, jdy, norm = dynamics_bundle
    n_points = 4
    rng = np.random.default_rng(0)
    traj = rng.uniform(-1, 1, size=(n_points, AUGMENTED_DIM))
    traj[:, 6] = 1.0  # keep mass sane
    residuals = continuity_residuals(traj, dy, norm, n_points=n_points, n_rk4_eval=8)
    assert np.all(residuals > 1e-3)


# ── switching_function ────────────────────────────────────────────────────────

def test_switching_function_shapes():
    from core import build_normalization

    norm = build_normalization()
    T = 10
    z = np.zeros((T, AUGMENTED_DIM))
    z[:, 6] = 1.0  # mass
    delta, alpha, S = switching_function(z, norm, eps=1e-4)
    assert delta.shape == (T,)
    assert alpha.shape == (T, 3)
    assert S.shape == (T,)


def test_switching_function_delta_in_unit_interval():
    from core import build_normalization

    norm = build_normalization()
    rng = np.random.default_rng(1)
    T = 20
    z = rng.uniform(-2, 2, size=(T, AUGMENTED_DIM))
    z[:, 6] = np.abs(z[:, 6]) + 0.1  # positive mass
    delta, alpha, S = switching_function(z, norm, eps=1e-4)
    assert np.all(delta >= 0.0) and np.all(delta <= 1.0)


def test_switching_function_delta_saturates_with_sign_of_S():
    norm = {"c_norm": 1.0}
    T = 4
    z = np.zeros((T, AUGMENTED_DIM))
    z[:, 6] = 1.0
    # S = c*|lambda_v|/m + lambda_m - 1; make it very negative (delta -> 0) then
    # very positive (delta -> 1) via lambda_m, keeping lambda_v tiny but nonzero.
    z[0, 10] = 1e-6      # lambda_v_x
    z[0, 13] = -100.0    # lambda_m -> S very negative
    z[1, 10] = 1e-6
    z[1, 13] = 100.0     # S very positive
    delta, alpha, S = switching_function(z, norm, eps=1e-6)
    assert delta[0] < 1e-3
    assert delta[1] > 1.0 - 1e-3


# ── IterCapture callback ──────────────────────────────────────────────────────

def test_iter_capture_records_matching_size_and_ignores_others():
    cb = IterCapture(n_x=6, n_g=0)
    x_match = np.arange(6, dtype=np.float64)
    x_mismatch = np.arange(3, dtype=np.float64)
    ret = cb.eval([x_match, 0.0, np.zeros((0, 1)), np.zeros((6, 1)), np.zeros((0, 1)), np.zeros((0, 1))])
    assert ret == [0.0]
    assert len(cb.iterates) == 1
    np.testing.assert_array_equal(cb.iterates[0], x_match)

    cb.eval([x_mismatch, 0.0, np.zeros((0, 1)), np.zeros((3, 1)), np.zeros((0, 1)), np.zeros((0, 1))])
    assert len(cb.iterates) == 1  # unchanged: size mismatch ignored


def test_iter_capture_sparsity_shapes():
    cb = IterCapture(n_x=14, n_g=0)
    assert cb.get_n_in() == ca.nlpsol_n_out()
    assert cb.get_n_out() == 1
    x_idx = [i for i in range(ca.nlpsol_n_out()) if ca.nlpsol_out(i) == "x"][0]
    assert cb.get_sparsity_in(x_idx).shape == (14, 1)


# ── BVPRefiner: small, fast end-to-end smoke test ────────────────────────────

@pytest.mark.slow
def test_bvp_refiner_solve_reduces_continuity_residual():
    """Not a full convergence test (that lives in test_known_trajectories.py);
    just verifies the NLP construction + IPOPT solve runs on a tiny problem and
    doesn't make the trajectory worse."""
    n_points = 6
    refiner = BVPRefiner(eps=1e-4, n_points=n_points, n_rk4_steps=4, ipopt_verbosity=0)
    norm = refiner.norm

    initial_state = np.concatenate([norm["r_i"], norm["v_i"], [norm["m0"]]])
    final_state = np.concatenate([norm["r_f"], norm["v_f"], [norm["m0"]]])

    rng = np.random.default_rng(0)
    z_init = np.zeros((n_points, AUGMENTED_DIM))
    for i in range(n_points):
        frac = i / (n_points - 1)
        z_init[i, :STATE_DIM] = (1 - frac) * initial_state + frac * final_state
    z_init[:, STATE_DIM:] = rng.uniform(-0.1, 0.1, size=(n_points, COSTATE_DIM))

    res_before = float(np.max(refiner.continuity_residuals(z_init)))
    z_opt, ws = refiner.solve(z_init, initial_state, final_state, verbose=False)
    res_after = float(np.max(refiner.continuity_residuals(z_opt)))

    assert z_opt.shape == (n_points, AUGMENTED_DIM)
    assert np.all(np.isfinite(z_opt))
    np.testing.assert_allclose(z_opt[0, :STATE_DIM], initial_state, atol=1e-6)
    np.testing.assert_allclose(z_opt[-1, :6], final_state[:6], atol=1e-6)
    assert res_after <= res_before + 1e-9
    assert "lam_x" in ws
