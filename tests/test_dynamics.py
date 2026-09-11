"""Unit tests for src/dynamics.py: TwoBodyCartesian symbolic dynamics.

These evaluate the CasADi expressions directly (no JAX compilation) so they
stay fast; core.py's jaxadi-compiled path is covered separately in test_core.py.
"""

import numpy as np
import casadi as ca
import pytest

from dynamics import TwoBodyCartesian
from core import build_normalization


@pytest.fixture(scope="module")
def dy_and_funcs():
    norm = build_normalization()
    dy = TwoBodyCartesian(n_x=7)
    dy.set_params(mu=norm["mu"], t_max=norm["t_max_norm"], c=norm["c_norm"], eps=1e-4)
    f = ca.Function("f_sub", [dy.states, dy.costates], [dy.augmented_dot_sub])
    h = ca.Function("h_sub", [dy.states, dy.costates], [dy.hamiltonian_sub])
    return dy, norm, f, h


def test_state_costate_dims(dy_and_funcs):
    dy, norm, f, h = dy_and_funcs
    assert dy.states.shape == (7, 1)
    assert dy.costates.shape == (7, 1)
    assert dy.augmented_dot_sub.shape == (14, 1)


def test_augmented_jac_shape(dy_and_funcs):
    dy, norm, f, h = dy_and_funcs
    assert dy.augmented_jac_sub.shape == (14, 14)


def test_thrust_off_reduces_to_keplerian_gravity(dy_and_funcs):
    dy, norm, f, h = dy_and_funcs
    r = np.asarray(norm["r_i"], dtype=np.float64)
    v = np.asarray(norm["v_i"], dtype=np.float64)
    m = norm["m0"]
    state = np.concatenate([r, v, [m]])
    # Strongly negative S forces delta -> 0 (thrust off); keep lambda_v nonzero
    # to avoid the 0/0 thrust-direction singularity.
    costate = np.array([0, 0, 0, 1e-3, 1e-3, 1e-3, -10.0])

    out = np.asarray(f(state, costate)).flatten()
    r_dot, v_dot, m_dot = out[0:3], out[3:6], out[6]

    np.testing.assert_allclose(r_dot, v, atol=1e-8)
    r_mag = np.linalg.norm(r)
    expected_g = -norm["mu"] / r_mag**3 * r
    # atol loosened slightly: S is very negative but not exactly -inf, so a
    # vanishingly small residual thrust remains.
    np.testing.assert_allclose(v_dot, expected_g, atol=1e-6)
    assert abs(m_dot) < 1e-6


def test_thrust_on_consumes_mass_at_full_rate(dy_and_funcs):
    dy, norm, f, h = dy_and_funcs
    r = np.asarray(norm["r_i"], dtype=np.float64)
    v = np.asarray(norm["v_i"], dtype=np.float64)
    m = norm["m0"]
    state = np.concatenate([r, v, [m]])
    # Strongly positive S forces delta -> 1 (thrust on, full throttle).
    costate = np.array([0, 0, 0, 1.0, 0, 0, 10.0])

    out = np.asarray(f(state, costate)).flatten()
    m_dot = out[6]
    expected_m_dot = -norm["t_max_norm"] / norm["c_norm"]
    assert m_dot == pytest.approx(expected_m_dot, rel=1e-3)


def test_thrust_direction_is_unit_vector_when_on(dy_and_funcs):
    dy, norm, f, h = dy_and_funcs
    r = np.asarray(norm["r_i"], dtype=np.float64)
    v = np.asarray(norm["v_i"], dtype=np.float64)
    m = norm["m0"]
    state = np.concatenate([r, v, [m]])
    lambda_v = np.array([0.3, -0.7, 0.2])
    costate = np.concatenate([[0, 0, 0], lambda_v, [10.0]])

    out = np.asarray(f(state, costate)).flatten()
    v_dot = out[3:6]
    r_mag = np.linalg.norm(r)
    gravity = -norm["mu"] / r_mag**3 * r
    thrust_accel = v_dot - gravity
    thrust_mag = np.linalg.norm(thrust_accel)
    expected_thrust_mag = norm["t_max_norm"] / m  # delta ≈ 1 here
    assert thrust_mag == pytest.approx(expected_thrust_mag, rel=1e-2)

    # Thrust direction should oppose lambda_v (alpha = -lambda_v / |lambda_v|)
    thrust_dir = thrust_accel / thrust_mag
    expected_dir = -lambda_v / np.linalg.norm(lambda_v)
    np.testing.assert_allclose(thrust_dir, expected_dir, atol=1e-2)


def test_hamiltonian_is_finite_and_scalar(dy_and_funcs):
    dy, norm, f, h = dy_and_funcs
    r = np.asarray(norm["r_i"], dtype=np.float64)
    v = np.asarray(norm["v_i"], dtype=np.float64)
    m = norm["m0"]
    state = np.concatenate([r, v, [m]])
    costate = np.array([0.1, -0.2, 0.05, 0.3, -0.1, 0.02, 0.5])
    value = float(np.asarray(h(state, costate)).flatten()[0])
    assert np.isfinite(value)


def test_configure_params_requires_mu():
    dy = TwoBodyCartesian(n_x=7)
    with pytest.raises(KeyError):
        dy.set_params(t_max=1.0, c=1.0, eps=1e-4)
