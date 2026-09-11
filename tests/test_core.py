"""Unit tests for src/core.py: normalization, dynamics construction, RHS builders."""

import numpy as np
import jax.numpy as jnp
import pytest

from core import (
    STATE_DIM,
    COSTATE_DIM,
    AUGMENTED_DIM,
    build_normalization,
    make_dynamics,
    make_dynamics_stm,
    build_rhs,
    build_reverse_rhs,
    build_rhs_stm,
    build_reverse_rhs_stm,
)

# lambda_v = 0 is a genuine 0/0 singularity in the optimal thrust direction
# (alpha = -lambda_v / |lambda_v|); tests that don't care about the specific
# costate value use this small-but-nonzero stand-in instead of all-zeros.
_NONSINGULAR_COSTATE = np.array([0, 0, 0, 1e-3, 1e-3, 1e-3, 0], dtype=np.float64)


def test_dimension_constants():
    assert STATE_DIM == 7
    assert COSTATE_DIM == 7
    assert AUGMENTED_DIM == 14


def test_build_normalization_keys_and_types():
    norm = build_normalization()
    expected_keys = {
        "AU", "AUm", "TU", "mu", "M", "m0", "g0", "I_sp",
        "t_max_norm", "c_norm", "r_i", "v_i", "r_f", "v_f", "t_f",
    }
    assert expected_keys <= set(norm.keys())
    assert norm["r_i"].shape == (3,)
    assert norm["v_i"].shape == (3,)
    assert norm["r_f"].shape == (3,)
    assert norm["v_f"].shape == (3,)
    assert norm["m0"] == pytest.approx(1.0)
    assert norm["t_f"] > 0.0
    assert norm["mu"] > 0.0


def test_build_normalization_is_deterministic():
    a = build_normalization()
    b = build_normalization()
    for key in ("mu", "t_max_norm", "c_norm", "t_f", "m0"):
        assert a[key] == pytest.approx(b[key])
    np.testing.assert_allclose(a["r_i"], b["r_i"])
    np.testing.assert_allclose(a["r_f"], b["r_f"])


def test_make_dynamics_returns_compiled_callable():
    dy, jdy, norm = make_dynamics(eps=1e-4, compile_jax=True)
    assert jdy is not None
    state = jnp.asarray(np.concatenate([norm["r_i"], norm["v_i"], [norm["m0"]]]), dtype=jnp.float32)
    # lambda_v must be nonzero: the optimal thrust direction alpha = -lambda_v/|lambda_v|
    # is a genuine 0/0 singularity at lambda_v=0.
    costate = jnp.asarray([0, 0, 0, 1e-3, 1e-3, 1e-3, 0], dtype=jnp.float32)
    out = jdy(state, costate)
    out = np.asarray(out).reshape(-1)
    assert out.shape == (AUGMENTED_DIM,)
    assert np.all(np.isfinite(out))


def test_make_dynamics_compile_jax_false_skips_compilation():
    dy, jdy, norm = make_dynamics(eps=1e-4, compile_jax=False)
    assert jdy is None
    assert dy.augmented_dot_sub is not None


def test_make_dynamics_zero_thrust_state_dot_matches_two_body_gravity(dynamics_bundle):
    """With costates such that S << 0 (delta≈0), r_dot=v and v_dot≈gravity only."""
    dy, jdy, norm = dynamics_bundle
    r = np.asarray(norm["r_i"], dtype=np.float64)
    v = np.asarray(norm["v_i"], dtype=np.float64)
    m = norm["m0"]
    state = jnp.asarray(np.concatenate([r, v, [m]]), dtype=jnp.float32)
    # S = c*|lambda_v|/m + lambda_m - 1; make it very negative to force delta -> 0.
    # lambda_v kept small-but-nonzero (not exactly 0) so the thrust direction
    # alpha = -lambda_v / |lambda_v| stays well defined instead of 0/0.
    costate = jnp.asarray([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, -10.0], dtype=jnp.float32)
    out = np.asarray(jdy(state, costate)).reshape(-1)
    r_dot = out[:3]
    v_dot = out[3:6]
    m_dot = out[6]

    np.testing.assert_allclose(r_dot, v, atol=1e-5)

    r_mag = np.linalg.norm(r)
    expected_g = -norm["mu"] / r_mag**3 * r
    # atol loosened slightly beyond float32 eps: S is very negative but not
    # exactly -inf, so a vanishingly small residual thrust remains.
    np.testing.assert_allclose(v_dot, expected_g, atol=1e-6)
    # Thrust off => no mass loss
    assert abs(m_dot) < 1e-4


def test_make_dynamics_stm_shapes(dynamics_stm_bundle):
    dy, jdy, jjac, norm = dynamics_stm_bundle
    state = jnp.asarray(np.concatenate([norm["r_i"], norm["v_i"], [norm["m0"]]]), dtype=jnp.float32)
    costate = jnp.asarray(_NONSINGULAR_COSTATE, dtype=jnp.float32)
    jac = np.asarray(jjac(state, costate)).reshape(AUGMENTED_DIM, AUGMENTED_DIM)
    assert jac.shape == (AUGMENTED_DIM, AUGMENTED_DIM)
    assert np.all(np.isfinite(jac))


def test_build_rhs_matches_jdy_and_ignores_t(dynamics_bundle):
    dy, jdy, norm = dynamics_bundle
    rhs = build_rhs(jdy)
    z0 = jnp.asarray(
        np.concatenate([norm["r_i"], norm["v_i"], [norm["m0"]], _NONSINGULAR_COSTATE]),
        dtype=jnp.float32,
    )
    out_t0 = np.asarray(rhs(0.0, z0, None))
    out_t5 = np.asarray(rhs(5.0, z0, None))
    np.testing.assert_allclose(out_t0, out_t5)

    direct = np.asarray(jdy(z0[:STATE_DIM], z0[STATE_DIM:AUGMENTED_DIM])).reshape(-1)
    np.testing.assert_allclose(out_t0, direct)


def test_build_reverse_rhs_is_negation_of_forward(dynamics_bundle):
    dy, jdy, norm = dynamics_bundle
    fwd = build_rhs(jdy)
    rev = build_reverse_rhs(jdy)
    z0 = jnp.asarray(
        np.concatenate([norm["r_i"], norm["v_i"], [norm["m0"]], _NONSINGULAR_COSTATE]),
        dtype=jnp.float32,
    )
    fwd_out = np.asarray(fwd(0.0, z0, None))
    rev_out = np.asarray(rev(0.0, z0, None))
    np.testing.assert_allclose(rev_out, -fwd_out)


def test_build_rhs_stm_initial_derivative_matches_build_rhs(dynamics_stm_bundle):
    dy, jdy, jjac, norm = dynamics_stm_bundle
    rhs_stm = build_rhs_stm(jdy, jjac)
    plain_rhs = build_rhs(jdy)

    z0 = jnp.asarray(
        np.concatenate([norm["r_i"], norm["v_i"], [norm["m0"]], _NONSINGULAR_COSTATE]),
        dtype=jnp.float32,
    )
    phi0 = jnp.eye(AUGMENTED_DIM, dtype=jnp.float32).reshape(-1)
    z_stm0 = jnp.concatenate([z0, phi0])

    out = np.asarray(rhs_stm(0.0, z_stm0, None))
    z_dot = out[:AUGMENTED_DIM]
    phi_dot = out[AUGMENTED_DIM:].reshape(AUGMENTED_DIM, AUGMENTED_DIM)

    expected_z_dot = np.asarray(plain_rhs(0.0, z0, None))
    np.testing.assert_allclose(z_dot, expected_z_dot, atol=1e-5)

    # At Phi=I, Phi_dot = A @ I = A (float32 jacobian entries near the
    # eps-regularized switching function can differ by a few 1e-4 in the
    # last significant digits; atol reflects float32 precision, not a bug).
    A = np.asarray(jjac(z0[:STATE_DIM], z0[STATE_DIM:AUGMENTED_DIM])).reshape(AUGMENTED_DIM, AUGMENTED_DIM)
    np.testing.assert_allclose(phi_dot, A, atol=5e-4)


def test_build_reverse_rhs_stm_is_negation_of_forward_stm(dynamics_stm_bundle):
    dy, jdy, jjac, norm = dynamics_stm_bundle
    fwd = build_rhs_stm(jdy, jjac)
    rev = build_reverse_rhs_stm(jdy, jjac)

    z0 = jnp.asarray(
        np.concatenate([norm["r_i"], norm["v_i"], [norm["m0"]], _NONSINGULAR_COSTATE]),
        dtype=jnp.float32,
    )
    phi0 = jnp.eye(AUGMENTED_DIM, dtype=jnp.float32).reshape(-1)
    z_stm0 = jnp.concatenate([z0, phi0])

    fwd_out = np.asarray(fwd(0.0, z_stm0, None))
    rev_out = np.asarray(rev(0.0, z_stm0, None))
    np.testing.assert_allclose(rev_out, -fwd_out, atol=1e-5)
