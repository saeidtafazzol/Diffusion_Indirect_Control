"""Shared constants, normalization, and dynamics construction.

Dimension conventions (14-dimensional augmented state):
    STATE_DIM       = 7   [r_x, r_y, r_z, v_x, v_y, v_z, m]
    COSTATE_DIM     = 7   [lambda_r (3), lambda_v (3), lambda_m]
    AUGMENTED_DIM   = 14  STATE_DIM + COSTATE_DIM
"""

import numpy as np
import jax.numpy as jnp
import jaxadi
import casadi as ca

from dynamics import TwoBodyCartesian

STATE_DIM = 7
COSTATE_DIM = 7
AUGMENTED_DIM = STATE_DIM + COSTATE_DIM  # 14
COSTATE_NORM_MIN = 1e-12


def build_normalization():
    AU = 1.496e8
    AUm = AU * 1e3
    TU = 31536000 / (2 * np.pi)
    mu = 132712440018 / (AU ** 3) * (TU ** 2)
    M = 1000

    m0 = 1000 / M
    g0 = 9.8065
    I_sp = 2000

    T_max = 0.5 / AUm * (TU ** 2) / M
    c = I_sp * g0 / AUm * TU

    r_i = np.array([-140699693, -51614428, 980]) / AU
    v_i = np.array([9.774596, -28.07828, 4.337725e-4]) / AU * TU

    r_f = np.array([-172682023, 176959469, 7948912]) / AU
    v_f = np.array([-16.427384, -14.860506, 9.21486e-2]) / AU * TU

    t_f = 348.795 * 24 * 3600 / TU

    return {
        "AU": AU,
        "AUm": AUm,
        "TU": TU,
        "mu": mu,
        "M": M,
        "m0": m0,
        "g0": g0,
        "I_sp": I_sp,
        "t_max_norm": T_max,
        "c_norm": c,
        "r_i": r_i,
        "v_i": v_i,
        "r_f": r_f,
        "v_f": v_f,
        "t_f": t_f,
    }


def make_dynamics(eps=1e-4, compile_jax=True):
    norm = build_normalization()
    dy = TwoBodyCartesian(n_x=7)
    dy.set_params(
        mu=norm["mu"],
        t_max=norm["t_max_norm"],
        c=norm["c_norm"],
        eps=eps,
    )
    if compile_jax:
        odefunc = ca.Function(
            "odefunc_sub",
            [dy.states, dy.costates],
            [dy.augmented_dot_sub],
        )
        jdy = jaxadi.convert(odefunc, compile=True)
    else:
        jdy = None
    return dy, jdy, norm


def make_dynamics_stm(eps=1e-4):
    """Like make_dynamics but also returns a compiled Jacobian function.

    The Jacobian function jjac(state, costate) returns the 14×14 matrix
    A = ∂Ż/∂Z, where Z = [state (7), costate (7)].
    Returns (dy, jdy, jjac, norm).
    """
    norm = build_normalization()
    dy = TwoBodyCartesian(n_x=7)
    dy.set_params(
        mu=norm["mu"],
        t_max=norm["t_max_norm"],
        c=norm["c_norm"],
        eps=eps,
    )
    odefunc = ca.Function(
        "odefunc_sub",
        [dy.states, dy.costates],
        [dy.augmented_dot_sub],
    )
    jdy = jaxadi.convert(odefunc, compile=True)

    jacfunc = ca.Function(
        "jacfunc_sub",
        [dy.states, dy.costates],
        [dy.augmented_jac_sub],
    )
    jjac = jaxadi.convert(jacfunc, compile=True)

    return dy, jdy, jjac, norm


def build_rhs(jdy):
    def ode_rhs(t, augmented_state, _):
        del t
        state = augmented_state[:STATE_DIM]
        costate = augmented_state[STATE_DIM:STATE_DIM + COSTATE_DIM]
        return jnp.asarray(jdy(state, costate), dtype=jnp.float32).reshape(-1)
    return ode_rhs


def build_reverse_rhs(jdy):
    """Negated ODE for time-reversed integration.

    Integrating dz/ds = build_reverse_rhs(jdy)(s, z) from s=0 (z = y(t_f))
    to s=t_f yields z(s) = y(t_f - s).  Reversing sol.ys then recovers
    y(t) in forward-time order [t=0 ... t=t_f].
    """
    def ode_rhs(t, augmented_state, _):
        del t
        state = augmented_state[:STATE_DIM]
        costate = augmented_state[STATE_DIM:STATE_DIM + COSTATE_DIM]
        return -jnp.asarray(jdy(state, costate), dtype=jnp.float32).reshape(-1)
    return ode_rhs


def build_rhs_stm(jdy, jjac):
    """ODE RHS for joint (Z, Φ) integration.

    State layout: [Z (AUGMENTED_DIM), vec(Φ) (AUGMENTED_DIM²)]
    where Φ is the state transition matrix flattened row-major.

    Equations:
        dZ/dt  = f(Z)
        dΦ/dt  = A(Z) @ Φ,   A = ∂f/∂Z  (AUGMENTED_DIM × AUGMENTED_DIM)

    Initial conditions: Z(0) = z0, Φ(0) = I.
    """
    def ode_rhs(t, state_stm, _):
        del t
        state   = state_stm[:STATE_DIM]
        costate = state_stm[STATE_DIM:AUGMENTED_DIM]
        phi     = state_stm[AUGMENTED_DIM:].reshape(AUGMENTED_DIM, AUGMENTED_DIM)

        z_dot = jnp.asarray(jdy(state, costate), dtype=jnp.float32).reshape(-1)
        A     = jnp.asarray(jjac(state, costate), dtype=jnp.float32).reshape(AUGMENTED_DIM, AUGMENTED_DIM)
        phi_dot = A @ phi

        return jnp.concatenate([z_dot, phi_dot.reshape(-1)])
    return ode_rhs


def build_reverse_rhs_stm(jdy, jjac):
    """Negated ODE for time-reversed (Z, Φ) integration.

    Mirrors build_reverse_rhs but also propagates the STM.  Both the
    trajectory and the STM equation are negated (substituting s = T - t):

        dZ/ds  = -f(Z)
        dΦ/ds  = -A(Z) @ Φ

    Integrating from s=0 to s=T with Φ(0)=I recovers the forward STM
    after reversing sol.ys.
    """
    def ode_rhs(t, state_stm, _):
        del t
        state   = state_stm[:STATE_DIM]
        costate = state_stm[STATE_DIM:AUGMENTED_DIM]
        phi     = state_stm[AUGMENTED_DIM:].reshape(AUGMENTED_DIM, AUGMENTED_DIM)

        z_dot = -jnp.asarray(jdy(state, costate), dtype=jnp.float32).reshape(-1)
        A     = jnp.asarray(jjac(state, costate), dtype=jnp.float32).reshape(AUGMENTED_DIM, AUGMENTED_DIM)
        phi_dot = -(A @ phi)

        return jnp.concatenate([z_dot, phi_dot.reshape(-1)])
    return ode_rhs
