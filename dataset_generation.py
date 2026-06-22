"""Dataset generation for min-fuel indirect trajectories.

Generates Earth-Mars min-fuel trajectory samples satisfying per-sample
constraints enforced by batch gradient descent:

    lambda_m(t=0)  <=  tau        tau     ~ Uniform[LAM_THRESH_LOW,  LAM_THRESH_HIGH]
    ||r(t=0)||     <=  pv_thresh  pv_thresh ~ Uniform[PV_THRESH_LOW, PV_THRESH_HIGH]
    ||v(t=0)||     <=  pv_thresh  (same threshold, both position and velocity)
    hard gate:  ||r||, ||v||  <=  PV_HARD_REJECT  (absolute acceptance ceiling)

GD objective per sample:
    loss = relu(lam_m(t=0) - tau)
           + PV_ALPHA * relu(||r(t=0)|| - pv_thresh)
           + PV_ALPHA * relu(||v(t=0)|| - pv_thresh)

Gradients are computed analytically via the State Transition Matrix (default,
--no-stm disables) or by central finite differences.

Free variables:   yf[3:6]  (terminal velocity),  costate_f[0:6]  (rv-costates)
Fixed variables:  yf[0:3]  (terminal position),   yf[6] (mass),   costate_f[6] = 0

STM convention
--------------
Backward integration from t_f to t_0 (substituting s = t_f - t) yields the
matrix Phi satisfying:

    delta_Z(t_0) = Phi(t_0, t_f) @ delta_Z(t_f)

where Z = [state (7) | costate (7)].

Z_f layout:   [r_f(3) | v_f(3) | m_f(1) | lam_r_f(3) | lam_v_f(3) | lam_m_f(1)]
              idx:  0-2      3-5      6       7-9         10-12         13

Free-param columns of Z_f:  [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]  (terminal r + v + rv-costates)
Loss-relevant rows of Z_0:  [0:3] = r(t_0),  [3:6] = v(t_0),  [13] = lam_m(t_0)

Gradient computation:
    d(Z_0[i])/d(p[k])   = Phi[i, FREE_COLS[k]]
    d(lam_m)/d(p)        = Phi[13, FREE_COLS]                          (N_FREE,)
    d(||r||)/d(p)        = r_unit @ Phi[0:3, FREE_COLS]                (N_FREE,)
    d(||v||)/d(p)        = v_unit @ Phi[3:6, FREE_COLS]                (N_FREE,)
"""

import argparse
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from diffrax import RESULTS, diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController

from core import (
    STATE_DIM, COSTATE_DIM, AUGMENTED_DIM,
    make_dynamics_stm,
    build_reverse_rhs, build_reverse_rhs_stm,
)

# ── Chunk size ─────────────────────────────────────────────────────────────────
CHUNK_SAVE_SIZE = 2**8

# ── Per-sample lambda_m convergence threshold ──────────────────────────────────
LAM_THRESH_LOW  = 0.3
LAM_THRESH_HIGH = 0.9

# ── Per-sample pos/vel soft-penalty threshold ──────────────────────────────────
PV_THRESH_LOW  = 1.5
PV_THRESH_HIGH = 2.0

# ── Absolute acceptance gate for ||r(t=0)|| and ||v(t=0)|| ────────────────────
PV_HARD_REJECT = 2.7

# ── Penalty coefficient (lam_m has implicit weight 1.0) ───────────────────────
PV_ALPHA = 0.1

# ── Terminal mass bounds ───────────────────────────────────────────────────────
MASS_FINAL_LOW  = 0.2
MASS_FINAL_HIGH = 1.0

# ── Free-parameter structure ───────────────────────────────────────────────────
# Z_f = [r_f(3) | v_f(3) | m_f(1) | lam_r_f(3) | lam_v_f(3) | lam_m_f(1)]
#  idx:   0-2      3-5      6         7-9          10-12          13
# Free: r_f (indices 0-2), v_f (indices 3-5), rv-costates lam_rv_f (indices 7-12).
# Fixed: m_f (index 6), lam_m_f = 0 (index 13, Pontryagin transversality).
FREE_COLS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
N_FREE    = len(FREE_COLS)

# ── GD hyper-parameters ────────────────────────────────────────────────────────
GD_MAX_ITER      = 25
GD_LR            = 1e-1
GD_FD_STEP       = 1e-4
GD_NOISE_STD     = 1e-4
GD_MAX_GRAD_NORM = 5.0


# ─────────────────────────────────────────────────────────────────────────────
# State / costate sampling
# ─────────────────────────────────────────────────────────────────────────────

def _perp_basis(r_vec):
    r_hat = r_vec / np.linalg.norm(r_vec)
    aux   = np.array([1.0, 0.0, 0.0]) if abs(r_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u1    = np.cross(r_hat, aux);  u1 /= np.linalg.norm(u1)
    u2    = np.cross(r_hat, u1);   u2 /= np.linalg.norm(u2)
    return u1, u2


def _sample_orbital_state(rng, low, high, min_radius, mu, m_val):
    """Orbital position/velocity via vis-viva equation + given mass."""
    while True:
        pos = rng.uniform(low, high, size=3).astype(np.float32)
        if np.linalg.norm(pos) > min_radius:
            break
    r_mag  = float(np.linalg.norm(pos))
    a      = rng.uniform(r_mag / 2.0 + 1e-8, r_mag * 2.0)
    v_mag  = np.sqrt(max(mu * (2.0 / r_mag - 1.0 / a), 0.0))
    u1, u2 = _perp_basis(pos)
    theta  = rng.uniform(0.0, 2.0 * np.pi)
    vel    = (v_mag * (np.cos(theta) * u1 + np.sin(theta) * u2)).astype(np.float32)
    return np.concatenate([pos, vel, [np.float32(m_val)]])


def sample_final_state_batch(rng, batch_size, low, high, min_radius, mu):
    """Terminal states: orbital pos/vel + mass ~ Uniform[MASS_FINAL_LOW, MASS_FINAL_HIGH]."""
    states = [
        _sample_orbital_state(rng, low, high, min_radius, mu,
                               rng.uniform(MASS_FINAL_LOW, MASS_FINAL_HIGH))
        for _ in range(batch_size)
    ]
    return jnp.asarray(np.stack(states), dtype=jnp.float32)


def sample_final_costate_batch(rng, batch_size):
    """lam_rv ~ Uniform[-2, 2]^6;  lam_m = 0 (Pontryagin transversality)."""
    lam_rv = rng.uniform(-2.0, 2.0, size=(batch_size, 6)).astype(np.float32)
    lam_m  = np.zeros((batch_size, 1), dtype=np.float32)
    return jnp.asarray(np.concatenate([lam_rv, lam_m], axis=1))


# ─────────────────────────────────────────────────────────────────────────────
# Integrator factories
# ─────────────────────────────────────────────────────────────────────────────

def make_traj_integrator(jdy, t_final, n_points, rtol, atol):
    """Vmapped backward integrator saving the full trajectory at n_points steps.

    Integrates dz/ds = -f(z) from s=0 (z = Z_f) to s=t_f.
    sol.ys[::-1] reorders to forward time: index 0 → t=0, index -1 → t=t_f.

    Returns: states (B, T, 7),  costates (B, T, 7),  ok (B,)
    """
    ts         = jnp.linspace(0.0, float(t_final), n_points)
    term       = ODETerm(build_reverse_rhs(jdy))
    solver     = Dopri5()
    controller = PIDController(rtol=rtol, atol=atol)
    saveat     = SaveAt(ts=ts)

    def _single(yf, cf):
        sol = diffeqsolve(
            term, solver, t0=0.0, t1=float(t_final), dt0=None,
            y0=jnp.concatenate([yf, cf]), args=None,
            stepsize_controller=controller, saveat=saveat, throw=False,
        )
        ys = jnp.asarray(sol.ys[::-1], dtype=jnp.float32)  # forward-time order
        ok = sol.result == RESULTS.successful
        return ys[:, :STATE_DIM], ys[:, STATE_DIM:AUGMENTED_DIM], ok

    return jax.jit(jax.vmap(_single))


def make_fast_integrator(jdy, t_final, rtol, atol):
    """Vmapped backward integrator saving only Z at t=0 (for FD gradient eval).

    Saves at s=t_f which corresponds to forward time t=0.
    Returns: Z_0 (B, 14),  ok (B,)
    """
    # Save only at the endpoint s=t_f (forward time t=0).
    ts         = jnp.array([float(t_final)])
    term       = ODETerm(build_reverse_rhs(jdy))
    solver     = Dopri5()
    controller = PIDController(rtol=rtol, atol=atol)
    saveat     = SaveAt(ts=ts)

    def _single(yf, cf):
        sol = diffeqsolve(
            term, solver, t0=0.0, t1=float(t_final), dt0=None,
            y0=jnp.concatenate([yf, cf]), args=None,
            stepsize_controller=controller, saveat=saveat, throw=False,
        )
        Z_0 = jnp.asarray(sol.ys[0], dtype=jnp.float32)  # (14,) at s=t_f ≡ t=0
        ok  = sol.result == RESULTS.successful
        return Z_0, ok

    return jax.jit(jax.vmap(_single))


def make_stm_integrator(jdy, jjac, t_final, rtol, atol):
    """Vmapped backward STM integrator saving Z_0 and Phi(t_0, t_f) at t=0.

    Integrates the 210-dim (Z, Phi) system:
        dZ/ds  = -f(Z)           Phi(s=0) = I
        dPhi/ds = -A(Z) @ Phi    where A = df/dZ

    Saves at s=t_f (forward time t=0).  Result satisfies:
        delta_Z(t_0) = Phi @ delta_Z(t_f)

    Returns: Z_0 (B, 14),  Phi (B, 14, 14),  ok (B,)
    """
    phi_init_flat = jnp.eye(AUGMENTED_DIM, dtype=jnp.float32).reshape(-1)
    ts         = jnp.array([float(t_final)])
    term       = ODETerm(build_reverse_rhs_stm(jdy, jjac))
    solver     = Dopri5()
    controller = PIDController(rtol=rtol, atol=atol)
    saveat     = SaveAt(ts=ts)

    def _single(yf, cf):
        Z_f = jnp.concatenate([yf, cf])
        sol = diffeqsolve(
            term, solver, t0=0.0, t1=float(t_final), dt0=None,
            y0=jnp.concatenate([Z_f, phi_init_flat]), args=None,
            stepsize_controller=controller, saveat=saveat, throw=False,
        )
        # sol.ys[0]: 210-dim state at s=t_f (forward time t=0)
        final = jnp.asarray(sol.ys[0], dtype=jnp.float32)
        Z_0   = final[:AUGMENTED_DIM]
        Phi   = final[AUGMENTED_DIM:].reshape(AUGMENTED_DIM, AUGMENTED_DIM)
        ok    = sol.result == RESULTS.successful
        return Z_0, Phi, ok

    return jax.jit(jax.vmap(_single))


# ─────────────────────────────────────────────────────────────────────────────
# Loss and gradient helpers
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_loss(Z_0, lam_t, pv_t, pv_alpha):
    """Composite loss and diagnostics from Z at t=0.

    Z_0:    (B, 14) float64  – augmented state at t=0
    lam_t:  (B,)    float64  – per-sample lambda_m threshold
    pv_t:   (B,)    float64  – per-sample pos/vel soft-penalty threshold

    Returns: loss, lam_m, r_norm, v_norm  all (B,) float64
    """
    r_norm = np.linalg.norm(Z_0[:, 0:3],  axis=-1)
    v_norm = np.linalg.norm(Z_0[:, 3:6],  axis=-1)
    lam_m  = Z_0[:, 13]
    loss   = (
        np.maximum(0.0, lam_m  - lam_t)
        + pv_alpha * np.maximum(0.0, r_norm - pv_t)
        + pv_alpha * np.maximum(0.0, v_norm - pv_t)
    )
    return loss, lam_m, r_norm, v_norm


def _stm_gradient(Z_0, Phi, lam_t, pv_t, pv_alpha):
    """Analytical gradient of the loss w.r.t. the N_FREE free params via Phi.

    The STM satisfies  delta_Z(t_0) = Phi @ delta_Z(t_f), so:

        d(Z_0[i])/d(p[k])  = Phi[i, FREE_COLS[k]]

    where p is the free-parameter vector and FREE_COLS selects the free
    columns of Z_f.

    Gradients used:
        d(lam_m)/d(p)   = Phi[13, FREE_COLS]                     (B, 12)
        d(||r||)/d(p)   = r_unit @ Phi[0:3, FREE_COLS]           (B, 12)
        d(||v||)/d(p)   = v_unit @ Phi[3:6, FREE_COLS]           (B, 12)

    Subgradient of relu activates only where the penalty term is positive.

    Z_0, Phi: float64 numpy arrays
    Returns: gradient (B, N_FREE) float64
    """
    Phi_f  = Phi[:, :, FREE_COLS]                       # (B, 14, 12)

    r      = Z_0[:, 0:3]
    v      = Z_0[:, 3:6]
    lam_m  = Z_0[:, 13]
    r_norm = np.linalg.norm(r, axis=-1)
    v_norm = np.linalg.norm(v, axis=-1)

    h_lam  = (lam_m  > lam_t).astype(np.float64)        # Heaviside sub-gradients
    h_r    = (r_norm > pv_t ).astype(np.float64)
    h_v    = (v_norm > pv_t ).astype(np.float64)

    g_lam  = h_lam[:, None] * Phi_f[:, 13, :]           # (B, 9)

    r_unit = r / np.maximum(r_norm[:, None], 1e-12)
    g_r    = h_r[:, None] * np.einsum('bi,bij->bj', r_unit, Phi_f[:, 0:3, :])  # (B, 9)

    v_unit = v / np.maximum(v_norm[:, None], 1e-12)
    g_v    = h_v[:, None] * np.einsum('bi,bij->bj', v_unit, Phi_f[:, 3:6, :])  # (B, 9)

    return g_lam + pv_alpha * g_r + pv_alpha * g_v      # (B, 9)


def _fd_gradient(fast_integrator, params, active_idx,
                 yf_mass, lam_t, pv_t, pv_alpha, fd_h):
    """Central-difference gradient using stacked 2A-sample integration calls.

    Only active (needs-GD) samples are evaluated; inactive rows are zero.
    Each of N_FREE parameters is perturbed ±fd_h independently.

    Returns: gradient (B, N_FREE) float64
    """
    B  = len(params)
    A  = len(active_idx)
    ap = params[active_idx]          # (A, N_FREE)

    a_mass  = yf_mass[active_idx]    # (A, 1)
    a_lam_t = lam_t[active_idx]
    a_pv_t  = pv_t[active_idx]

    def _build(p_a):
        yf_ = np.concatenate([p_a[:, :6], a_mass], axis=1).astype(np.float32)  # r+v free, mass fixed
        cf_ = np.concatenate([p_a[:, 6:], np.zeros((A, 1), np.float32)], axis=1)
        return yf_, cf_

    def _loss_from_Z0(Z, lam_t_, pv_t_):
        r_pen = np.maximum(0.0, np.linalg.norm(Z[:, 0:3], axis=-1) - pv_t_)
        v_pen = np.maximum(0.0, np.linalg.norm(Z[:, 3:6], axis=-1) - pv_t_)
        return np.maximum(0.0, Z[:, 13] - lam_t_) + pv_alpha * (r_pen + v_pen)

    grad = np.zeros((B, N_FREE))
    for j in range(N_FREE):
        p_p, p_m = ap.copy(), ap.copy()
        p_p[:, j] += fd_h
        p_m[:, j] -= fd_h
        yf_p, cf_p = _build(p_p)
        yf_m, cf_m = _build(p_m)
        # Stack ±h into a single 2A-sample call for efficiency
        Z_2a, _ = fast_integrator(
            jnp.asarray(np.vstack([yf_p, yf_m])),
            jnp.asarray(np.vstack([cf_p, cf_m])),
        )
        Z_2a_np = np.asarray(jax.device_get(Z_2a), dtype=np.float64)
        t2 = np.concatenate([a_lam_t, a_lam_t])
        p2 = np.concatenate([a_pv_t,  a_pv_t])
        grad[active_idx, j] = (
            _loss_from_Z0(Z_2a_np[:A], t2[:A], p2[:A])
            - _loss_from_Z0(Z_2a_np[A:], t2[A:], p2[A:])
        ) / (2.0 * fd_h)

    return grad


# ─────────────────────────────────────────────────────────────────────────────
# Gradient descent
# ─────────────────────────────────────────────────────────────────────────────

def run_gradient_descent(
    stm_integrator,
    fast_integrator,
    yf_np,
    cf_np,
    lam_thresholds,
    pv_thresholds,
    rng,
    use_stm,
    max_iter       = GD_MAX_ITER,
    lr             = GD_LR,
    fd_h           = GD_FD_STEP,
    noise_std      = GD_NOISE_STD,
    max_grad_norm  = GD_MAX_GRAD_NORM,
    pv_hard_reject = PV_HARD_REJECT,
    pv_alpha       = PV_ALPHA,
):
    """Batch GD minimising the composite loss over the N_FREE free parameters.

    Free parameters (N_FREE = 12):
        [r_fx, r_fy, r_fz,  v_fx, v_fy, v_fz,
         lam_rx_f, lam_ry_f, lam_rz_f,  lam_vx_f, lam_vy_f, lam_vz_f]
    Fixed: m_f (terminal mass),  lam_m_f = 0 (Pontryagin transversality).

    Convergence per sample requires:
        lam_m(t=0) <= lam_thresholds[i]
        ||r(t=0)||, ||v(t=0)|| <= pv_hard_reject

    Returns:
        yf_out    (B, 7) float32 – optimised terminal states
        cf_out    (B, 7) float32 – optimised terminal costates (lam_m = 0)
        converged (B,)   bool
    """
    B       = len(yf_np)
    yf_mass = yf_np[:, 6:7].astype(np.float64)
    params  = np.concatenate([yf_np[:, :6], cf_np[:, :6]], axis=1).astype(np.float64)
    lam_t   = lam_thresholds.astype(np.float64)
    pv_t    = pv_thresholds.astype(np.float64)

    def _reconstruct(p):
        yf_ = np.concatenate([p[:, :6], yf_mass], axis=1).astype(np.float32)  # r+v free, mass fixed
        cf_ = np.concatenate([p[:, 6:], np.zeros((B, 1), np.float32)], axis=1)
        return yf_, cf_

    def _eval(p):
        yf_, cf_ = _reconstruct(p)
        if use_stm:
            Z_j, Ph_j, ok_j = stm_integrator(jnp.asarray(yf_), jnp.asarray(cf_))
            Z_0 = np.asarray(jax.device_get(Z_j),  dtype=np.float64)
            Phi = np.asarray(jax.device_get(Ph_j), dtype=np.float64)
            ok  = np.asarray(jax.device_get(ok_j), dtype=bool)
        else:
            Z_j, ok_j = fast_integrator(jnp.asarray(yf_), jnp.asarray(cf_))
            Z_0 = np.asarray(jax.device_get(Z_j),  dtype=np.float64)
            Phi = None
            ok  = np.asarray(jax.device_get(ok_j), dtype=bool)
        return Z_0, Phi, ok

    # ── Initial evaluation ────────────────────────────────────────────────────
    Z_0, Phi, ok = _eval(params)
    _, lam_m, r_norm, v_norm = _evaluate_loss(Z_0, lam_t, pv_t, pv_alpha)

    perm_failed = ~np.isfinite(lam_m) | ~np.isfinite(r_norm) | ~np.isfinite(v_norm)
    needs_gd    = (
        (lam_m > lam_t) | (r_norm > pv_hard_reject) | (v_norm > pv_hard_reject)
    ) & ok & ~perm_failed

    n_lam = int(np.sum((lam_m > lam_t) & ok & ~perm_failed))
    n_pv  = int(np.sum(((r_norm > pv_hard_reject) | (v_norm > pv_hard_reject)) & ok & ~perm_failed))
    print(
        f"  GD start: {int(np.sum(needs_gd))}/{B} need GD "
        f"(lam_m={n_lam}, r/v={n_pv})  "
        f"lam_m=[{lam_m.min():.4f},{lam_m.max():.4f}]  "
        f"r=[{r_norm.min():.3f},{r_norm.max():.3f}]  "
        f"v=[{v_norm.min():.3f},{v_norm.max():.3f}]"
    )

    # ── GD loop ───────────────────────────────────────────────────────────────
    for it in range(max_iter):
        if not np.any(needs_gd):
            print(f"  GD: all converged before iter {it + 1}")
            break

        active_idx = np.where(needs_gd)[0]

        # Compute gradient (STM: one eval; FD: 2*N_FREE evals for active samples)
        if use_stm:
            grad = np.zeros((B, N_FREE))
            grad[active_idx] = _stm_gradient(
                Z_0[active_idx], Phi[active_idx],
                lam_t[active_idx], pv_t[active_idx], pv_alpha,
            )
        else:
            grad = _fd_gradient(
                fast_integrator, params, active_idx,
                yf_mass, lam_t, pv_t, pv_alpha, fd_h,
            )

        # Drop samples with non-finite gradients (numerical singularity)
        bad_grad     = ~np.all(np.isfinite(grad), axis=1)
        perm_failed |= bad_grad & needs_gd
        needs_gd    &= ~perm_failed

        # Per-sample gradient clipping + noise
        gn_col = np.linalg.norm(grad, axis=1, keepdims=True)    # (B, 1)
        clip   = np.minimum(1.0, max_grad_norm / (gn_col + 1e-8))
        noise  = rng.normal(0.0, noise_std, size=(B, N_FREE))
        params[needs_gd] -= (lr * grad * clip + noise)[needs_gd]

        # Evaluate at updated params
        Z_0, Phi, ok = _eval(params)
        _, lam_m, r_norm, v_norm = _evaluate_loss(Z_0, lam_t, pv_t, pv_alpha)
        bad          = ~np.isfinite(lam_m) | ~np.isfinite(r_norm) | ~np.isfinite(v_norm)
        perm_failed |= bad
        needs_gd     = (
            (lam_m > lam_t) | (r_norm > pv_hard_reject) | (v_norm > pv_hard_reject) | ~ok
        ) & ~perm_failed

        n_conv    = int(np.sum(~needs_gd & ~perm_failed))
        n_dropped = int(np.sum(perm_failed))
        active    = needs_gd
        gn        = np.linalg.norm(grad, axis=1)
        n_clip    = int(np.sum((gn > max_grad_norm) & active))
        if np.any(active):
            print(
                f"  GD iter {it + 1:3d}/{max_iter}  "
                f"conv={n_conv}/{B}  drop={n_dropped}  "
                f"lam_m=[{lam_m[active].min():.4f},{lam_m[active].max():.4f}]  "
                f"r=[{r_norm[active].min():.3f},{r_norm[active].max():.3f}]  "
                f"v=[{v_norm[active].min():.3f},{v_norm[active].max():.3f}]  "
                f"|g|=[{gn[active].min():.2e},{gn[active].max():.2e}]  "
                f"clip={n_clip}/{len(active_idx)}"
            )
        else:
            print(f"  GD iter {it + 1:3d}/{max_iter}  all done  drop={n_dropped}")

    # ── Final acceptance ──────────────────────────────────────────────────────
    converged = (
        (lam_m <= lam_t)
        & (r_norm <= pv_hard_reject)
        & (v_norm <= pv_hard_reject)
        & ok
        & ~perm_failed
    )
    print(f"  GD done: {int(np.sum(converged))}/{B} converged")

    yf_out, cf_out = _reconstruct(params)
    return np.asarray(yf_out), np.asarray(cf_out), converged


# ─────────────────────────────────────────────────────────────────────────────
# Chunk I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _create_chunk_buffers(chunk_size, n_points):
    return {
        "states":           np.empty((chunk_size, n_points, STATE_DIM),   np.float32),
        "costates":         np.empty((chunk_size, n_points, COSTATE_DIM), np.float32),
        "initial_states":   np.empty((chunk_size, STATE_DIM),             np.float32),
        "final_states":     np.empty((chunk_size, STATE_DIM),             np.float32),
        "sampled_costates": np.empty((chunk_size, COSTATE_DIM),           np.float32),
        "size": 0,
    }


def _flush_chunk(chunk_dir, buf, chunk_idx):
    n = buf["size"]
    if n == 0:
        return chunk_idx
    path = chunk_dir / f"chunk_{chunk_idx:06d}.npz"
    np.savez_compressed(
        path,
        states           = buf["states"][:n],
        costates         = buf["costates"][:n],
        initial_states   = buf["initial_states"][:n],
        final_states     = buf["final_states"][:n],
        sampled_costates = buf["sampled_costates"][:n],
    )
    print(f"  saved chunk {chunk_idx} ({n} samples) → {path}")
    buf["size"] = 0
    return chunk_idx + 1


def _write_to_chunk(buf, states_np, costates_np):
    wi = buf["size"]
    buf["states"][wi]           = states_np           # (T, 7) forward-time order
    buf["costates"][wi]         = costates_np         # (T, 7)
    buf["initial_states"][wi]   = states_np[0]        # t=0
    buf["final_states"][wi]     = states_np[-1]       # t=t_f
    buf["sampled_costates"][wi] = costates_np[-1]     # costate at t=t_f
    buf["size"] += 1


# ─────────────────────────────────────────────────────────────────────────────
# Main generation loop
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(
    num_samples,
    num_points,
    seed,
    output_path,
    box_low,
    box_high,
    min_radius,
    batch_size,
    rtol,
    atol,
    log_every,
    eps,
    use_stm          = True,
    gd_max_iter      = GD_MAX_ITER,
    gd_lr            = GD_LR,
    gd_fd_step       = GD_FD_STEP,
    gd_noise_std     = GD_NOISE_STD,
    gd_max_grad_norm = GD_MAX_GRAD_NORM,
    pv_hard_reject   = PV_HARD_REJECT,
    pv_alpha         = PV_ALPHA,
):
    _, jdy, jjac, norm = make_dynamics_stm(eps=eps)
    mu  = norm["mu"]
    rng = np.random.default_rng(seed)

    traj_integrator = make_traj_integrator(jdy, norm["t_f"], num_points, rtol, atol)
    stm_integrator  = make_stm_integrator(jdy, jjac, norm["t_f"], rtol, atol) if use_stm else None
    fast_integrator = make_fast_integrator(jdy, norm["t_f"], rtol, atol) if not use_stm else None

    # ── Output directory and metadata ─────────────────────────────────────────
    chunk_dir = output_path.parent / output_path.stem
    chunk_dir.mkdir(parents=True, exist_ok=True)
    time_grid = np.linspace(0.0, float(norm["t_f"]), num_points, dtype=np.float32)
    np.savez_compressed(
        chunk_dir / "metadata.npz",
        num_samples          = np.int64(num_samples),
        num_points           = np.int64(num_points),
        chunk_size           = np.int64(CHUNK_SAVE_SIZE),
        time_grid            = time_grid,
        t_final              = np.float32(norm["t_f"]),
        au                   = np.float32(norm["AU"]),
        tu                   = np.float32(norm["TU"]),
        state_box_low        = np.float32(box_low),
        state_box_high       = np.float32(box_high),
        min_radius           = np.float32(min_radius),
        seed                 = np.int64(seed),
        sampling_method      = "stm_constrained" if use_stm else "fd_constrained",
        mass_final_low       = np.float32(MASS_FINAL_LOW),
        mass_final_high      = np.float32(MASS_FINAL_HIGH),
        eps                  = np.float32(eps),
        lam_thresh_low       = np.float32(LAM_THRESH_LOW),
        lam_thresh_high      = np.float32(LAM_THRESH_HIGH),
        pv_thresh_low        = np.float32(PV_THRESH_LOW),
        pv_thresh_high       = np.float32(PV_THRESH_HIGH),
        pv_hard_reject       = np.float32(pv_hard_reject),
        pv_alpha             = np.float32(pv_alpha),
        gd_max_iter          = np.int64(gd_max_iter),
        gd_lr                = np.float32(gd_lr),
        gd_fd_step           = np.float32(gd_fd_step),
        gd_noise_std         = np.float32(gd_noise_std),
        gd_max_grad_norm     = np.float32(gd_max_grad_norm),
        use_stm              = np.bool_(use_stm),
    )

    buf       = _create_chunk_buffers(CHUNK_SAVE_SIZE, num_points)
    chunk_idx = 0
    accepted  = 0
    skipped   = 0
    gd_count  = 0

    while accepted < num_samples:
        # ── Sample terminal states; keep re-sampling until all solve ──────────
        while True:
            yf_batch = sample_final_state_batch(rng, batch_size, box_low, box_high, min_radius, mu)
            cf_batch = sample_final_costate_batch(rng, batch_size)
            try:
                s_b, c_b, ok_b = traj_integrator(yf_batch, cf_batch)
            except Exception as exc:
                if "maximum number of solver steps" not in str(exc).lower():
                    raise
                print("  traj solve hit step limit; resampling")
                continue
            ok_np = np.asarray(jax.device_get(ok_b), dtype=bool)
            if np.all(ok_np):
                break
            print(f"  {int(np.sum(~ok_np))}/{batch_size} traj solves failed; resampling")

        states_np   = np.asarray(jax.device_get(s_b),      dtype=np.float32)
        costates_np = np.asarray(jax.device_get(c_b),      dtype=np.float32)
        yf_np       = np.asarray(jax.device_get(yf_batch), dtype=np.float32)
        cf_np_arr   = np.asarray(jax.device_get(cf_batch), dtype=np.float32)

        # ── Per-sample thresholds ─────────────────────────────────────────────
        lam_thresh = rng.uniform(LAM_THRESH_LOW,  LAM_THRESH_HIGH, batch_size).astype(np.float64)
        pv_thresh  = rng.uniform(PV_THRESH_LOW,   PV_THRESH_HIGH,  batch_size).astype(np.float64)

        # Diagnostics at t=0 from the initial integration
        lam_m_now = costates_np[:, 0, 6].astype(np.float64)   # costate[6] at t=0 (index 0 = t=0)
        r_now     = np.linalg.norm(states_np[:, 0, :3],  axis=-1)
        v_now     = np.linalg.norm(states_np[:, 0, 3:6], axis=-1)
        needs_gd  = (
            (lam_m_now > lam_thresh)
            | (r_now    > pv_hard_reject)
            | (v_now    > pv_hard_reject)
        )

        # Working copies updated for GD-derived samples
        final_states   = states_np.copy()
        final_costates = costates_np.copy()
        skip_mask      = np.zeros(batch_size, dtype=bool)

        # ── Run GD for samples that need it ───────────────────────────────────
        if np.any(needs_gd):
            gd_idx = np.where(needs_gd)[0]
            gd_count += len(gd_idx)
            print(
                f"[batch] {len(gd_idx)}/{batch_size} need GD  "
                f"(accepted so far: {accepted})"
            )
            yf_gd, cf_gd, converged_gd = run_gradient_descent(
                stm_integrator  = stm_integrator,
                fast_integrator = fast_integrator,
                yf_np           = yf_np[gd_idx],
                cf_np           = cf_np_arr[gd_idx],
                lam_thresholds  = lam_thresh[gd_idx],
                pv_thresholds   = pv_thresh[gd_idx],
                rng             = rng,
                use_stm         = use_stm,
                max_iter        = gd_max_iter,
                lr              = gd_lr,
                fd_h            = gd_fd_step,
                noise_std       = gd_noise_std,
                max_grad_norm   = gd_max_grad_norm,
                pv_hard_reject  = pv_hard_reject,
                pv_alpha        = pv_alpha,
            )

            # Re-integrate converged GD samples at full resolution for the dataset
            conv_local = np.where(converged_gd)[0]
            if len(conv_local) > 0:
                s_re, c_re, ok_re = traj_integrator(
                    jnp.asarray(yf_gd[conv_local]),
                    jnp.asarray(cf_gd[conv_local]),
                )
                s_re_np  = np.asarray(jax.device_get(s_re),  dtype=np.float32)
                c_re_np  = np.asarray(jax.device_get(c_re),  dtype=np.float32)
                ok_re_np = np.asarray(jax.device_get(ok_re), dtype=bool)
                for k, local_idx in enumerate(conv_local):
                    global_idx = gd_idx[local_idx]
                    if ok_re_np[k]:
                        final_states[global_idx]   = s_re_np[k]
                        final_costates[global_idx] = c_re_np[k]
                    else:
                        skip_mask[global_idx] = True
                        skipped += 1

            for local_idx in np.where(~converged_gd)[0]:
                skip_mask[gd_idx[local_idx]] = True
                skipped += 1

        # ── Write accepted samples ────────────────────────────────────────────
        for i in range(batch_size):
            if accepted >= num_samples:
                break
            if skip_mask[i]:
                continue
            # Final hard-reject gate (safety check, should already be satisfied)
            if (np.linalg.norm(final_states[i, 0, :3])  > pv_hard_reject or
                    np.linalg.norm(final_states[i, 0, 3:6]) > pv_hard_reject):
                skipped += 1
                continue
            _write_to_chunk(buf, final_states[i], final_costates[i])
            accepted += 1
            if buf["size"] == CHUNK_SAVE_SIZE:
                chunk_idx = _flush_chunk(chunk_dir, buf, chunk_idx)

        if log_every > 0:
            total = accepted + skipped
            pct = 100.0 * gd_count / total if total else 0.0
            if accepted > 0 and (accepted % log_every == 0 or accepted >= num_samples):
                print(
                    f"  accepted {min(accepted, num_samples)}/{num_samples}  "
                    f"GD: {gd_count}  skipped: {skipped}  ({pct:.1f}% needed GD)"
                )

    _flush_chunk(chunk_dir, buf, chunk_idx)
    total = accepted + skipped
    print(
        f"done — {accepted} samples saved to {chunk_dir}  "
        f"(GD: {gd_count}, skipped: {skipped}, "
        f"{100.0 * gd_count / total if total else 0.0:.1f}% needed GD)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate Earth-Mars min-fuel trajectory dataset with per-sample "
            "lambda_m and pos/vel constraints enforced by batch gradient descent.\n"
            f"  lambda_m threshold: Uniform[{LAM_THRESH_LOW}, {LAM_THRESH_HIGH}]\n"
            f"  pos/vel soft threshold: Uniform[{PV_THRESH_LOW}, {PV_THRESH_HIGH}]\n"
            f"  pos/vel hard gate: {PV_HARD_REJECT}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Dataset
    parser.add_argument("--num-samples",  type=int,   default=2**20)
    parser.add_argument("--num-points",   type=int,   default=32)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--output",       type=Path,
                        default=Path(__file__).resolve().parent
                                / "earth_mars_minfuel_constrained_32pts.npz")

    # State space
    parser.add_argument("--box-low",      type=float, default=-3.0)
    parser.add_argument("--box-high",     type=float, default=3.0)
    parser.add_argument("--min-radius",   type=float, default=1e-1)

    # Integration
    parser.add_argument("--batch-size",   type=int,   default=64)
    parser.add_argument("--rtol",         type=float, default=1e-7)
    parser.add_argument("--atol",         type=float, default=1e-9)
    parser.add_argument("--log-every",    type=int,   default=1)
    parser.add_argument("--eps",          type=float, default=1e-4)

    # Gradient method
    parser.add_argument("--no-stm",       action="store_true", default=False,
                        help="Use finite differences instead of STM for gradients "
                             "(default: use STM)")

    # GD
    parser.add_argument("--gd-max-iter",      type=int,   default=GD_MAX_ITER)
    parser.add_argument("--gd-lr",            type=float, default=GD_LR)
    parser.add_argument("--gd-fd-step",       type=float, default=GD_FD_STEP,
                        help="Finite-difference step (ignored when using STM)")
    parser.add_argument("--gd-noise-std",     type=float, default=GD_NOISE_STD)
    parser.add_argument("--gd-max-grad-norm", type=float, default=GD_MAX_GRAD_NORM)

    # Penalty
    parser.add_argument("--pv-hard-reject",   type=float, default=PV_HARD_REJECT,
                        help="Hard acceptance ceiling for ||r(t=0)|| and ||v(t=0)||")
    parser.add_argument("--pv-alpha",         type=float, default=PV_ALPHA,
                        help="Penalty coefficient for pos/vel terms (lam_m has weight 1.0)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_points < 2:
        raise ValueError("--num-points must be >= 2")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.box_low >= args.box_high:
        raise ValueError("--box-low must be < --box-high")

    generate_dataset(
        num_samples      = args.num_samples,
        num_points       = args.num_points,
        seed             = args.seed,
        output_path      = args.output,
        box_low          = args.box_low,
        box_high         = args.box_high,
        min_radius       = args.min_radius,
        batch_size       = args.batch_size,
        rtol             = args.rtol,
        atol             = args.atol,
        log_every        = args.log_every,
        eps              = args.eps,
        use_stm          = not args.no_stm,
        gd_max_iter      = args.gd_max_iter,
        gd_lr            = args.gd_lr,
        gd_fd_step       = args.gd_fd_step,
        gd_noise_std     = args.gd_noise_std,
        gd_max_grad_norm = args.gd_max_grad_norm,
        pv_hard_reject   = args.pv_hard_reject,
        pv_alpha         = args.pv_alpha,
    )


if __name__ == "__main__":
    main()
