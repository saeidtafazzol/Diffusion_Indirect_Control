"""CasADi multiple-shooting BVP refinement for diffusion model trajectories.

Multiple-shooting NLP
---------------------
Decision variables : Z_k ∈ R^14  for k = 0 … T-1
                     Z_k = [r(3), v(3), m(1), λ_r(3), λ_v(3), λ_m(1)]
                     indices 0–6 = state, 7–13 = costate

Continuity constraints (RK4, n_rk4_steps per interval):
    F(Z_k, Δt_k) = Z_{k+1},   k = 0 … T-2

Fixed boundary conditions (enforced via lbw = ubw):
    Z[0,   0:7 ] = initial_state   (r₀, v₀, m₀)
    Z[T-1, 0:6 ] = final_state     (r_f, v_f)    — terminal mass is FREE
    Z[T-1, 13  ] = 0               (final λ_m, Pontryagin transversality)

Objective:
    min  Σ_k ‖F(Z_k, Δt_k) − Z_{k+1}‖²   (dynamic consistency, RK4 defects)

Constraints:
    only boundary conditions, enforced via variable bounds (lbw = ubw):
        Z[0,   0:7 ] = initial_state   (r₀, v₀, m₀)
        Z[T-1, 0:6 ] = final_state     (r_f, v_f)  — terminal mass is FREE
        Z[T-1, 13  ] = 0               (final λ_m, Pontryagin transversality)

The dynamics are NOT hard equality constraints — they are the objective.
IPOPT sees a bound-constrained nonlinear least-squares problem; BCs are the
only constraints.  Given the diffusion model's prediction as x0, IPOPT drives
the trajectory to be dynamically consistent while anchoring the endpoints.
"""

import numpy as np
import casadi as ca

from core import make_dynamics, STATE_DIM, COSTATE_DIM, AUGMENTED_DIM

# ── Unique-name counter for CasADi callbacks ─────────────────────────────────
_CB_ID = 0


# ─────────────────────────────────────────────────────────────────────────────
# Iteration-capture callback
# ─────────────────────────────────────────────────────────────────────────────

class IterCapture(ca.Callback):
    """Records the primal iterate x at every IPOPT major iteration.

    CasADi callback convention
    --------------------------
    ``get_n_in()`` must return ``ca.nlpsol_n_out()`` (= 6).
    The inputs, in nlpsol output order, are:
        0 → x       (n_x × 1)
        1 → f       (scalar)
        2 → g       (n_g × 1)
        3 → lam_x   (n_x × 1)
        4 → lam_g   (n_g × 1)
        5 → lam_p   (n_p × 1)
    ``eval`` returns [0] to continue, [nonzero] to abort.
    """

    def __init__(self, n_x: int, n_g: int = 0):
        global _CB_ID
        _CB_ID += 1
        ca.Callback.__init__(self)
        self._n_x = n_x
        self._n_g = n_g   # 0 when there are no NLP constraints
        self.iterates: list = []
        self.construct(f"IterCapture_{_CB_ID}", {})

    def get_n_in(self):
        return ca.nlpsol_n_out()

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, i):
        name = ca.nlpsol_out(i)
        if name == "f":
            return ca.Sparsity.scalar()
        if name in ("x", "lam_x"):
            return ca.Sparsity.dense(self._n_x)
        if name in ("g", "lam_g"):
            # No NLP constraints in this formulation (dynamics are the objective)
            return ca.Sparsity(self._n_g, 1) if self._n_g > 0 else ca.Sparsity(0, 0)
        # lam_p: no NLP parameter in this formulation → always empty
        return ca.Sparsity(0, 0)

    def get_sparsity_out(self, i):
        return ca.Sparsity.scalar()

    def eval(self, arg):
        # arg[0] = x (current IPOPT iterate, primal variables)
        x = np.asarray(arg[0]).flatten()
        if x.size == self._n_x:
            self.iterates.append(x.copy())
        return [0.0]


# ─────────────────────────────────────────────────────────────────────────────
# NLP construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_nlp_problem(dy, t_f: float, n_points: int, n_rk4_steps: int):
    """Build the CasADi NLP symbolic problem dict.

    Objective  : min −m(t_f)  [maximize final mass = minimize fuel]
    Constraints: (T-1)×14 continuity equations from RK4 multiple shooting
    No NLP parameter; the initial guess is passed only as x0 at solve time.

    Returns
    -------
    prob      : CasADi NLP problem dict  {'f', 'x', 'g'}
    dts       : (T-1,) float64 — per-interval time steps
    time_grid : (T,)   float64
    """
    T = n_points
    N = T - 1
    time_grid = np.linspace(0.0, float(t_f), T)
    dts = np.diff(time_grid).astype(np.float64)

    # Dynamics: f(state[7], costate[7]) → Ż[14]
    f_dyn = ca.Function(
        "f_dyn_bvp",
        [dy.states, dy.costates],
        [dy.augmented_dot_sub],
    )

    # ── RK4 single step ───────────────────────────────────────────────────────
    def rk4(Z, h):
        k1 = f_dyn(Z[:7], Z[7:])
        Z2 = Z + 0.5 * h * k1
        k2 = f_dyn(Z2[:7], Z2[7:])
        Z3 = Z + 0.5 * h * k2
        k3 = f_dyn(Z3[:7], Z3[7:])
        Z4 = Z + h * k3
        k4 = f_dyn(Z4[:7], Z4[7:])
        return Z + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # ── Multiple RK4 steps per shooting interval ──────────────────────────────
    def integrate(Z0, dt):
        h = dt / n_rk4_steps
        Z = Z0
        for _ in range(n_rk4_steps):
            Z = rk4(Z, h)
        return Z

    # ── Decision variables: Z_0, Z_1, … Z_{T-1} each ∈ R^14 ─────────────────
    Z_sym = [ca.MX.sym(f"Z{k}", AUGMENTED_DIM) for k in range(T)]
    w = ca.vertcat(*Z_sym)  # (T × 14,)

    # ── Objective: squared RK4 defects (dynamic consistency) ─────────────────
    # F(Z_k, Δt_k) is the RK4 propagation of Z_k over one shooting interval.
    # The defect d_k = F(Z_k) − Z_{k+1} is zero iff the arc is dynamically
    # consistent.  Minimising Σ ‖d_k‖² drives all arcs to satisfy the ODE
    # while the BC variable bounds anchor the boundary conditions.
    J = sum(
        ca.sumsqr(integrate(Z_sym[k], dts[k]) - Z_sym[k + 1])
        for k in range(N)
    )

    # ── No equality constraints — BCs are enforced only via lbw = ubw bounds ──
    prob = {"f": J, "x": w}   # no g
    return prob, dts, time_grid


# ─────────────────────────────────────────────────────────────────────────────
# Bounds helper
# ─────────────────────────────────────────────────────────────────────────────

def make_bounds(
    initial_state: np.ndarray,     # (7,)  r₀, v₀, m₀
    final_state: np.ndarray,       # (7,)  r_f, v_f, m_f  — only r_f/v_f pinned
    n_points: int = 32,
    large: float = 1e9,
    fix_final_mass: bool = False,
) -> tuple:
    """Build (lbw, ubw, lbg, ubg) that encode the boundary conditions.

    Fixed via lbw = ubw (equality):
        Z[0,   0:7 ] = initial_state   (r₀, v₀, m₀ — full initial state)
        Z[T-1, 0:6 ] = final_state[0:6] (r_f, v_f  — terminal mass is free)
        Z[T-1, 13  ] = 0                (final λ_m, Pontryagin transversality)

    Parameters
    ----------
    fix_final_mass : if True, also pin Z[T-1, 6] = final_state[6].  Set to
                     True only when comparing against dataset ground truth.
    """
    T = n_points
    D = AUGMENTED_DIM

    lbw = np.full(T * D, -large, dtype=np.float64)
    ubw = np.full(T * D,  large, dtype=np.float64)

    # ── Pin initial state (all 7 components) ─────────────────────────────────
    lbw[0:STATE_DIM] = np.asarray(initial_state, dtype=np.float64)
    ubw[0:STATE_DIM] = np.asarray(initial_state, dtype=np.float64)

    # ── Pin final r_f and v_f (indices 0–5 of the last Z block) ──────────────
    start_last = (T - 1) * D
    lbw[start_last : start_last + 6] = np.asarray(final_state[:6], dtype=np.float64)
    ubw[start_last : start_last + 6] = np.asarray(final_state[:6], dtype=np.float64)

    if fix_final_mass:
        lbw[start_last + 6] = float(final_state[6])
        ubw[start_last + 6] = float(final_state[6])

    # ── Pin final λ_m = 0 (index 13 of the last Z block) ────────────────────
    lbw[start_last + D - 1] = 0.0
    ubw[start_last + D - 1] = 0.0

    return lbw, ubw


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def continuity_residuals(
    z: np.ndarray,       # (T, 14) trajectory
    dy,                  # TwoBodyCartesian with params set
    norm: dict,
    n_points: int = 32,
    n_rk4_eval: int = 16,
) -> np.ndarray:
    """Per-interval ‖F(Z_k, Δt_k) − Z_{k+1}‖₂ using numpy/CasADi evaluation.

    Returns (T-1,) float64 continuity residuals.
    """
    T = n_points
    t_f = float(norm["t_f"])
    dts = np.diff(np.linspace(0.0, t_f, T))

    f_np = ca.Function("f_diag", [dy.states, dy.costates], [dy.augmented_dot_sub])

    def _f(Z):
        return np.asarray(f_np(Z[:7], Z[7:])).flatten()

    def _rk4(Z, h):
        k1 = _f(Z)
        k2 = _f(Z + 0.5 * h * k1)
        k3 = _f(Z + 0.5 * h * k2)
        k4 = _f(Z + h * k3)
        return Z + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _integrate(Z0, dt):
        h = dt / n_rk4_eval
        Z = Z0.astype(np.float64).copy()
        for _ in range(n_rk4_eval):
            Z = _rk4(Z, h)
        return Z

    residuals = np.empty(T - 1, dtype=np.float64)
    for k in range(T - 1):
        Z_prop = _integrate(z[k], dts[k])
        residuals[k] = np.linalg.norm(Z_prop - z[k + 1].astype(np.float64))
    return residuals


def switching_function(z: np.ndarray, norm: dict, eps: float = 1e-4) -> tuple:
    """Compute optimal-thrust diagnostics from a (T, 14) trajectory.

    Returns
    -------
    delta : (T,)    smoothed thrust magnitude ∈ [0, 1]
    alpha : (T, 3)  thrust direction unit vector
    S     : (T,)    switching function  S = c·‖λ_v‖/m + λ_m − 1
    """
    c    = float(norm["c_norm"])
    mass = np.maximum(z[:, 6], 1e-12)
    lv   = z[:, 10:13]
    lm   = z[:, 13]
    lv_n = np.linalg.norm(lv, axis=1)
    S    = c * lv_n / mass + lm - 1.0
    delta = 0.5 * (1.0 + S / np.sqrt(S**2 + eps**2))
    alpha = -lv / (lv_n[:, None] + 1e-30)
    return delta, alpha, S


# ─────────────────────────────────────────────────────────────────────────────
# Main public class
# ─────────────────────────────────────────────────────────────────────────────

class BVPRefiner:
    """Multiple-shooting BVP refiner for 32-point discretized trajectories.

    Parameters
    ----------
    eps          : smoothing param for the optimal thrust (must match training)
    n_points     : number of discretization points (32 to match policy/dataset)
    n_rk4_steps  : RK4 sub-steps per shooting interval
    ipopt_verbosity : IPOPT print level (0 = silent)

    Example
    -------
    >>> refiner = BVPRefiner()
    >>> # One-shot full solve
    >>> z_opt, ws = refiner.solve(z_init_T14, initial_state, final_state)
    >>> # Iterative solve with per-iterate snapshots for animation
    >>> frames = refiner.solve_iterative(z_init_T14, initial_state, final_state)
    """

    def __init__(
        self,
        eps: float = 1e-4,
        n_points: int = 32,
        n_rk4_steps: int = 8,
        ipopt_verbosity: int = 0,
    ):
        self.n_points    = n_points
        self.n_rk4_steps = n_rk4_steps
        self._eps        = eps
        self._n_x        = n_points * AUGMENTED_DIM
        self._n_g        = 0   # no NLP constraints; BCs are variable bounds
        self._ipopt_verb = ipopt_verbosity

        self.dy, _, self.norm = make_dynamics(eps=eps)
        self._prob, self.dts, self.time_grid = _build_nlp_problem(
            self.dy,
            float(self.norm["t_f"]),
            n_points,
            n_rk4_steps,
        )
        # Build and cache the full-budget solver (no callback)
        self._full_solver = self._make_solver(max_iter=500, cb=None)

    # ── Internal solver factory ───────────────────────────────────────────────

    def _make_solver(self, max_iter: int, cb) -> ca.Function:
        opts = {
            "ipopt.print_level"           : self._ipopt_verb,
            "print_time"                  : 0,
            "ipopt.tol"                   : 1e-7,
            "ipopt.constr_viol_tol"       : 1e-7,
            "ipopt.max_iter"              : max_iter,
            "ipopt.warm_start_init_point" : "yes",
            "ipopt.mu_init"               : 0.1,
        }
        if cb is not None:
            opts["iteration_callback"]      = cb
            opts["iteration_callback_step"] = 1
        return ca.nlpsol("bvp_nlp", "ipopt", self._prob, opts)

    # ── Convenience: continuity residuals ─────────────────────────────────────

    def continuity_residuals(
        self,
        z: np.ndarray,
        n_rk4_eval: int = 16,
    ) -> np.ndarray:
        """Per-interval ‖F(Z_k) − Z_{k+1}‖₂.  Returns (T-1,) float64."""
        return continuity_residuals(
            z, self.dy, self.norm, self.n_points, n_rk4_eval
        )

    # ── One-shot full solve ───────────────────────────────────────────────────

    def solve(
        self,
        z_init: np.ndarray,          # (T, 14) or (T*14,) initial guess
        initial_state: np.ndarray,   # (7,)
        final_state: np.ndarray,     # (7,)  only r_f, v_f are pinned
        fix_final_mass: bool = False,
        verbose: bool = True,
        warm_start: dict = None,
    ) -> tuple:
        """Solve the BVP NLP in one call (up to 500 IPOPT iterations).

        Returns
        -------
        z_opt      : (T, 14) float64  — refined trajectory
        warm_start : dict {'lam_x', 'lam_g'} for subsequent warm starts
        """
        T  = self.n_points
        D  = AUGMENTED_DIM
        z_flat = np.asarray(z_init, dtype=np.float64).reshape(-1)
        lbw, ubw = make_bounds(
            initial_state, final_state, T,
            fix_final_mass=fix_final_mass,
        )

        kw = dict(x0=z_flat, lbx=lbw, ubx=ubw)
        if warm_start is not None:
            kw["lam_x0"] = warm_start["lam_x"]

        sol    = self._full_solver(**kw)
        z_opt  = sol["x"].full().reshape(T, D).astype(np.float64)

        if verbose:
            st = self._full_solver.stats()
            res_max = float(np.max(self.continuity_residuals(z_opt)))
            print(
                f"BVP solve: {st['return_status']}  "
                f"iter={st['iter_count']}  J={float(sol['f']):.3e}  "
                f"max_cont_residual={res_max:.3e}"
            )

        ws = {
            "lam_x": sol["lam_x"].full().flatten(),
        }
        return z_opt, ws

    # ── Iterative solve with per-iterate snapshots ────────────────────────────

    def solve_iterative(
        self,
        z_init: np.ndarray,          # (T, 14) initial guess
        initial_state: np.ndarray,   # (7,)
        final_state: np.ndarray,     # (7,)
        fix_final_mass: bool = False,
        max_iter: int = 150,
        stride: int = 1,
    ) -> list:
        """Solve BVP while capturing every IPOPT iterate for animation.

        Parameters
        ----------
        stride : record one snapshot per ``stride`` major iterations
                 (stride=1 captures every iterate; use larger values to thin
                 the frame list for fast / many-iteration solves)

        Returns
        -------
        list of (T, 14) float64 arrays:
            [z_init (frame 0), iterate_1, ..., z_opt (last frame)]
        The first element is the diffusion model's initial guess;
        the last is the NLP solution.
        """
        T = self.n_points
        D = AUGMENTED_DIM
        z_flat = np.asarray(z_init, dtype=np.float64).reshape(-1)
        lbw, ubw = make_bounds(
            initial_state, final_state, T,
            fix_final_mass=fix_final_mass,
        )

        cb = IterCapture(n_x=T * D)
        solver = self._make_solver(max_iter=max_iter, cb=cb)

        cb.iterates.clear()
        sol = solver(x0=z_flat, lbx=lbw, ubx=ubw)

        # Assemble frame list: initial guess + every stride-th iterate + final
        z_final = sol["x"].full().flatten()
        raw = [z_flat] + cb.iterates
        selected = [raw[0]] + raw[1::stride]
        if len(raw) > 1 and not np.allclose(selected[-1], z_final, atol=1e-10):
            selected.append(z_final)

        st = solver.stats()
        res_max = float(np.max(self.continuity_residuals(
            z_final.reshape(T, D)
        )))
        print(
            f"BVP iterative: {st['return_status']}  "
            f"iter={st['iter_count']}  frames={len(selected)}  "
            f"max_cont_residual={res_max:.3e}"
        )

        return [it.reshape(T, D).astype(np.float64) for it in selected]

    # ── Batch API (processes one sample at a time, returns list) ──────────────

    def solve_batch(
        self,
        z_inits: np.ndarray,          # (B, T, 14)
        initial_states: np.ndarray,   # (B, 7)
        final_states: np.ndarray,     # (B, 7)
        fix_final_mass: bool = False,
        verbose: bool = True,
    ) -> tuple:
        """Solve BVP independently for each sample in a batch.

        Returns
        -------
        z_opts     : (B, T, 14) float64
        statuses   : list of B solver status strings
        """
        B = len(z_inits)
        z_opts  = np.zeros_like(z_inits, dtype=np.float64)
        statuses = []
        for b in range(B):
            if verbose:
                print(f"  Sample {b + 1}/{B}")
            z_opt, _ = self.solve(
                z_inits[b], initial_states[b], final_states[b],
                fix_final_mass=fix_final_mass,
                verbose=verbose,
            )
            z_opts[b]  = z_opt
            st = self._full_solver.stats()
            statuses.append(st["return_status"])
        return z_opts, statuses
