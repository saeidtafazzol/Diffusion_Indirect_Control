"""plot_3d_trajectories.py
==========================
Generate 3-D trajectory EPS figures for selected time-shift cases.
One figure per shift; only the spacecraft path + celestial body markers
are drawn (no state/costate panels).

Customisation blocks are labelled  ── STYLE ──  for easy editing.
"""
import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(1, str(_root / "experiments"))
del _root


import types
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from core import build_normalization
from casadi_bvp_refine import BVPRefiner, make_bounds
from compare_convergence import load_checkpoint, build_jit_apply, run_diffusion_final


# ── STYLE ─────────────────────────────────────────────────────────────────────
# Figure geometry
FIG_SIZE   = (5.5, 4.5)       # inches  (width, height)
DPI_SCREEN = 150               # for rasterised preview; EPS ignores this

# Trajectory line
TRAJ_COLOR     = "#1f77b4"     # blue
TRAJ_LW        = 1.6           # line width
TRAJ_LS        = "-"           # line style

# Celestial body markers
SUN_COLOR      = "#FDB813"     # gold
SUN_MARKER     = "*"
SUN_SIZE       = 200

EARTH_COLOR    = "#2ca02c"     # green
EARTH_MARKER   = "o"
EARTH_SIZE     = 80

MARS_COLOR     = "#d62728"     # red
MARS_MARKER    = "o"
MARS_SIZE      = 80

# Axis limits  (AU)
XLIM = (-2.0, 2.0)
YLIM = (-2.0, 2.0)
ZLIM = (-0.2, 0.2)

# 3-D view angle
ELEV = 20        # degrees above the ecliptic plane
AZIM = 45        # azimuth

# Labels
XLABEL = r"$x$ [AU]"
YLABEL = r"$y$ [AU]"
ZLABEL = r"$z$ [AU]"
LABEL_FONTSIZE = 8
TICK_FONTSIZE  = 7
LEGEND_FONTSIZE = 8

# Grid  (True / False)
SHOW_GRID = True
GRID_ALPHA = 0.25

# Output
OUTPUT_DIR = Path("trajectory_eps")
# ─────────────────────────────────────────────────────────────────────────────


# ── Keplerian propagation ─────────────────────────────────────────────────────

def _kepler_rhs(t, y, mu):
    r, v = y[:3], y[3:6]
    return np.concatenate([v, -mu * r / np.linalg.norm(r) ** 3])


def propagate_kepler(r, v, mu, dt):
    if dt == 0.0:
        return np.array(r, np.float64), np.array(v, np.float64)
    y0  = np.concatenate([np.array(r, np.float64), np.array(v, np.float64)])
    sol = solve_ivp(_kepler_rhs, (0.0, dt), y0, args=(mu,),
                    method="DOP853", rtol=1e-12, atol=1e-13)
    return sol.y[:3, -1], sol.y[3:6, -1]


def shifted_bcs(norm, shift_days):
    shift_tu = shift_days * 24.0 * 3600.0 / norm["TU"]
    mu       = float(norm["mu"])
    r_i, v_i = propagate_kepler(norm["r_i"], norm["v_i"], mu, shift_tu)
    r_f, v_f = propagate_kepler(norm["r_f"], norm["v_f"], mu, shift_tu)
    m0       = float(norm["m0"])
    return (
        np.concatenate([r_i, v_i, [m0]]).astype(np.float32),
        np.concatenate([r_f, v_f, [m0]]).astype(np.float32),
    )


# ── CasADi one-shot solve ─────────────────────────────────────────────────────

def solve_bvp_oneshot(refiner, z_init, initial_state, final_state, max_iter=500):
    T      = refiner.n_points
    z_flat = z_init.astype(np.float64).reshape(-1)
    lbw, ubw = make_bounds(initial_state, final_state, T)
    solver = refiner._make_solver(max_iter=max_iter, cb=None)
    sol    = solver(x0=z_flat, lbx=lbw, ubx=ubw)
    z_opt  = sol["x"].full().reshape(T, 14).astype(np.float64)
    status = solver.stats()["return_status"]
    res    = float(np.max(refiner.continuity_residuals(z_opt, n_rk4_eval=8)))
    return z_opt, status, res


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_3d_trajectory(z_opt, initial_state, final_state, shift_days, out_path):
    """Render and save a single 3-D trajectory EPS figure.

    Parameters
    ----------
    z_opt         : (T, 14) optimised augmented trajectory
    initial_state : (7,)  Earth departure state [r, v, m]
    final_state   : (7,)  Mars arrival  state [r, v, m]
    shift_days    : scalar, used only for the figure title
    out_path      : Path  – destination .eps file
    """
    r     = z_opt[:, 0:3]
    r_dep = initial_state[:3]
    r_arr = final_state[:3]

    fig = plt.figure(figsize=FIG_SIZE)
    ax  = fig.add_subplot(111, projection="3d")

    # ── Trajectory ────────────────────────────────────────────────────────────
    ax.plot(
        r[:, 0], r[:, 1], r[:, 2],
        color=TRAJ_COLOR, lw=TRAJ_LW, ls=TRAJ_LS,
        label="Spacecraft",
        zorder=3,
    )

    # ── Departure / arrival markers on the trajectory ─────────────────────────
    ax.scatter(*r[0],  color=EARTH_COLOR, s=EARTH_SIZE * 0.6,
               marker="^", zorder=5)
    ax.scatter(*r[-1], color=MARS_COLOR,  s=MARS_SIZE  * 0.6,
               marker="v", zorder=5)

    # ── Celestial bodies ──────────────────────────────────────────────────────
    ax.scatter(0, 0, 0,
               color=SUN_COLOR, s=SUN_SIZE, marker=SUN_MARKER,
               label="Sun", zorder=6, depthshade=False)
    ax.scatter(*r_dep,
               color=EARTH_COLOR, s=EARTH_SIZE, marker=EARTH_MARKER,
               label="Earth (departure)", zorder=6, depthshade=False)
    ax.scatter(*r_arr,
               color=MARS_COLOR, s=MARS_SIZE, marker=MARS_MARKER,
               label="Mars (arrival)", zorder=6, depthshade=False)

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_zlim(*ZLIM)
    ax.set_xlabel(XLABEL, fontsize=LABEL_FONTSIZE, labelpad=4)
    ax.set_ylabel(YLABEL, fontsize=LABEL_FONTSIZE, labelpad=4)
    ax.set_zlabel(ZLABEL, fontsize=LABEL_FONTSIZE, labelpad=2)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.view_init(elev=ELEV, azim=AZIM)

    if SHOW_GRID:
        ax.grid(True, alpha=GRID_ALPHA)

    # ── Title & legend ────────────────────────────────────────────────────────
    sign = "+" if shift_days >= 0 else ""
    ax.set_title(
        rf"$\Delta t = {sign}{int(shift_days)}$ days",
        fontsize=LABEL_FONTSIZE + 1, pad=6,
    )
    ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper left",
              framealpha=0.7, edgecolor="none")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), format="eps", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── STYLE: which shifts to plot ───────────────────────────────────────────
    SHIFTS_DAYS       = [-300, +100, +500]
    DIFFUSION_SEED    = 0
    NUM_DIFF_STEPS    = 30
    CASADI_MAX_ITER   = 500
    CHECKPOINT        = Path("checkpoints/indiff_ctrl_latest.msgpack")
    # ─────────────────────────────────────────────────────────────────────────

    norm = build_normalization()

    ckpt_args = types.SimpleNamespace(
        no_integration_signals=False,
        embd_dim=512, num_layers=12, num_heads=4,
        eps=1e-4, inference_rtol=1e-7, inference_atol=1e-9,
    )
    params, policy, time_grid, _ = load_checkpoint(CHECKPOINT, ckpt_args)
    _apply   = build_jit_apply(policy)
    n_points = len(time_grid)
    refiner  = BVPRefiner(n_points=n_points, n_rk4_steps=8, ipopt_verbosity=0)

    for shift in SHIFTS_DAYS:
        print(f"\n── shift = {shift:+d} days ──────────────────────────────")
        initial_state, final_state = shifted_bcs(norm, shift)
        sample = {"initial_state": initial_state, "final_state": final_state}

        z_diff = run_diffusion_final(
            policy, params, sample, _apply,
            num_steps=NUM_DIFF_STEPS, rng_seed=DIFFUSION_SEED,
        )

        z_opt, status, res = solve_bvp_oneshot(
            refiner, z_diff,
            initial_state.astype(np.float64),
            final_state.astype(np.float64),
            max_iter=CASADI_MAX_ITER,
        )
        print(f"  {status}   max_res={res:.3e}")

        out_path = OUTPUT_DIR / f"trajectory_{shift:+d}d.eps"
        plot_3d_trajectory(z_opt, initial_state, final_state, shift, out_path)


if __name__ == "__main__":
    main()
