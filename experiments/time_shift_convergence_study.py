"""time_shift_convergence_study.py
=================================
Convergence study across shifted Earth–Mars launch windows (Diffusion only).

Earth (r_i, v_i) and Mars (r_f, v_f) are both propagated backward/forward in
time by the same Δt under pure two-body (Keplerian) gravity — no thrust.
This slides the whole mission's launch/arrival epoch by Δt while the transfer
duration t_f (fixed by the trained network / BVPRefiner) stays the same.

For each shift:
  Method — Diffusion (N steps, seed s) → CasADi IPOPT refinement
  repeated --num-trials times (different seeds) to check convergence
  robustness of the diffusion model's initial guess at that boundary
  condition.

"converged" = IPOPT Solve_Succeeded  AND  max_cont_residual < --res-tol

At the end, a table reports, per shift, how many of the trials converged.

Usage
-----
python time_shift_convergence_study.py \\
    --checkpoint checkpoints/indiff_ctrl_latest.msgpack \\
    [--shifts-days -50 -40 -30 -20 -10 10 20 30 40 50] \\
    [--num-trials 3] \\
    [--num-diffusion-steps 30] \\
    [--casadi-max-iter 500] \\
    [--res-tol 1e-3]
"""
import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(1, str(_root / "experiments"))
del _root


import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from core import build_normalization
from casadi_bvp_refine import BVPRefiner, switching_function
from compare_convergence import (
    load_checkpoint, build_jit_apply, run_diffusion_final, solve_bvp,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pure two-body (Keplerian) propagation — used only to shift Earth/Mars in time
# ─────────────────────────────────────────────────────────────────────────────

def _kepler_rhs(t, y, mu):
    r = y[:3]
    v = y[3:6]
    a = -mu * r / np.linalg.norm(r) ** 3
    return np.concatenate([v, a])


def propagate_kepler(r: np.ndarray, v: np.ndarray, mu: float, dt: float):
    """Propagate (r, v) under pure two-body gravity by dt (dt<0 → backward)."""
    if dt == 0.0:
        return np.asarray(r, dtype=np.float64).copy(), np.asarray(v, dtype=np.float64).copy()
    y0 = np.concatenate([np.asarray(r, dtype=np.float64), np.asarray(v, dtype=np.float64)])
    sol = solve_ivp(_kepler_rhs, (0.0, dt), y0, args=(mu,),
                     method="DOP853", rtol=1e-12, atol=1e-13)
    yf = sol.y[:, -1]
    return yf[:3], yf[3:]


def shifted_bcs(norm: dict, shift_days: float):
    """Return (initial_state, final_state) with Earth/Mars shifted by shift_days."""
    shift_tu = shift_days * 24.0 * 3600.0 / norm["TU"]
    mu = float(norm["mu"])
    r_i, v_i = propagate_kepler(norm["r_i"], norm["v_i"], mu, shift_tu)
    r_f, v_f = propagate_kepler(norm["r_f"], norm["v_f"], mu, shift_tu)
    m0 = float(norm["m0"])
    initial_state = np.concatenate([r_i, v_i, [m0]]).astype(np.float32)
    final_state   = np.concatenate([r_f, v_f, [m0]]).astype(np.float32)
    return initial_state, final_state


# ─────────────────────────────────────────────────────────────────────────────
# Per-shift plot: overlay all trials' trajectories
# ─────────────────────────────────────────────────────────────────────────────

_TRIAL_COLORS = ["limegreen", "dodgerblue", "darkorange", "mediumorchid", "gold"]


def _plot_shift(
    out_dir: Path,
    shift_days: float,
    z_trials: list,            # list of (T, 14) arrays, one per trial
    conv_trials: list,         # list of bool
    res_trials: list,          # list of float
    initial_state: np.ndarray,
    final_state: np.ndarray,
    time_grid: np.ndarray,
    norm: dict,
):
    t = np.asarray(time_grid)
    n_trials = len(z_trials)

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, wspace=0.38, hspace=0.50)

    # ── 3D trajectory ─────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[:, 0], projection="3d")
    ax3.scatter(0, 0, 0, color="gold", s=120, marker="*", zorder=5, label="Sun")
    ax3.scatter(*initial_state[:3], color="dodgerblue", s=60, zorder=5, label="Earth (shifted)")
    ax3.scatter(*final_state[:3],   color="tomato",     s=60, zorder=5, label="Mars (shifted)")
    for i, (z, conv) in enumerate(zip(z_trials, conv_trials)):
        c = _TRIAL_COLORS[i % len(_TRIAL_COLORS)]
        ls = "o-" if conv else "x--"
        ax3.plot(z[:, 0], z[:, 1], z[:, 2], ls, ms=2.5, lw=1.5, color=c,
                  alpha=0.9, label=f"trial {i} ({'OK' if conv else 'fail'})")
    ax3.set_xlim(-2, 2); ax3.set_ylim(-2, 2)
    ax3.set_xlabel("x [AU]", fontsize=7); ax3.set_ylabel("y [AU]", fontsize=7)
    ax3.set_zlabel("z [AU]", fontsize=7); ax3.tick_params(labelsize=6)
    ax3.legend(fontsize=6, loc="upper left")
    ax3.set_title("3D Trajectory", fontsize=9)

    # ── r(t) ──────────────────────────────────────────────────────────────────
    ax_r = fig.add_subplot(gs[0, 1])
    xyz_colors = ["tab:red", "tab:green", "tab:blue"]
    for i, z in enumerate(z_trials):
        c = _TRIAL_COLORS[i % len(_TRIAL_COLORS)]
        for k in range(3):
            ax_r.plot(t, z[:, k], "-", color=c, lw=1.2, alpha=0.85)
    ax_r.set_title("Position r(t)  (R/G/B = x/y/z per trial color)", fontsize=8)
    ax_r.set_xlabel("time [TU/2π]", fontsize=7); ax_r.set_ylabel("AU", fontsize=7)
    ax_r.tick_params(labelsize=6); ax_r.grid(True, alpha=0.25)

    # ── mass + thrust ─────────────────────────────────────────────────────────
    ax_m  = fig.add_subplot(gs[0, 2])
    ax_th = ax_m.twinx()
    for i, z in enumerate(z_trials):
        c = _TRIAL_COLORS[i % len(_TRIAL_COLORS)]
        delta, _, _ = switching_function(z, norm, eps=1e-4)
        ax_m.plot(t, z[:, 6], "-", color=c, lw=1.4)
        ax_th.plot(t, delta, "--", color=c, lw=1.0, alpha=0.7)
    ax_m.set_ylim(0, 1.05); ax_th.set_ylim(0, 1.5)
    ax_m.set_title("Mass (solid) & Thrust δ (dashed)", fontsize=9)
    ax_m.set_xlabel("time [TU/2π]", fontsize=7)
    ax_m.set_ylabel("mass", fontsize=7)
    ax_th.set_ylabel("thrust δ", fontsize=7)
    ax_m.tick_params(labelsize=6); ax_m.grid(True, alpha=0.25)

    # ── λ_m(t) ────────────────────────────────────────────────────────────────
    ax_lm = fig.add_subplot(gs[1, 1])
    ax_lm.axhline(0, color="gray", lw=0.7, ls="--")
    for i, z in enumerate(z_trials):
        c = _TRIAL_COLORS[i % len(_TRIAL_COLORS)]
        ax_lm.plot(t, z[:, 13], "-", color=c, lw=1.4,
                   label=f"trial {i}  λ_m(tf)={z[-1, 13]:.4f}")
    ax_lm.set_title("Costate λ_m(t)", fontsize=9)
    ax_lm.set_xlabel("time [TU/2π]", fontsize=7)
    ax_lm.set_ylim(-0.2, 1.2); ax_lm.tick_params(labelsize=6)
    ax_lm.grid(True, alpha=0.25); ax_lm.legend(fontsize=6)

    # ── continuity residuals ──────────────────────────────────────────────────
    ax_res = fig.add_subplot(gs[1, 2])
    labels = [f"trial {i}" for i in range(n_trials)]
    colors = [_TRIAL_COLORS[i % len(_TRIAL_COLORS)] for i in range(n_trials)]
    ax_res.bar(labels, res_trials, color=colors, alpha=0.85)
    ax_res.axhline(1e-3, color="gray", lw=0.8, ls="--", label="res-tol 1e-3")
    ax_res.set_yscale("log")
    ax_res.set_ylabel("max ‖F(Z_k)−Z_{k+1}‖", fontsize=7)
    ax_res.set_title("Max continuity residual per trial", fontsize=9)
    ax_res.tick_params(labelsize=7); ax_res.grid(True, alpha=0.25, axis="y")
    ax_res.legend(fontsize=6)

    n_ok = sum(conv_trials)
    fig.suptitle(
        f"Shift = {shift_days:+.1f} days   |   converged trials: {n_ok}/{n_trials}",
        fontsize=11, fontweight="bold",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "trajectory.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def _plot_summary(out_dir: Path, shifts: list, n_ok: list, n_trials: int):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["seagreen" if k == n_trials else ("goldenrod" if k > 0 else "firebrick")
              for k in n_ok]
    ax.bar([f"{s:+.0f}d" for s in shifts], n_ok, color=colors)
    ax.axhline(n_trials, color="gray", lw=0.8, ls="--")
    ax.set_ylabel(f"converged trials (out of {n_trials})")
    ax.set_xlabel("time shift")
    ax.set_title("Diffusion+CasADi convergence vs. Earth/Mars time shift")
    ax.set_ylim(0, n_trials + 0.5)
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "summary.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path,
                   default=Path("checkpoints/indiff_ctrl_latest.msgpack"))
    p.add_argument("--shifts-days", type=float, nargs="+",
                   default=[-50, -40, -30, -20, -10, 10, 20, 30, 40, 50],
                   help="Time shifts (days) applied to both Earth and Mars")
    p.add_argument("--num-trials",           type=int,   default=3,
                   help="Diffusion+CasADi repeats per shift (different seeds)")
    p.add_argument("--seed-offset",          type=int,   default=0,
                   help="Trial k of shift i uses seed = seed_offset + i*num_trials + k")
    p.add_argument("--num-diffusion-steps",  type=int,   default=30)
    p.add_argument("--casadi-max-iter",      type=int,   default=500)
    p.add_argument("--res-tol",              type=float, default=1e-3,
                   help="Max continuity residual threshold for 'converged'")
    p.add_argument("--output-dir", type=Path, default=Path("time_shift_convergence_study"),
                   help="Root folder; each shift gets a subfolder shift_XXX/")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip saving per-shift trajectory plots")
    return p.parse_args()


def main():
    args = parse_args()

    norm = build_normalization()
    print("Baseline (unshifted) Earth-Mars BCs:")
    print(f"  r_i={norm['r_i'].round(4)}  v_i={norm['v_i'].round(4)}")
    print(f"  r_f={norm['r_f'].round(4)}  v_f={norm['v_f'].round(4)}")

    # ── Load checkpoint & build JIT forward ───────────────────────────────────
    if not args.checkpoint.exists():
        sys.exit(f"Checkpoint not found: {args.checkpoint}")
    params, policy, time_grid, _norm_from_ckpt = load_checkpoint(args.checkpoint, args)
    _apply = build_jit_apply(policy)

    # ── Build BVP refiner (t_f fixed — only Earth/Mars r,v are shifted) ───────
    n_points = len(time_grid)
    refiner  = BVPRefiner(n_points=n_points, n_rk4_steps=8, ipopt_verbosity=0)
    print(f"\nBVPRefiner: n_points={n_points}  n_rk4_steps=8  t_f={norm['t_f']:.4f} (fixed)")

    shifts   = args.shifts_days
    n_shift  = len(shifts)
    n_trials = args.num_trials

    print(f"\n── Running {n_shift} time shifts × {n_trials} trials "
          f"(diffusion steps={args.num_diffusion_steps}, "
          f"casadi_max_iter={args.casadi_max_iter}) ──")
    print(f"{'shift[d]':>9}  {'trial':>5}  {'seed':>5}  {'status':>16}  "
          f"{'res':>10}  {'conv':>5}  {'time':>6}")
    print("-" * 70)

    table_rows = []   # (shift_days, [conv per trial], [res per trial])
    t_study = time.perf_counter()

    for si, shift in enumerate(shifts):
        initial_state, final_state = shifted_bcs(norm, shift)
        sample = {"initial_state": initial_state, "final_state": final_state}

        z_trials, conv_trials, res_trials = [], [], []
        for k in range(n_trials):
            seed = args.seed_offset + si * n_trials + k
            t0 = time.perf_counter()
            z_diff = run_diffusion_final(
                policy, params, sample, _apply,
                num_steps=args.num_diffusion_steps,
                rng_seed=seed,
            )
            z_opt, status, res = solve_bvp(
                refiner, z_diff,
                initial_state.astype(np.float64),
                final_state.astype(np.float64),
                max_iter=args.casadi_max_iter,
            )
            dt = time.perf_counter() - t0

            conv = ("Succeed" in status) and (res < args.res_tol)
            z_trials.append(z_opt)
            conv_trials.append(conv)
            res_trials.append(res)

            print(f"{shift:>+9.1f}  {k:>5}  {seed:>5}  {status:>16}  "
                  f"{res:>10.3e}  {'YES' if conv else 'no':>5}  {dt:>5.1f}s")

        table_rows.append((shift, conv_trials, res_trials))

        if not args.no_plots:
            _plot_shift(
                args.output_dir / f"shift_{si:02d}_{shift:+.0f}d",
                shift_days=shift,
                z_trials=z_trials, conv_trials=conv_trials, res_trials=res_trials,
                initial_state=initial_state, final_state=final_state,
                time_grid=time_grid, norm=norm,
            )

    t_total = time.perf_counter() - t_study

    # ── Summary table ─────────────────────────────────────────────────────────
    n_ok_list = [sum(c) for _, c, _ in table_rows]
    if not args.no_plots:
        _plot_summary(args.output_dir, shifts, n_ok_list, n_trials)

    print("\n" + "=" * 78)
    print(f"  Diffusion+CasADi convergence vs. Earth/Mars time shift  "
          f"({n_trials} trials/shift, res-tol={args.res_tol:.0e})")
    print("-" * 78)
    header = f"  {'shift [days]':>12}  {'converged':>11}  " + \
             "  ".join(f"t{k}" for k in range(n_trials)) + "   verdict"
    print(header)
    print("-" * 78)
    for shift, conv_trials, res_trials in table_rows:
        n_ok = sum(conv_trials)
        marks = "  ".join("OK" if c else "no" for c in conv_trials)
        if n_ok == n_trials:
            verdict = "SUCCESS"
        elif n_ok > 0:
            verdict = "PARTIAL"
        else:
            verdict = "FAILED"
        print(f"  {shift:>+12.1f}  {n_ok:>8}/{n_trials}  {marks:>{3*n_trials}}   {verdict}")
    print("=" * 78)
    n_full   = sum(1 for n in n_ok_list if n == n_trials)
    n_part   = sum(1 for n in n_ok_list if 0 < n < n_trials)
    n_none   = sum(1 for n in n_ok_list if n == 0)
    print(f"  SUCCESS (all {n_trials} trials converged) : {n_full}/{n_shift} shifts")
    print(f"  PARTIAL (some trials converged)          : {n_part}/{n_shift} shifts")
    print(f"  FAILED  (no trials converged)             : {n_none}/{n_shift} shifts")
    print(f"  Total study time: {t_total/60:.1f} min")


if __name__ == "__main__":
    main()
