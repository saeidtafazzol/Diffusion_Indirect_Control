"""Generate one MP4 per shift from shift_eps_study results.

Usage:
    python experiments/make_shift_videos.py --study-dir shift_eps_study \
        --out-dir shift_videos --res-tol 1e-8 --fps 20

Frame layout (18×9 figure):
  [3D orbit | r(t) | v(t) | mass+thrust ]
  [         | λr   | λv   | λm          ]
  [         | continuity residuals (wide) ]
"""

import argparse
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────
AUGDIM      = 14          # [r(3) v(3) m(1) λr(3) λv(3) λm(1)]
N_RK4_SUB  = 8           # sub-steps per interval for residual recomputation
_XYZ_COLORS = ["tab:red", "tab:green", "tab:blue"]
_XYZ_LABELS = ["x", "y", "z"]
_DIFF_COLOR  = "tab:green"
_IPOPT_COLOR = "tab:orange"
_FINAL_COLOR = "tab:orange"


# ── Physics ───────────────────────────────────────────────────────────────────
def _ode(norm):
    """Return f(Z) → Zdot for the augmented state [r,v,m,λr,λv,λm]."""
    mu = float(norm["mu"])
    c  = float(norm["c_norm"])
    tm = float(norm["t_max_norm"])
    eps = 1e-4

    def f(Z):
        r, v, m   = Z[0:3], Z[3:6], Z[6]
        lr, lv, lm = Z[7:10], Z[10:13], Z[13]
        rm  = np.linalg.norm(r)
        lvm = np.linalg.norm(lv)
        S   = c * lvm / m + lm - 1.0
        d   = 0.5 * (1.0 + S / np.sqrt(S**2 + eps))
        a   = -lv / (lvm + 1e-20)
        return np.concatenate([
            v,
            -mu * r / rm**3 + d * (tm / m) * a,
            [-d * tm / c],
            mu / rm**3 * lv - 3.0 * mu * (lv @ r) / rm**5 * r,
            -lr,
            [d * tm * (lv @ a) / m**2],
        ])
    return f


def _rk4(Z, h, f):
    k1 = f(Z);           k2 = f(Z + 0.5*h*k1)
    k3 = f(Z + 0.5*h*k2); k4 = f(Z + h*k3)
    return Z + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)


def _continuity_residuals(z, norm, n_sub=N_RK4_SUB):
    """Per-interval ‖F(Z_k)−Z_{k+1}‖₂.  Returns array of length T-1."""
    f   = _ode(norm)
    t_f = float(norm["t_f"])
    N   = len(z)
    dt  = t_f / (N - 1)
    h   = dt / n_sub
    res = np.empty(N - 1)
    for k in range(N - 1):
        s = z[k].astype(np.float64).copy()
        for _ in range(n_sub):
            s = _rk4(s, h, f)
        res[k] = np.linalg.norm(s - z[k + 1])
    return res


def _switching(z, norm):
    """Return (S, delta_bb, mass_bb).

    delta_bb : bang-bang thrust at each node (0 or 1)
    mass_bb  : mass reconstructed from bang-bang delta and initial node mass,
               giving perfectly flat coast segments and clean linear thrust arcs.
    """
    c  = float(norm["c_norm"])
    tm = float(norm["t_max_norm"])
    t_f = float(norm["t_f"])
    lv = z[:, 10:13]
    lm = z[:, 13]
    m  = z[:, 6]
    N  = len(z)
    S  = c * np.linalg.norm(lv, axis=1) / m + lm - 1.0
    delta_bb = (S > 0).astype(float)
    dt = t_f / (N - 1)           # interval length in TU
    mass_bb  = np.empty(N)
    mass_bb[0] = float(z[0, 6])
    for k in range(N - 1):
        mass_bb[k + 1] = mass_bb[k] - delta_bb[k] * tm / c * dt
    return S, delta_bb, mass_bb


# ── Global axis limits ────────────────────────────────────────────────────────
def _compute_global_limits(jsons, res_tol):
    """Scan all converged method-A z_opt to fix axis ranges across all frames."""
    r_all, v_all, lr_all, lv_all, lm_all = [], [], [], [], []
    res_all = []
    for jp in jsons:
        try:
            d  = json.load(open(jp))
            ma = d["method_A"]
            if not ma["ok"] or float(ma["res"]) >= res_tol:
                continue
            z = np.array(ma["z_opt"], dtype=np.float64)
            r_all.append(z[:, 0:3]); v_all.append(z[:, 3:6])
            lr_all.append(z[:, 7:10]); lv_all.append(z[:, 10:13])
            lm_all.append(z[:, 13])
        except Exception:
            continue

    def _lim(arrays, pad=0.05):
        a = np.concatenate(arrays)
        lo, hi = a.min(), a.max()
        m = (hi - lo) * pad
        return (lo - m, hi + m)

    glims = {
        "r":     _lim(r_all),
        "v":     _lim(v_all),
        "lam_r": _lim(lr_all),
        "lam_v": _lim(lv_all),
        "lam_m": _lim(lm_all),
        "res":   (1e-16, 1e1),
    }
    return glims


# ── Single frame renderer ─────────────────────────────────────────────────────
def _render_frame(z, t_day, ri, rf, norm, title, color, out_path, glims,
                  true_res=None):
    """Write one PNG frame."""
    z   = np.asarray(z, dtype=np.float64)
    N   = len(z)
    TU  = float(norm["TU"])

    # ── diagnostics ───────────────────────────────────────────────────────────
    S, delta_bb, mass_bb = _switching(z, norm)
    res_arr = _continuity_residuals(z, norm)

    # ── figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 9))
    gs  = gridspec.GridSpec(3, 4, figure=fig, wspace=0.44, hspace=0.60)
    ax3d = fig.add_subplot(gs[0:2, 0], projection="3d")
    ax_r  = fig.add_subplot(gs[0, 1])
    ax_v  = fig.add_subplot(gs[0, 2])
    ax_mt = fig.add_subplot(gs[0, 3])
    ax_lr = fig.add_subplot(gs[1, 1])
    ax_lv = fig.add_subplot(gs[1, 2])
    ax_lm = fig.add_subplot(gs[1, 3])
    ax_rs = fig.add_subplot(gs[2, 1:])
    ax_t2 = ax_mt.twinx()
    fig.suptitle(title, fontsize=8, fontweight="bold", y=0.999)

    # ── 3D orbit ──────────────────────────────────────────────────────────────
    lim3d = 2.0
    ax3d.set_xlim(-lim3d, lim3d); ax3d.set_ylim(-lim3d, lim3d); ax3d.set_zlim(-lim3d, lim3d)
    ax3d.scatter(0, 0, 0, color="gold",       s=120, marker="*", label="Sun")
    ax3d.scatter(*ri[:3], color="dodgerblue", s=60,              label="Earth")
    ax3d.scatter(*rf[:3], color="tomato",     s=60,              label="Mars")
    r3 = z[:, 0:3]
    ax3d.plot(r3[:, 0], r3[:, 1], r3[:, 2], "o-", color=color, ms=2.5, lw=1.8)
    on = delta_bb > 0.5
    if np.any(on):
        lv_n = z[:, 10:13]
        lvm  = np.linalg.norm(lv_n, axis=1, keepdims=True) + 1e-20
        alpha = -lv_n / lvm
        ax3d.quiver(r3[on, 0], r3[on, 1], r3[on, 2],
                    alpha[on, 0], alpha[on, 1], alpha[on, 2],
                    length=0.05, normalize=True, color="tab:red", alpha=0.5)
    ax3d.set_xlabel("x [AU]", fontsize=6); ax3d.set_ylabel("y [AU]", fontsize=6)
    ax3d.set_zlabel("z [AU]", fontsize=6); ax3d.tick_params(labelsize=5)
    ax3d.legend(fontsize=6, loc="upper left")

    xl = (float(t_day[0]), float(t_day[-1]))

    def _setup(ax, ttl, yl, ylim=None):
        ax.set_title(ttl, fontsize=8)
        ax.set_xlabel("time (days)", fontsize=6)
        ax.set_ylabel(yl, fontsize=6)
        ax.set_xlim(*xl)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.tick_params(labelsize=5)
        ax.grid(True, alpha=0.25)

    # ── position ──────────────────────────────────────────────────────────────
    _setup(ax_r, "Position r(t)", "AU", glims.get("r"))
    for k, (c, lb) in enumerate(zip(_XYZ_COLORS, _XYZ_LABELS)):
        ax_r.plot(t_day, z[:, k], color=c, lw=1.3, label=lb)
    ax_r.legend(fontsize=6)

    # ── velocity ──────────────────────────────────────────────────────────────
    _setup(ax_v, "Velocity v(t)", "AU/TU", glims.get("v"))
    for k, (c, lb) in enumerate(zip(_XYZ_COLORS, _XYZ_LABELS)):
        ax_v.plot(t_day, z[:, 3+k], color=c, lw=1.3, label=lb)
    ax_v.legend(fontsize=6)

    # ── mass + thrust (bang-bang) ──────────────────────────────────────────────
    ax_mt.set_title("Mass & Thrust δ", fontsize=8)
    ax_mt.set_xlabel("time (days)", fontsize=6)
    ax_mt.set_ylabel("mass (norm)", fontsize=6, color="tab:green")
    ax_mt.set_ylim(0.0, 1.05); ax_mt.set_xlim(*xl)
    ax_mt.tick_params(labelsize=5, axis="y", labelcolor="tab:green")
    ax_mt.grid(True, alpha=0.25)
    ax_t2.set_ylabel("thrust δ", fontsize=6, color="tab:red")
    ax_t2.yaxis.set_label_position("right")
    ax_t2.set_ylim(-0.05, 1.15)
    ax_t2.tick_params(labelsize=5, axis="y", labelcolor="tab:red")
    ax_mt.plot(t_day, mass_bb, color="tab:green", lw=1.3)
    ax_t2.step(t_day, delta_bb, color="tab:red",   lw=1.5, where="mid")

    # ── costate λ_r ───────────────────────────────────────────────────────────
    _setup(ax_lr, "Costate λ_r(t)", "", glims.get("lam_r"))
    for k, (c, lb) in enumerate(zip(_XYZ_COLORS, _XYZ_LABELS)):
        ax_lr.plot(t_day, z[:, 7+k], color=c, lw=1.3, label=f"λ_r{lb}")
    ax_lr.legend(fontsize=6)

    # ── costate λ_v ───────────────────────────────────────────────────────────
    _setup(ax_lv, "Costate λ_v(t)", "", glims.get("lam_v"))
    for k, (c, lb) in enumerate(zip(_XYZ_COLORS, _XYZ_LABELS)):
        ax_lv.plot(t_day, z[:, 10+k], color=c, lw=1.3, label=f"λ_v{lb}")
    ax_lv.legend(fontsize=6)

    # ── costate λ_m ───────────────────────────────────────────────────────────
    ax_lm.axhline(0.0, color="gray", lw=0.8, ls="--")
    _setup(ax_lm, "Costate λ_m(t)  [→0 at t_f]", "", glims.get("lam_m"))
    ax_lm.plot(t_day, z[:, 13], color="tab:purple", lw=1.8)
    lm_f = float(z[-1, 13])
    ax_lm.scatter(t_day[-1], lm_f, color="tab:purple", s=35, zorder=5,
                  label=f"λ_m(tf)={lm_f:.4f}")
    ax_lm.legend(fontsize=6)

    # ── continuity residuals ───────────────────────────────────────────────────
    ax_rs.set_xlabel("interval k", fontsize=7)
    ax_rs.set_ylabel("‖F(Z_k)−Z_{k+1}‖", fontsize=7)
    ax_rs.tick_params(labelsize=6)
    bars = np.maximum(res_arr, 1e-16)
    ax_rs.bar(np.arange(len(bars)), bars, color=color, alpha=0.75, width=0.8)
    ax_rs.set_yscale("log")
    ax_rs.set_ylim(glims.get("res", (1e-16, 1e1)))
    label = f"{true_res:.2e}" if true_res is not None else f"{res_arr.max():.2e}"
    ax_rs.set_title(f"Continuity residuals  (max={label})", fontsize=8)

    try:
        fig.tight_layout(rect=[0, 0, 1, 0.97])
    except Exception:
        pass
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=80)
    plt.close(fig)


# ── Video assembly ─────────────────────────────────────────────────────────────
def _make_video(frame_dir, out_mp4, fps, tail_seconds=4):
    """Encode PNG frames → MP4, holding the last frame for tail_seconds."""
    ffmpeg = "/usr/bin/ffmpeg"
    cmd = [
        ffmpeg, "-y", "-framerate", str(fps),
        "-pattern_type", "glob", "-i", str(frame_dir / "frame_*.png"),
        "-vf", f"tpad=stop_mode=clone:stop_duration={tail_seconds}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18", str(out_mp4),
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{r.stderr.decode()}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study-dir", type=Path, default=Path("shift_eps_study"))
    p.add_argument("--out-dir",   type=Path, default=Path("shift_videos"))
    p.add_argument("--res-tol",   type=float, default=1e-8)
    p.add_argument("--fps",       type=int,   default=10)
    p.add_argument("--workers",   type=int,   default=6)
    args = p.parse_args()

    # ── collect JSONs ─────────────────────────────────────────────────────────
    jsons = sorted(args.study_dir.rglob("trial_data.json"))
    if not jsons:
        sys.exit(f"No trial_data.json found under {args.study_dir}")
    print(f"Found {len(jsons)} trials in {args.study_dir}")

    # ── global axis limits ────────────────────────────────────────────────────
    glims = _compute_global_limits(jsons, args.res_tol)

    # ── pick best converged trial per shift ───────────────────────────────────
    from collections import defaultdict
    shift_best = {}
    for jp in jsons:
        try:
            d  = json.load(open(jp))
            ma = d["method_A"]
            if not ma["ok"] or float(ma["res"]) >= args.res_tol:
                continue
            sh = int(d["shift_days"])
            rv = float(ma["res"])
            if sh not in shift_best or rv < shift_best[sh][0]:
                shift_best[sh] = (rv, jp, d)
        except Exception:
            continue

    if not shift_best:
        sys.exit(f"No converged trials found (res_tol={args.res_tol})")

    shifts = sorted(shift_best)
    print(f"\nConverged shifts ({len(shifts)}): {shifts}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for shift in shifts:
        res_a, jp, d = shift_best[shift]
        seed  = int(d["seed"])
        norm  = d["norm"]
        TU    = float(norm["TU"])
        t_day = np.array(d["time_grid_tu"]) * TU / 86400.0
        ri    = np.array(d["initial_state"][:3])
        rf    = np.array(d["final_state"][:3])

        ma = d["method_A"]
        diff_frames  = [np.array(f, dtype=np.float64) for f in ma["diffusion_frames"]]
        ipopt_frames = [np.array(f, dtype=np.float64) for f in ma["ipopt_iterates"]]
        z_opt        = np.array(ma["z_opt"], dtype=np.float64)

        all_frames = diff_frames + ipopt_frames + [z_opt]
        n_diff  = len(diff_frames)
        n_ipopt = len(ipopt_frames)
        n_all   = len(all_frames)

        print(f"\n[shift {shift:+d}d]  seed={seed}  res={res_a:.2e}"
              f"  frames={n_diff} diff + {n_ipopt} IPOPT + 1 final")

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futs = []
                for fi, zf in enumerate(all_frames):
                    if fi < n_diff:
                        stage = f"Diffusion step {fi+1}/{n_diff}"
                        color = _DIFF_COLOR
                        tr    = None
                    elif fi < n_diff + n_ipopt:
                        stage = f"IPOPT iter {fi - n_diff + 1}/{n_ipopt}"
                        color = _IPOPT_COLOR
                        tr    = None
                    else:
                        stage = "Converged solution"
                        color = _FINAL_COLOR
                        tr    = res_a

                    title = (f"Shift {shift:+d} d  |  {stage}"
                             f"  [{fi+1}/{n_all}]  seed={seed}"
                             f"  res_final={res_a:.2e}")
                    out_f = tmp / f"frame_{fi:04d}.png"
                    futs.append(pool.submit(
                        _render_frame,
                        zf, t_day, ri, rf, norm, title, color, out_f, glims, tr
                    ))

                for fut in futs:
                    fut.result()

            out_mp4 = args.out_dir / f"shift_{shift:+d}d.mp4"
            _make_video(tmp, out_mp4, fps=args.fps)
            print(f"  → {out_mp4}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
