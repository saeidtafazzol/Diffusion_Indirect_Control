"""
Generate 3 square 3D EPS plots for shifts -300, +100, +500.
4 trajectories per plot: Earth orbit, Mars orbit, diffusion output, IPOPT refined.
Only shift -300 gets a legend.
Uses best-residual trial for each shift.
"""
import sys, json
import numpy as np
from pathlib import Path
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from core import build_normalization

# ── Normalization ─────────────────────────────────────────────────────────────
norm = build_normalization()
mu   = float(norm["mu"])
TU   = float(norm["TU"])
r_i  = np.asarray(norm["r_i"], dtype=np.float64)   # Earth base pos (norm units = AU)
v_i  = np.asarray(norm["v_i"], dtype=np.float64)
r_f  = np.asarray(norm["r_f"], dtype=np.float64)   # Mars base pos
v_f  = np.asarray(norm["v_f"], dtype=np.float64)


def _kepler_rhs(t, y):
    r, v = y[:3], y[3:]
    return np.concatenate([v, -mu / np.linalg.norm(r) ** 3 * r])


def orbital_period(r0, v0):
    """Kepler 3rd law from specific energy."""
    E = 0.5 * np.dot(v0, v0) - mu / np.linalg.norm(r0)
    a = -mu / (2.0 * E)
    return 2.0 * np.pi * np.sqrt(a ** 3 / mu)


def propagate_orbit(r0, v0, n_pts=800):
    """One full Keplerian orbit → (n_pts, 3)."""
    T = orbital_period(r0, v0)
    sol = solve_ivp(
        _kepler_rhs, (0, T), np.concatenate([r0, v0]),
        method="DOP853", t_eval=np.linspace(0, T, n_pts),
        rtol=1e-12, atol=1e-13,
    )
    return sol.y[:3].T   # (n_pts, 3)


earth_orbit = propagate_orbit(r_i, v_i)
mars_orbit  = propagate_orbit(r_f, v_f)

# ── Best-residual trial per shift ─────────────────────────────────────────────
#   -300d: trial_00 (res=2.17e-09)
#   +100d: trial_00 (res=1.06e-11)
#   +500d: trial_09 (res=1.72e-12)
TARGETS = [
    ("-300", "shift_-300d", "00"),
    ("+100", "shift_+100d", "00"),
    ("+500", "shift_+500d", "09"),
]

out_dir = Path("orbit_plots")
out_dir.mkdir(exist_ok=True)

# ── Colours ───────────────────────────────────────────────────────────────────
C_EARTH  = "#93C572"   # pistachio green
C_MARS   = "#CC5500"   # burnt orange
C_DIFF   = "#0F52BA"   # space blue
C_IPOPT  = "#8B008B"   # dark magenta
C_SUN    = "#F5C518"   # gold

for shift_label, shift_dir, trial_id in TARGETS:
    json_path = Path("shift_eps_study") / shift_dir / f"trial_{trial_id}" / "trial_data.json"
    with open(json_path) as f:
        d = json.load(f)

    i_st = np.asarray(d["initial_state"], dtype=np.float64)   # shifted Earth pos
    f_st = np.asarray(d["final_state"],   dtype=np.float64)   # shifted Mars pos

    diff_frames = np.asarray(d["method_A"]["diffusion_frames"], dtype=np.float64)
    z_diff = diff_frames[-1]                                   # (32, 14) last diffusion step
    z_opt  = np.asarray(d["method_A"]["z_opt"], dtype=np.float64)  # (32, 14) IPOPT output

    r_diff = z_diff[:, :3]
    r_opt  = z_opt[:,  :3]

    show_legend = (shift_label == "-300")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(5, 5))
    ax  = fig.add_subplot(111, projection="3d")

    ax.plot(earth_orbit[:, 0], earth_orbit[:, 1], earth_orbit[:, 2],
            color=C_EARTH, lw=1.2, ls=":", label="Earth orbit")
    ax.plot(mars_orbit[:, 0],  mars_orbit[:, 1],  mars_orbit[:, 2],
            color=C_MARS,  lw=1.2, ls=":", label="Mars orbit")
    ax.plot(r_opt[:, 0],  r_opt[:, 1],  r_opt[:, 2],
            color=C_IPOPT, lw=1.8, ls="-",  label="IPOPT refined")
    ax.plot(r_diff[:, 0], r_diff[:, 1], r_diff[:, 2],
            color=C_DIFF,  lw=1.4, ls="--", label="Diffusion")

    # Start / end markers
    ax.scatter(*i_st[:3], color=C_EARTH, s=35, zorder=5)
    ax.scatter(*f_st[:3], color=C_MARS,  s=35, zorder=5)
    ax.scatter(0, 0, 0, color=C_SUN, s=80, marker="*", zorder=5, label="Sun")

    # Equal-aspect cube: find max range across all plotted data
    all_pts = np.vstack([earth_orbit, mars_orbit, r_diff, r_opt])
    lo, hi  = all_pts.min(axis=0), all_pts.max(axis=0)
    mid     = 0.5 * (lo + hi)
    half    = 0.55 * (hi - lo).max()
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)

    # Axes: keep ticks and labels, remove grids and pane fills
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#cccccc")
    ax.yaxis.pane.set_edgecolor("#cccccc")
    ax.zaxis.pane.set_edgecolor("#cccccc")
    ax.grid(False)

    # Tick density: 5 ticks per axis, rounded to 1 decimal
    for axis, lims in zip([ax.xaxis, ax.yaxis, ax.zaxis],
                          [(mid[0]-half, mid[0]+half),
                           (mid[1]-half, mid[1]+half),
                           (mid[2]-half, mid[2]+half)]):
        ticks = np.linspace(lims[0], lims[1], 5)
        ticks = np.round(ticks, 1)
        axis.set_ticks(ticks)

    ax.tick_params(axis="both", labelsize=6, pad=2)
    ax.set_xlabel("X (AU)", fontsize=8, labelpad=6)
    ax.set_ylabel("Y (AU)", fontsize=8, labelpad=6)
    ax.set_zlabel("Z (AU)", fontsize=8, labelpad=6)

    ax.set_title(f"shift = {shift_label} d", fontsize=10, pad=6)

    if show_legend:
        ax.legend(fontsize=10, loc="upper left", framealpha=0.85, edgecolor="#aaaaaa")

    out_path = out_dir / f"shift_{shift_label}d.eps"
    fig.savefig(str(out_path), format="eps", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"saved → {out_path}")

print("done")
