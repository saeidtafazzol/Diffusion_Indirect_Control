"""
Single figure: 3 trajectories (-300, +100, +500 d shifts) stacked vertically.
Shared XY extent and view angle; each panel has its own Z axis, -0.1 to +0.1 AU.
"""
import sys, json
import numpy as np
from pathlib import Path
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from core import build_normalization

norm = build_normalization()
mu  = float(norm["mu"])
r_i = np.asarray(norm["r_i"], dtype=np.float64)
v_i = np.asarray(norm["v_i"], dtype=np.float64)
r_f = np.asarray(norm["r_f"], dtype=np.float64)
v_f = np.asarray(norm["v_f"], dtype=np.float64)


def _kepler_rhs(t, y):
    r, v = y[:3], y[3:]
    return np.concatenate([v, -mu / np.linalg.norm(r) ** 3 * r])


def orbital_period(r0, v0):
    E = 0.5 * np.dot(v0, v0) - mu / np.linalg.norm(r0)
    a = -mu / (2.0 * E)
    return 2.0 * np.pi * np.sqrt(a ** 3 / mu)


def propagate_orbit(r0, v0, n_pts=800):
    T = orbital_period(r0, v0)
    sol = solve_ivp(
        _kepler_rhs, (0, T), np.concatenate([r0, v0]),
        method="DOP853", t_eval=np.linspace(0, T, n_pts),
        rtol=1e-12, atol=1e-13,
    )
    return sol.y[:3].T


earth_orbit = propagate_orbit(r_i, v_i)
mars_orbit  = propagate_orbit(r_f, v_f)

# Shared XY limits from the two planet orbits
all_orb = np.vstack([earth_orbit, mars_orbit])
x_mid = 0.5 * (all_orb[:, 0].max() + all_orb[:, 0].min())
y_mid = 0.5 * (all_orb[:, 1].max() + all_orb[:, 1].min())
xy_half = 0.58 * max(all_orb[:, 0].max() - all_orb[:, 0].min(),
                     all_orb[:, 1].max() - all_orb[:, 1].min())
X_LO, X_HI = x_mid - xy_half, x_mid + xy_half
Y_LO, Y_HI = y_mid - xy_half, y_mid + xy_half
Z_LO, Z_HI = -0.1, 0.1

TARGETS = [
    ("-300 days", "shift_-300d", "00"),
    ("+100 days", "shift_+100d", "00"),
    ("+500 days", "shift_+500d", "09"),
]

C_EARTH = "#93C572"
C_MARS  = "#CC5500"
C_DIFF  = "#0F52BA"
C_IPOPT = "#8B008B"
C_SUN   = "#F5C518"

STUDY_DIR = Path("shift_eps_study")

ELEV, AZIM = 14, 225

fig = plt.figure(figsize=(5.5, 11))

for row, (shift_label, shift_dir, trial_id) in enumerate(TARGETS):
    json_path = STUDY_DIR / shift_dir / f"trial_{trial_id}" / "trial_data.json"
    with open(json_path) as f:
        d = json.load(f)

    i_st = np.asarray(d["initial_state"],  dtype=np.float64)
    f_st = np.asarray(d["final_state"],    dtype=np.float64)
    z_diff = np.asarray(d["method_A"]["diffusion_frames"], dtype=np.float64)[-1]
    z_opt  = np.asarray(d["method_A"]["z_opt"],           dtype=np.float64)

    r_diff = z_diff[:, :3]
    r_opt  = z_opt[:,  :3]

    ax = fig.add_subplot(3, 1, row + 1, projection="3d")
    ax.view_init(elev=ELEV, azim=AZIM)

    ax.plot(earth_orbit[:, 0], earth_orbit[:, 1], earth_orbit[:, 2],
            color=C_EARTH, lw=1.0, ls=":", label="Earth orbit")
    ax.plot(mars_orbit[:, 0],  mars_orbit[:, 1],  mars_orbit[:, 2],
            color=C_MARS,  lw=1.0, ls=":", label="Mars orbit")
    ax.plot(r_opt[:, 0],  r_opt[:, 1],  r_opt[:, 2],
            color=C_IPOPT, lw=1.8, ls="-",  label="IPOPT refined",  zorder=4)
    ax.plot(r_diff[:, 0], r_diff[:, 1], r_diff[:, 2],
            color=C_DIFF,  lw=1.3, ls="--", label="Diffusion",       zorder=5)

    ax.scatter(*i_st[:3], color=C_EARTH, s=25, zorder=6)
    ax.scatter(*f_st[:3], color=C_MARS,  s=25, zorder=6)
    ax.scatter(0, 0, 0,   color=C_SUN,   s=60, marker="*", zorder=6,
               label="Sun" if row == 0 else None)

    ax.set_xlim(X_LO, X_HI)
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_zlim(Z_LO, Z_HI)

    # True scale: Z visual size proportional to actual AU range
    xy_span = 2 * xy_half
    ax.set_box_aspect([1, 1, (Z_HI - Z_LO) / xy_span])

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#cccccc")
    ax.yaxis.pane.set_edgecolor("#cccccc")
    ax.zaxis.pane.set_edgecolor("#cccccc")
    ax.grid(False)

    # X, Y ticks: 5 values across shared range
    xy_ticks = np.round(np.linspace(X_LO, X_HI, 5), 1)
    ax.xaxis.set_ticks(xy_ticks)
    ax.yaxis.set_ticks(xy_ticks)
    ax.zaxis.set_ticks([-0.1, -0.05, 0.0, 0.05, 0.1])

    ax.tick_params(labelsize=6, pad=1)

    # Only label X,Y on bottom panel; always label Z
    if row == 2:
        ax.set_xlabel("X (AU)", fontsize=8, labelpad=5)
        ax.set_ylabel("Y (AU)", fontsize=8, labelpad=5)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    ax.set_zlabel("Z (AU)", fontsize=8, labelpad=5)

    # Label inside the plot, sitting just above the XY plane level
    ax.text2D(0.04, 0.10, f"shift: {shift_label}",
              transform=ax.transAxes, fontsize=9, ha="left", va="bottom",
              color="#222222")

    if row == 0:
        ax.legend(fontsize=7.5, loc="upper left", framealpha=0.85,
                  edgecolor="#aaaaaa", ncol=1)

fig.subplots_adjust(hspace=-0.35)

out_dir = Path("orbit_plots")
out_dir.mkdir(exist_ok=True)
for ext in ("eps", "png"):
    out_path = out_dir / f"stacked_orbits.{ext}"
    fig.savefig(str(out_path), format=ext, bbox_inches="tight", pad_inches=0.1, dpi=150)
    print(f"saved → {out_path}")
plt.close(fig)
print("done")
