"""
Single 3D axes: 3 trajectories stacked along Z with display offsets.
True visual scale: box_aspect proportional to actual AU ranges.
Each trajectory band spans ±0.1 AU (0.2 AU) relative to its floor.
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
    sol = solve_ivp(_kepler_rhs, (0, T), np.concatenate([r0, v0]),
                    method="DOP853", t_eval=np.linspace(0, T, n_pts),
                    rtol=1e-12, atol=1e-13)
    return sol.y[:3].T


earth_orbit = propagate_orbit(r_i, v_i)
mars_orbit  = propagate_orbit(r_f, v_f)

all_orb  = np.vstack([earth_orbit, mars_orbit])
x_mid    = 0.5 * (all_orb[:, 0].max() + all_orb[:, 0].min())
y_mid    = 0.5 * (all_orb[:, 1].max() + all_orb[:, 1].min())
xy_half  = 0.58 * max(all_orb[:, 0].max() - all_orb[:, 0].min(),
                      all_orb[:, 1].max() - all_orb[:, 1].min())
X_LO, X_HI = x_mid - xy_half, x_mid + xy_half
Y_LO, Y_HI = y_mid - xy_half, y_mid + xy_half
DZ     = 0.10    # half-width of each Z band (band spans ±0.1 AU)
Z_STEP = 0.35    # floor-to-floor offset (> 2*DZ so bands never overlap)

TARGETS = [
    ("-300 days", "shift_-300d", "00"),
    ("+100 days", "shift_+100d", "00"),
    ("+500 days", "shift_+500d", "09"),
]
N = len(TARGETS)
Z_LO_TOT = 0 - DZ - 0.02
Z_HI_TOT = (N - 1) * Z_STEP + DZ + 0.02

C_EARTH = "#93C572"; C_MARS = "#CC5500"
C_DIFF  = "#0F52BA"; C_IPOPT = "#8B008B"; C_SUN = "#F5C518"

STUDY_DIR = Path("shift_eps_study")

fig = plt.figure(figsize=(6, 5))
ax  = fig.add_subplot(111, projection="3d")
ax.view_init(elev=20, azim=225)

for level, (shift_label, shift_dir, trial_id) in enumerate(TARGETS):
    z_off = level * Z_STEP

    with open(STUDY_DIR / shift_dir / f"trial_{trial_id}" / "trial_data.json") as f:
        d = json.load(f)

    i_st   = np.asarray(d["initial_state"],  dtype=np.float64)
    f_st   = np.asarray(d["final_state"],    dtype=np.float64)
    r_diff = np.asarray(d["method_A"]["diffusion_frames"], dtype=np.float64)[-1, :, :3].copy()
    r_opt  = np.asarray(d["method_A"]["z_opt"], dtype=np.float64)[:, :3]

    # Clip diffusion Z to this band to avoid spikes breaking the layout
    out = (r_diff[:, 2] < -DZ - 0.05) | (r_diff[:, 2] > DZ + 0.05)
    r_diff[out] = np.nan

    lbl = level == 0   # only label first set for legend
    ax.plot(earth_orbit[:, 0], earth_orbit[:, 1], earth_orbit[:, 2] + z_off,
            color=C_EARTH, lw=0.9, ls=":", label="Earth orbit" if lbl else None, alpha=0.75)
    ax.plot(mars_orbit[:, 0],  mars_orbit[:, 1],  mars_orbit[:, 2] + z_off,
            color=C_MARS,  lw=0.9, ls=":", label="Mars orbit"  if lbl else None, alpha=0.75)
    ax.scatter(0, 0, z_off, color=C_SUN, s=50, marker="*", zorder=6,
               label="Sun" if lbl else None)

    ax.plot(r_opt[:, 0],  r_opt[:, 1],  r_opt[:, 2]  + z_off,
            color=C_IPOPT, lw=1.8, ls="-",  label="IPOPT refined" if lbl else None, zorder=4)
    ax.plot(r_diff[:, 0], r_diff[:, 1], r_diff[:, 2] + z_off,
            color=C_DIFF,  lw=1.2, ls="--", label="Diffusion"      if lbl else None, zorder=5)

    ax.scatter(i_st[0], i_st[1], i_st[2] + z_off, color=C_EARTH, s=20, zorder=7)
    ax.scatter(f_st[0], f_st[1], f_st[2] + z_off, color=C_MARS,  s=20, zorder=7)

    # Shift label: spread horizontally so they don't overlap (Z is too compressed)
    x_lbl = X_LO + 0.10 + level * (X_HI - X_LO) * 0.28
    ax.text(x_lbl, Y_LO + 0.05, z_off - DZ + 0.005,
            f"shift: {shift_label}",
            fontsize=8.5, ha="left", va="bottom",
            color="#111111", fontweight="bold")

ax.set_xlim(X_LO, X_HI)
ax.set_ylim(Y_LO, Y_HI)
ax.set_zlim(Z_LO_TOT, Z_HI_TOT)

# True visual scale: Z proportional to AU range
xy_span = 2 * xy_half
z_span  = Z_HI_TOT - Z_LO_TOT
ax.set_box_aspect([1, 1, z_span / xy_span])

ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor("#cccccc")
ax.yaxis.pane.set_edgecolor("#cccccc")
ax.zaxis.pane.set_edgecolor("#cccccc")
ax.grid(False)

xy_ticks = np.round(np.linspace(X_LO, X_HI, 5), 1)
ax.xaxis.set_ticks(xy_ticks)
ax.yaxis.set_ticks(xy_ticks)
# Z ticks relative to each floor (0, 0.35, 0.70) showing ±0.1 per band
z_ticks = [z - DZ for z in [0, Z_STEP, 2*Z_STEP]] + \
          [z       for z in [0, Z_STEP, 2*Z_STEP]] + \
          [z + DZ  for z in [0, Z_STEP, 2*Z_STEP]]
z_ticks = sorted(set(z_ticks))
ax.zaxis.set_ticks(z_ticks)
ax.zaxis.set_ticklabels([f"{v - round(v/Z_STEP)*Z_STEP:+.2f}" for v in z_ticks])

ax.tick_params(labelsize=6, pad=1)
ax.set_xlabel("X (AU)", fontsize=8, labelpad=5)
ax.set_ylabel("Y (AU)", fontsize=8, labelpad=5)
ax.set_zlabel("Z (AU)", fontsize=8, labelpad=5)

ax.legend(fontsize=7.5, loc="upper left", framealpha=0.85, edgecolor="#aaaaaa")

out_dir = Path("orbit_plots")
out_dir.mkdir(exist_ok=True)
for ext in ("eps", "png"):
    out_path = out_dir / f"stacked_orbits.{ext}"
    fig.savefig(str(out_path), format=ext, bbox_inches="tight", pad_inches=0.1, dpi=150)
    print(f"saved → {out_path}")
plt.close(fig)
print("done")
