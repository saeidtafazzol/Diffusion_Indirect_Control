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
Z_STEP = 1.30    # floor-to-floor offset — larger = more breathing room

LBL_DX = [0.05, 0.05, 0.05]  # x offset: push all labels right of centre
LBL_DY = [-0.1, -0.1, -0.1]  # y offset: push all labels toward viewer

TARGETS = [
    ("−300 days", "shift_-300d", "00"),
    ("+100 days", "shift_+100d", "00"),
    ("+500 days", "shift_+500d", "09"),
]
N = len(TARGETS)
Z_LO_TOT = 0 - DZ - 0.02
Z_HI_TOT = (N - 1) * Z_STEP + DZ + 0.02

C_EARTH = "#93C572"; C_MARS = "#CC5500"
C_DIFF  = "#0F52BA"; C_IPOPT = "#8B008B"; C_SUN = "#F5C518"

STUDY_DIR = Path("shift_eps_study")

fig = plt.figure(figsize=(8, 6))
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


    ax.plot(r_opt[:, 0],  r_opt[:, 1],  r_opt[:, 2]  + z_off,
            color=C_IPOPT, lw=1.8, ls="-",  label="IPOPT refined" if lbl else None, zorder=4)
    ax.plot(r_diff[:, 0], r_diff[:, 1], r_diff[:, 2] + z_off,
            color=C_DIFF,  lw=1.2, ls="--", label="Diffusion"      if lbl else None, zorder=5)

    ax.scatter(i_st[0], i_st[1], i_st[2] + z_off, color=C_EARTH, s=20, zorder=7)
    ax.scatter(f_st[0], f_st[1], f_st[2] + z_off, color=C_MARS,  s=20, zorder=7)

    # Label at XY origin of each floor level ("+500 days" offset to avoid collision)
    ax.text(LBL_DX[level], LBL_DY[level], z_off,
            shift_label,
            fontsize=9, ha="center", va="center",
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
# ±0.1 tick marks and labels on every floor
all_z_ticks = sorted([lv * Z_STEP + dz for lv in range(N) for dz in (-DZ, DZ)])
ax.zaxis.set_ticks(all_z_ticks)
ax.zaxis.set_ticklabels(["-0.1", "+0.1"] * N)

ax.tick_params(labelsize=8, pad=1)
ax.set_xlabel("X (AU)", fontsize=11, labelpad=6)
ax.set_ylabel("Y (AU)", fontsize=11, labelpad=6)
ax.set_zlabel("")          # suppress default (unreliable in 3D)
ax.text2D(-0.08, 0.5, "Z (AU)", transform=ax.transAxes,
          fontsize=11, rotation=90, va="center", ha="center")

ax.legend(fontsize=10, loc="upper left", framealpha=0.85, edgecolor="#aaaaaa")

out_dir = Path("orbit_plots")
out_dir.mkdir(exist_ok=True)
for ext in ("eps", "png"):
    out_path = out_dir / f"stacked_orbits.{ext}"
    fig.savefig(str(out_path), format=ext, bbox_inches="tight", pad_inches=0.3, dpi=150)
    print(f"saved → {out_path}")
plt.close(fig)
print("done")
