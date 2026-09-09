"""
2×2 comparison for the −300-day shift:
  top-left:     r(t) + m(t)    |  top-right:   v(t)
  bottom-left:  λ_r(t)+λ_m(t)  |  bottom-right: λ_v(t)

Dashed  = diffusion (last frame, pre-IPOPT)
Solid   = IPOPT-refined solution
"""
import sys, json
import numpy as np
from pathlib import Path
from matplotlib.lines import Line2D
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

# ── Load data ─────────────────────────────────────────────────────────────────
json_path = _root / "shift_eps_study" / "shift_-300d" / "trial_00" / "trial_data.json"
with open(json_path) as f:
    d = json.load(f)

TU     = float(d["norm"]["TU"])
t_days = np.asarray(d["time_grid_tu"]) * TU / 86400.0

z_d = np.asarray(d["method_A"]["diffusion_frames"])[-1]  # (32, 14) last diffusion step
z_o = np.asarray(d["method_A"]["z_opt"])                 # (32, 14) IPOPT output

# State layout: rx ry rz  vx vy vz  m  λrx λry λrz  λvx λvy λvz  λm
r_d,  r_o  = z_d[:, 0:3], z_o[:, 0:3]
v_d,  v_o  = z_d[:, 3:6], z_o[:, 3:6]
m_d,  m_o  = z_d[:, 6],   z_o[:, 6]
lr_d, lr_o = z_d[:, 7:10], z_o[:, 7:10]
lv_d, lv_o = z_d[:, 10:13], z_o[:, 10:13]
lm_d, lm_o = z_d[:, 13],    z_o[:, 13]

# ── Palette ───────────────────────────────────────────────────────────────────
CX = "#E74C3C"; CY = "#27AE60"; CZ = "#2980B9"; CM = "#E67E22"

# ── Helpers ───────────────────────────────────────────────────────────────────
def add_xyz(ax, t, d3, o3):
    for i, c in enumerate((CX, CY, CZ)):
        ax.plot(t, d3[:, i], color=c, ls="--", lw=1.2, alpha=0.85)
        ax.plot(t, o3[:, i], color=c, ls="-",  lw=1.5)

def add_scalar_twin(ax, t, sd, so, ylabel):
    ax2 = ax.twinx()
    ax2.plot(t, sd, color=CM, ls="--", lw=1.2, alpha=0.85)
    ax2.plot(t, so, color=CM, ls="-",  lw=1.5)
    ax2.set_ylabel(ylabel, fontsize=11, color=CM)
    ax2.tick_params(labelsize=9, colors=CM)
    return ax2

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axs = plt.subplots(2, 2, figsize=(9, 6))

# Top-left: r(t) + m(t)
ax = axs[0, 0]
add_xyz(ax, t_days, r_d, r_o)
add_scalar_twin(ax, t_days, m_d, m_o, "mass (norm)")
ax.set_ylabel("$r$ (AU)", fontsize=11)
ax.set_title(r"Position $r(t)$ + Mass", fontsize=11)
ax.tick_params(labelsize=9)

# Top-right: v(t)
ax = axs[0, 1]
add_xyz(ax, t_days, v_d, v_o)
ax.set_ylabel("$v$ (AU/TU)", fontsize=11)
ax.set_title(r"Velocity $v(t)$", fontsize=11)
ax.tick_params(labelsize=9)

# Bottom-left: λ_r(t) + λ_m(t)
ax = axs[1, 0]
add_xyz(ax, t_days, lr_d, lr_o)
add_scalar_twin(ax, t_days, lm_d, lm_o, r"$\lambda_m$")
ax.set_xlabel("time (days)", fontsize=11)
ax.set_ylabel(r"$\lambda_r$", fontsize=11)
ax.set_title(r"Costate $\lambda_r(t)$ + $\lambda_m$", fontsize=11)
ax.tick_params(labelsize=9)

# Bottom-right: λ_v(t)
ax = axs[1, 1]
add_xyz(ax, t_days, lv_d, lv_o)
ax.set_xlabel("time (days)", fontsize=11)
ax.set_ylabel(r"$\lambda_v$", fontsize=11)
ax.set_title(r"Costate $\lambda_v(t)$", fontsize=11)
ax.tick_params(labelsize=9)

# ── Shared figure legend ──────────────────────────────────────────────────────
legend_handles = [
    Line2D([0], [0], color="black", ls="-",  lw=1.5, label="IPOPT Refinement"),
    Line2D([0], [0], color="black", ls="--", lw=1.2, label="Diffusion"),
    Line2D([0], [0], color=CX, ls="-", lw=2, label="$x$"),
    Line2D([0], [0], color=CY, ls="-", lw=2, label="$y$"),
    Line2D([0], [0], color=CZ, ls="-", lw=2, label="$z$"),
    Line2D([0], [0], color=CM, ls="-", lw=2, label=r"mass / $\lambda_m$"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=6, fontsize=10,
           framealpha=0.9, bbox_to_anchor=(0.5, 0.0))

fig.tight_layout(rect=[0, 0.07, 1, 1.0])

out_dir = _root / "orbit_plots"
out_dir.mkdir(exist_ok=True)
for ext in ("png", "eps"):
    p = out_dir / f"comparison_-300d.{ext}"
    fig.savefig(str(p), dpi=150, bbox_inches="tight", pad_inches=0.15, format=ext)
    print(f"saved → {p}")
plt.close(fig)
print("done")
