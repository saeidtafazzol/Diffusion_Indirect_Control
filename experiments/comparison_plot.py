"""
2×2 + wide-bottom comparison for the −300-day shift:
  top-left:     r(t) + m(t)    |  top-right:   v(t)
  bottom-left:  λ_r(t)+λ_m(t)  |  bottom-right: λ_v(t)
  bottom-wide:  switching function S(t) + thrust δ(t)  [half height]

Dashed  = diffusion (last frame, pre-IPOPT)
Solid   = IPOPT-refined solution
"""
import sys, json
import numpy as np
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
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

# ── Switching function & thrust ───────────────────────────────────────────────
def _switching(z, norm):
    c  = float(norm["c_norm"])
    lv = z[:, 10:13]; lm = z[:, 13]; m = z[:, 6]
    S  = c * np.linalg.norm(lv, axis=1) / m + lm - 1.0
    return S, (S > 0).astype(float)

norm  = d["norm"]
S_d, delta_d = _switching(z_d, norm)
S_o, delta_o = _switching(z_o, norm)

# ── Palette ───────────────────────────────────────────────────────────────────
CX = "#E74C3C"; CY = "#27AE60"; CZ = "#2980B9"; CM = "#E67E22"
CS = "#8E44AD"   # switching function colour
CD = "#E67E22"   # thrust colour

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
fig = plt.figure(figsize=(9, 7.5))
gs  = GridSpec(3, 2, figure=fig,
               height_ratios=[1, 1, 1.0],
               hspace=0.55, wspace=0.38)

ax_tl = fig.add_subplot(gs[0, 0])
ax_tr = fig.add_subplot(gs[0, 1])
ax_bl = fig.add_subplot(gs[1, 0])
ax_br = fig.add_subplot(gs[1, 1])
ax_sw = fig.add_subplot(gs[2, :])   # wide, half height

# Top-left: r(t) + m(t)
add_xyz(ax_tl, t_days, r_d, r_o)
add_scalar_twin(ax_tl, t_days, m_d, m_o, "mass (norm)")
ax_tl.set_ylabel("$r$ (AU)", fontsize=11)
ax_tl.set_title(r"Position $r(t)$ + Mass", fontsize=11)
ax_tl.tick_params(labelsize=9)

# Top-right: v(t)
add_xyz(ax_tr, t_days, v_d, v_o)
ax_tr.set_ylabel("$v$ (AU/TU)", fontsize=11)
ax_tr.set_title(r"Velocity $v(t)$", fontsize=11)
ax_tr.tick_params(labelsize=9)

# Bottom-left: λ_r(t) + λ_m(t)
add_xyz(ax_bl, t_days, lr_d, lr_o)
add_scalar_twin(ax_bl, t_days, lm_d, lm_o, r"$\lambda_m$")
ax_bl.set_xlabel("time (days)", fontsize=11)
ax_bl.set_ylabel(r"$\lambda_r$", fontsize=11)
ax_bl.set_title(r"Costate $\lambda_r(t)$ + $\lambda_m$", fontsize=11)
ax_bl.tick_params(labelsize=9)

# Bottom-right: λ_v(t)
add_xyz(ax_br, t_days, lv_d, lv_o)
ax_br.set_xlabel("time (days)", fontsize=11)
ax_br.set_ylabel(r"$\lambda_v$", fontsize=11)
ax_br.set_title(r"Costate $\lambda_v(t)$", fontsize=11)
ax_br.tick_params(labelsize=9)

# Wide bottom: switching function S(t) + thrust δ(t)
ax_sw.axhline(0, color="gray", lw=0.7, ls=":")
ax_sw.plot(t_days, S_d, color=CS, ls="--", lw=1.2, alpha=0.85, label=r"$S$ (diffusion)")
ax_sw.plot(t_days, S_o, color=CS, ls="-",  lw=1.5,              label=r"$S$ (IPOPT)")
ax_sw.set_ylabel(r"$S(t)$", fontsize=11, color=CS)
ax_sw.tick_params(labelsize=9, axis="y", labelcolor=CS)
ax_sw.set_xlabel("time (days)", fontsize=11)
ax_sw.set_title(r"Switching function $S(t)$ and thrust $\delta(t)$", fontsize=11)

ax_d = ax_sw.twinx()
ax_d.step(t_days, delta_d, color=CD, ls="--", lw=1.2, alpha=0.85, where="mid",
          label=r"$\delta$ (diffusion)")
ax_d.step(t_days, delta_o, color=CD, ls="-",  lw=1.5, where="mid",
          label=r"$\delta$ (IPOPT)")
ax_d.set_ylabel(r"$\delta(t)$", fontsize=11, color=CD)
ax_d.set_ylim(-0.15, 1.35)
ax_d.tick_params(labelsize=9, axis="y", labelcolor=CD)

# ── Shared figure legend ──────────────────────────────────────────────────────
legend_handles = [
    Line2D([0], [0], color="black", ls="-",  lw=1.5, label="IPOPT Refinement"),
    Line2D([0], [0], color="black", ls="--", lw=1.2, label="Diffusion"),
    Line2D([0], [0], color=CX, ls="-", lw=2, label="$x$"),
    Line2D([0], [0], color=CY, ls="-", lw=2, label="$y$"),
    Line2D([0], [0], color=CZ, ls="-", lw=2, label="$z$"),
    Line2D([0], [0], color=CM, ls="-", lw=2, label=r"mass / $\lambda_m$"),
    Line2D([0], [0], color=CS, ls="-", lw=2, label=r"$S(t)$"),
    Line2D([0], [0], color=CD, ls="-", lw=2, label=r"$\delta(t)$"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=8, fontsize=9,
           framealpha=0.9, bbox_to_anchor=(0.5, 0.0))

fig.tight_layout(rect=[0, 0.06, 1, 1.0])

out_dir = _root / "orbit_plots"
out_dir.mkdir(exist_ok=True)
for ext in ("png", "eps"):
    p = out_dir / f"comparison_-300d.{ext}"
    fig.savefig(str(p), dpi=150, bbox_inches="tight", pad_inches=0.15, format=ext)
    print(f"saved → {p}")
plt.close(fig)
print("done")
