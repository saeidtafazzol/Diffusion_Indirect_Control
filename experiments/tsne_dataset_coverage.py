"""
tsne_dataset_coverage.py
========================
t-SNE coverage plots: training dataset vs shift-experiment boundary conditions.

Two panels:
  Left  — initial states (7D: r, v, m)  projected to 2D
  Right — final states   (6D: r, v, no mass) projected to 2D

Dataset points: small semi-transparent dots
Shift-experiment points: coloured crosses (one per shift)

Saved to experiments/tsne_coverage.png
"""

import sys
import numpy as np
from pathlib import Path
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
from core import build_normalization

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR = _root / "earth_mars_minfuel_posvel_constrained_32pts"
N_TSNE      = 8000        # dataset points fed to t-SNE (random subset)
RANDOM_SEED = 42
PERPLEXITY  = 40

DEFAULT_SHIFTS = [
    -700, -600, -500, -400, -300, -200, -100, -50,
       0,   50,  100,  200,  300,  400,  500,  600,  700,
]

# ── Load dataset ──────────────────────────────────────────────────────────────
chunks = sorted(DATASET_DIR.glob("chunk_*.npz"))
print(f"Loading {len(chunks)} chunks …")

all_init, all_final = [], []
for ch in chunks:
    c = np.load(ch)
    all_init.append(c["initial_states"])    # (4096, 7)
    all_final.append(c["final_states"])     # (4096, 7)

init_all  = np.concatenate(all_init,  axis=0).astype(np.float32)   # (N, 7)
final_all = np.concatenate(all_final, axis=0).astype(np.float32)   # (N, 7)
N_total   = len(init_all)
print(f"Total trajectories: {N_total:,}")

rng = np.random.default_rng(RANDOM_SEED)
idx = rng.choice(N_total, size=min(N_TSNE, N_total), replace=False)
init_sub  = init_all[idx]          # (N_TSNE, 7)
final_sub = final_all[idx, :6]     # (N_TSNE, 6) — drop mass

# ── Shift experiment boundary conditions ─────────────────────────────────────
norm = build_normalization()
AU, TU, mu = float(norm["AU"]), float(norm["TU"]), float(norm["mu"])
r_i, v_i   = np.asarray(norm["r_i"]), np.asarray(norm["v_i"])
r_f, v_f   = np.asarray(norm["r_f"]), np.asarray(norm["v_f"])
m0          = float(norm["m0"])

def _kepler_rhs(t, y):
    r, v = y[:3], y[3:]
    return np.concatenate([v, -mu / np.linalg.norm(r)**3 * r])

def propagate(r0, v0, dt_norm):
    if abs(dt_norm) < 1e-12:
        return r0.copy(), v0.copy()
    sol = solve_ivp(_kepler_rhs, (0, dt_norm), np.concatenate([r0, v0]),
                    method="DOP853", rtol=1e-12, atol=1e-13)
    return sol.y[:3, -1], sol.y[3:6, -1]

shift_init, shift_final = [], []
for s in DEFAULT_SHIFTS:
    dt = s * 86400.0 / TU
    ri, vi = propagate(r_i, v_i, dt)
    rf, vf = propagate(r_f, v_f, dt)
    shift_init.append(np.concatenate([ri, vi, [m0]]))
    shift_final.append(np.concatenate([rf, vf]))          # 6D

shift_init  = np.array(shift_init,  dtype=np.float32)    # (17, 7)
shift_final = np.array(shift_final, dtype=np.float32)    # (17, 6)

# ── t-SNE: initial (7D) ───────────────────────────────────────────────────────
print("Running t-SNE on initial states (7D) …")
n_ds = len(init_sub)
combined_init = np.vstack([init_sub, shift_init])         # (N+17, 7)
emb_init = TSNE(n_components=2, perplexity=PERPLEXITY,
                random_state=RANDOM_SEED, n_jobs=-1).fit_transform(combined_init)
ds_init    = emb_init[:n_ds]
shift_emb_init = emb_init[n_ds:]

# ── t-SNE: final (6D) ────────────────────────────────────────────────────────
print("Running t-SNE on final states (6D) …")
combined_final = np.vstack([final_sub, shift_final])      # (N+17, 6)
emb_final = TSNE(n_components=2, perplexity=PERPLEXITY,
                 random_state=RANDOM_SEED, n_jobs=-1).fit_transform(combined_final)
ds_final       = emb_final[:n_ds]
shift_emb_final = emb_final[n_ds:]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle(
    f"t-SNE coverage  |  dataset: {N_total:,} trajectories  "
    f"(subset {N_TSNE:,})  vs  17 shift experiments",
    fontsize=10,
)

cmap   = cm.coolwarm
colors = [cmap(i / (len(DEFAULT_SHIFTS) - 1)) for i in range(len(DEFAULT_SHIFTS))]

for ax, ds_pts, sh_pts, title in [
    (axes[0], ds_init,  shift_emb_init,  "Initial states  (r, v, m)  7D → 2D"),
    (axes[1], ds_final, shift_emb_final, "Final states  (r, v)  6D → 2D"),
]:
    ax.scatter(ds_pts[:, 0], ds_pts[:, 1],
               s=2, alpha=0.25, color="#888888", linewidths=0, label="Dataset")
    for i, (pt, shift) in enumerate(zip(sh_pts, DEFAULT_SHIFTS)):
        ax.scatter(pt[0], pt[1], marker="x", s=60, linewidths=1.5,
                   color=colors[i], zorder=5,
                   label=f"{shift:+d}d" if i == 0 else f"{shift:+d}d")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("t-SNE 1", fontsize=8); ax.set_ylabel("t-SNE 2", fontsize=8)

# Single legend for shift crosses (right panel)
handles = [
    plt.Line2D([0], [0], marker="x", color=colors[i], markersize=6,
               linewidth=0, markeredgewidth=1.5, label=f"{s:+d}d")
    for i, s in enumerate(DEFAULT_SHIFTS)
]
handles.insert(0, plt.Line2D([0], [0], marker="o", color="#888888",
                              markersize=4, linewidth=0, alpha=0.5, label="Dataset"))
axes[1].legend(handles=handles, fontsize=6.5, ncol=2,
               loc="lower right", framealpha=0.8, title="Shift", title_fontsize=7)

fig.tight_layout()
out = Path(__file__).parent / "tsne_coverage.png"
fig.savefig(str(out), dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")
print(f"\nDataset size: {N_total:,} trajectories  ({len(chunks)} chunks × {N_total//len(chunks):,})")
