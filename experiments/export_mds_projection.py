"""
Extract the 13×2 linear projection matrix from classical MDS (= PCA).

Classical MDS on Euclidean distances is equivalent to PCA on the centered
data matrix.  We replicate the exact data pipeline from dataset_coverage.py:
  - 8000-point random subset of the training dataset   (13D: r0,v0,m0,rf,vf)
  - 17 shift-experiment boundary conditions
  - stacked → (8017, 13) combined matrix
  - SVD on centred combined matrix → W = Vt[:2,:].T  (13×2)
  - (x - mu) @ W  recovers the 2-D MDS embedding for any new 13D vector
"""
import sys, colorsys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
from core import build_normalization

DATASET_DIR = _root / "earth_mars_minfuel_posvel_constrained_32pts"
N_SUBSET    = 8000
RANDOM_SEED = 42

_SHIFTS_RAW = [
    -700, -600, -500, -400, -300, -200, -100, -50,
       0,   50,  100,  200,  300,  400,  500,  600,  700,
]
def _syn(s): return s + 780 if s < 0 else s
SHIFTS = sorted(_SHIFTS_RAW, key=_syn)

# ── Load dataset ──────────────────────────────────────────────────────────────
chunks = sorted(DATASET_DIR.glob("chunk_*.npz"))
print(f"Loading {len(chunks)} chunks …")
all_init, all_final = [], []
for ch in chunks:
    c = np.load(ch)
    all_init.append(c["initial_states"])
    all_final.append(c["final_states"])

init_all  = np.concatenate(all_init,  axis=0).astype(np.float64)
final_all = np.concatenate(all_final, axis=0).astype(np.float64)

rng = np.random.default_rng(RANDOM_SEED)
idx = rng.choice(len(init_all), size=min(N_SUBSET, len(init_all)), replace=False)
init_sub  = init_all[idx]         # (8000, 7)
final_sub = final_all[idx, :6]    # (8000, 6) — drop mass column

# ── Shift experiment boundary conditions ─────────────────────────────────────
norm = build_normalization()
TU, mu = float(norm["TU"]), float(norm["mu"])
r_i, v_i = np.asarray(norm["r_i"]), np.asarray(norm["v_i"])
r_f, v_f = np.asarray(norm["r_f"]), np.asarray(norm["v_f"])
m0 = float(norm["m0"])

def _rhs(t, y):
    r, v = y[:3], y[3:]
    return np.concatenate([v, -mu / np.linalg.norm(r)**3 * r])

def propagate(r0, v0, dt_norm):
    if abs(dt_norm) < 1e-12:
        return r0.copy(), v0.copy()
    sol = solve_ivp(_rhs, (0, dt_norm), np.concatenate([r0, v0]),
                    method="DOP853", rtol=1e-12, atol=1e-13)
    return sol.y[:3, -1], sol.y[3:6, -1]

shift_init, shift_final = [], []
for s in SHIFTS:
    dt = s * 86400.0 / TU
    ri, vi = propagate(r_i, v_i, dt)
    rf, vf = propagate(r_f, v_f, dt)
    shift_init.append(np.concatenate([ri, vi, [m0]]))
    shift_final.append(np.concatenate([rf, vf]))

shift_13 = np.hstack([np.array(shift_init), np.array(shift_final)])  # (17, 13)
ds_13    = np.hstack([init_sub, final_sub])                           # (8000, 13)
combined = np.vstack([ds_13, shift_13])                               # (8017, 13)

print(f"Combined matrix: {combined.shape}")

# ── Classical MDS = PCA ───────────────────────────────────────────────────────
# Use sklearn PCA so the projection is byte-for-byte identical to the
# classical_mds_coverage plot produced by dataset_coverage.py
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=RANDOM_SEED)
emb = pca.fit_transform(combined)   # (8017, 2)  — same as the plot

# W maps centred x → 2D:  (x - mu) @ W
W    = pca.components_.T   # (13, 2)
mu_x = pca.mean_           # (13,)

print("Projection matrix W shape:", W.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_.round(4))
print("Embedding sample — first 3 rows:\n", emb[:3].round(4))

# ── Write LaTeX ───────────────────────────────────────────────────────────────
row_labels = [
    r"r_x^0", r"r_y^0", r"r_z^0",
    r"v_x^0", r"v_y^0", r"v_z^0",
    r"m_0",
    r"r_x^f", r"r_y^f", r"r_z^f",
    r"v_x^f", r"v_y^f", r"v_z^f",
]

lines = []
lines.append(r"% 13×2 classical MDS (= PCA) projection matrix W")
lines.append(r"% Fitted on: 8000-point training subset + 17 shift boundary conditions (13D each)")
lines.append(r"% Usage: \hat{y} = (x - \mu) \mathbf{W},  x \in \mathbb{R}^{13},  \hat{y} \in \mathbb{R}^2")
lines.append(r"\[")
lines.append(r"  \mathbf{W} \;=\;")
lines.append(r"  \begin{array}{r|rr}")
lines.append(r"    & d_1 & d_2 \\ \hline")
for lbl, (w1, w2) in zip(row_labels, W):
    lines.append(f"    ${lbl}$ & {w1:9.5f} & {w2:9.5f} \\\\")
lines.append(r"  \end{array}")
lines.append(r"\]")

out = _root / "Indirect_Diffusion_Control" / "figures" / "mds_matrix.tex"
out.write_text("\n".join(lines) + "\n")
print(f"\nsaved → {out}")
print("\n" + "\n".join(lines))
