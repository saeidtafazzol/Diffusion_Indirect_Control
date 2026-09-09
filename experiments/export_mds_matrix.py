"""
Compute MDS on the 17 shift boundary conditions (13D) and write
the resulting 17×2 coordinate matrix as a LaTeX file.
"""
import sys, colorsys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.manifold import MDS

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
from core import build_normalization

RANDOM_SEED = 42

_SHIFTS_RAW = [
    -700, -600, -500, -400, -300, -200, -100, -50,
       0,   50,  100,  200,  300,  400,  500,  600,  700,
]

def _syn(s):
    return s + 780 if s < 0 else s

SHIFTS = sorted(_SHIFTS_RAW, key=_syn)

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

rows = []
for s in SHIFTS:
    dt = s * 86400.0 / TU
    ri, vi = propagate(r_i, v_i, dt)
    rf, vf = propagate(r_f, v_f, dt)
    rows.append(np.concatenate([ri, vi, [m0], rf, vf]))

X = np.array(rows, dtype=np.float32)   # (17, 13)
print(f"Computing MDS on {X.shape[0]}×{X.shape[1]} matrix …")
emb = MDS(n_components=2, metric=True, normalized_stress=False,
          random_state=RANDOM_SEED, n_jobs=-1).fit_transform(X)
print("done")

# ── Write LaTeX ───────────────────────────────────────────────────────────────
# Column headers: r_x r_y r_z  v_x v_y v_z  m   r_fx r_fy r_fz  v_fx v_fy v_fz
col_headers = [
    r"r_x^0", r"r_y^0", r"r_z^0",
    r"v_x^0", r"v_y^0", r"v_z^0",
    r"m_0",
    r"r_x^f", r"r_y^f", r"r_z^f",
    r"v_x^f", r"v_y^f", r"v_z^f",
]
n_col = len(col_headers)

lines = []
lines.append(r"% Original 17×13 boundary-condition matrix fed into MDS")
lines.append(r"% Each row is one shift; columns are (r^0, v^0, m_0, r^f, v^f) normalised.")
lines.append(r"\[")
lines.append(r"  \mathbf{X} \;=\;")
lines.append(r"  \begin{array}{r|" + "r" * n_col + "}")
header = r"    \Delta t\,(\text{d}) & " + " & ".join(f"${h}$" for h in col_headers) + r" \\ \hline"
lines.append(header)
for s, row in zip(SHIFTS, X):
    label = _syn(s)
    vals = " & ".join(f"{v:7.4f}" for v in row)
    lines.append(f"    {label:4d} & {vals} \\\\")
lines.append(r"  \end{array}")
lines.append(r"\]")

out = _root / "Indirect_Diffusion_Control" / "figures" / "mds_matrix.tex"
out.write_text("\n".join(lines) + "\n")
print(f"saved → {out}")

# Also print for inspection
print("\n" + "\n".join(lines))
