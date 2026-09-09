"""
Box plots of wall-clock time per shift for converged trials only.

Method A = DBIC + IPOPT    (diffusion warm-start)
Method B = IPOPT Baseline  (ε-continuation, no diffusion)

Only rows where A_ok / B_ok == 'OK' are included.
All shifts in one figure, ordered by shift value (−700 → +700).
"""

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Locate study dir ──────────────────────────────────────────────────────────
_root = Path(__file__).resolve().parent.parent
study_dir = _root / "shift_eps_study_200"

# ── Parse detail.txt files ────────────────────────────────────────────────────
ROW_RE = re.compile(
    r"^\s*\d+\s+"          # seed
    r"(OK|--)\s+"          # A_ok
    r"[\d.e+\-]+\s+"       # A_res
    r"([\d.]+)\s+"         # A_t(s)  ← group 2
    r"(OK|--)\s+"          # B_ok    ← group 3
    r"[\d.e+\-]+\s+"       # B_res
    r"([\d.]+)"            # B_t(s)  ← group 4
)

shift_data = {}   # int(days) → {"A": [...], "B": [...]}

for detail in sorted(study_dir.glob("shift_*/detail.txt")):
    # extract signed int from folder name, e.g. "shift_+400d" → 400
    m = re.search(r"shift_([+-]?\d+)d", detail.parent.name)
    if not m:
        continue
    shift = int(m.group(1))
    times_A, times_B = [], []
    for line in detail.read_text().splitlines():
        rm = ROW_RE.match(line)
        if not rm:
            continue
        a_ok, a_t, b_ok, b_t = rm.group(1), float(rm.group(2)), rm.group(3), float(rm.group(4))
        if a_ok == "OK":
            times_A.append(a_t)
        if b_ok == "OK":
            times_B.append(b_t)
    shift_data[shift] = {"A": times_A, "B": times_B}
    print(f"shift {shift:+4d}d  A_conv={len(times_A):3d}  B_conv={len(times_B):3d}")

shifts_sorted = sorted(shift_data.keys())
n = len(shifts_sorted)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 5))

COLOR_A = "#2980B9"   # blue  — DBIC + IPOPT
COLOR_B = "#E74C3C"   # red   — IPOPT Baseline

W = 0.30   # box half-width in x-units
GAP = 0.20 # gap between A and B boxes within one shift

positions_A, positions_B = [], []
data_A, data_B = [], []

for i, shift in enumerate(shifts_sorted):
    x = i + 1
    positions_A.append(x - GAP)
    positions_B.append(x + GAP)
    data_A.append(shift_data[shift]["A"])
    data_B.append(shift_data[shift]["B"])

def _bp(ax, data, positions, color, label):
    # filter out empty lists
    valid_pos  = [p for p, d in zip(positions, data) if d]
    valid_data = [d for d in data if d]
    if not valid_data:
        return
    bp = ax.boxplot(
        valid_data,
        positions=valid_pos,
        widths=W,
        patch_artist=True,
        notch=False,
        showfliers=True,
        flierprops=dict(marker=".", markersize=4, alpha=0.5, markerfacecolor=color,
                        markeredgecolor=color),
        medianprops=dict(color="white", linewidth=2),
        boxprops=dict(facecolor=color, alpha=0.75, linewidth=0),
        whiskerprops=dict(color=color, linewidth=1.2),
        capprops=dict(color=color, linewidth=1.5),
    )
    # invisible handle for legend
    ax.plot([], [], color=color, lw=6, alpha=0.75, label=label)

_bp(ax, data_A, positions_A, COLOR_A, "DBIC + IPOPT")
_bp(ax, data_B, positions_B, COLOR_B, "IPOPT Baseline")

ax.set_xticks(range(1, n + 1))
ax.set_xticklabels([f"{s:+d}d" for s in shifts_sorted], fontsize=9, rotation=45, ha="right")
ax.set_ylabel("Wall-clock time (s)", fontsize=12)
ax.set_xlabel("Time shift", fontsize=12)
ax.set_title("Runtime for converged trials — DBIC + IPOPT vs. IPOPT Baseline", fontsize=13)
ax.legend(fontsize=11, loc="upper left")
ax.grid(axis="y", alpha=0.3)
ax.set_xlim(0.5, n + 0.5)

fig.tight_layout()

out_dir = _root / "orbit_plots"
out_dir.mkdir(exist_ok=True)
for ext in ("png", "eps"):
    p = out_dir / f"runtime_boxplots.{ext}"
    fig.savefig(str(p), dpi=150, bbox_inches="tight")
    print(f"saved → {p}")
plt.close(fig)
