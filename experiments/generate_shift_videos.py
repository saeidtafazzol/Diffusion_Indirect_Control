"""generate_shift_videos.py
==========================
Batch-generate diffusion+CasADi videos for every successful time-shift case.

Earth/Mars states are propagated by Δt (Keplerian two-body, no thrust) to
produce shifted boundary conditions.  For each shift the pipeline is identical
to generate_video_refined.py:

  Phase 1 — Diffusion denoising  (frame-captured every --diffusion-capture-every steps)
  Phase 2 — CasADi IPOPT refinement (frame-captured every --casadi-stride iters)

Videos are saved under --output-dir / shift_<+XXX>d.mp4.

Usage
-----
python generate_shift_videos.py \\
    --checkpoint checkpoints/indiff_ctrl_latest.msgpack \\
    [--shifts-days -700 -600 -300 -200 -100 -50 -40 -30 -20 -10 \\
                   10  20  30  40  50 100 500 600 700] \\
    [--seed 0] \\
    [--num-diffusion-steps 30] \\
    [--diffusion-capture-every 1] \\
    [--casadi-max-iter 150] \\
    [--casadi-stride 2] \\
    [--fps 12] \\
    [--dpi 110] \\
    [--output-dir shift_videos]
"""
import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(1, str(_root / "experiments"))
del _root


import argparse
import sys
import time
import types
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

import jax
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from core import build_normalization, make_dynamics, STATE_DIM, COSTATE_DIM
from casadi_bvp_refine import BVPRefiner

# Import video-building helpers from generate_video_refined
from generate_video_refined import (
    load_checkpoint as _load_checkpoint_video,
    run_diffusion_loop,
    build_animation,
)


# ─────────────────────────────────────────────────────────────────────────────
# Keplerian propagation (mirrors time_shift_convergence_study.py)
# ─────────────────────────────────────────────────────────────────────────────

def _kepler_rhs(t, y, mu):
    r = y[:3]; v = y[3:6]
    return np.concatenate([v, -mu * r / np.linalg.norm(r) ** 3])


def propagate_kepler(r, v, mu, dt):
    if dt == 0.0:
        return np.array(r, dtype=np.float64), np.array(v, dtype=np.float64)
    y0 = np.concatenate([np.array(r, np.float64), np.array(v, np.float64)])
    sol = solve_ivp(_kepler_rhs, (0.0, dt), y0, args=(mu,),
                     method="DOP853", rtol=1e-12, atol=1e-13)
    yf = sol.y[:, -1]
    return yf[:3], yf[3:]


def shifted_bcs(norm, shift_days):
    shift_tu = shift_days * 24.0 * 3600.0 / norm["TU"]
    mu = float(norm["mu"])
    r_i, v_i = propagate_kepler(norm["r_i"], norm["v_i"], mu, shift_tu)
    r_f, v_f = propagate_kepler(norm["r_f"], norm["v_f"], mu, shift_tu)
    m0 = float(norm["m0"])
    return (
        np.concatenate([r_i, v_i, [m0]]).astype(np.float32),
        np.concatenate([r_f, v_f, [m0]]).astype(np.float32),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Minimal args namespace expected by generate_video_refined.load_checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def _video_args_ns(**overrides):
    defaults = dict(
        no_integration_signals=False,
        embd_dim=512,
        num_layers=12,
        num_heads=4,
        eps=1e-4,
        inference_rtol=1e-7,
        inference_atol=1e-9,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path,
                   default=Path("checkpoints/indiff_ctrl_latest.msgpack"))
    p.add_argument("--shifts-days", type=float, nargs="+",
                   default=[-700, -600, -300, -200, -100,
                            -50, -40, -30, -20, -10,
                            10, 20, 30, 40, 50,
                            100, 500, 600, 700],
                   help="Time shifts (days) to generate videos for")
    p.add_argument("--seed",                   type=int, default=0)
    p.add_argument("--num-diffusion-steps",    type=int, default=30)
    p.add_argument("--diffusion-capture-every",type=int, default=1)
    p.add_argument("--casadi-max-iter",        type=int, default=150)
    p.add_argument("--casadi-stride",          type=int, default=2)
    p.add_argument("--casadi-rk4-steps",       type=int, default=8)
    p.add_argument("--fps",                    type=int, default=12)
    p.add_argument("--dpi",                    type=int, default=110)
    p.add_argument("--output-dir", type=Path,  default=Path("shift_videos"))
    return p.parse_args()


def main():
    args = parse_args()

    if not args.checkpoint.exists():
        sys.exit(f"Checkpoint not found: {args.checkpoint}")

    # ── Load checkpoint once ──────────────────────────────────────────────────
    print("Loading checkpoint …")
    ckpt_args = _video_args_ns()
    params, policy, time_grid, norm = _load_checkpoint_video(args.checkpoint, ckpt_args)
    n_points = len(time_grid)

    # ── Build BVPRefiner once (t_f fixed, shared across all shifts) ───────────
    refiner = BVPRefiner(n_points=n_points, n_rk4_steps=args.casadi_rk4_steps,
                          ipopt_verbosity=0)
    print(f"BVPRefiner: n_points={n_points}  t_f={norm['t_f']:.4f}")
    print(f"\nJAX devices: {jax.devices()}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shifts = args.shifts_days
    n_total = len(shifts)
    t_wall0 = time.perf_counter()

    for si, shift in enumerate(shifts):
        tag = f"{shift:+.0f}d"
        out_path = args.output_dir / f"shift_{tag}.mp4"
        print(f"\n{'='*70}")
        print(f"[{si+1}/{n_total}]  shift = {tag}  →  {out_path}")
        print(f"{'='*70}")

        # ── Shifted boundary conditions ───────────────────────────────────────
        initial_state, final_state = shifted_bcs(norm, shift)
        sample = {"initial_state": initial_state, "final_state": final_state}
        print(f"  r_i={initial_state[:3].round(4)}  v_i={initial_state[3:6].round(4)}")
        print(f"  r_f={final_state[:3].round(4)}  v_f={final_state[3:6].round(4)}")

        # Modified norm for visualization (Earth/Mars markers at shifted positions)
        norm_vis = dict(norm)
        norm_vis["r_i"] = initial_state[:3].astype(np.float64)
        norm_vis["r_f"] = final_state[:3].astype(np.float64)

        # ── Phase 1: diffusion ────────────────────────────────────────────────
        t0 = time.perf_counter()
        diff_frames = run_diffusion_loop(
            policy=policy, params=params, sample=sample,
            num_steps=args.num_diffusion_steps,
            capture_every=args.diffusion_capture_every,
            rng_seed=args.seed,
        )
        t1 = time.perf_counter()
        print(f"  Phase 1: {len(diff_frames)} frames  ({t1-t0:.1f}s)")

        # ── Phase 2: CasADi BVP refinement with frame capture ─────────────────
        z_init = diff_frames[-1].astype(np.float64)
        init_res = refiner.continuity_residuals(z_init).max()
        print(f"  Initial continuity residual: {init_res:.4e}")
        t0 = time.perf_counter()
        cas_frames = refiner.solve_iterative(
            z_init=z_init,
            initial_state=initial_state.astype(np.float64),
            final_state=final_state.astype(np.float64),
            max_iter=args.casadi_max_iter,
            stride=args.casadi_stride,
        )
        t1 = time.perf_counter()
        print(f"  Phase 2: {len(cas_frames)} frames  ({t1-t0:.1f}s)")

        # ── Build & save animation ────────────────────────────────────────────
        total_frames = len(diff_frames) + len(cas_frames)
        print(f"  Encoding {total_frames} frames → {out_path} …")
        t0 = time.perf_counter()
        anim = build_animation(
            diffusion_frames=diff_frames,
            casadi_frames=cas_frames,
            sample=sample,
            refiner=refiner,
            norm=norm_vis,
            time_grid=time_grid,
            fps=args.fps,
            dpi=args.dpi,
        )
        writer = animation.FFMpegWriter(
            fps=args.fps, bitrate=2000, codec="mpeg4",
            extra_args=["-q:v", "4"],
            metadata={"title": f"Diffusion+CasADi  shift={tag}"},
        )
        anim.save(str(out_path), writer=writer, dpi=args.dpi)
        plt.close("all")
        t1 = time.perf_counter()
        print(f"  Saved ({t1-t0:.1f}s encode)")

    t_total = time.perf_counter() - t_wall0
    print(f"\n{'='*70}")
    print(f"All {n_total} videos saved to {args.output_dir}/")
    print(f"Total time: {t_total/60:.1f} min")


if __name__ == "__main__":
    main()
