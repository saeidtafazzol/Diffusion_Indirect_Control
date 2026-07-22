#!/usr/bin/env python3
"""
generate_video.py  —  Visualise the DDPM denoising process as an MP4.

Forces JAX to CPU (JAX_PLATFORMS=cpu set before any JAX import) so the GPU
stays free for training.

Usage
-----
    python generate_video.py
    python generate_video.py --checkpoint-path checkpoints/indiff_ctrl_latest.msgpack \
                             --output diffusion.mp4 --num-inference-steps 100 --fps 20

Requirements: ffmpeg must be on PATH (apt install ffmpeg).
"""
import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(1, str(_root / "experiments"))
del _root

import os

# ── Force CPU before any JAX import ─────────────────────────────────────────
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from flax import serialization

from core import COSTATE_DIM, STATE_DIM, build_normalization, build_rhs
from diffrax import Dopri5, ODETerm, PIDController, RESULTS, SaveAt, diffeqsolve
from jax_ddpm_scheduler import JaxDDPMScheduler
from policy import IndiffCtrlPolicy
from transformer_diffusion_model import DiffusionTransformer, DiffusionTransformerConfig, final_lambda_m_costate_mask, default_state_known_mask_no_final_mass


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Generate DDPM denoising video on CPU.")
    p.add_argument("--checkpoint-path", type=Path,
                   default=Path(__file__).resolve().parent / "checkpoints" / "indiff_ctrl_latest.msgpack")
    p.add_argument("--output", type=Path,
                   default=Path(__file__).resolve().parent / "diffusion_video.mp4")
    p.add_argument("--num-inference-steps", type=int, default=100)
    p.add_argument("--fps",         type=int,   default=5)
    p.add_argument("--hold-frames", type=int,   default=3,
                   help="Repeat each denoising frame N times to slow the video.")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--num-points",  type=int,   default=32,
                   help="Fallback seq_len if checkpoint peek fails.")
    p.add_argument("--initial-mass", type=float, default=None,
                   help="Normalized initial spacecraft mass m(t0).  "
                        "Defaults to norm['m0']=1.0 (full tank).  "
                        "Ignored when --dataset-index is set.")
    p.add_argument("--final-mass", type=float, default=0.7,
                   help="Normalized terminal spacecraft mass m(t_f).  "
                        "Must be < initial-mass (fuel is burned).  "
                        "Training data: final mass ∈ [0.20, 1.00], mean ≈ 0.675. "
                        "Setting this to initial-mass (= no fuel burned) is out-of-"
                        "distribution and will corrupt costate predictions.  "
                        "Ignored when --dataset-index is set.")
    # Model config — must match the checkpoint
    p.add_argument("--embd-dim",     type=int,   default=512)
    p.add_argument("--num-layers",   type=int,   default=12)
    p.add_argument("--num-heads",    type=int,   default=4)
    p.add_argument("--mlp-ratio",    type=int,   default=4)
    p.add_argument("--eps",          type=float, default=1e-4)
    p.add_argument("--segment-rtol", type=float, default=1e-7)
    p.add_argument("--segment-atol", type=float, default=1e-9)
    # Dataset sample conditioning
    p.add_argument("--dataset-dir", type=Path,
                   default=Path(__file__).resolve().parent / "earth_mars_minfuel_dataset_32pts",
                   help="Path to the chunked dataset directory (contains metadata.npz + chunk_*.npz).")
    p.add_argument("--dataset-index", type=int, default=None,
                   help="If given, use this sample index from the dataset as the initial/final state "
                        "instead of the hard-coded Earth–Mars endpoints.  Also stores the ground-truth "
                        "trajectory for the title label.")
    return p.parse_args()


# ── Dataset sample loader ────────────────────────────────────────────────────
def _load_dataset_sample(dataset_dir: Path, index: int):
    """
    Load a single sample (by global index) from the chunked dataset.

    Returns
    -------
    initial_state  : np.ndarray  shape (7,)  float32  [r(3), v(3), m]
    final_state    : np.ndarray  shape (7,)  float32
    gt_states      : np.ndarray  shape (T, 7) float32  ground-truth state trajectory
    gt_costates    : np.ndarray  shape (T, 7) float32  ground-truth costate trajectory
    time_grid      : np.ndarray  shape (T,)  float32  or None
    """
    metadata = np.load(dataset_dir / "metadata.npz")
    chunk_size  = int(metadata["chunk_size"].item())
    num_samples = int(metadata["num_samples"].item())
    time_grid   = metadata["time_grid"].astype(np.float32) if "time_grid" in metadata else None

    if index < 0 or index >= num_samples:
        raise ValueError(f"--dataset-index {index} is out of range [0, {num_samples}).")

    chunk_idx  = index // chunk_size
    local_idx  = index %  chunk_size
    chunk_paths = sorted(dataset_dir.glob("chunk_*.npz"))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunk_*.npz files found in {dataset_dir}")
    chunk = np.load(chunk_paths[chunk_idx])

    initial_state = chunk["initial_states"][local_idx].astype(np.float32)
    final_state   = chunk["final_states"][local_idx].astype(np.float32)
    gt_states     = chunk["states"][local_idx].astype(np.float32)
    gt_costates   = chunk["costates"][local_idx].astype(np.float32)
    return initial_state, final_state, gt_states, gt_costates, time_grid


# ── Checkpoint peek (avoids needing a full target for just metadata) ─────────
def _peek_checkpoint(raw_bytes):
    """Return (time_grid: np.ndarray | None, scheduler_config: dict | None)."""
    try:
        # flax >= 0.6
        from flax.serialization import msgpack_restore
    except ImportError:
        try:
            from flax.serialization import _msgpack_restore as msgpack_restore
        except ImportError:
            return None, None

    def _s(v):
        return v.decode() if isinstance(v, bytes) else v

    try:
        raw = msgpack_restore(raw_bytes)
        tg  = raw.get("time_grid", None)
        sc  = raw.get("scheduler_config", None)
        if tg is not None:
            tg = np.asarray(tg, dtype=np.float32).ravel()
        if sc is not None:
            sc = {_s(k): (_s(v) if isinstance(v, bytes) else v) for k, v in sc.items()}
        return tg, sc
    except Exception as exc:
        print(f"  [warn] checkpoint peek failed ({exc}); using defaults.")
        return None, None


# ── Physical derived quantities ──────────────────────────────────────────────
def compute_quantities(states, costates, norm, eps):
    """
    states   : (T, 7)  float32  [r(3), v(3), m]
    costates : (T, 7)  float32  [λ_r(3), λ_v(3), λ_m]
    Returns a dict with both scalar (T,) and vector (T,3) arrays.
    """
    r  = states[:, 0:3]
    v  = states[:, 3:6]
    m  = states[:, 6]
    lr = costates[:, 0:3]   # λ_r  (T, 3)
    lv = costates[:, 3:6]   # λ_v  (T, 3)
    lm = costates[:, 6]     # λ_m  (T,)

    speed   = np.linalg.norm(v,  axis=-1)
    lv_norm = np.linalg.norm(lv, axis=-1)

    # Switching function: S = c·|λ_v|/m + λ_m − 1
    # ε-smoothed optimal thrust fraction (matches dynamics.py exactly):
    #   δ = (1 + S / sqrt(S² + ε)) / 2
    S = norm["c_norm"] * lv_norm / np.maximum(m, 1e-12) + lm - 1.0
    thrust_frac  = (1.0 + S / np.sqrt(S**2 + eps)) * 0.5
    thrust       = norm["t_max_norm"] * thrust_frac
    thrust_alpha = -lv / (lv_norm[:, None] + 1e-30)  # unit thrust direction

    return dict(
        r=r, v=v, speed=speed, mass=m, thrust=thrust,
        thrust_frac=thrust_frac, thrust_alpha=thrust_alpha,
        lr=lr, lv=lv, lm=lm,
    )


# ── Per-segment arc integrator (mirrors train.py make_segment_arc_integrator) ──
def make_segment_arc_integrator(policy, arc_points, rtol, atol):
    segment_term       = ODETerm(build_rhs(policy.jdy))
    segment_solver     = Dopri5()
    segment_controller = PIDController(rtol=rtol, atol=atol)
    segment_ts_base    = jnp.linspace(0.0, 1.0, arc_points, dtype=jnp.float32)

    def single(y0, costate, dt):
        augmented_y0 = jnp.concatenate([y0, costate], axis=0)
        ts  = segment_ts_base * dt
        sol = diffeqsolve(
            segment_term, segment_solver,
            t0=0.0, t1=dt, dt0=None,
            y0=augmented_y0, args=None,
            stepsize_controller=segment_controller,
            saveat=SaveAt(ts=ts), throw=False,
            max_steps=512,
        )
        success = jnp.asarray(sol.result == RESULTS.successful, dtype=bool)
        return jnp.asarray(sol.ys, dtype=jnp.float32)[:, :STATE_DIM], success

    return jax.jit(jax.vmap(single))


# ── Denoising loop that captures every intermediate step ─────────────────────
def collect_frames(policy, params, initial_state, final_state,
                   num_inference_steps, rng_key, segment_arc_integrator=None):
    cfg = policy.model.config
    skm, ckm = policy._resolve_known_masks(
        1, cfg.seq_len,
        state_known_mask=default_state_known_mask_no_final_mass(1, cfg.seq_len),
        costate_known_mask=final_lambda_m_costate_mask(1, cfg.seq_len),
    )
    segment_dt = getattr(policy, "segment_dt", None)

    # Conditioned fixed points  [Earth start, Mars end]
    cst8 = np.zeros((1, cfg.seq_len, STATE_DIM),   np.float32)
    ccs  = np.zeros((1, cfg.seq_len, COSTATE_DIM), np.float32)
    cst8[0, 0, :]  = initial_state
    cst8[0, -1, :] = final_state

    cond = jnp.asarray(np.concatenate([cst8, ccs], axis=-1), dtype=jnp.float32)
    mask = policy._build_condition_mask(skm, ckm)

    rng_key, sk = jax.random.split(rng_key)
    traj = policy.noise_scheduler.sample_noise(sk, cond.shape, dtype=jnp.float32)
    policy.noise_scheduler.set_timesteps(num_inference_steps,power=2.7)

    frames = []
    total  = len(policy.noise_scheduler.timesteps)
    for i, ts in enumerate(policy.noise_scheduler.timesteps):
        traj = jnp.where(mask, cond, traj)
        st   = traj[..., :STATE_DIM]
        cst  = traj[..., STATE_DIM:STATE_DIM + COSTATE_DIM]

        int_st = int_cst = int_fail = None
        if cfg.use_integration_signals:
            int_st, int_cst, int_fail = policy.compute_integration_signals(st, cst)

        out = policy.model.apply(
            {"params": params},
            noisy_states=st, noisy_costates=cst,
            diffusion_steps=jnp.full((1,), int(ts), dtype=jnp.int32),
            state_known_mask=skm, costate_known_mask=ckm,
            integrated_states=int_st, integrated_costates=int_cst,
            integration_failed=int_fail, train=False,
        )

        model_out = policy._combine_trajectory(out["state_eps"], out["costate_eps"])
        rng_key, step_key = jax.random.split(rng_key)
        traj = policy.noise_scheduler.step(
            model_output=model_out, timestep=int(ts),
            sample=traj, rng_key=step_key,
        ).prev_sample
        traj = jnp.where(mask, cond, traj)  # re-pin after step (matches predict_trajectory)

        arr = np.asarray(traj[0], dtype=np.float32)
        frame = {
            "states":   arr[:, :STATE_DIM],
            "costates": arr[:, STATE_DIM:STATE_DIM + COSTATE_DIM],
            "t_noise":  int(ts),
            "step_idx": i,
        }
        if segment_arc_integrator is not None and segment_dt is not None:
            _arcs, _ok = segment_arc_integrator(
                jnp.asarray(arr[:cfg.seq_len - 1, :STATE_DIM]),
                jnp.asarray(arr[:cfg.seq_len - 1, STATE_DIM:STATE_DIM + COSTATE_DIM]),
                jnp.asarray(segment_dt, dtype=jnp.float32),
            )
            frame["arcs"]        = np.asarray(_arcs, dtype=np.float32)
            frame["arc_success"] = np.asarray(_ok,   dtype=bool)
        frames.append(frame)
        print(f"\r  step {i+1:3d}/{total}  t_noise={int(ts):4d}", end="", flush=True)
    print()
    return frames


# ── Video assembly ────────────────────────────────────────────────────────────
def build_video(frames, norm, eps, time_grid, output_path, fps, hold_frames=3,
                gt_states=None, gt_costates=None, dataset_index=None):
    # Expand each denoising frame by repeating it `hold_frames` times so the
    # video is easier to scrub through.
    expanded = [f for f in frames for _ in range(hold_frames)]
    n        = len(expanded)

    all_d = [compute_quantities(f["states"], f["costates"], norm, eps)
             for f in expanded]

    def _ylim(key):
        vals = np.stack([d[key] for d in all_d])
        lo, hi = float(vals.min()), float(vals.max())
        pad = max((hi - lo) * 0.06, 1e-6)
        return lo - pad, hi + pad

    def _ylim2d(key):   # for (T,3) arrays
        vals = np.stack([d[key] for d in all_d])
        lo, hi = float(vals.min()), float(vals.max())
        pad = max((hi - lo) * 0.06, 1e-6)
        return lo - pad, hi + pad

    sp_lim  = (0.0, max(float(np.stack([d["speed"]  for d in all_d]).max()), 1e-12) * 1.1)
    th_lim  = (0.0, 1.5)           # fixed: thrust fraction δ ∈ [0,1], headroom to 1.5
    m_lim   = (0.0, 1.0)
    r_ylim  = _ylim2d("r")
    v_ylim  = _ylim2d("v")
    lr_ylim = _ylim2d("lr")
    lv_ylim = _ylim2d("lv")
    lm_ylim = (-0.2, 1.2)          # fixed: λ_m transversality range

    t  = time_grid
    ri = norm["r_i"]
    rf = norm["r_f"]

    # ── Layout ────────────────────────────────────────────────────────────────
    # Row 0: 3D (spans 3 rows) | r xyz | v xyz | mass+thrust
    # Row 1:                   | λ_r xyz | λ_v xyz | λ_m
    fig = plt.figure(figsize=(22, 10))
    gs  = gridspec.GridSpec(2, 4, figure=fig, wspace=0.38, hspace=0.50)

    ax3d  = fig.add_subplot(gs[:, 0], projection="3d")
    ax_r  = fig.add_subplot(gs[0, 1])   # r_x r_y r_z
    ax_v  = fig.add_subplot(gs[0, 2])   # v_x v_y v_z
    ax_mt = fig.add_subplot(gs[0, 3])   # mass + thrust (twin axis)
    ax_lr = fig.add_subplot(gs[1, 1])   # λ_rx λ_ry λ_rz
    ax_lv = fig.add_subplot(gs[1, 2])   # λ_vx λ_vy λ_vz
    ax_lm = fig.add_subplot(gs[1, 3])   # λ_m

    xyz_colors = ["tab:red", "tab:green", "tab:blue"]
    xyz_labels = ["x", "y", "z"]

    def _setup(ax, title, yl, ylim):
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("time [TU/2π]", fontsize=7)
        ax.set_ylabel(yl, fontsize=7)
        ax.set_ylim(*ylim)
        ax.set_xlim(float(t[0]), float(t[-1]))
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.25)

    _setup(ax_r,  "Position  r",      "AU",  r_ylim)
    _setup(ax_v,  "Velocity  v",      "AU/TU", v_ylim)
    _setup(ax_lr, "Costate  λ_r",     "",    lr_ylim)
    _setup(ax_lv, "Costate  λ_v",     "",    lv_ylim)
    _setup(ax_lm, "Costate  λ_m",     "",    lm_ylim)

    # mass (left axis) + thrust (right axis) share one panel
    ax_mt.set_title("Mass & Thrust", fontsize=9)
    ax_mt.set_xlabel("time [TU/2π]", fontsize=7)
    ax_mt.set_ylabel("mass (norm)", fontsize=7, color="tab:green")
    ax_mt.set_ylim(*m_lim)
    ax_mt.set_xlim(float(t[0]), float(t[-1]))
    ax_mt.tick_params(labelsize=6, axis="y", labelcolor="tab:green")
    ax_mt.grid(True, alpha=0.25)
    ax_t2 = ax_mt.twinx()
    ax_t2.set_ylabel("thrust fraction δ", fontsize=7, color="tab:red")
    ax_t2.set_ylim(*th_lim)
    ax_t2.tick_params(labelsize=6, axis="y", labelcolor="tab:red")

    # ── 3-D axes ──────────────────────────────────────────────────────────────
    ax3d.set_xlim(-1, 1)
    ax3d.set_ylim(-1, 1)
    ax3d.set_zlim(-1, 1)
    ax3d.set_xlabel("x [AU]", fontsize=7)
    ax3d.set_ylabel("y [AU]", fontsize=7)
    ax3d.set_zlabel("z [AU]", fontsize=7)
    ax3d.tick_params(labelsize=6)
    ax3d.scatter(*ri, c="dodgerblue", s=70,  zorder=5, label="Earth")
    ax3d.scatter(*rf, c="tomato",     s=70,  zorder=5, label="Mars")
    ax3d.scatter(0, 0, 0, c="gold",   s=130, marker="*", zorder=5, label="Sun")
    ax3d.legend(fontsize=6, loc="upper right")

    title_txt = fig.suptitle("", fontsize=11, y=0.998)

    # ── Animated line objects ─────────────────────────────────────────────────
    (l3,) = ax3d.plot([], [], [], c="limegreen", lw=1.5)

    lr_lines = [ax_r.plot([], [], c=c, lw=1.4, label=lb)[0]
                for c, lb in zip(xyz_colors, xyz_labels)]
    ax_r.legend(fontsize=6, loc="upper right")

    lv_lines = [ax_v.plot([], [], c=c, lw=1.4, label=lb)[0]
                for c, lb in zip(xyz_colors, xyz_labels)]
    ax_v.legend(fontsize=6, loc="upper right")

    (lm_,) = ax_mt.plot([], [], c="tab:green", lw=1.4, label="mass")
    (lt,)  = ax_t2.plot( [], [], c="tab:red",   lw=1.4, label="thrust")

    llr_lines = [ax_lr.plot([], [], c=c, lw=1.4, label=f"λ_r{lb}")[0]
                 for c, lb in zip(xyz_colors, xyz_labels)]
    ax_lr.legend(fontsize=6, loc="upper right")

    llv_lines = [ax_lv.plot([], [], c=c, lw=1.4, label=f"λ_v{lb}")[0]
                 for c, lb in zip(xyz_colors, xyz_labels)]
    ax_lv.legend(fontsize=6, loc="upper right")

    (llm,) = ax_lm.plot([], [], c="tab:purple", lw=1.4)
    # ── Ground-truth overlay (static dashed lines, dataset mode only) ─────────
    has_gt = gt_states is not None and gt_costates is not None
    if has_gt:
        gt_q = compute_quantities(gt_states, gt_costates, norm, eps)
        # 3D GT path
        ax3d.plot(gt_q["r"][:, 0], gt_q["r"][:, 1], gt_q["r"][:, 2],
                  c="white", lw=1.0, ls="--", alpha=0.7, label="GT")
        ax3d.legend(fontsize=6, loc="upper right")
        # 2D GT overlays
        for k, c in enumerate(xyz_colors):
            ax_r.plot(t, gt_q["r"][:, k],  c=c, lw=1.0, ls="--", alpha=0.6)
            ax_v.plot(t, gt_q["v"][:, k],  c=c, lw=1.0, ls="--", alpha=0.6)
            ax_lr.plot(t, gt_q["lr"][:, k], c=c, lw=1.0, ls="--", alpha=0.6)
            ax_lv.plot(t, gt_q["lv"][:, k], c=c, lw=1.0, ls="--", alpha=0.6)
        ax_mt.plot(t, gt_q["mass"],         c="tab:green", lw=1.0, ls="--", alpha=0.6)
        ax_t2.plot(t, gt_q["thrust_frac"],  c="tab:red",   lw=1.0, ls="--", alpha=0.6)
        ax_lm.plot(t, gt_q["lm"],           c="tab:purple", lw=1.0, ls="--", alpha=0.6)
        # Expand y-limits to include GT values (static lines don't trigger auto-scale)
        def _expand(ax, lo, hi):
            cur = ax.get_ylim()
            ax.set_ylim(min(cur[0], lo), max(cur[1], hi))
        for k in range(3):
            _expand(ax_r,  gt_q["r"][:, k].min(),  gt_q["r"][:, k].max())
            _expand(ax_v,  gt_q["v"][:, k].min(),  gt_q["v"][:, k].max())
            _expand(ax_lr, gt_q["lr"][:, k].min(), gt_q["lr"][:, k].max())
            _expand(ax_lv, gt_q["lv"][:, k].min(), gt_q["lv"][:, k].max())

    T        = len(time_grid)
    has_arcs = "arcs" in expanded[0]
    arc_lines = [ax3d.plot([], [], [], c="tab:orange", lw=0.8, alpha=0.5)[0]
                 for _ in range(T - 1)]
    quiver_holder = [None]  # mutable ref so update() can remove/re-add each frame

    all_artists = (
        [l3, lm_, lt, llm]
        + lr_lines + lv_lines + llr_lines + llv_lines
        + arc_lines
        + [title_txt]
    )

    def update(fi):
        d = all_d[fi]
        f = expanded[fi]
        r = d["r"]
        l3.set_data(r[:, 0], r[:, 1])
        l3.set_3d_properties(r[:, 2])
        for k, ln in enumerate(lr_lines):
            ln.set_data(t, d["r"][:, k])
        for k, ln in enumerate(lv_lines):
            ln.set_data(t, d["v"][:, k])
        lm_.set_data(t, d["mass"])
        lt.set_data(t,  d["thrust_frac"])
        for k, ln in enumerate(llr_lines):
            ln.set_data(t, d["lr"][:, k])
        for k, ln in enumerate(llv_lines):
            ln.set_data(t, d["lv"][:, k])
        llm.set_data(t, d["lm"])
        # Segment arcs (yellow)
        if has_arcs and "arcs" in f:
            for seg_idx, ln in enumerate(arc_lines):
                if f["arc_success"][seg_idx]:
                    arc_r = f["arcs"][seg_idx, :, :3]
                    ln.set_data(arc_r[:, 0], arc_r[:, 1])
                    ln.set_3d_properties(arc_r[:, 2])
                else:
                    ln.set_data([], [])
                    ln.set_3d_properties([])
        # Thrust direction quivers (red arrows, engine-on waypoints only)
        if quiver_holder[0] is not None:
            quiver_holder[0].remove()
            quiver_holder[0] = None
        on = d["thrust_frac"] > 0.5
        if np.any(on):
            r_on = d["r"][on]
            a_on = d["thrust_alpha"][on]
            quiver_holder[0] = ax3d.quiver(
                r_on[:, 0], r_on[:, 1], r_on[:, 2],
                a_on[:, 0], a_on[:, 1], a_on[:, 2],
                length=0.08, normalize=True, color="tab:red", alpha=0.7,
            )
        title_txt.set_text(
            f"Diffusion denoising"
            + (f"  |  dataset sample #{dataset_index}" if dataset_index is not None else "")
            + ("  (dashed = GT)" if has_gt else "")
            + f"  |  frame {f['step_idx'] + 1}/{len(frames)}"
            f"  |  t_noise = {f['t_noise']}"
        )
        return all_artists

    ani = animation.FuncAnimation(
        fig, update, frames=n,
        interval=max(1, 1000 // fps), blit=False,
    )
    writer = animation.FFMpegWriter(
        fps=fps, codec="libx264", bitrate=6000,
        extra_args=["-pix_fmt", "yuv420p"],
        metadata={"title": "IndiffCtrl diffusion denoising"},
    )
    import shutil
    system_ffmpeg = shutil.which("ffmpeg", path="/usr/bin:/usr/local/bin")
    if system_ffmpeg:
        matplotlib.rcParams["animation.ffmpeg_path"] = system_ffmpeg
    print(f"Using ffmpeg: {matplotlib.rcParams['animation.ffmpeg_path']}")

    print(f"Writing {output_path}  ({n} frames @ {fps} fps  ≈ {n/fps:.1f}s) …")
    ani.save(str(output_path), writer=writer, dpi=120)
    plt.close(fig)
    print("Done.")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    print("JAX devices:", jax.devices())   # should show only CPU

    norm = build_normalization()
    m0   = norm["m0"]

    gt_states = gt_costates = None
    if args.dataset_index is not None:
        print(f"Loading dataset sample {args.dataset_index} from {args.dataset_dir} …")
        initial_state, final_state, gt_states, gt_costates, ds_time_grid = \
            _load_dataset_sample(args.dataset_dir, args.dataset_index)
        print(f"  initial_state = {initial_state}")
        print(f"  final_state   = {final_state}")
        # Dataset time grid takes precedence over checkpoint time grid when available
        if ds_time_grid is not None:
            print(f"  using dataset time grid ({len(ds_time_grid)} pts)")
        if args.initial_mass is not None or args.final_mass != 0.7:
            print(f"  [note] --initial-mass / --final-mass flags are ignored when --dataset-index is set")
    else:
        ds_time_grid = None
        # Hard-coded Earth → Mars endpoints
        m_i = args.initial_mass if args.initial_mass is not None else m0
        m_f = args.final_mass
        if m_f >= m_i:
            print(f"  [warn] --final-mass ({m_f:.3f}) >= --initial-mass ({m_i:.3f}): "
                  f"implies no fuel burned — out-of-distribution, costates will be unreliable.")
        initial_state = np.concatenate([norm["r_i"], norm["v_i"], [m_i]]).astype(np.float32)
        final_state   = np.concatenate([norm["r_f"], norm["v_f"], [m_f]]).astype(np.float32)
        print(f"r_i = {norm['r_i']}  v_i = {norm['v_i']}  m_i = {m_i:.4f}")
        print(f"r_f = {norm['r_f']}  v_f = {norm['v_f']}  m_f = {m_f:.4f}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f"Loading {args.checkpoint_path} …")
    raw_bytes = args.checkpoint_path.read_bytes()

    time_grid_raw, sch_raw = _peek_checkpoint(raw_bytes)

    # Fallbacks (dataset time grid > checkpoint time grid > linspace default)
    if ds_time_grid is not None and ds_time_grid.ndim == 1 and ds_time_grid.shape[0] >= 2:
        time_grid_raw = ds_time_grid
    elif time_grid_raw is None or time_grid_raw.ndim != 1 or time_grid_raw.shape[0] < 2:
        print(f"  using default time grid ({args.num_points} pts, t_f={norm['t_f']:.4f} TU)")
        time_grid_raw = np.linspace(0.0, norm["t_f"], args.num_points, dtype=np.float32)
    if not sch_raw:
        sch_raw = {}

    seq_len = int(time_grid_raw.shape[0])

    def _sc(key, default):
        v = sch_raw.get(key, default)
        return v if v is not None else default

    noise_scheduler = JaxDDPMScheduler(
        num_train_timesteps=int(_sc("num_train_timesteps", 5000)),
        beta_schedule      =str(_sc("beta_schedule",       "squaredcos_cap_v2")),
        prediction_type    =str(_sc("prediction_type",     "epsilon")),
        clip_sample        =bool(_sc("clip_sample",        True)),
        clip_sample_range  =float(_sc("clip_sample_range", 5.0)),
        noise_scale        =float(_sc("noise_scale",       2.0)),
    )

    model_config = DiffusionTransformerConfig(
        seq_len=seq_len,
        state_dim=STATE_DIM,
        costate_dim=COSTATE_DIM,
        embd_dim=args.embd_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        p_drop_embd=0.0,   # no dropout at inference
        p_drop_attn=0.0,
        use_integration_signals=True,
        zero_known_state_eps=True,
    )
    model  = DiffusionTransformer(config=model_config)
    policy = IndiffCtrlPolicy(
        model=model, noise_scheduler=noise_scheduler,
        time_grid=time_grid_raw,
        segment_rtol=args.segment_rtol, segment_atol=args.segment_atol,
        num_inference_steps=args.num_inference_steps, eps=args.eps,
    )
    seg_arc_integrator = make_segment_arc_integrator(
        policy, arc_points=8, rtol=args.segment_rtol, atol=args.segment_atol,
    )

    # Init params + optimizer → get pytree structures → load from bytes
    print("Initialising model (needed for pytree structure) …")
    key             = jax.random.PRNGKey(args.seed)
    params_template = policy.init_params(key, batch_size=1)
    lr_sched        = optax.linear_schedule(init_value=1e-4, end_value=5e-6,
                                             transition_steps=1)
    opt_state_tpl   = optax.adamw(learning_rate=lr_sched,
                                   weight_decay=1e-6).init(params_template)

    target = {
        "params":    params_template,
        "opt_state": opt_state_tpl,
        "step":      np.int32(0),
        "time_grid": np.zeros(seq_len, np.float32),
        "scheduler_config": {
            "num_train_timesteps": 5000,
            "beta_schedule":       "squaredcos_cap_v2",
            "prediction_type":     "epsilon",
            "clip_sample":         True,
            "clip_sample_range":   5.0,
            "noise_scale":         2.0,
        },
    }
    loaded = serialization.from_bytes(target, raw_bytes)
    params = loaded["params"]
    print(f"Checkpoint at training step {int(loaded['step'])} loaded.")

    # ── Run denoising on CPU ──────────────────────────────────────────────────
    rng = jax.random.PRNGKey(args.seed)
    print(f"Running {args.num_inference_steps} denoising steps on {jax.devices()[0]} …")
    frames = collect_frames(policy, params, initial_state, final_state,
                            args.num_inference_steps, rng,
                            segment_arc_integrator=seg_arc_integrator)

    # ── Render video ──────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_video(frames, norm, args.eps, time_grid_raw, args.output, args.fps,
                hold_frames=args.hold_frames,
                gt_states=gt_states, gt_costates=gt_costates,
                dataset_index=args.dataset_index)


if __name__ == "__main__":
    main()
