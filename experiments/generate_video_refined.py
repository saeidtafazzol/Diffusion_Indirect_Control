"""Generate a video showing diffusion denoising followed by CasADi BVP refinement.

The animation has two phases rendered back-to-back:

  Phase 1 — Diffusion denoising
      One frame per captured denoising step (every ``--diffusion-capture-every``
      steps).  The trajectory evolves from pure noise to the model's best guess.

  Phase 2 — CasADi BVP refinement
      Starts from the last diffusion frame.  Every IPOPT major iteration (or
      every ``--casadi-stride`` iterations) produces one frame, showing the NLP
      solver driving the trajectory to satisfy the multiple-shooting continuity
      constraints and boundary conditions.

Four subplots per frame:
    (0,0)  3D position trajectory in AU  (with thrust direction quivers)
    (0,1)  Mass m(t) vs time
    (1,0)  λ_m(t) vs time  (should reach 0 at t_f)
    (1,1)  Per-interval continuity residual ‖F(Z_k)−Z_{k+1}‖

Usage
-----
python generate_video_refined.py \\
    --checkpoint  checkpoints/indiff_ctrl_latest.msgpack \\
    --dataset     earth_mars_minfuel_constrained_32pts \\
    [--sample-idx 0] \\
    [--output     refined_trajectory.mp4] \\
    [--num-diffusion-steps 30] \\
    [--diffusion-capture-every 1] \\
    [--casadi-max-iter 100] \\
    [--casadi-stride 1] \\
    [--fps 8] \\
    [--dpi 120] \\
    [--embd-dim 512] \\
    [--num-layers 12] \\
    [--num-heads 4] \\
    [--no-integration-signals]
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
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

# ── Persistent XLA compilation cache ─────────────────────────────────────────
# Compiled GPU kernels are cached to disk; subsequent runs skip recompilation
# and are ~10-15 s faster.
_JAX_CACHE = Path.home() / ".cache" / "jax_compile_cache"
_JAX_CACHE.mkdir(parents=True, exist_ok=True)
try:
    jax.config.update("jax_compilation_cache_dir", str(_JAX_CACHE))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
except Exception:
    pass   # older JAX versions may not support these options
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  registers the 3-D projection

import optax
from flax import serialization

from core import STATE_DIM, COSTATE_DIM, AUGMENTED_DIM, make_dynamics
from jax_ddpm_scheduler import JaxDDPMScheduler
from policy import IndiffCtrlPolicy
from transformer_diffusion_model import (
    DiffusionTransformer,
    DiffusionTransformerConfig,
    default_state_known_mask_no_final_mass,
    final_lambda_m_costate_mask,
)
from casadi_bvp_refine import BVPRefiner, switching_function


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loader (chunk-based, mirrors ChunkedEarthMarsDataset in train.py)
# ─────────────────────────────────────────────────────────────────────────────

class _ChunkedDataset:
    def __init__(self, dataset_dir: Path):
        self.dir = Path(dataset_dir)
        meta = np.load(self.dir / "metadata.npz")
        self.num_points   = int(meta["num_points"])
        self.time_grid    = meta["time_grid"].astype(np.float32)
        self.t_final      = float(meta["t_final"])
        self.chunk_paths  = sorted(self.dir.glob("chunk_*.npz"))
        if not self.chunk_paths:
            raise FileNotFoundError(f"No chunk files in {self.dir}")

    def get_sample(self, idx: int) -> dict:
        chunk_size_default = 256
        chunk_idx   = idx // chunk_size_default
        local_idx   = idx %  chunk_size_default
        chunk_idx   = min(chunk_idx, len(self.chunk_paths) - 1)
        data        = np.load(self.chunk_paths[chunk_idx])
        n = data["states"].shape[0]
        local_idx   = min(local_idx, n - 1)
        return {
            "states":         data["states"][local_idx].astype(np.float32),      # (T, 7)
            "costates":       data["costates"][local_idx].astype(np.float32),    # (T, 7)
            "initial_state":  data["initial_states"][local_idx].astype(np.float32),
            "final_state":    data["final_states"][local_idx].astype(np.float32),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

def _peek_checkpoint(raw_bytes: bytes) -> tuple:
    """Extract (time_grid, scheduler_config) from a raw checkpoint.

    Returns (None, {}) if extraction fails.
    """
    try:
        from flax.serialization import msgpack_restore
    except ImportError:
        try:
            from flax.serialization import _msgpack_restore as msgpack_restore  # noqa
        except ImportError:
            return None, {}

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
        return tg, (sc or {})
    except Exception as exc:
        print(f"  [warn] checkpoint peek failed ({exc}); using CLI defaults")
        return None, {}


def load_checkpoint(ckpt_path: Path, args) -> tuple:
    """Load checkpoint; return (params, policy, time_grid, norm).

    The model architecture is inferred from CLI args (embd_dim, num_layers,
    num_heads, no_integration_signals).  The noise-scheduler config is read
    from the checkpoint if available, otherwise defaults are used.
    """
    raw_bytes = ckpt_path.read_bytes()
    tg_ckpt, sch_raw = _peek_checkpoint(raw_bytes)

    # Time grid
    if tg_ckpt is not None and len(tg_ckpt) >= 2:
        time_grid = tg_ckpt
    else:
        from core import build_normalization
        norm_defaults = build_normalization()
        time_grid = np.linspace(0.0, float(norm_defaults["t_f"]), 32, dtype=np.float32)
    seq_len = len(time_grid)

    # Scheduler
    def _sc(k, d):
        v = sch_raw.get(k, d)
        return d if v is None else v

    noise_scheduler = JaxDDPMScheduler(
        num_train_timesteps = int(_sc("num_train_timesteps", 5000)),
        beta_schedule       = str(_sc("beta_schedule",       "squaredcos_cap_v2")),
        prediction_type     = str(_sc("prediction_type",     "epsilon")),
        clip_sample         = bool(_sc("clip_sample",        True)),
        clip_sample_range   = float(_sc("clip_sample_range", 5.0)),
        noise_scale         = float(_sc("noise_scale",       2.0)),
    )

    # Model
    use_integration = not args.no_integration_signals
    model = DiffusionTransformer(DiffusionTransformerConfig(
        seq_len               = seq_len,
        state_dim             = STATE_DIM,
        costate_dim           = COSTATE_DIM,
        embd_dim              = args.embd_dim,
        num_layers            = args.num_layers,
        num_heads             = args.num_heads,
        mlp_ratio             = 4,
        p_drop_embd           = 0.0,
        p_drop_attn           = 0.0,
        use_integration_signals = use_integration,
        zero_known_state_eps  = True,
    ))

    policy = IndiffCtrlPolicy(
        model=model,
        noise_scheduler=noise_scheduler,
        time_grid=time_grid,
        segment_rtol=args.inference_rtol,
        segment_atol=args.inference_atol,
        eps=args.eps,
    )

    # Reconstruct param + opt_state template for serialization.from_bytes
    key_init = jax.random.PRNGKey(0)
    params_tpl = policy.init_params(key_init, batch_size=1)
    opt_tpl = optax.adamw(
        optax.linear_schedule(1e-4, 5e-6, 1), weight_decay=1e-6
    ).init(params_tpl)
    sch_tpl = {
        "num_train_timesteps": 5000,
        "beta_schedule":       "squaredcos_cap_v2",
        "prediction_type":     "epsilon",
        "clip_sample":         True,
        "clip_sample_range":   5.0,
        "noise_scale":         2.0,
    }
    state_tpl = {
        "params":           params_tpl,
        "opt_state":        opt_tpl,
        "step":             np.int32(0),
        "time_grid":        np.zeros(seq_len, np.float32),
        "scheduler_config": sch_tpl,
    }
    loaded = serialization.from_bytes(state_tpl, raw_bytes)
    params = loaded["params"]
    step   = int(loaded["step"])
    print(f"Checkpoint loaded: step={step}  seq_len={seq_len}")
    print(f"  scheduler: {noise_scheduler.config.beta_schedule}  "
          f"T={noise_scheduler.config.num_train_timesteps}")

    _, _, norm = make_dynamics(eps=args.eps)
    return params, policy, time_grid, norm


# ─────────────────────────────────────────────────────────────────────────────
# Diffusion denoising loop with frame capture
# ─────────────────────────────────────────────────────────────────────────────

def run_diffusion_loop(
    policy: IndiffCtrlPolicy,
    params,
    sample: dict,
    num_steps: int,
    capture_every: int = 1,
    rng_seed: int = 42,
) -> list:
    """Run the reverse diffusion denoising loop for one sample.

    Returns a list of (T, 14) float32 arrays — one per captured step.
    The first element is the initial noise; the last is the final prediction.
    """
    seq_len = len(policy.time_grid)
    cfg     = policy.model.config
    key     = jax.random.PRNGKey(rng_seed)

    # Build condition
    skm = default_state_known_mask_no_final_mass(1, seq_len)
    ckm = final_lambda_m_costate_mask(1, seq_len)
    skm, ckm = policy._resolve_known_masks(1, seq_len,
                                           state_known_mask=skm,
                                           costate_known_mask=ckm)

    cond_st       = np.zeros((1, seq_len, STATE_DIM),   np.float32)
    cond_cs       = np.zeros((1, seq_len, COSTATE_DIM), np.float32)
    cond_st[0, 0, :]  = sample["initial_state"]
    cond_st[0, -1, :] = sample["final_state"]
    # Pin final λ_m = 0 in condition
    cond_cs[0, -1, COSTATE_DIM - 1] = 0.0

    cond_traj = jnp.asarray(
        np.concatenate([cond_st, cond_cs], axis=-1), dtype=jnp.float32
    )  # (1, T, 14)
    cond_mask = policy._build_condition_mask(skm, ckm)

    # Sample initial noise
    key, sk = jax.random.split(key)
    traj    = policy.noise_scheduler.sample_noise(sk, cond_traj.shape, dtype=jnp.float32)
    # power=2.7 matches generate_video.py — concentrates steps at high noise levels
    policy.noise_scheduler.set_timesteps(num_steps, power=2.7)

    # JIT-compile the model forward pass (13× faster than eager on GPU)
    if cfg.use_integration_signals:
        @jax.jit
        def _apply(params, st, cst, ts_int, skm, ckm, int_st, int_cst, int_fail):
            return policy.model.apply(
                {"params": params},
                noisy_states=st, noisy_costates=cst,
                diffusion_steps=jnp.full((1,), ts_int, dtype=jnp.int32),
                state_known_mask=skm, costate_known_mask=ckm,
                integrated_states=int_st, integrated_costates=int_cst,
                integration_failed=int_fail, train=False,
            )
    else:
        @jax.jit
        def _apply(params, st, cst, ts_int, skm, ckm, int_st, int_cst, int_fail):
            return policy.model.apply(
                {"params": params},
                noisy_states=st, noisy_costates=cst,
                diffusion_steps=jnp.full((1,), ts_int, dtype=jnp.int32),
                state_known_mask=skm, costate_known_mask=ckm,
                integrated_states=None, integrated_costates=None,
                integration_failed=None, train=False,
            )

    frames = []
    for i, ts in enumerate(policy.noise_scheduler.timesteps):
        traj = jnp.where(cond_mask, cond_traj, traj)
        st   = traj[..., :STATE_DIM]
        cst  = traj[..., STATE_DIM:STATE_DIM + COSTATE_DIM]

        int_st = int_cst = int_fail = None
        if cfg.use_integration_signals:
            int_st, int_cst, int_fail = policy.compute_integration_signals(st, cst)

        out = _apply(params, st, cst, int(ts), skm, ckm, int_st, int_cst, int_fail)
        model_out = policy._combine_trajectory(out["state_eps"], out["costate_eps"])
        key, step_key = jax.random.split(key)
        traj = policy.noise_scheduler.step(
            model_output = model_out,
            timestep     = int(ts),
            sample       = traj,
            rng_key      = step_key,
        ).prev_sample
        traj = jnp.where(cond_mask, cond_traj, traj)

        # Keep as JAX array — avoid per-step GPU→CPU transfer
        if i % capture_every == 0 or i == len(policy.noise_scheduler.timesteps) - 1:
            frames.append(traj[0])  # (T, 14) on device

    # Single batch GPU→CPU transfer after the loop
    jax.block_until_ready(frames[-1])
    frames_np = [np.asarray(f, dtype=np.float32) for f in frames]

    # Always include the very last step
    final_np = np.asarray(traj[0], dtype=np.float32)
    if not np.allclose(frames_np[-1], final_np, atol=1e-8):
        frames_np.append(final_np)

    print(f"Diffusion frames captured: {len(frames_np)}")
    return frames_np


# ─────────────────────────────────────────────────────────────────────────────
# Animation builder
# ─────────────────────────────────────────────────────────────────────────────

_DIFF_COLOR  = "limegreen"    # diffusion phase  — 3D trajectory and suptitle
_CAS_COLOR   = "darkorange"   # CasADi phase     — 3D trajectory and suptitle
_GT_COLOR    = "#d62728"      # ground truth (dashed red)
_XYZ_COLORS  = ["tab:red", "tab:green", "tab:blue"]   # x / y / z component colours
_XYZ_LABELS  = ["x", "y", "z"]


def _frame_diagnostics(z: np.ndarray, refiner: BVPRefiner, norm: dict) -> dict:
    """Pre-compute all quantities shown in the animation for one frame."""
    delta, alpha, S = switching_function(z, norm, eps=1e-4)
    on = delta > 0.5
    res = refiner.continuity_residuals(z, n_rk4_eval=8)
    return dict(
        r      = z[:, 0:3],
        v      = z[:, 3:6],
        mass   = z[:, 6],
        lam_r  = z[:, 7:10],
        lam_v  = z[:, 10:13],
        lam_m  = z[:, 13],
        delta  = delta,
        alpha  = alpha,
        on     = on,
        S      = S,
        res    = res,
    )


def build_animation(
    diffusion_frames: list,        # list of (T, 14) float32
    casadi_frames:    list,        # list of (T, 14) float64
    sample:           dict,
    refiner:          BVPRefiner,
    norm:             dict,
    time_grid:        np.ndarray,  # (T,)
    fps:              int  = 8,
    dpi:              int  = 120,
) -> animation.FuncAnimation:
    """Build a matplotlib FuncAnimation with two phases.

    Layout  (matches generate_video.py)
    ------
    Row 0 | col 0 (spans rows 0-1): 3D trajectory
    Row 0 | col 1: position r(t)  — x/y/z
    Row 0 | col 2: velocity v(t)  — x/y/z
    Row 0 | col 3: mass m(t) + thrust δ(t)  (twin axis)
    Row 1 | col 1: costate λ_r(t) — x/y/z
    Row 1 | col 2: costate λ_v(t) — x/y/z
    Row 1 | col 3: costate λ_m(t)
    Row 2 | cols 1-3: per-interval continuity residuals
    """
    # Ground truth — only available when loaded from a dataset chunk
    has_gt = "states" in sample and "costates" in sample
    if has_gt:
        gt_z    = np.concatenate(
            [sample["states"], sample["costates"]], axis=-1
        ).astype(np.float64)
        gt_diag = _frame_diagnostics(gt_z, refiner, norm)
    else:
        gt_diag = None

    # All frame metadata
    meta = []
    for i, z in enumerate(diffusion_frames):
        meta.append(("diffusion", i + 1, len(diffusion_frames),
                     z.astype(np.float64)))
    for i, z in enumerate(casadi_frames):
        meta.append(("casadi", i, len(casadi_frames) - 1,
                     z.astype(np.float64)))

    # Pre-compute diagnostics for every frame
    diags = [_frame_diagnostics(m[3], refiner, norm) for m in meta]

    # Pre-compute y-limits across all frames (+ GT if available)
    def _ylim(key):
        vals = np.concatenate([d[key].ravel() for d in diags])
        if gt_diag is not None:
            vals = np.concatenate([vals, gt_diag[key].ravel()])
        lo, hi = float(vals.min()), float(vals.max())
        pad = max((hi - lo) * 0.06, 1e-6)
        return lo - pad, hi + pad

    r_lim  = _ylim("r")
    v_lim  = _ylim("v")
    lr_lim = _ylim("lam_r")
    lv_lim = _ylim("lam_v")
    m_lim  = (0.0, 1.0)
    th_lim = (0.0, 1.5)
    lm_lim = (-0.2, 1.2)

    t  = time_grid
    ri = norm["r_i"]
    rf = norm["r_f"]

    # ── Figure layout: 3 rows × 4 cols ───────────────────────────────────────
    fig = plt.figure(figsize=(22, 12))
    gs  = gridspec.GridSpec(3, 4, figure=fig, wspace=0.40, hspace=0.55)

    ax3d  = fig.add_subplot(gs[0:2, 0], projection="3d")
    ax_r  = fig.add_subplot(gs[0, 1])   # position r(t)  — x/y/z
    ax_v  = fig.add_subplot(gs[0, 2])   # velocity v(t)  — x/y/z
    ax_mt = fig.add_subplot(gs[0, 3])   # mass + thrust (twin axis)
    ax_lr = fig.add_subplot(gs[1, 1])   # costate λ_r(t) — x/y/z
    ax_lv = fig.add_subplot(gs[1, 2])   # costate λ_v(t) — x/y/z
    ax_lm = fig.add_subplot(gs[1, 3])   # costate λ_m(t)
    ax_rs = fig.add_subplot(gs[2, 1:])  # continuity residuals
    ax_t2 = ax_mt.twinx()              # thrust axis (created once)

    title_txt = fig.suptitle("", fontsize=11, fontweight="bold", y=0.998)

    # ── Static 3-D setup (done once; never cleared so view angle is stable) ───
    all_r_3d = np.vstack([d["r"] for d in diags])
    if gt_diag is not None:
        all_r_3d = np.vstack([all_r_3d, gt_diag["r"]])
    all_r_3d = np.vstack([all_r_3d, np.array([[0, 0, 0], ri, rf])])
    xyz_lo  = all_r_3d.min(axis=0)
    xyz_hi  = all_r_3d.max(axis=0)
    pad3d   = np.maximum((xyz_hi - xyz_lo) * 0.12, 0.1)
    ax3d.set_xlim(-2.0, 2.0)
    ax3d.set_ylim(-2.0, 2.0)
    ax3d.set_zlim(-0.1, 0.1)
    ax3d.set_xlabel("x [AU]", fontsize=7)
    ax3d.set_ylabel("y [AU]", fontsize=7)
    ax3d.set_zlabel("z [AU]", fontsize=7)
    ax3d.tick_params(labelsize=6)
    ax3d.scatter(0, 0, 0,  color="gold",       s=130, marker="*", zorder=5, label="Sun")
    ax3d.scatter(*ri,       color="dodgerblue", s=70,              zorder=5, label="Earth")
    ax3d.scatter(*rf,       color="tomato",     s=70,              zorder=5, label="Mars")
    if gt_diag is not None:
        gr = gt_diag["r"]
        ax3d.plot(gr[:, 0], gr[:, 1], gr[:, 2],
                  "--", lw=1.0, color="white", alpha=0.6, label="GT")
    ax3d.legend(fontsize=6, loc="upper left")
    (l3,)          = ax3d.plot([], [], [], "o-", ms=2.0, lw=1.8)  # animated trajectory
    quiver_holder  = [None]   # holds the current Quiver3D so we can remove it each frame

    # ── Shared helper ─────────────────────────────────────────────────────────
    def _setup(ax, title, yl, ylim):
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("time [TU/2π]", fontsize=7)
        ax.set_ylabel(yl, fontsize=7)
        ax.set_ylim(*ylim)
        ax.set_xlim(float(t[0]), float(t[-1]))
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.25)

    # ── Update function ───────────────────────────────────────────────────────
    def _update(frame_idx):
        phase, step_n, step_total, _ = meta[frame_idx]
        d     = diags[frame_idx]
        color = _DIFF_COLOR if phase == "diffusion" else _CAS_COLOR

        # ── 3D trajectory (axes never cleared — view angle stays fixed) ───────
        l3.set_data(d["r"][:, 0], d["r"][:, 1])
        l3.set_3d_properties(d["r"][:, 2])
        l3.set_color(color)
        # Replace quivers each frame
        if quiver_holder[0] is not None:
            quiver_holder[0].remove()
            quiver_holder[0] = None
        if np.any(d["on"]):
            quiver_holder[0] = ax3d.quiver(
                d["r"][d["on"], 0], d["r"][d["on"], 1], d["r"][d["on"], 2],
                d["alpha"][d["on"], 0], d["alpha"][d["on"], 1], d["alpha"][d["on"], 2],
                length=0.05, normalize=True, color="tab:red", alpha=0.6,
            )

        # ── Position r(t) ─────────────────────────────────────────────────────
        ax_r.cla()
        _setup(ax_r, "Position  r(t)", "AU", r_lim)
        if gt_diag is not None:
            for k, c in enumerate(_XYZ_COLORS):
                ax_r.plot(t, gt_diag["r"][:, k], "--", color=c, lw=1.0, alpha=0.5)
        for k, (c, lb) in enumerate(zip(_XYZ_COLORS, _XYZ_LABELS)):
            ax_r.plot(t, d["r"][:, k], color=c, lw=1.4, label=lb)
        ax_r.legend(fontsize=6, loc="upper right")

        # ── Velocity v(t) ─────────────────────────────────────────────────────
        ax_v.cla()
        _setup(ax_v, "Velocity  v(t)", "AU/TU", v_lim)
        if gt_diag is not None:
            for k, c in enumerate(_XYZ_COLORS):
                ax_v.plot(t, gt_diag["v"][:, k], "--", color=c, lw=1.0, alpha=0.5)
        for k, (c, lb) in enumerate(zip(_XYZ_COLORS, _XYZ_LABELS)):
            ax_v.plot(t, d["v"][:, k], color=c, lw=1.4, label=lb)
        ax_v.legend(fontsize=6, loc="upper right")

        # ── Mass + thrust (twin axis) ─────────────────────────────────────────
        ax_mt.cla()
        ax_t2.cla()
        ax_mt.set_title("Mass & Thrust", fontsize=9)
        ax_mt.set_xlabel("time [TU/2π]", fontsize=7)
        ax_mt.set_ylabel("mass (norm)", fontsize=7, color="tab:green")
        ax_mt.set_ylim(*m_lim)
        ax_mt.set_xlim(float(t[0]), float(t[-1]))
        ax_mt.tick_params(labelsize=6, axis="y", labelcolor="tab:green")
        ax_mt.grid(True, alpha=0.25)
        ax_t2.set_ylabel("thrust fraction δ", fontsize=7, color="tab:red")
        ax_t2.set_ylim(*th_lim)
        ax_t2.tick_params(labelsize=6, axis="y", labelcolor="tab:red")
        if gt_diag is not None:
            ax_mt.plot(t, gt_diag["mass"],  "--", color="tab:green", lw=1.0, alpha=0.5)
            ax_t2.plot(t, gt_diag["delta"], "--", color="tab:red",   lw=1.0, alpha=0.5)
        ax_mt.plot(t, d["mass"],  color="tab:green", lw=1.4, label="mass")
        ax_t2.plot(t, d["delta"], color="tab:red",   lw=1.4, label="δ")

        # ── Costate λ_r(t) ────────────────────────────────────────────────────
        ax_lr.cla()
        _setup(ax_lr, "Costate  λ_r(t)", "", lr_lim)
        if gt_diag is not None:
            for k, c in enumerate(_XYZ_COLORS):
                ax_lr.plot(t, gt_diag["lam_r"][:, k], "--", color=c, lw=1.0, alpha=0.5)
        for k, (c, lb) in enumerate(zip(_XYZ_COLORS, _XYZ_LABELS)):
            ax_lr.plot(t, d["lam_r"][:, k], color=c, lw=1.4, label=f"λ_r{lb}")
        ax_lr.legend(fontsize=6, loc="upper right")

        # ── Costate λ_v(t) ────────────────────────────────────────────────────
        ax_lv.cla()
        _setup(ax_lv, "Costate  λ_v(t)", "", lv_lim)
        if gt_diag is not None:
            for k, c in enumerate(_XYZ_COLORS):
                ax_lv.plot(t, gt_diag["lam_v"][:, k], "--", color=c, lw=1.0, alpha=0.5)
        for k, (c, lb) in enumerate(zip(_XYZ_COLORS, _XYZ_LABELS)):
            ax_lv.plot(t, d["lam_v"][:, k], color=c, lw=1.4, label=f"λ_v{lb}")
        ax_lv.legend(fontsize=6, loc="upper right")

        # ── Costate λ_m(t) ────────────────────────────────────────────────────
        ax_lm.cla()
        ax_lm.axhline(0.0, color="gray", lw=0.8, ls="--")
        _setup(ax_lm, "Costate  λ_m(t)  [→ 0 at t_f]", "", lm_lim)
        if gt_diag is not None:
            ax_lm.plot(t, gt_diag["lam_m"], "--", color="tab:purple", lw=1.0, alpha=0.5)
        ax_lm.plot(t, d["lam_m"], color="tab:purple", lw=1.8)
        lm_f = float(d["lam_m"][-1])
        ax_lm.scatter(t[-1], lm_f, color="tab:purple", s=40, zorder=5,
                      label=f"λ_m(tf) = {lm_f:.4f}")
        ax_lm.legend(fontsize=7)

        # ── Continuity residuals ──────────────────────────────────────────────
        ax_rs.cla()
        intervals = np.arange(len(d["res"]))
        ax_rs.bar(intervals, d["res"], color=color, alpha=0.75, width=0.8)
        ax_rs.set_ylabel("‖F(Z_k) − Z_{k+1}‖", fontsize=8)
        ax_rs.set_xlabel("interval k",            fontsize=8)
        ax_rs.set_title(f"Continuity residuals  (max = {d['res'].max():.2e})", fontsize=9)
        ax_rs.set_yscale("symlog", linthresh=1e-8)
        ax_rs.grid(alpha=0.3, axis="y")
        ax_rs.tick_params(labelsize=7)

        # ── Super-title ───────────────────────────────────────────────────────
        if phase == "diffusion":
            title = f"Phase 1 — Diffusion denoising   step {step_n}/{step_total}"
        else:
            label = ("initial guess (from diffusion)" if step_n == 0
                     else f"IPOPT iter {step_n}/{step_total}")
            title = f"Phase 2 — Fine-tuning with CasADi   {label}"
        title_txt.set_text(title)
        title_txt.set_color(color)
        return []

    n_frames = len(meta)
    anim = animation.FuncAnimation(
        fig, _update,
        frames   = n_frames,
        interval = int(1000 / fps),
        blit     = False,
        repeat   = False,
    )
    return anim


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint",  type=Path,
                   default=Path("checkpoints/indiff_ctrl_latest.msgpack"))
    p.add_argument("--dataset",     type=Path,
                   default=Path("earth_mars_minfuel_constrained_32pts"))
    p.add_argument("--sample-idx",  type=int,   default=0)
    p.add_argument("--output",      type=Path,  default=Path("refined_trajectory.mp4"))
    # Diffusion
    p.add_argument("--num-diffusion-steps",     type=int, default=30)
    p.add_argument("--diffusion-capture-every", type=int, default=1,
                   help="Record one diffusion frame every N denoising steps")
    # CasADi
    p.add_argument("--casadi-max-iter",    type=int, default=100)
    p.add_argument("--casadi-stride",      type=int, default=1,
                   help="Record one CasADi frame every N IPOPT iterations")
    p.add_argument("--casadi-rk4-steps",   type=int, default=8)
    # Model architecture
    p.add_argument("--embd-dim",    type=int, default=512)
    p.add_argument("--num-layers",  type=int, default=12)
    p.add_argument("--num-heads",   type=int, default=4)
    p.add_argument("--no-integration-signals", action="store_true")
    p.add_argument("--eps",         type=float, default=1e-4)
    p.add_argument(
        "--inference-rtol", type=float, default=1e-7,
        help="ODE rtol for integration signals during inference (default 1e-7)",
    )
    p.add_argument(
        "--inference-atol", type=float, default=1e-9,
        help="ODE atol for integration signals during inference (default 1e-9)",
    )
    # Video
    p.add_argument("--fps",  type=int, default=8)
    p.add_argument("--dpi",  type=int, default=120)
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for initial diffusion noise (default 42, matches generate_video.py)")
    p.add_argument(
        "--earth-mars", action="store_true",
        help="Use hardcoded Earth-Mars BCs from core.py (no chunk files needed).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ── Sample: dataset chunk OR hardcoded Earth-Mars BCs ─────────────────────
    if args.earth_mars:
        from core import build_normalization
        _n    = build_normalization()
        m0    = float(_n["m0"])
        initial_state = np.concatenate([_n["r_i"], _n["v_i"], [m0]]).astype(np.float32)
        final_state   = np.concatenate([_n["r_f"], _n["v_f"], [m0]]).astype(np.float32)
        sample = {"initial_state": initial_state, "final_state": final_state}
        print("Earth-Mars BCs (norm units):")
        print(f"  r0={initial_state[:3].round(4)}  v0={initial_state[3:6].round(4)}  m0={m0:.4f}")
        print(f"  rf={final_state[:3].round(4)}  vf={final_state[3:6].round(4)}")
        time_grid = None   # filled from checkpoint below
    else:
        if not args.dataset.exists():
            sys.exit(f"Dataset not found: {args.dataset}  (tip: use --earth-mars)")
        ds        = _ChunkedDataset(args.dataset)
        sample    = ds.get_sample(args.sample_idx)
        time_grid = ds.time_grid
        print(f"Loaded sample {args.sample_idx}  "
              f"initial_state={sample['initial_state'].round(4)}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    if not args.checkpoint.exists():
        sys.exit(f"Checkpoint not found: {args.checkpoint}")
    params, policy, tg_ckpt, norm = load_checkpoint(args.checkpoint, args)
    # Use time grid from checkpoint (may differ from dataset)
    time_grid = tg_ckpt.astype(np.float32)

    # ── Report JAX device ─────────────────────────────────────────────────────
    devices = jax.devices()
    print(f"\nJAX devices: {devices}")
    print(f"Diffusion (Phase 1) will run on: {devices[0]}")

    # ── Phase 1: diffusion denoising with frame capture ───────────────────────
    print("\n── Phase 1: diffusion denoising ─────────────────────────────────────")
    _t0_diff = time.perf_counter()
    diff_frames = run_diffusion_loop(
        policy         = policy,
        params         = params,
        sample         = sample,
        num_steps      = args.num_diffusion_steps,
        capture_every  = args.diffusion_capture_every,
        rng_seed       = args.seed,
    )
    _t1_diff = time.perf_counter()
    jax.block_until_ready(diff_frames[-1])   # ensure GPU work is finished
    _t1_diff = time.perf_counter()
    print(f"Phase 1 time: {_t1_diff - _t0_diff:.2f} s  "
          f"({args.num_diffusion_steps} steps, "
          f"{((_t1_diff - _t0_diff) / args.num_diffusion_steps * 1000):.1f} ms/step)")

    # ── Phase 2: CasADi BVP refinement ───────────────────────────────────────
    print("\n── Phase 2: CasADi BVP refinement ──────────────────────────────────")
    _t0_cas = time.perf_counter()
    refiner = BVPRefiner(
        eps          = args.eps,
        n_points     = len(time_grid),
        n_rk4_steps  = args.casadi_rk4_steps,
    )

    z_init = diff_frames[-1].astype(np.float64)  # (T, 14) from diffusion
    print(f"Initial continuity residual (max): "
          f"{refiner.continuity_residuals(z_init).max():.4e}")

    cas_frames = refiner.solve_iterative(
        z_init        = z_init,
        initial_state = sample["initial_state"].astype(np.float64),
        final_state   = sample["final_state"].astype(np.float64),
        max_iter      = args.casadi_max_iter,
        stride        = args.casadi_stride,
    )
    _t1_cas = time.perf_counter()
    print(f"Phase 2 time: {_t1_cas - _t0_cas:.2f} s  (CPU, CasADi/IPOPT)")
    print(f"\n{'='*60}")
    print(f"  Phase 1 (diffusion, {devices[0]}): {_t1_diff - _t0_diff:.2f} s")
    print(f"  Phase 2 (CasADi/IPOPT, CPU):      {_t1_cas  - _t0_cas:.2f} s")
    print(f"{'='*60}")

    # ── Build and save animation ──────────────────────────────────────────────
    print("\n── Building animation ───────────────────────────────────────────────")
    total_frames = len(diff_frames) + len(cas_frames)
    print(f"Total frames: {total_frames}  "
          f"({len(diff_frames)} diffusion + {len(cas_frames)} CasADi)")

    anim = build_animation(
        diffusion_frames = diff_frames,
        casadi_frames    = cas_frames,
        sample           = sample,
        refiner          = refiner,
        norm             = norm,
        time_grid        = time_grid,
        fps              = args.fps,
        dpi              = args.dpi,
    )

    suffix = args.output.suffix.lower()
    if suffix == ".gif":
        writer = animation.PillowWriter(fps=args.fps)
    else:
        # mpeg4 is universally available in most ffmpeg builds
        writer = animation.FFMpegWriter(
            fps=args.fps, bitrate=1800, codec="mpeg4",
            extra_args=["-q:v", "4"],
            metadata={"title": "Diffusion + CasADi BVP"},
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(args.output), writer=writer, dpi=args.dpi)
    print(f"\nVideo saved → {args.output}")


if __name__ == "__main__":
    main()
