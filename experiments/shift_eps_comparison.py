"""shift_eps_comparison.py
==========================
Convergence-rate study: Method A vs. Method B across time-shifted Earth–Mars transfers.

For each Δt shift, Earth/Mars states are Keplerian-propagated by Δt, then
--num-trials independent seeds are run for each method.

Method A — Diffusion (N steps) → CasADi BVP (eps=1e-4, one solve)
Method B — ε-continuation CasADi (no diffusion):
    1. Sample λ(t_0) ~ U[-2,2]^7 and forward-integrate eps=1.0 augmented
       Hamiltonian ODE over [0, t_f] → T-point initial guess
    2-6. Five successive BVP solves, warm-starting from previous:
         eps = 1.0 → 1e-1 → 1e-2 → 1e-3 → 1e-4
    Non-fatal IPOPT exits (max-iter exceeded, acceptable level) at intermediate
    stages pass the current iterate forward.  Only fatal errors abort early.
    Convergence / residual check applied only at eps=1e-4.

Max iters: --casadi-max-iter (A, default 500) vs.
           --casadi-max-iter-b per stage (B, default 1000 → 5000 total max)

Output
------
  <output-dir>/
    summary.png    — convergence rate vs shift (both methods)
    results.txt    — per-shift table
"""

import sys
import argparse
import time
import json
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import casadi as ca
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from concurrent.futures import ThreadPoolExecutor
import optax
from flax import serialization

_JAX_CACHE = Path.home() / ".cache" / "jax_compile_cache"
_JAX_CACHE.mkdir(parents=True, exist_ok=True)
try:
    jax.config.update("jax_compilation_cache_dir", str(_JAX_CACHE))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
except Exception:
    pass

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
del _root

from core import STATE_DIM, COSTATE_DIM, AUGMENTED_DIM, make_dynamics  # noqa: E402
from jax_ddpm_scheduler import JaxDDPMScheduler  # noqa: E402
from policy import IndiffCtrlPolicy  # noqa: E402
from transformer_diffusion_model import (  # noqa: E402
    DiffusionTransformer,
    DiffusionTransformerConfig,
    default_state_known_mask_no_final_mass,
    final_lambda_m_costate_mask,
)
from casadi_bvp_refine import BVPRefiner, make_bounds, IterCapture, switching_function  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: Path):
    """Return (params, policy, time_grid, norm)."""
    from flax.serialization import msgpack_restore

    raw_bytes = ckpt_path.read_bytes()
    try:
        raw = msgpack_restore(raw_bytes)
        tg  = np.asarray(raw["time_grid"], dtype=np.float32).ravel()
        sc  = raw.get("scheduler_config", {})
        def _s(v): return v.decode() if isinstance(v, bytes) else v
        sc = {_s(k): (_s(v) if isinstance(v, bytes) else v) for k, v in sc.items()}
    except Exception as exc:
        sys.exit(f"Cannot read checkpoint: {exc}")

    seq_len = len(tg)
    def _sc(k, d):
        v = sc.get(k, d)
        return d if v is None else v

    noise_scheduler = JaxDDPMScheduler(
        num_train_timesteps = int(_sc("num_train_timesteps", 5000)),
        beta_schedule       = str(_sc("beta_schedule",       "squaredcos_cap_v2")),
        prediction_type     = str(_sc("prediction_type",     "epsilon")),
        clip_sample         = bool(_sc("clip_sample",        True)),
        clip_sample_range   = float(_sc("clip_sample_range", 5.0)),
        noise_scale         = float(_sc("noise_scale",       2.0)),
    )
    model = DiffusionTransformer(DiffusionTransformerConfig(
        seq_len=seq_len, state_dim=STATE_DIM, costate_dim=COSTATE_DIM,
        embd_dim=512, num_layers=12, num_heads=4, mlp_ratio=4,
        p_drop_embd=0.0, p_drop_attn=0.0,
        use_integration_signals=True, zero_known_state_eps=True,
    ))
    policy = IndiffCtrlPolicy(
        model=model, noise_scheduler=noise_scheduler, time_grid=tg,
        segment_rtol=1e-7, segment_atol=1e-9, eps=1e-4,
    )
    key_init = jax.random.PRNGKey(0)
    params_tpl = policy.init_params(key_init, batch_size=1)
    opt_tpl = optax.adamw(
        optax.linear_schedule(1e-4, 5e-6, 1), weight_decay=1e-6
    ).init(params_tpl)
    sch_tpl = {"num_train_timesteps": 5000, "beta_schedule": "squaredcos_cap_v2",
               "prediction_type": "epsilon", "clip_sample": True,
               "clip_sample_range": 5.0, "noise_scale": 2.0}
    state_tpl = {"params": params_tpl, "opt_state": opt_tpl,
                 "step": np.int32(0), "time_grid": np.zeros(seq_len, np.float32),
                 "scheduler_config": sch_tpl}
    loaded = serialization.from_bytes(state_tpl, raw_bytes)
    params = loaded["params"]
    print(f"Checkpoint loaded: step={int(loaded['step'])}  seq_len={seq_len}")

    _, _, norm = make_dynamics(eps=1e-4, compile_jax=False)
    return params, policy, tg, norm


# ─────────────────────────────────────────────────────────────────────────────
# Method A — Diffusion + one-shot CasADi BVP
# ─────────────────────────────────────────────────────────────────────────────

def build_jit_apply(policy: IndiffCtrlPolicy):
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
    return _apply


def run_diffusion(policy, params, sample, _apply, num_steps, rng_seed):
    """Run full reverse diffusion; return list of (T,14) float32 frames (one per step)."""
    seq_len = len(policy.time_grid)
    cfg     = policy.model.config
    key     = jax.random.PRNGKey(rng_seed)

    skm = default_state_known_mask_no_final_mass(1, seq_len)
    ckm = final_lambda_m_costate_mask(1, seq_len)
    skm, ckm = policy._resolve_known_masks(1, seq_len,
                                            state_known_mask=skm,
                                            costate_known_mask=ckm)

    cond_st = np.zeros((1, seq_len, STATE_DIM),   np.float32)
    cond_cs = np.zeros((1, seq_len, COSTATE_DIM), np.float32)
    cond_st[0,  0, :] = sample["initial_state"]
    cond_st[0, -1, :] = sample["final_state"]
    cond_cs[0, -1, COSTATE_DIM - 1] = 0.0

    cond_traj = jnp.asarray(
        np.concatenate([cond_st, cond_cs], axis=-1), dtype=jnp.float32
    )
    cond_mask = policy._build_condition_mask(skm, ckm)

    key, sk = jax.random.split(key)
    traj = policy.noise_scheduler.sample_noise(sk, cond_traj.shape, dtype=jnp.float32)
    policy.noise_scheduler.set_timesteps(num_steps, power=2.7)

    captured = []
    for ts in policy.noise_scheduler.timesteps:
        traj = jnp.where(cond_mask, cond_traj, traj)
        st   = traj[..., :STATE_DIM]
        cst  = traj[..., STATE_DIM:STATE_DIM + COSTATE_DIM]

        int_st = int_cst = int_fail = None
        if cfg.use_integration_signals:
            int_st, int_cst, int_fail = policy.compute_integration_signals(st, cst)

        out       = _apply(params, st, cst, int(ts), skm, ckm, int_st, int_cst, int_fail)
        model_out = policy._combine_trajectory(out["state_eps"], out["costate_eps"])
        key, step_key = jax.random.split(key)
        traj = policy.noise_scheduler.step(
            model_output=model_out, timestep=int(ts),
            sample=traj, rng_key=step_key,
        ).prev_sample
        traj = jnp.where(cond_mask, cond_traj, traj)
        captured.append(traj[0])  # keep on device

    jax.block_until_ready(captured[-1])
    return [np.asarray(f, dtype=np.float32) for f in captured]


# ─────────────────────────────────────────────────────────────────────────────
# Full 8-panel frame builder  (matches generate_video_refined.py layout)
# ─────────────────────────────────────────────────────────────────────────────

_XYZ_COLORS = ["tab:red", "tab:green", "tab:blue"]
_XYZ_LABELS = ["x", "y", "z"]
_DIFF_COLOR  = "limegreen"
_IPOPT_COLOR = "darkorange"
_CONT_COLOR  = "mediumpurple"

_SAVE_POOL = ThreadPoolExecutor(max_workers=3)


def _worker_save_frame(z, t_day, ri, rf, diags, res_arr, title, color, out_path):
    """Thread worker: build full 8-panel figure and save PNG."""
    fig = plt.figure(figsize=(18, 9))
    gs  = gridspec.GridSpec(3, 4, figure=fig, wspace=0.42, hspace=0.58)
    ax3d  = fig.add_subplot(gs[0:2, 0], projection="3d")
    ax_r  = fig.add_subplot(gs[0, 1])
    ax_v  = fig.add_subplot(gs[0, 2])
    ax_mt = fig.add_subplot(gs[0, 3])
    ax_lr = fig.add_subplot(gs[1, 1])
    ax_lv = fig.add_subplot(gs[1, 2])
    ax_lm = fig.add_subplot(gs[1, 3])
    ax_rs = fig.add_subplot(gs[2, 1:])
    ax_t2 = ax_mt.twinx()
    fig.suptitle(title, fontsize=8, fontweight="bold", y=0.999)

    # ── 3D trajectory ─────────────────────────────────────────────────────────
    ax3d.set_xlim(-2.0, 2.0); ax3d.set_ylim(-2.0, 2.0); ax3d.set_zlim(-2.0, 2.0)
    ax3d.scatter(0, 0, 0,  color="gold",       s=120, marker="*", zorder=5, label="Sun")
    ax3d.scatter(*ri,       color="dodgerblue", s=60,              zorder=5, label="Earth")
    ax3d.scatter(*rf,       color="tomato",     s=60,              zorder=5, label="Mars")
    r = diags["r"]
    ax3d.plot(r[:, 0], r[:, 1], r[:, 2], "o-", color=color, ms=2.5, lw=1.8)
    on = diags["on"]
    if np.any(on):
        ax3d.quiver(r[on, 0], r[on, 1], r[on, 2],
                    diags["alpha"][on, 0], diags["alpha"][on, 1], diags["alpha"][on, 2],
                    length=0.05, normalize=True, color="tab:red", alpha=0.5)
    ax3d.set_xlabel("x [AU]", fontsize=6); ax3d.set_ylabel("y [AU]", fontsize=6)
    ax3d.set_zlabel("z [AU]", fontsize=6); ax3d.tick_params(labelsize=5)
    ax3d.legend(fontsize=6, loc="upper left")

    xl = (float(t_day[0]), float(t_day[-1]))

    def _setup(ax, ttl, yl):
        ax.set_title(ttl, fontsize=8); ax.set_xlabel("time (days)", fontsize=6)
        ax.set_ylabel(yl, fontsize=6); ax.set_xlim(*xl)
        ax.tick_params(labelsize=5); ax.grid(True, alpha=0.25)

    # ── Position r(t) ─────────────────────────────────────────────────────────
    _setup(ax_r, "Position r(t)", "AU")
    for k, (c, lb) in enumerate(zip(_XYZ_COLORS, _XYZ_LABELS)):
        ax_r.plot(t_day, diags["r"][:, k], color=c, lw=1.3, label=lb)
    ax_r.legend(fontsize=6, loc="upper right")

    # ── Velocity v(t) ─────────────────────────────────────────────────────────
    _setup(ax_v, "Velocity v(t)", "AU/TU")
    for k, (c, lb) in enumerate(zip(_XYZ_COLORS, _XYZ_LABELS)):
        ax_v.plot(t_day, diags["v"][:, k], color=c, lw=1.3, label=lb)
    ax_v.legend(fontsize=6, loc="upper right")

    # ── Mass + thrust (twin axis) ─────────────────────────────────────────────
    ax_mt.set_title("Mass & Thrust δ", fontsize=8)
    ax_mt.set_xlabel("time (days)", fontsize=6)
    ax_mt.set_ylabel("mass (norm)", fontsize=6, color="tab:green")
    ax_mt.set_ylim(0.0, 1.05); ax_mt.set_xlim(*xl)
    ax_mt.tick_params(labelsize=5, axis="y", labelcolor="tab:green")
    ax_mt.grid(True, alpha=0.25)
    ax_t2.set_ylabel("thrust δ", fontsize=6, color="tab:red")
    ax_t2.yaxis.set_label_position("right")
    ax_t2.set_ylim(-0.05, 1.3); ax_t2.tick_params(labelsize=5, axis="y", labelcolor="tab:red")
    ax_mt.plot(t_day, diags["mass"],  color="tab:green", lw=1.3)
    ax_t2.plot(t_day, diags["delta"], color="tab:red",   lw=1.3)

    # ── Costate λ_r(t) ────────────────────────────────────────────────────────
    _setup(ax_lr, "Costate λ_r(t)", "")
    for k, (c, lb) in enumerate(zip(_XYZ_COLORS, _XYZ_LABELS)):
        ax_lr.plot(t_day, diags["lam_r"][:, k], color=c, lw=1.3, label=f"λ_r{lb}")
    ax_lr.legend(fontsize=6, loc="upper right")

    # ── Costate λ_v(t) ────────────────────────────────────────────────────────
    _setup(ax_lv, "Costate λ_v(t)", "")
    for k, (c, lb) in enumerate(zip(_XYZ_COLORS, _XYZ_LABELS)):
        ax_lv.plot(t_day, diags["lam_v"][:, k], color=c, lw=1.3, label=f"λ_v{lb}")
    ax_lv.legend(fontsize=6, loc="upper right")

    # ── Costate λ_m(t) ────────────────────────────────────────────────────────
    ax_lm.axhline(0.0, color="gray", lw=0.8, ls="--")
    _setup(ax_lm, "Costate λ_m(t)  [→0 at t_f]", "")
    ax_lm.set_ylim(-0.3, 1.3)
    ax_lm.plot(t_day, diags["lam_m"], color="tab:purple", lw=1.8)
    lm_f = float(diags["lam_m"][-1])
    ax_lm.scatter(t_day[-1], lm_f, color="tab:purple", s=35, zorder=5,
                  label=f"λ_m(tf)={lm_f:.4f}")
    ax_lm.legend(fontsize=6)

    # ── Continuity residuals ──────────────────────────────────────────────────
    ax_rs.set_xlabel("interval k", fontsize=7); ax_rs.tick_params(labelsize=6)
    ax_rs.set_ylabel("‖F(Z_k)−Z_{k+1}‖", fontsize=7)
    res_plot = np.maximum(res_arr, 1e-16)
    ax_rs.bar(np.arange(len(res_plot)), res_plot, color=color, alpha=0.75, width=0.8)
    ax_rs.set_yscale("log")
    ax_rs.set_title(f"Continuity residuals  (max={res_arr.max():.2e})", fontsize=8)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    except Exception:
        pass
    fig.savefig(str(out_path), dpi=82)
    plt.close(fig)


def _precompute_diags(z, norm, eps=1e-4):
    """Pure-numpy diagnostics from (T,14) trajectory."""
    delta, alpha, S = switching_function(z, norm, eps=eps)
    return dict(r=z[:, 0:3], v=z[:, 3:6], mass=z[:, 6],
                lam_r=z[:, 7:10], lam_v=z[:, 10:13], lam_m=z[:, 13],
                delta=delta, alpha=alpha, on=delta > 0.5, S=S)


def enqueue_frame(z, time_grid_tu, r_i_actual, r_f_actual, norm,
                  refiner, title, out_path, color):
    """Compute all diagnostics + residuals in main thread, submit PNG to pool.

    refiner: BVPRefiner instance whose continuity_residuals() is used.
    The function is cached inside the refiner so repeated calls are fast.
    """
    TU    = float(norm["TU"])
    t_day = np.asarray(time_grid_tu) * TU / 86400.0
    z     = np.asarray(z, dtype=np.float64)
    diags = _precompute_diags(z, norm, eps=float(refiner._eps))
    res_arr = refiner.continuity_residuals(z, n_rk4_eval=8)
    ri = np.asarray(r_i_actual, dtype=np.float64).ravel()[:3]
    rf = np.asarray(r_f_actual, dtype=np.float64).ravel()[:3]
    _SAVE_POOL.submit(
        _worker_save_frame,
        z.copy(), t_day.copy(), ri, rf,
        {k: np.array(v) for k, v in diags.items()},
        res_arr, title, color, Path(out_path),
    )


def solve_bvp_oneshot(refiner, solver, z_init, initial_state, final_state):
    """One-shot BVP solve; returns (z_opt, status_str, max_cont_residual)."""
    T      = refiner.n_points
    D      = AUGMENTED_DIM
    z_flat = np.asarray(z_init, dtype=np.float64).reshape(-1)
    lbw, ubw = make_bounds(initial_state, final_state, T)
    sol    = solver(x0=z_flat, lbx=lbw, ubx=ubw)
    z_opt  = sol["x"].full().reshape(T, D).astype(np.float64)
    stats  = solver.stats()
    res    = float(np.max(refiner.continuity_residuals(z_opt, n_rk4_eval=8)))
    return z_opt, stats["return_status"], res


# ─────────────────────────────────────────────────────────────────────────────
# Method B — ε-continuation CasADi BVP
# ─────────────────────────────────────────────────────────────────────────────

EPS_STAGES = [1.0, 1e-1, 1e-2, 1e-3, 1e-4]

NON_FATAL = frozenset({
    "Solve_Succeeded",
    "Solved_To_Acceptable_Level",
    "Maximum_Iterations_Exceeded",
})


def make_continuation_guess(f_fwd, initial_state, final_state, norm, n_points, rng_seed):
    """Sample λ(t_0) ~ U[-2,2]^7, forward-integrate eps=1.0 dynamics; return (T,14)."""
    rng       = np.random.default_rng(rng_seed)
    costate_0 = rng.uniform(-2.0, 2.0, 7).astype(np.float64)
    Z0        = np.concatenate([initial_state[:7].astype(np.float64), costate_0])
    t_f       = float(norm["t_f"])
    t_eval    = np.linspace(0.0, t_f, n_points)

    def rhs(_, Z):
        return np.asarray(f_fwd(Z[:7], Z[7:])).flatten()

    try:
        sol = solve_ivp(rhs, [0.0, t_f], Z0, t_eval=t_eval,
                        method="DOP853", rtol=1e-8, atol=1e-9)
        if sol.success and sol.y.shape[1] == n_points:
            return sol.y.T.astype(np.float64)
    except Exception:
        pass

    # Fallback: linear states, constant costate
    z = np.zeros((n_points, 14), dtype=np.float64)
    for j in range(6):
        z[:, j] = np.linspace(float(initial_state[j]), float(final_state[j]), n_points)
    z[:, 6]   = np.linspace(float(initial_state[6]), float(initial_state[6]) * 0.5, n_points)
    z[:, 7:]  = costate_0[None, :]
    z[-1, 13] = 0.0
    return z


def build_continuation_solvers(refiners, max_iter_per_stage, iter_cbs=None):
    """Pre-compile one CasADi solver per eps stage."""
    if iter_cbs is None:
        return [r._make_solver(max_iter=max_iter_per_stage, cb=None) for r in refiners]
    return [r._make_solver(max_iter=max_iter_per_stage, cb=cb)
            for r, cb in zip(refiners, iter_cbs)]


def solve_bvp_continuation(refiners, solvers, z_init, initial_state, final_state):
    """Five-stage ε-continuation BVP.

    Returns (z_final, status, max_res_at_1e-4, stage_log, z_stages).
    stage_log  : list of (eps, status_str, n_iter).
    z_stages   : list of (T,14) arrays — final trajectory after each stage.
    """
    T   = refiners[0].n_points
    D   = AUGMENTED_DIM
    lbw, ubw = make_bounds(initial_state.astype(np.float64),
                           final_state.astype(np.float64), T)
    z         = z_init.astype(np.float64)
    stage_log = []
    z_stages  = []

    for i, (_, solver, eps) in enumerate(zip(refiners, solvers, EPS_STAGES)):
        is_final = (i == len(refiners) - 1)
        sol    = solver(x0=z.reshape(-1), lbx=lbw, ubx=ubw)
        z      = sol["x"].full().reshape(T, D).astype(np.float64)
        stats  = solver.stats()
        status = stats["return_status"]
        n_iter = stats.get("iter_count", -1)
        stage_log.append((eps, status, n_iter))
        z_stages.append(z.copy())

        if not is_final and status not in NON_FATAL:
            res = float(np.max(refiners[-1].continuity_residuals(z, n_rk4_eval=8)))
            return z, f"EarlyAbort_{status}", res, stage_log, z_stages

    res = float(np.max(refiners[-1].continuity_residuals(z, n_rk4_eval=8)))
    return z, status, res, stage_log, z_stages


# ─────────────────────────────────────────────────────────────────────────────
# Keplerian propagation + shifted BCs
# ─────────────────────────────────────────────────────────────────────────────

def _kepler_rhs(_, y, mu):
    r, v = y[:3], y[3:6]
    return np.concatenate([v, -mu * r / np.linalg.norm(r) ** 3])


def propagate_kepler(r, v, mu, dt):
    if dt == 0.0:
        return np.array(r, np.float64), np.array(v, np.float64)
    y0  = np.concatenate([np.array(r, np.float64), np.array(v, np.float64)])
    sol = solve_ivp(_kepler_rhs, (0.0, dt), y0, args=(mu,),
                    method="DOP853", rtol=1e-12, atol=1e-13)
    return sol.y[:3, -1], sol.y[3:6, -1]


def shifted_bcs(norm, shift_days):
    """Keplerian-propagate Earth/Mars by shift_days; return (initial, final) float32."""
    shift_tu = shift_days * 24.0 * 3600.0 / norm["TU"]
    mu       = float(norm["mu"])
    r_i, v_i = propagate_kepler(norm["r_i"], norm["v_i"], mu, shift_tu)
    r_f, v_f = propagate_kepler(norm["r_f"], norm["v_f"], mu, shift_tu)
    m0       = float(norm["m0"])
    return (
        np.concatenate([r_i, v_i, [m0]]).astype(np.float32),
        np.concatenate([r_f, v_f, [m0]]).astype(np.float32),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-shift text summary (always written)
# ─────────────────────────────────────────────────────────────────────────────

def write_shift_detail_txt(shift, trial_log, num_trials, out_dir, res_tol=1e-8):
    """Write per-trial table + B stage convergence to detail.txt for one shift."""
    n = len(trial_log)
    if n == 0:
        return
    shift_dir = out_dir / f"shift_{shift:+d}d"
    shift_dir.mkdir(parents=True, exist_ok=True)
    fpath = shift_dir / "detail.txt"

    conv_a = sum(r[0] for r in trial_log)
    conv_b = sum(r[1] for r in trial_log)

    lines = []
    lines.append(f"shift = {shift:+d} d   ({n}/{num_trials} trials done)")
    lines.append(f"res_tol = {res_tol:.0e}")
    lines.append(f"A: {conv_a}/{n} converged ({100*conv_a/n:.0f}%)    "
                 f"B: {conv_b}/{n} converged ({100*conv_b/n:.0f}%)")
    lines.append("")
    lines.append(f"{'seed':>4}  {'A_ok':>4}  {'A_res':>10}  {'A_t(s)':>7}  "
                 f"{'B_ok':>4}  {'B_res':>10}  {'B_t(s)':>7}  B_stages")
    lines.append("-" * 80)
    for i, (ok_a, ok_b, res_a, res_b, stage_log, t_a, t_b) in enumerate(trial_log):
        stages = " ".join(f"{e:.0e}:{s[:4]}" for e, s, _ in stage_log)
        lines.append(f"{i:>4}  {'OK' if ok_a else '--':>4}  {res_a:>10.3e}  {t_a:>7.1f}  "
                     f"{'OK' if ok_b else '--':>4}  {res_b:>10.3e}  {t_b:>7.1f}  {stages}")

    if trial_log and trial_log[0][4]:
        lines.append("")
        lines.append("B stage convergence:")
        stage_logs = [r[4] for r in trial_log]
        eps_list = [e for e, _, _ in stage_logs[0]]
        for si, eps in enumerate(eps_list):
            cnt = sum(1 for sl in stage_logs
                      if si < len(sl) and ("Succeed" in sl[si][1] or "Acceptable" in sl[si][1]))
            lines.append(f"  eps={eps:.0e}:  {cnt}/{n}")

    with open(fpath, "w") as fp:
        fp.write("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Per-shift figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_shift_detail(shift, trial_log, num_trials, out_dir, res_tol=1e-4):
    """Save residual scatter + stage-success bar for one shift value."""
    n = len(trial_log)
    if n == 0:
        return
    seeds  = np.arange(n)
    res_a  = np.array([r[2] for r in trial_log])
    res_b  = np.array([r[3] for r in trial_log])
    ok_a   = np.array([r[0] for r in trial_log], dtype=bool)
    ok_b   = np.array([r[1] for r in trial_log], dtype=bool)
    t_a    = np.array([r[5] for r in trial_log])
    t_b    = np.array([r[6] for r in trial_log])
    stage_logs = [r[4] for r in trial_log]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"shift = {shift:+d} d   ({n}/{num_trials} trials done)", fontsize=11)

    # ── Left: residuals scatter ──────────────────────────────────────────────
    ax = axes[0]
    ax.semilogy(seeds[ok_a],  res_a[ok_a],  "g^", ms=7, label="A OK")
    ax.semilogy(seeds[~ok_a], res_a[~ok_a], "gv", ms=7, alpha=0.4, label="A fail")
    ax.semilogy(seeds[ok_b],  res_b[ok_b],  "bs", ms=7, label="B OK")
    ax.semilogy(seeds[~ok_b], res_b[~ok_b], "bx", ms=7, alpha=0.4, label="B fail")
    ax.axhline(res_tol, color="red", lw=0.8, ls="--", label=f"tol={res_tol:.0e}")
    ax.set_xlabel("trial seed"); ax.set_ylabel("max cont. residual")
    ax.set_title("Residuals"); ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.3)

    # ── Middle: wall-clock time ──────────────────────────────────────────────
    ax = axes[1]
    w = 0.38
    x = seeds
    ax.bar(x - w/2, t_a, w, color="limegreen", alpha=0.8, label="A")
    ax.bar(x + w/2, t_b, w, color="darkorange", alpha=0.8, label="B")
    ax.set_xlabel("trial seed"); ax.set_ylabel("wall-clock (s)")
    ax.set_title("Time per trial"); ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # ── Right: B stage success ───────────────────────────────────────────────
    ax = axes[2]
    if stage_logs and stage_logs[0]:
        eps_list = [e for e, _, _ in stage_logs[0]]
        n_stages = len(eps_list)
        stage_conv = np.zeros(n_stages, dtype=int)
        for sl in stage_logs:
            for si2, (_, st, _) in enumerate(sl):
                if "Succeed" in st or "Acceptable" in st:
                    stage_conv[si2] += 1
        ax.bar(range(n_stages), stage_conv, color="darkorange", alpha=0.85)
        ax.set_xticks(range(n_stages))
        ax.set_xticklabels([f"ε={e:.0e}" for e in eps_list], fontsize=7, rotation=30)
        ax.set_ylabel("# trials converged"); ax.set_ylim(0, n + 1)
        ax.set_title("B stage convergence")
        ax.grid(True, axis="y", alpha=0.3)
    else:
        ax.set_visible(False)

    shift_dir = out_dir / f"shift_{shift:+d}d"
    shift_dir.mkdir(parents=True, exist_ok=True)
    fpath = shift_dir / "detail.png"
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(str(fpath), dpi=120)
    plt.close(fig)
    print(f"  fig → {fpath}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary(shift_results, num_trials, out_path):
    shifts = [r["shift"] for r in shift_results]
    pct_a  = [r["conv_A"] / num_trials * 100 for r in shift_results]
    pct_b  = [r["conv_B"] / num_trials * 100 for r in shift_results]

    x, w = np.arange(len(shifts)), 0.38
    fig, ax = plt.subplots(figsize=(max(10, len(shifts) * 0.75), 4.5))
    ba = ax.bar(x - w/2, pct_a, w, label="A: Diff+CasADi",
                color="limegreen", alpha=0.85, edgecolor="white")
    bb = ax.bar(x + w/2, pct_b, w, label="B: ε-Cont. BVP",
                color="darkorange", alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:+d}d" for s in shifts], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Convergence rate (%)", fontsize=10)
    ax.set_xlabel("Time shift Δt (days)", fontsize=10)
    ax.set_title(
        f"Convergence rate vs. time shift  ({num_trials} trials/shift)\n"
        "A: Diffusion + CasADi   |   B: ε-continuation CasADi",
        fontsize=10,
    )
    ax.set_ylim(0, 115)
    ax.axhline(100, color="gray", lw=0.7, ls="--", alpha=0.4)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(ba, pct_a):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=6.5, color="darkgreen")
    for bar, val in zip(bb, pct_b):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=6.5, color="saddlebrown")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(str(out_path), dpi=130)
    plt.close(fig)
    print(f"  saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI + main
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SHIFTS = [-700, -600, -500, -400, -300, -200, -100, -50, 0,
                  +50, +100, +200, +300, +400, +500, +600, +700]


def save_trial_json(trial_dir, shift, seed, initial_state, final_state,
                    time_grid_tu, norm,
                    diff_frames, a_ipopt_frames,
                    z_opt_a, st_a, res_a, t_a, ok_a,
                    z_b_init, b_stage_iterates, z_stages_b, stage_log,
                    st_b, res_b, t_b, ok_b):
    """Persist all raw trial data to JSON for offline plot regeneration."""

    def _f32(x):
        return np.asarray(x, dtype=np.float32).tolist()

    def _serialize_norm(n):
        out = {}
        for k, v in n.items():
            if isinstance(v, np.ndarray):
                out[k] = v.astype(np.float64).tolist()
            elif isinstance(v, (float, np.floating)):
                out[k] = float(v)
            elif isinstance(v, (int, np.integer)):
                out[k] = int(v)
            else:
                out[k] = str(v)
        return out

    # Build stage dicts for B
    stages_data = []
    for si, ((eps_b, st_b_s, ni_b), iters_b) in enumerate(
            zip(stage_log, b_stage_iterates)):
        stages_data.append({
            "eps": float(eps_b),
            "status": st_b_s,
            "n_iter": int(ni_b),
            "iterates": [_f32(z) for z in iters_b],
            "z_final": _f32(z_stages_b[si]),
        })

    data = {
        "shift_days": int(shift),
        "seed": int(seed),
        "initial_state": _f32(initial_state),
        "final_state": _f32(final_state),
        "time_grid_tu": _f32(time_grid_tu),
        "norm": _serialize_norm(norm),
        "method_A": {
            "status": str(st_a),
            "res": float(res_a),
            "ok": bool(ok_a),
            "time_s": float(t_a),
            "diffusion_frames": [_f32(f) for f in diff_frames],
            "ipopt_iterates": [_f32(f) for f in a_ipopt_frames],
            "z_opt": _f32(z_opt_a),
        },
        "method_B": {
            "status": str(st_b),
            "res": float(res_b),
            "ok": bool(ok_b),
            "time_s": float(t_b),
            "z_init": _f32(z_b_init),
            "stages": stages_data,
        },
    }

    out_path = Path(trial_dir) / "trial_data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump(data, fp, separators=(",", ":"))
    kb = out_path.stat().st_size / 1024
    print(f"    json → {out_path}  ({kb:.0f} KB)", flush=True)


def write_live_summary(out_path, args, shifts, completed_results,
                       cur_si, cur_shift, cur_trial_log, t_start):
    """Overwrite summary.txt after each trial with current running totals."""
    N_trials = args.num_trials
    N_shifts  = len(shifts)
    elapsed   = time.perf_counter() - t_start

    lines = []
    lines.append("shift_eps_comparison — live progress")
    lines.append(f"  A: diff_steps={args.num_diffusion_steps}  "
                 f"casadi_iter={args.casadi_max_iter}")
    lines.append(f"  B: {len(EPS_STAGES)} stages x {args.casadi_max_iter_b} "
                 f"iter/stage")
    lines.append(f"  res_tol={args.res_tol}  shifts={N_shifts}  "
                 f"trials/shift={N_trials}")
    lines.append(f"  elapsed: {elapsed/60:.1f} min")
    lines.append("")
    lines.append(f"  {'shift':>7}  {'A conv':>8}  {'A%':>5}  "
                 f"{'B conv':>8}  {'B%':>5}  status")
    lines.append("  " + "-" * 64)

    for r in completed_results:
        pct_a = 100.0 * r["conv_A"] / N_trials
        pct_b = 100.0 * r["conv_B"] / N_trials
        lines.append(f"  {r['shift']:>+7d}  "
                     f"{r['conv_A']:>2}/{N_trials:<5}  {pct_a:>4.0f}%  "
                     f"{r['conv_B']:>2}/{N_trials:<5}  {pct_b:>4.0f}%  DONE")

    if cur_trial_log:
        n_done = len(cur_trial_log)
        conv_a = sum(r[0] for r in cur_trial_log)
        conv_b = sum(r[1] for r in cur_trial_log)
        pct_a  = 100.0 * conv_a / n_done
        pct_b  = 100.0 * conv_b / n_done
        lines.append(f"  {cur_shift:>+7d}  "
                     f"{conv_a:>2}/{n_done:<5}  {pct_a:>4.0f}%  "
                     f"{conv_b:>2}/{n_done:<5}  {pct_b:>4.0f}%  "
                     f"[{n_done}/{N_trials} trials]")

    for shift in shifts[cur_si + 1:]:
        lines.append(f"  {shift:>+7d}  {'':8}  {'':5}  {'':8}  {'':5}  ...")

    lines.append("")
    lines.append(f"  -- current shift={cur_shift:+d}d trial detail --")
    for i, (ok_a, ok_b, res_a, res_b, stage_log, ta, tb) in enumerate(
            cur_trial_log):
        stages = " ".join(f"{e:.0e}:{s[:4]}" for e, s, _ in stage_log)
        lines.append(f"    s{i:02d}  "
                     f"A={'OK' if ok_a else '--'}({res_a:.1e} {ta:.0f}s)  "
                     f"B={'OK' if ok_b else '--'}({res_b:.1e} {tb:.0f}s)"
                     f"  [{stages}]")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path,
                   default=Path("checkpoints/indiff_ctrl_latest.msgpack"))
    p.add_argument("--shifts-days", type=int, nargs="+", default=DEFAULT_SHIFTS,
                   metavar="DAYS")
    p.add_argument("--num-trials",          type=int,   default=10)
    p.add_argument("--num-diffusion-steps", type=int,   default=30)
    p.add_argument("--no-plots",            action="store_true",
                   help="Disable all PNG/JSON output; write detail.txt per shift only")
    p.add_argument("--no-frames",           action="store_true",
                   help="Skip frame PNGs but still save trial_data.json")
    p.add_argument("--casadi-max-iter",     type=int,   default=500,
               help="IPOPT max iters for Method A")
    p.add_argument("--casadi-max-iter-b",   type=int,   default=240,
                   help="IPOPT max iters per stage for Method B (5 stages)")
    p.add_argument("--res-tol",             type=float, default=1e-8)
    p.add_argument("--seed-offset",         type=int,   default=0,
                   help="Start seeds at this value (use to continue after a previous run)")
    p.add_argument("--seeds",              type=int,   nargs="+", default=None,
                   help="Explicit seed values to run for BOTH methods (overrides --seed-offset / --num-trials)")
    p.add_argument("--seeds-a",            type=int,   nargs="+", default=None,
                   help="Seed values for Method A only (overrides --seeds for A)")
    p.add_argument("--seeds-b",            type=int,   nargs="+", default=None,
                   help="Seed values for Method B only (overrides --seeds for B)")
    p.add_argument("--output-dir",          type=Path,  default=Path("shift_eps_study"))
    return p.parse_args()


def main():
    args = parse_args()

    if not args.checkpoint.exists():
        sys.exit(f"Checkpoint not found: {args.checkpoint}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    params, policy, time_grid, norm = load_checkpoint(args.checkpoint)
    _apply   = build_jit_apply(policy)
    n_points = len(time_grid)

    # ── Build refiners once (reused across all shifts and trials) ─────────────
    t0 = time.perf_counter()
    print(f"\nBuilding refiners and solvers (compiled once, reused for all trials) …")

    print(f"  Method A (eps=1e-4) …", end="", flush=True)
    refiner_A  = BVPRefiner(n_points=n_points, n_rk4_steps=8, ipopt_verbosity=0)
    iter_cb_A  = IterCapture(n_x=n_points * AUGMENTED_DIM)
    solver_A   = refiner_A._make_solver(max_iter=args.casadi_max_iter, cb=iter_cb_A)
    print(f" done")

    print(f"  Method B ({len(EPS_STAGES)} stages) …")
    cont_refiners = []
    iter_cbs_b    = []
    for eps in EPS_STAGES:
        print(f"    eps={eps:.0e} …", end="", flush=True)
        cont_refiners.append(
            BVPRefiner(eps=eps, n_points=n_points, n_rk4_steps=8, ipopt_verbosity=0)
        )
        iter_cbs_b.append(IterCapture(n_x=n_points * AUGMENTED_DIM))
        print(" done")
    cont_solvers = build_continuation_solvers(
        cont_refiners, args.casadi_max_iter_b, iter_cbs_b)
    print(f"  Total build+compile: {time.perf_counter()-t0:.1f}s")

    # ── Forward-dynamics function for Method B guess (eps=1.0, built once) ───
    dy_1, _, _ = make_dynamics(eps=1.0, compile_jax=False)
    f_fwd = ca.Function("f_fwd", [dy_1.states, dy_1.costates], [dy_1.augmented_dot_sub])

    # ── Study ─────────────────────────────────────────────────────────────────
    shifts   = args.shifts_days
    N_shifts = len(shifts)
    N_trials = args.num_trials
    _base_seeds = args.seeds if args.seeds else list(range(args.seed_offset, args.seed_offset + N_trials))
    seeds_a = args.seeds_a if args.seeds_a is not None else _base_seeds
    seeds_b = args.seeds_b if args.seeds_b is not None else _base_seeds
    seeds_a_set = set(seeds_a)
    seeds_b_set = set(seeds_b)
    # union of all seeds to iterate over; each method runs only its own subset
    all_seeds = sorted(set(seeds_a) | set(seeds_b))

    print("\n" + "=" * 72)
    print(f"  {N_shifts} shifts  A: {len(seeds_a)} seeds  B: {len(seeds_b)} seeds")
    print(f"  A: diff_steps={args.num_diffusion_steps}  max_iter={args.casadi_max_iter}")
    print(f"  B: {len(EPS_STAGES)} stages × {args.casadi_max_iter_b} iter/stage  "
          f"({len(EPS_STAGES) * args.casadi_max_iter_b} total max)")
    print(f"{'='*72}")

    shift_results = []
    t_study = time.perf_counter()

    for si, shift in enumerate(shifts):
        initial_state, final_state = shifted_bcs(norm, shift)
        i_st   = initial_state.astype(np.float64)
        f_st   = final_state.astype(np.float64)
        sample = {"initial_state": initial_state, "final_state": final_state}

        print(f"\n[{si+1}/{N_shifts}]  shift={shift:+d}d  "
              f"r_i={initial_state[:3].round(3)}  r_f={final_state[:3].round(3)}")

        trial_log = []  # (ok_a, ok_b, res_a, res_b, stage_log, t_a, t_b)
        lbw_b, ubw_b = make_bounds(i_st, f_st, n_points)  # fixed per shift

        for seed in all_seeds:
            run_a = seed in seeds_a_set
            run_b = seed in seeds_b_set
            trial_dir = args.output_dir / f"shift_{shift:+d}d" / f"trial_{seed:02d}"
            save_frames = not args.no_plots and not args.no_frames

            # ── Method A ───────────────────────────────────────────────────────
            if run_a:
                t_a = time.perf_counter()
                diff_frames = run_diffusion(policy, params, sample, _apply,
                                            num_steps=args.num_diffusion_steps, rng_seed=seed)
                iter_cb_A.iterates.clear()
                z_diff  = diff_frames[-1]
                z_opt_a, st_a, res_a = solve_bvp_oneshot(
                    refiner_A, solver_A, z_diff, i_st, f_st)
                t_a  = time.perf_counter() - t_a
                ok_a = res_a < args.res_tol

                # Always collect IPOPT iterates for JSON (needed for video generation)
                a_ipopt = (
                    [np.asarray(z_diff, np.float64).reshape(n_points, AUGMENTED_DIM)]
                    + [it.reshape(n_points, AUGMENTED_DIM) for it in iter_cb_A.iterates]
                    + [z_opt_a]
                )
                if len(a_ipopt) >= 2 and np.allclose(a_ipopt[-1], a_ipopt[-2], atol=1e-12):
                    a_ipopt = a_ipopt[:-1]

                if save_frames:
                    diff_dir = trial_dir / "A" / "diffusion"
                    n_df = len(diff_frames)
                    for fi, z_df in enumerate(diff_frames):
                        enqueue_frame(
                            z_df, refiner_A.time_grid, i_st[:3], f_st[:3], norm,
                            refiner_A,
                            f"A s{seed:02d}  diffusion step {fi+1}/{n_df}",
                            diff_dir / f"frame_{fi:03d}.png",
                            color=_DIFF_COLOR,
                        )
                    ipopt_dir = trial_dir / "A" / "ipopt"
                    n_ai = len(a_ipopt)
                    for fi, z_af in enumerate(a_ipopt):
                        enqueue_frame(
                            z_af, refiner_A.time_grid, i_st[:3], f_st[:3], norm,
                            refiner_A,
                            f"A s{seed:02d}  IPOPT iter {fi}/{n_ai-1}  {st_a[:16]}  res={res_a:.1e}",
                            ipopt_dir / f"iter_{fi:03d}.png",
                            color=_IPOPT_COLOR,
                        )
            else:
                diff_frames = []
                z_opt_a = np.zeros((n_points, AUGMENTED_DIM), np.float64)
                st_a, res_a, t_a, ok_a = "skipped", float("inf"), 0.0, False
                a_ipopt = []

            # ── Method B ───────────────────────────────────────────────────────
            if run_b:
                t_b = time.perf_counter()
                z_b_init = make_continuation_guess(
                    f_fwd, initial_state, final_state, norm, n_points, seed)
                for cb_b in iter_cbs_b:
                    cb_b.iterates.clear()
                _, st_b, res_b, stage_log, z_stages_b = solve_bvp_continuation(
                    cont_refiners, cont_solvers, z_b_init, i_st, f_st)
                t_b  = time.perf_counter() - t_b
                ok_b = res_b < args.res_tol

                if save_frames:
                    b_stage_iterates_all = []
                    for si_b, (eps_b, st_b_s, _) in enumerate(stage_log):
                        stage_dir = trial_dir / "B" / f"stage_{si_b}_eps{eps_b:.0e}"
                        prev_z = (z_b_init.reshape(n_points, AUGMENTED_DIM)
                                  if si_b == 0 else z_stages_b[si_b - 1])
                        b_iters = (
                            [prev_z]
                            + [it.reshape(n_points, AUGMENTED_DIM)
                               for it in iter_cbs_b[si_b].iterates]
                            + [z_stages_b[si_b]]
                        )
                        if len(b_iters) >= 2 and np.allclose(b_iters[-1], b_iters[-2], atol=1e-12):
                            b_iters = b_iters[:-1]
                        b_stage_iterates_all.append(b_iters)
                        n_bi = len(b_iters)
                        for fi_b, z_bf in enumerate(b_iters):
                            enqueue_frame(
                                z_bf, refiner_A.time_grid, i_st[:3], f_st[:3], norm,
                                cont_refiners[si_b],
                                (f"B s{seed:02d}  stage {si_b+1}/5  ε={eps_b:.0e}"
                                 f"  iter {fi_b}/{n_bi-1}  {st_b_s[:16]}"),
                                stage_dir / f"iter_{fi_b:03d}.png",
                                color=_CONT_COLOR,
                            )
                else:
                    b_stage_iterates_all = []
            else:
                z_b_init = np.zeros((n_points, AUGMENTED_DIM), np.float64)
                z_stages_b, stage_log = [], []
                st_b, res_b, t_b, ok_b = "skipped", float("inf"), 0.0, False
                b_stage_iterates_all = []

            if not args.no_plots:
                # Save all raw data to JSON (blocking — data integrity)
                save_trial_json(
                    trial_dir, shift, seed, initial_state, final_state,
                    refiner_A.time_grid, norm,
                    diff_frames, a_ipopt,
                    z_opt_a, st_a, res_a, t_a, ok_a,
                    z_b_init, b_stage_iterates_all, z_stages_b, stage_log,
                    st_b, res_b, t_b, ok_b,
                )

            run_a_str = f"A={'OK' if ok_a else ('--' if run_a else 'sk')}({res_a:.1e} {t_a:.0f}s)"
            run_b_str = f"B={'OK' if ok_b else ('--' if run_b else 'sk')}({res_b:.1e} {t_b:.0f}s)"
            stages_str = " ".join(f"{e:.0e}:{s[:4]}" for e, s, _ in stage_log) if stage_log else "skipped"
            print(f"  s{seed:02d}  {run_a_str}  {run_b_str}  [{stages_str}]", flush=True)
            trial_log.append((ok_a, ok_b, res_a, res_b, stage_log, t_a, t_b))

            stages = " ".join(f"{e:.0e}:{s[:4]}" for e, s, _ in stage_log)
            print(f"  s{seed:02d}  "
                  f"A={'OK' if ok_a else '--'}({res_a:.1e} {t_a:.0f}s)  "
                  f"B={'OK' if ok_b else '--'}({res_b:.1e} {t_b:.0f}s)  [{stages}]",
                  flush=True)

            write_live_summary(
                args.output_dir / "summary.txt", args, shifts,
                shift_results, si, shift, trial_log, t_study)

        conv_A = sum(r[0] for r in trial_log)
        conv_B = sum(r[1] for r in trial_log)
        shift_results.append({
            "shift":      shift,
            "conv_A":     conv_A,
            "conv_B":     conv_B,
            "res_A_mean": float(np.mean([r[2] for r in trial_log])),
            "res_B_mean": float(np.mean([r[3] for r in trial_log])),
        })
        print(f"  ── A: {conv_A}/{N_trials} ({100*conv_A/N_trials:.0f}%)  "
              f"B: {conv_B}/{N_trials} ({100*conv_B/N_trials:.0f}%)")
        write_shift_detail_txt(shift, trial_log, N_trials, args.output_dir, res_tol=args.res_tol)
        if not args.no_plots:
            plot_shift_detail(shift, trial_log, N_trials, args.output_dir, res_tol=args.res_tol)

    t_elapsed = time.perf_counter() - t_study

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(f"  {'shift':>7}  {'A':>8}  {'A%':>5}  {'B':>8}  {'B%':>5}  "
          f"{'A res':>9}  {'B res':>9}")
    print(f"  {'-'*7}  {'-'*8}  {'-'*5}  {'-'*8}  {'-'*5}  {'-'*9}  {'-'*9}")
    for r in shift_results:
        print(f"  {r['shift']:>+7d}  "
              f"{r['conv_A']:>2}/{N_trials:<5}  {100*r['conv_A']/N_trials:>4.0f}%  "
              f"{r['conv_B']:>2}/{N_trials:<5}  {100*r['conv_B']/N_trials:>4.0f}%  "
              f"{r['res_A_mean']:>9.2e}  {r['res_B_mean']:>9.2e}")
    print(f"{'='*68}")
    print(f"  Total: {t_elapsed/60:.1f} min")

    # ── Save ──────────────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    txt = args.output_dir / "results.txt"
    with open(txt, "w") as fp:
        fp.write(f"shifts={N_shifts}  trials={N_trials}  res_tol={args.res_tol}\n")
        fp.write(f"A: diff_steps={args.num_diffusion_steps}  "
                 f"casadi_iter={args.casadi_max_iter}\n")
        fp.write(f"B: {len(EPS_STAGES)} stages x {args.casadi_max_iter_b} iter/stage\n\n")
        fp.write(f"{'shift':>7}  {'A_conv':>8}  {'A%':>5}  {'B_conv':>8}  {'B%':>5}  "
                 f"{'A_res':>12}  {'B_res':>12}\n")
        for r in shift_results:
            fp.write(f"{r['shift']:>+7d}  "
                     f"{r['conv_A']:>2}/{N_trials:<5}  {100*r['conv_A']/N_trials:>4.0f}%  "
                     f"{r['conv_B']:>2}/{N_trials:<5}  {100*r['conv_B']/N_trials:>4.0f}%  "
                     f"{r['res_A_mean']:>12.3e}  {r['res_B_mean']:>12.3e}\n")
    print(f"  saved → {txt}")

    if not args.no_plots:
        plot_summary(shift_results, N_trials, args.output_dir / "summary.png")


if __name__ == "__main__":
    main()
