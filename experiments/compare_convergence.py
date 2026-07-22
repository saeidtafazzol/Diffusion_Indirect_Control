"""compare_convergence.py
========================
Convergence-rate study:

  Method A — Diffusion (N steps, seed s) → CasADi IPOPT
  Method B — CasADi alone   (linear-state + random-costate initial guess, seed s)

For Method B the initial guess is:
  states  z[:, 0:7]  : linspace from initial_state → final_state  (T points)
  costates z[:, 7:13] : uniform random in [-1, 1]  (seeded)
  λ_m     z[:, 13]   : 0.5 everywhere (final value is pinned to 0 by the BVP bounds)

Baseline correct solution: seed=42, Method A.

"converged" = IPOPT Solve_Succeeded  AND  max_cont_residual < --res-tol
"correct"   = converged  AND  max |z_opt[:, 0:6] − z_base[:, 0:6]| < --sol-tol
              (compares only r+v so costate scale differences don't inflate the norm)

Usage
-----
python compare_convergence.py \\
    --checkpoint checkpoints/indiff_ctrl_latest.msgpack \\
    --earth-mars \\
    [--num-seeds 100] \\
    [--num-diffusion-steps 30] \\
    [--casadi-max-iter 200] \\
    [--res-tol 1e-3] \\
    [--sol-tol 0.1]
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── JAX persistent compilation cache ─────────────────────────────────────────
_JAX_CACHE = Path.home() / ".cache" / "jax_compile_cache"
_JAX_CACHE.mkdir(parents=True, exist_ok=True)
try:
    jax.config.update("jax_compilation_cache_dir", str(_JAX_CACHE))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
except Exception:
    pass

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
from casadi_bvp_refine import BVPRefiner


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loader (minimal — same as generate_video_refined.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: Path, args):
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
    def _sc(k, d): v = sc.get(k, d); return d if v is None else v

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
    opt_tpl = optax.adamw(optax.linear_schedule(1e-4, 5e-6, 1), weight_decay=1e-6).init(params_tpl)
    sch_tpl = {"num_train_timesteps": 5000, "beta_schedule": "squaredcos_cap_v2",
               "prediction_type": "epsilon", "clip_sample": True,
               "clip_sample_range": 5.0, "noise_scale": 2.0}
    state_tpl = {"params": params_tpl, "opt_state": opt_tpl,
                 "step": np.int32(0), "time_grid": np.zeros(seq_len, np.float32),
                 "scheduler_config": sch_tpl}
    loaded = serialization.from_bytes(state_tpl, raw_bytes)
    params = loaded["params"]
    print(f"Checkpoint loaded: step={int(loaded['step'])}  seq_len={seq_len}")

    _, _, norm = make_dynamics(eps=1e-4)
    return params, policy, tg, norm


# ─────────────────────────────────────────────────────────────────────────────
# Build JIT-compiled diffusion forward pass (once, reused across all seeds)
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


# ─────────────────────────────────────────────────────────────────────────────
# Run diffusion: return final (T, 14) float32 array only
# ─────────────────────────────────────────────────────────────────────────────

def run_diffusion_final(
    policy: IndiffCtrlPolicy,
    params,
    sample: dict,
    _apply,           # JIT-compiled forward pass
    num_steps: int,
    rng_seed: int,
) -> np.ndarray:
    """Run full reverse diffusion, return final (T, 14) float32."""
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

    for ts in policy.noise_scheduler.timesteps:
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
            model_output=model_out, timestep=int(ts),
            sample=traj, rng_key=step_key,
        ).prev_sample
        traj = jnp.where(cond_mask, cond_traj, traj)

    jax.block_until_ready(traj)
    return np.asarray(traj[0], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# CasADi-alone initial guess
# ─────────────────────────────────────────────────────────────────────────────

def make_casadi_only_guess(
    initial_state: np.ndarray,   # (7,)
    final_state: np.ndarray,     # (7,)
    n_points: int,
    rng_seed: int,
) -> np.ndarray:
    """Build (T, 14) initial guess without diffusion.

    States   z[:, 0:7]  : linspace from initial_state → final_state
    λ_r, λ_v z[:, 7:13] : uniform random in [-1, 1]
    λ_m      z[:, 13]   : 0.5 (BVP pins final λ_m=0 via bounds anyway)
    """
    T   = n_points
    rng = np.random.default_rng(rng_seed)
    z   = np.zeros((T, AUGMENTED_DIM), dtype=np.float64)
    for j in range(6):   # r and v only — linspace initial → final
        z[:, j] = np.linspace(float(initial_state[j]), float(final_state[j]), T)
    # mass: linspace from m0 down to half m0 (final mass is free in BVP)
    z[:, 6] = np.linspace(float(initial_state[6]), float(initial_state[6]) * 0.5, T)
    z[:, STATE_DIM:] = rng.uniform(-1.0, 1.0, (T, COSTATE_DIM))  # all costates random
    z[-1, AUGMENTED_DIM - 1] = 0.0   # terminal λ_m = 0  (BVP transversality)
    return z


# ─────────────────────────────────────────────────────────────────────────────
# Single BVP solve: returns (z_opt, status, max_cont_residual)
# ─────────────────────────────────────────────────────────────────────────────

def solve_bvp(
    refiner: BVPRefiner,
    z_init: np.ndarray,
    initial_state: np.ndarray,
    final_state: np.ndarray,
    max_iter: int,
) -> tuple:
    """One-shot BVP solve. Returns (z_opt, status_str, max_cont_residual)."""
    from casadi_bvp_refine import make_bounds
    import casadi as ca

    T = refiner.n_points
    D = AUGMENTED_DIM
    z_flat = np.asarray(z_init, dtype=np.float64).reshape(-1)
    lbw, ubw = make_bounds(initial_state, final_state, T)

    # Use a fresh solver with the requested max_iter
    solver = refiner._make_solver(max_iter=max_iter, cb=None)
    sol    = solver(x0=z_flat, lbx=lbw, ubx=ubw)
    z_opt  = sol["x"].full().reshape(T, D).astype(np.float64)
    st     = solver.stats()
    res    = float(np.max(refiner.continuity_residuals(z_opt, n_rk4_eval=8)))
    return z_opt, st["return_status"], res


# ─────────────────────────────────────────────────────────────────────────────
# Per-seed trajectory plot
# ─────────────────────────────────────────────────────────────────────────────

_C_A    = "limegreen"    # Method A (Diffusion+CasADi)
_C_B    = "darkorange"   # Method B (CasADi alone)
_C_BASE = "#d62728"      # Baseline (dashed red)
_XYZ    = ["tab:red", "tab:green", "tab:blue"]


def _plot_seed(
    out_dir: Path,
    seed: int,
    z_base: np.ndarray,
    z_a: np.ndarray,    # Method A final solution
    z_b: np.ndarray,    # Method B final solution
    time_grid: np.ndarray,
    norm: dict,
    conv_a: bool, corr_a: bool, res_a: float,
    conv_b: bool, corr_b: bool, res_b: float,
):
    """Save a 2-row × 3-col trajectory summary figure to out_dir."""
    from casadi_bvp_refine import switching_function

    ri = np.asarray(norm["r_i"])
    rf = np.asarray(norm["r_f"])
    t  = np.asarray(time_grid)

    def _sw(z):
        d, _, _ = switching_function(z, norm, eps=1e-4)
        return d

    delta_a    = _sw(z_a)
    delta_b    = _sw(z_b)
    delta_base = _sw(z_base)

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, wspace=0.38, hspace=0.50)

    # ── 3D trajectory ─────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[:, 0], projection="3d")
    ax3.plot(z_base[:, 0], z_base[:, 1], z_base[:, 2],
             "--", lw=1.5, color=_C_BASE, alpha=0.8, label="Baseline")
    ax3.plot(z_a[:, 0], z_a[:, 1], z_a[:, 2],
             "o-", ms=2.5, lw=1.8, color=_C_A, label=f"A (Diff+CasADi)")
    ax3.plot(z_b[:, 0], z_b[:, 1], z_b[:, 2],
             "s-", ms=2.5, lw=1.8, color=_C_B, alpha=0.85, label=f"B (CasADi only)")
    ax3.scatter(0, 0, 0,   color="gold",       s=120, marker="*", zorder=5)
    ax3.scatter(*ri,        color="dodgerblue", s=60,              zorder=5, label="Earth")
    ax3.scatter(*rf,        color="tomato",     s=60,              zorder=5, label="Mars")
    ax3.set_xlim(-2, 2); ax3.set_ylim(-2, 2)
    ax3.set_xlabel("x [AU]", fontsize=7); ax3.set_ylabel("y [AU]", fontsize=7)
    ax3.set_zlabel("z [AU]", fontsize=7); ax3.tick_params(labelsize=6)
    ax3.legend(fontsize=6, loc="upper left")
    ax3.set_title("3D Trajectory", fontsize=9)

    # ── r(t) ──────────────────────────────────────────────────────────────────
    ax_r = fig.add_subplot(gs[0, 1])
    for k, c in enumerate(_XYZ):
        ax_r.plot(t, z_base[:, k],   "--", color=c, lw=1.0, alpha=0.6)
        ax_r.plot(t, z_a[:, k],      "-",  color=c, lw=1.4)
        ax_r.plot(t, z_b[:, k],      ":",  color=c, lw=1.4)
    ax_r.set_title("Position r(t)", fontsize=9)
    ax_r.set_xlabel("time [TU/2π]", fontsize=7); ax_r.set_ylabel("AU", fontsize=7)
    ax_r.tick_params(labelsize=6); ax_r.grid(True, alpha=0.25)

    # ── mass + thrust ─────────────────────────────────────────────────────────
    ax_m  = fig.add_subplot(gs[0, 2])
    ax_th = ax_m.twinx()
    ax_m.plot(t, z_base[:, 6], "--", color="tab:green", lw=1.0, alpha=0.6)
    ax_m.plot(t, z_a[:, 6],    "-",  color=_C_A,        lw=1.4, label="m A")
    ax_m.plot(t, z_b[:, 6],    ":",  color=_C_B,        lw=1.4, label="m B")
    ax_th.plot(t, delta_base,  "--", color="tab:red",   lw=1.0, alpha=0.6)
    ax_th.plot(t, delta_a,     "-",  color="tab:red",   lw=1.0, alpha=0.8)
    ax_th.plot(t, delta_b,     ":",  color="tab:red",   lw=1.0, alpha=0.8)
    ax_m.set_ylim(0, 1.05); ax_th.set_ylim(0, 1.5)
    ax_m.set_title("Mass & Thrust", fontsize=9)
    ax_m.set_xlabel("time [TU/2π]", fontsize=7)
    ax_m.set_ylabel("mass", fontsize=7, color="tab:green")
    ax_th.set_ylabel("thrust δ", fontsize=7, color="tab:red")
    ax_m.tick_params(labelsize=6); ax_m.grid(True, alpha=0.25)

    # ── λ_m(t) ────────────────────────────────────────────────────────────────
    ax_lm = fig.add_subplot(gs[1, 1])
    ax_lm.axhline(0, color="gray", lw=0.7, ls="--")
    ax_lm.plot(t, z_base[:, 13], "--", color="tab:purple", lw=1.0, alpha=0.6, label="baseline")
    ax_lm.plot(t, z_a[:, 13],    "-",  color=_C_A,         lw=1.4, label=f"A  λ_m(tf)={z_a[-1,13]:.4f}")
    ax_lm.plot(t, z_b[:, 13],    ":",  color=_C_B,         lw=1.4, label=f"B  λ_m(tf)={z_b[-1,13]:.4f}")
    ax_lm.set_title("Costate λ_m(t)", fontsize=9)
    ax_lm.set_xlabel("time [TU/2π]", fontsize=7)
    ax_lm.set_ylim(-0.2, 1.2); ax_lm.tick_params(labelsize=6)
    ax_lm.grid(True, alpha=0.25); ax_lm.legend(fontsize=6)

    # ── continuity residuals ──────────────────────────────────────────────────
    ax_res = fig.add_subplot(gs[1, 2])
    intervals = np.arange(len(z_a) - 1)
    from casadi_bvp_refine import continuity_residuals as _cr
    # reuse already-computed via BVPRefiner — pass via closure unavailable here;
    # just recompute quickly with 8 RK4 steps.
    # (We already have res_a, res_b as scalars; plot bar charts.)
    # Since the refiner isn't passed in, use the stored scalar values in the title.
    ax_res.set_title(
        f"Max cont. residual\nA={res_a:.2e}  B={res_b:.2e}", fontsize=9
    )
    # Simple bar: just show the two scalars as a bar chart
    ax_res.bar(["A: Diff+CasADi", "B: CasADi only"],
               [res_a, res_b],
               color=[_C_A, _C_B], alpha=0.8)
    ax_res.axhline(1e-3, color="gray", lw=0.8, ls="--", label="res-tol 1e-3")
    ax_res.set_yscale("log")
    ax_res.set_ylabel("max ‖F(Z_k)−Z_{k+1}‖", fontsize=7)
    ax_res.tick_params(labelsize=7); ax_res.grid(True, alpha=0.25, axis="y")
    ax_res.legend(fontsize=6)

    # ── Super-title ───────────────────────────────────────────────────────────
    a_tag = ("CORRECT" if corr_a else ("converged" if conv_a else "FAILED"))
    b_tag = ("CORRECT" if corr_b else ("converged" if conv_b else "FAILED"))
    fig.suptitle(
        f"Seed {seed}   |   A (Diff+CasADi): {a_tag}   |   B (CasADi only): {b_tag}\n"
        "-- dashed = baseline, solid = A, dotted = B",
        fontsize=10, fontweight="bold",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "trajectory.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path,
                   default=Path("checkpoints/indiff_ctrl_latest.msgpack"))
    p.add_argument("--earth-mars", action="store_true",
                   help="Use hardcoded Earth-Mars BCs (no dataset needed)")
    p.add_argument("--num-seeds",            type=int,   default=100)
    p.add_argument("--num-diffusion-steps",  type=int,   default=30)
    p.add_argument("--casadi-max-iter",      type=int,   default=500)
    p.add_argument("--res-tol",              type=float, default=1e-3,
                   help="Max continuity residual threshold for 'converged'")
    p.add_argument("--sol-tol",              type=float, default=0.1,
                   help="Max |r+v - baseline| inf-norm threshold for 'correct'")
    p.add_argument("--baseline-seed",        type=int,   default=42,
                   help="Seed used to generate the baseline correct solution")
    p.add_argument("--output-dir", type=Path, default=Path("convergence_study"),
                   help="Root folder; each seed gets a subfolder seed_NNN/")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip saving per-seed trajectory plots")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Sample BCs ────────────────────────────────────────────────────────────
    if args.earth_mars:
        from core import build_normalization
        _n = build_normalization()
        m0 = float(_n["m0"])
        initial_state = np.concatenate([_n["r_i"], _n["v_i"], [m0]]).astype(np.float32)
        final_state   = np.concatenate([_n["r_f"], _n["v_f"], [m0]]).astype(np.float32)
        sample = {"initial_state": initial_state, "final_state": final_state}
        print("Earth-Mars BCs:")
        print(f"  r0={initial_state[:3].round(4)}  v0={initial_state[3:6].round(4)}")
        print(f"  rf={final_state[:3].round(4)}  vf={final_state[3:6].round(4)}")
    else:
        sys.exit("Only --earth-mars is supported for now; add --earth-mars flag.")

    # ── Load checkpoint & build JIT forward ───────────────────────────────────
    if not args.checkpoint.exists():
        sys.exit(f"Checkpoint not found: {args.checkpoint}")
    params, policy, time_grid, norm = load_checkpoint(args.checkpoint, args)
    _apply = build_jit_apply(policy)

    # ── Build BVP refiner ─────────────────────────────────────────────────────
    n_points = len(time_grid)
    refiner  = BVPRefiner(n_points=n_points, n_rk4_steps=8, ipopt_verbosity=0)
    print(f"\nBVPRefiner: n_points={n_points}  n_rk4_steps=8")

    # ── Baseline: seed=42, Method A ───────────────────────────────────────────
    print(f"\n── Computing baseline (seed={args.baseline_seed}, Diffusion+CasADi) ──")
    t0 = time.perf_counter()
    z_diff_base = run_diffusion_final(
        policy, params, sample, _apply,
        num_steps=args.num_diffusion_steps,
        rng_seed=args.baseline_seed,
    )
    z_base, status_base, res_base = solve_bvp(
        refiner, z_diff_base,
        initial_state.astype(np.float64),
        final_state.astype(np.float64),
        max_iter=args.casadi_max_iter,
    )
    print(f"  status={status_base}  max_cont_res={res_base:.3e}  "
          f"time={time.perf_counter()-t0:.1f}s")
    if "Succeed" not in status_base:
        sys.exit("Baseline solve failed — cannot proceed.")

    # ── Save baseline plot ────────────────────────────────────────────────────
    if not args.no_plots:
        _plot_seed(
            args.output_dir / f"seed_{args.baseline_seed:03d}_baseline",
            seed=args.baseline_seed,
            z_base=z_base, z_a=z_base, z_b=z_base,
            time_grid=time_grid, norm=norm,
            conv_a=True, corr_a=True, res_a=res_base,
            conv_b=True, corr_b=True, res_b=res_base,
        )

    # ── Study ─────────────────────────────────────────────────────────────────
    seeds = list(range(args.num_seeds))
    N     = len(seeds)

    # Result containers: (converged, correct) per method per seed
    res_A = []   # Method A: Diffusion + CasADi
    res_B = []   # Method B: CasADi alone

    i_st  = initial_state.astype(np.float64)
    f_st  = final_state.astype(np.float64)

    print(f"\n── Running {N} seeds  (diffusion steps={args.num_diffusion_steps}, "
          f"casadi_max_iter={args.casadi_max_iter}) ──")
    print(f"{'seed':>5}  {'A_conv':>6} {'A_corr':>6}  {'A_res':>10}  "
          f"{'B_conv':>6} {'B_corr':>6}  {'B_res':>10}  {'t_A':>6} {'t_B':>6}")
    print("-" * 78)

    t_study = time.perf_counter()
    for seed in seeds:
        # ── Method A: Diffusion + CasADi ─────────────────────────────────────
        t_a0 = time.perf_counter()
        z_diff = run_diffusion_final(
            policy, params, sample, _apply,
            num_steps=args.num_diffusion_steps,
            rng_seed=seed,
        )
        z_a, st_a, res_a = solve_bvp(refiner, z_diff, i_st, f_st,
                                      max_iter=args.casadi_max_iter)
        t_a = time.perf_counter() - t_a0

        conv_a = ("Succeed" in st_a) and (res_a < args.res_tol)
        corr_a = conv_a and (np.max(np.abs(z_a[:, :6] - z_base[:, :6])) < args.sol_tol)

        # ── Method B: CasADi alone ────────────────────────────────────────────
        t_b0 = time.perf_counter()
        z_b_init = make_casadi_only_guess(initial_state, final_state, n_points, seed)
        z_b, st_b, res_b = solve_bvp(refiner, z_b_init, i_st, f_st,
                                      max_iter=args.casadi_max_iter)
        t_b = time.perf_counter() - t_b0

        conv_b = ("Succeed" in st_b) and (res_b < args.res_tol)
        corr_b = conv_b and (np.max(np.abs(z_b[:, :6] - z_base[:, :6])) < args.sol_tol)

        res_A.append((conv_a, corr_a))
        res_B.append((conv_b, corr_b))

        if not args.no_plots:
            _plot_seed(
                args.output_dir / f"seed_{seed:03d}",
                seed=seed,
                z_base=z_base, z_a=z_a, z_b=z_b,
                time_grid=time_grid, norm=norm,
                conv_a=conv_a, corr_a=corr_a, res_a=res_a,
                conv_b=conv_b, corr_b=corr_b, res_b=res_b,
            )

        print(f"{seed:>5}  "
              f"{'YES' if conv_a else 'no':>6} {'YES' if corr_a else 'no':>6}  "
              f"{res_a:>10.3e}  "
              f"{'YES' if conv_b else 'no':>6} {'YES' if corr_b else 'no':>6}  "
              f"{res_b:>10.3e}  "
              f"{t_a:>6.1f}s {t_b:>6.1f}s")

    t_total = time.perf_counter() - t_study

    # ── Summary ───────────────────────────────────────────────────────────────
    conv_A = sum(c for c, _ in res_A)
    corr_A = sum(k for _, k in res_A)
    conv_B = sum(c for c, _ in res_B)
    corr_B = sum(k for _, k in res_B)

    print("\n" + "=" * 60)
    print(f"  {'':32s}  {'Method A':>10}  {'Method B':>10}")
    print(f"  {'':32s}  {'Diff+CasADi':>10}  {'CasADi only':>10}")
    print("-" * 60)
    print(f"  {'Converged (IPOPT + res<tol)':32s}  {conv_A:>9}/{N}  {conv_B:>9}/{N}")
    print(f"  {'% converged':32s}  {100*conv_A/N:>9.1f}%  {100*conv_B/N:>9.1f}%")
    print(f"  {'Correct (converged + close to baseline)':32s}  {corr_A:>9}/{N}  {corr_B:>9}/{N}")
    print(f"  {'% correct':32s}  {100*corr_A/N:>9.1f}%  {100*corr_B/N:>9.1f}%")
    print("=" * 60)
    print(f"  res-tol={args.res_tol:.0e}  sol-tol={args.sol_tol}  "
          f"diff-steps={args.num_diffusion_steps}  casadi-max-iter={args.casadi_max_iter}")
    print(f"  Total study time: {t_total/60:.1f} min")


if __name__ == "__main__":
    main()
