"""
Compute final propellant mass for converged cases (method A and B) using
seeds taken from the shift_eps_study_50 detail.txt files.

Since all converged seeds for a given shift converge to the same optimal
trajectory, we run ONE converged seed per shift per method.

Outputs a LaTeX table to Indirect_Diffusion_Control/figures/final_mass_table.tex
"""
import sys, re, time
from pathlib import Path

import numpy as np
import jax, jax.numpy as jnp
import casadi as ca
from scipy.integrate import solve_ivp
from flax import serialization

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from core import STATE_DIM, COSTATE_DIM, AUGMENTED_DIM, make_dynamics
from jax_ddpm_scheduler import JaxDDPMScheduler
from policy import IndiffCtrlPolicy
from transformer_diffusion_model import (
    DiffusionTransformer, DiffusionTransformerConfig,
    default_state_known_mask_no_final_mass, final_lambda_m_costate_mask,
)
from casadi_bvp_refine import BVPRefiner, make_bounds, IterCapture

_JAX_CACHE = Path.home() / ".cache" / "jax_compile_cache"
_JAX_CACHE.mkdir(parents=True, exist_ok=True)
try:
    jax.config.update("jax_compilation_cache_dir", str(_JAX_CACHE))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
except Exception:
    pass

# ── Paste minimal helpers from shift_eps_comparison.py ───────────────────────

CKPT = _root / "checkpoints" / "indiff_ctrl_latest.msgpack"
STUDY_50 = _root / "shift_eps_study_50"
RES_TOL  = 1e-8
N_DIFF_STEPS = 30
MAX_ITER_A   = 160
MAX_ITER_B   = 240
EPS_STAGES   = [1.0, 1e-1, 1e-2, 1e-3, 1e-4]
NON_FATAL    = frozenset({"Solve_Succeeded", "Solved_To_Acceptable_Level",
                          "Maximum_Iterations_Exceeded"})

_SHIFTS_RAW = [-700,-600,-500,-400,-300,-200,-100,-50,0,50,100,200,300,400,500,600,700]
def _syn(s): return s + 780 if s < 0 else s
ALL_SHIFTS = sorted(_SHIFTS_RAW, key=_syn)

ROW_RE = re.compile(
    r"^\s*(\d+)\s+(OK|--)\s+[\d.e+\-]+\s+[\d.]+\s+(OK|--)\s+[\d.e+\-]+\s+[\d.]+"
)

def parse_converged_seeds(shift):
    """Return (first_A_seed, first_B_seed) from 50-trial detail.txt, or None."""
    f = STUDY_50 / f"shift_{shift:+d}d" / "detail.txt"
    if not f.exists():
        return None, None
    first_a = first_b = None
    for line in f.read_text().splitlines():
        m = ROW_RE.match(line)
        if not m:
            continue
        seed, a_ok, b_ok = int(m.group(1)), m.group(2), m.group(3)
        if first_a is None and a_ok == "OK":
            first_a = seed
        if first_b is None and b_ok == "OK":
            first_b = seed
        if first_a is not None and first_b is not None:
            break
    return first_a, first_b


# ── Load checkpoint ───────────────────────────────────────────────────────────

def load_checkpoint():
    from flax.serialization import msgpack_restore
    raw = CKPT.read_bytes()
    restored = msgpack_restore(raw)
    tg = np.asarray(restored["time_grid"], dtype=np.float32).ravel()
    sc = restored.get("scheduler_config", {})
    def _s(v): return v.decode() if isinstance(v, bytes) else v
    sc = {_s(k): (_s(v) if isinstance(v, bytes) else v) for k, v in sc.items()}
    def _sc(k, d):
        v = sc.get(k, d); return d if v is None else v
    noise_scheduler = JaxDDPMScheduler(
        num_train_timesteps=int(_sc("num_train_timesteps", 5000)),
        beta_schedule=str(_sc("beta_schedule", "squaredcos_cap_v2")),
        prediction_type=str(_sc("prediction_type", "epsilon")),
        clip_sample=bool(_sc("clip_sample", True)),
        clip_sample_range=float(_sc("clip_sample_range", 5.0)),
        noise_scale=float(_sc("noise_scale", 2.0)),
    )
    seq_len = len(tg)
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
    import optax
    opt_tpl = optax.adamw(optax.linear_schedule(1e-4, 5e-6, 1), weight_decay=1e-6).init(params_tpl)
    sch_tpl = {"num_train_timesteps": 5000, "beta_schedule": "squaredcos_cap_v2",
               "prediction_type": "epsilon", "clip_sample": True,
               "clip_sample_range": 5.0, "noise_scale": 2.0}
    state_tpl = {"params": params_tpl, "opt_state": opt_tpl,
                 "step": np.int32(0), "time_grid": np.zeros(seq_len, np.float32),
                 "scheduler_config": sch_tpl}
    loaded = serialization.from_bytes(state_tpl, raw)
    _, _, norm = make_dynamics(eps=1e-4, compile_jax=False)
    print(f"Checkpoint loaded: step={int(loaded['step'])}  seq_len={seq_len}")
    return loaded["params"], policy, tg, norm


def build_jit_apply(policy):
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
        np.concatenate([cond_st, cond_cs], axis=-1), dtype=jnp.float32)
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
        captured.append(traj[0])

    jax.block_until_ready(captured[-1])
    return [np.asarray(f, dtype=np.float32) for f in captured]


def propagate_kepler(r, v, mu, dt):
    if dt == 0.0:
        return np.array(r, np.float64), np.array(v, np.float64)
    def rhs(_, y, mu=mu):
        return np.concatenate([y[3:6], -mu * y[:3] / np.linalg.norm(y[:3])**3])
    sol = solve_ivp(rhs, (0.0, dt), np.concatenate([np.array(r, np.float64), np.array(v, np.float64)]),
                    method="DOP853", rtol=1e-12, atol=1e-13)
    return sol.y[:3, -1], sol.y[3:6, -1]


def shifted_bcs(norm, shift_days):
    shift_tu = shift_days * 86400.0 / norm["TU"]
    mu = float(norm["mu"])
    r_i, v_i = propagate_kepler(norm["r_i"], norm["v_i"], mu, shift_tu)
    r_f, v_f = propagate_kepler(norm["r_f"], norm["v_f"], mu, shift_tu)
    m0 = float(norm["m0"])
    return (np.concatenate([r_i, v_i, [m0]]).astype(np.float32),
            np.concatenate([r_f, v_f, [m0]]).astype(np.float32))


def solve_A(refiner, solver, policy, params, _apply, sample, i_st, f_st, seed):
    frames = run_diffusion(policy, params, sample, _apply, N_DIFF_STEPS, seed)
    z_init = frames[-1]
    lbw, ubw = make_bounds(i_st, f_st, refiner.n_points)
    sol = solver(x0=z_init.reshape(-1), lbx=lbw, ubx=ubw)
    z = sol["x"].full().reshape(refiner.n_points, AUGMENTED_DIM)
    res = float(np.max(refiner.continuity_residuals(z, n_rk4_eval=8)))
    ok = res < RES_TOL
    return z, ok, res


def solve_B(cont_refiners, cont_solvers, norm, i_st, f_st, seed):
    dy_1, _, _ = make_dynamics(eps=1.0, compile_jax=False)
    f_fwd = ca.Function("f_fwd_b", [dy_1.states, dy_1.costates], [dy_1.augmented_dot_sub])
    n_points = cont_refiners[0].n_points
    rng = np.random.default_rng(seed)
    costate_0 = rng.uniform(-2.0, 2.0, 7).astype(np.float64)
    Z0 = np.concatenate([i_st[:7].astype(np.float64), costate_0])
    t_f = float(norm["t_f"])
    t_eval = np.linspace(0.0, t_f, n_points)
    def rhs(_, Z):
        return np.asarray(f_fwd(Z[:7], Z[7:])).flatten()
    try:
        sol = solve_ivp(rhs, [0.0, t_f], Z0, t_eval=t_eval, method="DOP853", rtol=1e-8, atol=1e-9)
        z = sol.y.T.astype(np.float64) if sol.success and sol.y.shape[1] == n_points else None
    except Exception:
        z = None
    if z is None:
        z = np.zeros((n_points, 14), dtype=np.float64)
        for j in range(6):
            z[:, j] = np.linspace(float(i_st[j]), float(f_st[j]), n_points)
        z[:, 6]  = np.linspace(float(i_st[6]), float(i_st[6]) * 0.5, n_points)
        z[:, 7:] = costate_0[None, :]
        z[-1, 13] = 0.0

    lbw, ubw = make_bounds(i_st.astype(np.float64), f_st.astype(np.float64), n_points)
    for idx, (refiner, solver, eps) in enumerate(zip(cont_refiners, cont_solvers, EPS_STAGES)):
        sol = solver(x0=z.reshape(-1), lbx=lbw, ubx=ubw)
        z = sol["x"].full().reshape(n_points, AUGMENTED_DIM).astype(np.float64)
        status = solver.stats()["return_status"]
        if idx < len(EPS_STAGES) - 1 and status not in NON_FATAL:
            break
    res = float(np.max(cont_refiners[-1].continuity_residuals(z, n_rk4_eval=8)))
    ok = res < RES_TOL
    return z, ok, res


# ── Main ──────────────────────────────────────────────────────────────────────

print("Loading checkpoint …")
params, policy, time_grid, norm = load_checkpoint()
_apply = build_jit_apply(policy)
n_points = len(time_grid)
M_kg = float(norm["M"])

print("Building solvers …")
refiner_A = BVPRefiner(n_points=n_points, n_rk4_steps=8, ipopt_verbosity=0)
solver_A  = refiner_A._make_solver(max_iter=MAX_ITER_A, cb=None)

cont_refiners = [BVPRefiner(eps=eps, n_points=n_points, n_rk4_steps=8, ipopt_verbosity=0)
                 for eps in EPS_STAGES]
cont_solvers  = [r._make_solver(max_iter=MAX_ITER_B, cb=None) for r in cont_refiners]
print("Solvers ready.\n")

results = {}  # shift → {A_mf, B_mf}

for shift in ALL_SHIFTS:
    seed_a, seed_b = parse_converged_seeds(shift)
    entry = {}
    initial_state, final_state = shifted_bcs(norm, shift)
    i_st = initial_state.astype(np.float64)
    f_st = final_state.astype(np.float64)
    sample = {"initial_state": initial_state, "final_state": final_state}

    if seed_a is not None:
        t0 = time.perf_counter()
        z_a, ok_a, res_a = solve_A(refiner_A, solver_A, policy, params, _apply,
                                    sample, i_st, f_st, seed_a)
        dt = time.perf_counter() - t0
        mf_a = z_a[-1, 6] * M_kg
        entry["A"] = mf_a if ok_a else None
        print(f"  shift={shift:+4d}d  A seed={seed_a}  ok={ok_a}  res={res_a:.2e}  "
              f"mf={mf_a:.2f} kg  ({dt:.1f}s)")
    else:
        entry["A"] = None
        print(f"  shift={shift:+4d}d  A  no converged seed")

    if seed_b is not None:
        t0 = time.perf_counter()
        z_b, ok_b, res_b = solve_B(cont_refiners, cont_solvers, norm, i_st, f_st, seed_b)
        dt = time.perf_counter() - t0
        mf_b = z_b[-1, 6] * M_kg
        entry["B"] = mf_b if ok_b else None
        print(f"  shift={shift:+4d}d  B seed={seed_b}  ok={ok_b}  res={res_b:.2e}  "
              f"mf={mf_b:.2f} kg  ({dt:.1f}s)")
    else:
        entry["B"] = None
        print(f"  shift={shift:+4d}d  B  no converged seed")

    results[shift] = entry

# ── LaTeX table ───────────────────────────────────────────────────────────────

lines = [
    r"\begin{table}[ht]",
    r"  \centering",
    r"  \caption{Final spacecraft mass for converged trials (initial mass $m_0 = 1000$ kg).}",
    r"  \label{tab:final_mass}",
    r"  \begin{tabular}{r r r}",
    r"    \toprule",
    r"    $\Delta t_{\text{syn}}$ (days) & DBIC + IPOPT $m_f$ (kg) & IPOPT Baseline $m_f$ (kg) \\",
    r"    \midrule",
]
for shift in ALL_SHIFTS:
    label = _syn(shift)
    e = results.get(shift, {})
    a_str = f"{e['A']:.1f}" if e.get("A") is not None else "---"
    b_str = f"{e['B']:.1f}" if e.get("B") is not None else "---"
    lines.append(f"    {label} & {a_str} & {b_str} \\\\")
lines += [
    r"    \bottomrule",
    r"  \end{tabular}",
    r"\end{table}",
]

out = _root / "Indirect_Diffusion_Control" / "figures" / "final_mass_table.tex"
out.write_text("\n".join(lines) + "\n")
print(f"\nSaved → {out}")
print("\n" + "\n".join(lines))
