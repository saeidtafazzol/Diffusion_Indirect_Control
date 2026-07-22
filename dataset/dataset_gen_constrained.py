"""Dataset generation with lambda_m(t=0) upper-bound constraint via gradient descent.

After each backward integration the initial costate lambda_m(t=0) is checked
against a per-sample threshold tau ~ Uniform(LAMBDA_M_THRESH_LOW,
LAMBDA_M_THRESH_HIGH).  When lambda_m(t=0) exceeds tau, gradient descent
(central-difference numerical gradients + additive Gaussian noise) adjusts the
free terminal-boundary variables until the constraint is satisfied.

Free variables (optimised by GD):
    yf[3:6]         — terminal velocity  (vx, vy, vz)
    costate_f[0:6]  — terminal rv-costates  (lambda_rx, lambda_ry, lambda_rz,
                                              lambda_vx, lambda_vy, lambda_vz)

Fixed variables (never modified):
    yf[0:3]         — terminal position
    yf[6]           — terminal mass
    costate_f[6]    — lambda_m_final = 0  (Pontryagin transversality condition)
"""
import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

del _root


import argparse
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from core import STATE_DIM, COSTATE_DIM, make_dynamics
from dataset_gen import (
    CHUNK_SAVE_SIZE,
    MASS_FINAL_LOW,
    MASS_FINAL_HIGH,
    sample_final_state_batch,
    sample_final_costate_batch,
    flush_chunk,
    create_chunk_buffers,
    is_step_limit_error,
    make_backward_batch_integrator,
)

# ── Per-sample threshold bounds for lambda_m(t=0) ────────────────────────────
LAMBDA_M_THRESH_LOW  = 0.2
LAMBDA_M_THRESH_HIGH = 0.9

# ── Gradient-descent hyper-parameters ────────────────────────────────────────
GD_MAX_ITER  = 100     # max iterations before declaring non-convergence
GD_LR        = 2.5e-1   # gradient-descent step size
GD_FD_STEP   = 1e-4   # finite-difference perturbation h (central differences)
GD_NOISE_STD = 1e-4   # std of Gaussian noise added to params each step

# Augmented-vector index for lambda_m: layout = [state(0:7), costate(7:14)]
_LAM_M_AUG_IDX = STATE_DIM + COSTATE_DIM - 1  # = 13


# ─────────────────────────────────────────────────────────────────────────────
# Batched gradient-descent routine
# ─────────────────────────────────────────────────────────────────────────────

def run_gradient_descent_batched(
    batch_integrator,
    yf_np, cf_np, thresholds, rng,
    max_iter  = GD_MAX_ITER,
    lr        = GD_LR,
    fd_h      = GD_FD_STEP,
    noise_std = GD_NOISE_STD,
):
    """Batched GD to drive lambda_m(t=0) below per-sample thresholds.

    For each parameter j (9 total), all B samples are perturbed ±h simultaneously
    and integrated in a single 2B-sample batch call — 9 such calls per GD step
    instead of B×18 sequential scalar calls.

    Only active (not-yet-converged) samples are updated each step; already-
    converged samples are frozen in place.

    Returns
    -------
    states_out   : (B, T, 7)
    costates_out : (B, T, 7)
    converged    : (B,) bool
    """
    B, n_params = len(yf_np), 9

    yf_pos  = yf_np[:, :3].astype(np.float64)   # (B, 3)  fixed
    yf_mass = yf_np[:, 6:7].astype(np.float64)  # (B, 1)  fixed

    # Free params (B, 9): [vx,vy,vz | lam_rx,lam_ry,lam_rz,lam_vx,lam_vy,lam_vz]
    params = np.concatenate([
        yf_np[:, 3:6].astype(np.float64),
        cf_np[:, :6].astype(np.float64),
    ], axis=1)

    def _reconstruct(p):
        yf_ = np.concatenate([yf_pos, p[:, :3], yf_mass], axis=1).astype(np.float32)
        cf_ = np.concatenate([p[:, 3:9], np.zeros((len(p), 1), dtype=np.float32)], axis=1)
        return yf_, cf_

    def _eval_lam_m(p):
        yf_, cf_ = _reconstruct(p)
        _, c, ok = batch_integrator(jnp.asarray(yf_), jnp.asarray(cf_))
        lam = np.asarray(jax.device_get(c[:, 0, 6]), dtype=np.float64)
        ok_ = np.asarray(jax.device_get(ok), dtype=bool)
        return lam, ok_

    def _eval_full(p):
        yf_, cf_ = _reconstruct(p)
        s, c, ok = batch_integrator(jnp.asarray(yf_), jnp.asarray(cf_))
        s_np  = np.asarray(jax.device_get(s),  dtype=np.float32)
        c_np  = np.asarray(jax.device_get(c),  dtype=np.float32)
        ok_np = np.asarray(jax.device_get(ok), dtype=bool)
        lam   = c_np[:, 0, 6].astype(np.float64)
        return s_np, c_np, ok_np, lam

    lam_m, ok = _eval_lam_m(params)
    needs_gd  = (lam_m > thresholds) & ok
    print(
        f"  Batch GD start: {int(np.sum(needs_gd))}/{B} need GD  "
        f"lam_m=[{lam_m.min():.4f}, {lam_m.max():.4f}]  "
        f"tau=[{thresholds.min():.4f}, {thresholds.max():.4f}]"
    )

    for it in range(max_iter):
        if not np.any(needs_gd):
            print(f"  All {B} samples converged before iter {it + 1}")
            break

        # Central-difference gradient: one 2B integrator call per parameter
        grad = np.zeros((B, n_params))
        for j in range(n_params):
            p_p = params.copy(); p_p[:, j] += fd_h
            p_m = params.copy(); p_m[:, j] -= fd_h
            yf_p, cf_p = _reconstruct(p_p)
            yf_m, cf_m = _reconstruct(p_m)
            _, c2b, _ = batch_integrator(
                jnp.asarray(np.vstack([yf_p, yf_m])),
                jnp.asarray(np.vstack([cf_p, cf_m])),
            )
            lam2b = np.asarray(jax.device_get(c2b[:, 0, 6]), dtype=np.float64)
            grad[:, j] = (lam2b[:B] - lam2b[B:]) / (2.0 * fd_h)

        noise = rng.normal(0.0, noise_std, size=(B, n_params))
        # Update only active (still-above-threshold) samples
        params[needs_gd] -= lr * grad[needs_gd] + noise[needs_gd]

        lam_m, ok = _eval_lam_m(params)
        needs_gd  = (lam_m > thresholds) | ~ok

        n_conv      = int(np.sum(~needs_gd))
        grad_norms  = np.linalg.norm(grad, axis=1)
        if np.any(needs_gd):
            print(
                f"  GD iter {it + 1:3d}/{max_iter}: converged={n_conv}/{B}  "
                f"lam_m(active)=[{lam_m[needs_gd].min():.4f}, {lam_m[needs_gd].max():.4f}]  "
                f"|grad|=[{grad_norms.min():.3e}, {grad_norms.max():.3e}]"
            )
        else:
            print(f"  GD iter {it + 1:3d}/{max_iter}: all {B} converged")

    # Final full integration to retrieve accepted trajectories
    states_out, costates_out, ok_out, lam_m_final = _eval_full(params)
    converged = (lam_m_final <= thresholds) & ok_out
    print(f"  Batch GD done: {int(np.sum(converged))}/{B} converged")
    return states_out, costates_out, converged


# ─────────────────────────────────────────────────────────────────────────────
# Dataset generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset_constrained(
    num_samples,
    num_points,
    seed,
    output_path,
    box_low,
    box_high,
    min_radius,
    batch_size,
    rtol,
    atol,
    log_every,
    eps,
    gd_max_iter  = GD_MAX_ITER,
    gd_lr        = GD_LR,
    gd_fd_step   = GD_FD_STEP,
    gd_noise_std = GD_NOISE_STD,
):
    _, jdy, norm = make_dynamics(eps=eps)
    mu  = norm["mu"]
    rng = np.random.default_rng(seed)

    # Batch integrator used for both initial generation and GD perturbations
    _, batch_integrator = make_backward_batch_integrator(
        jdy=jdy, t_final=norm["t_f"],
        n_points=num_points, rtol=rtol, atol=atol,
    )

    chunk_dir = output_path.parent / output_path.stem
    chunk_dir.mkdir(parents=True, exist_ok=True)

    time_grid = np.linspace(0.0, np.float32(norm["t_f"]), num_points, dtype=np.float32)
    np.savez_compressed(
        chunk_dir / "metadata.npz",
        num_samples        = np.int64(num_samples),
        num_points         = np.int64(num_points),
        chunk_size         = np.int64(CHUNK_SAVE_SIZE),
        time_grid          = time_grid,
        t_final            = np.float32(norm["t_f"]),
        au                 = np.float32(norm["AU"]),
        tu                 = np.float32(norm["TU"]),
        state_box_low      = np.float32(box_low),
        state_box_high     = np.float32(box_high),
        min_radius         = np.float32(min_radius),
        seed               = np.int64(seed),
        sampling_method    = "backward_lambda_m_constrained_gd",
        mass_final_low     = np.float32(MASS_FINAL_LOW),
        mass_final_high    = np.float32(MASS_FINAL_HIGH),
        eps                = np.float32(eps),
        lambda_m_thresh_low  = np.float32(LAMBDA_M_THRESH_LOW),
        lambda_m_thresh_high = np.float32(LAMBDA_M_THRESH_HIGH),
        gd_max_iter        = np.int64(gd_max_iter),
        gd_lr              = np.float32(gd_lr),
        gd_fd_step         = np.float32(gd_fd_step),
        gd_noise_std       = np.float32(gd_noise_std),
    )

    chunk_buffers = create_chunk_buffers(CHUNK_SAVE_SIZE, num_points)
    chunk_index = 0
    accepted    = 0   # samples successfully written
    gd_count    = 0   # samples that required gradient descent
    skip_count  = 0   # samples dropped (GD non-convergence or integration failure)

    while accepted < num_samples:
        # ── Batch sample & backward-integrate ────────────────────────────────
        while True:
            yf_batch = sample_final_state_batch(
                rng=rng, batch_size=batch_size,
                low=box_low, high=box_high,
                min_radius=min_radius, mu=mu,
            )
            cf_batch = sample_final_costate_batch(rng, batch_size)
            try:
                traj_b, ctraj_b, ok_b = batch_integrator(yf_batch, cf_batch)
            except Exception as exc:
                if not is_step_limit_error(exc):
                    raise
                print("batch solve hit step limit; resampling")
                continue
            ok_np = np.asarray(jax.device_get(ok_b), dtype=bool)
            if not np.all(ok_np):
                print("some batch solves failed; resampling")
                continue
            break

        states_np   = np.asarray(jax.device_get(traj_b),   dtype=np.float32)
        costates_np = np.asarray(jax.device_get(ctraj_b),  dtype=np.float32)
        yf_np       = np.asarray(jax.device_get(yf_batch), dtype=np.float32)
        cf_np       = np.asarray(jax.device_get(cf_batch), dtype=np.float32)

        # ── Sample per-trajectory thresholds, identify which need GD ─────────
        lam_m_batch = costates_np[:, 0, 6]  # (batch_size,) lambda_m at t=0
        thresholds  = rng.uniform(
            LAMBDA_M_THRESH_LOW, LAMBDA_M_THRESH_HIGH, size=batch_size
        ).astype(np.float64)
        needs_gd_mask = lam_m_batch > thresholds  # (batch_size,) bool

        # Final trajectory arrays — updated in-place for GD samples
        final_states   = states_np.copy()
        final_costates = costates_np.copy()
        should_skip    = np.zeros(batch_size, dtype=bool)

        # ── Batched GD for all samples that exceed their threshold ────────────
        if np.any(needs_gd_mask):
            gd_idx   = np.where(needs_gd_mask)[0]
            gd_count += len(gd_idx)
            print(
                f"[batch] {len(gd_idx)}/{batch_size} samples need GD  "
                f"(accepted so far: {accepted})"
            )
            states_gd, costates_gd, converged_gd = run_gradient_descent_batched(
                batch_integrator = batch_integrator,
                yf_np            = yf_np[gd_idx],
                cf_np            = cf_np[gd_idx],
                thresholds       = thresholds[gd_idx],
                rng              = rng,
                max_iter         = gd_max_iter,
                lr               = gd_lr,
                fd_h             = gd_fd_step,
                noise_std        = gd_noise_std,
            )
            for k, idx in enumerate(gd_idx):
                if converged_gd[k]:
                    final_states[idx]   = states_gd[k]
                    final_costates[idx] = costates_gd[k]
                else:
                    should_skip[idx] = True
                    skip_count += 1

        # ── Write accepted samples to chunk ───────────────────────────────────
        for i in range(batch_size):
            if accepted >= num_samples:
                break
            if should_skip[i]:
                continue
            wi = chunk_buffers["size"]
            chunk_buffers["states"][wi]           = final_states[i]
            chunk_buffers["costates"][wi]         = final_costates[i]
            chunk_buffers["initial_states"][wi]   = final_states[i, 0]
            chunk_buffers["final_states"][wi]     = final_states[i, -1]
            chunk_buffers["sampled_costates"][wi] = final_costates[i, -1]
            chunk_buffers["size"] += 1
            if chunk_buffers["size"] == CHUNK_SAVE_SIZE:
                chunk_index = flush_chunk(chunk_dir, chunk_buffers, chunk_index)
            accepted += 1

        if log_every > 0 and accepted > 0:
            if accepted % log_every == 0 or accepted >= num_samples:
                total_attempted = accepted + skip_count
                pct = 100.0 * gd_count / total_attempted if total_attempted else 0.0
                print(
                    f"accepted {min(accepted, num_samples)}/{num_samples}  "
                    f"(GD invoked: {gd_count}, skipped: {skip_count}, "
                    f"{pct:.1f}% needed GD)"
                )

    chunk_index = flush_chunk(chunk_dir, chunk_buffers, chunk_index)
    total_attempted = accepted + skip_count
    pct = 100.0 * gd_count / total_attempted if total_attempted else 0.0
    print(
        f"done — {accepted} samples saved to {chunk_dir}  "
        f"| GD invocations: {gd_count}, skipped: {skip_count}, {pct:.1f}% needed GD"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate Earth-Mars min-fuel trajectories with a lambda_m(t=0) "
            "upper-bound constraint.  After each backward integration a per-sample "
            f"threshold tau ~ Uniform({LAMBDA_M_THRESH_LOW}, {LAMBDA_M_THRESH_HIGH}) "
            "is drawn; if lambda_m(t=0) > tau, gradient descent on terminal velocity "
            "and rv-costates drives it below tau."
        )
    )
    parser.add_argument("--num-samples",   type=int,   default=2**20)
    parser.add_argument("--num-points",    type=int,   default=32)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "earth_mars_minfuel_constrained_32pts.npz",
    )
    parser.add_argument("--box-low",       type=float, default=-3.0)
    parser.add_argument("--box-high",      type=float, default=3.0)
    parser.add_argument("--min-radius",    type=float, default=1e-1)
    parser.add_argument("--batch-size",    type=int,   default=64)
    parser.add_argument("--rtol",          type=float, default=1e-7)
    parser.add_argument("--atol",          type=float, default=1e-9)
    parser.add_argument("--log-every",     type=int,   default=1)
    parser.add_argument("--eps",           type=float, default=1e-4)
    parser.add_argument("--gd-max-iter",   type=int,   default=GD_MAX_ITER,
                        help="Max GD iterations per sample before skipping")
    parser.add_argument("--gd-lr",         type=float, default=GD_LR,
                        help="Gradient-descent learning rate")
    parser.add_argument("--gd-fd-step",    type=float, default=GD_FD_STEP,
                        help="Finite-difference step h for central differences")
    parser.add_argument("--gd-noise-std",  type=float, default=GD_NOISE_STD,
                        help="Std of Gaussian noise injected each GD step")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_points < 2:
        raise ValueError("num_points must be at least 2")
    if args.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if args.box_low >= args.box_high:
        raise ValueError("box_low must be smaller than box_high")
    generate_dataset_constrained(
        num_samples  = args.num_samples,
        num_points   = args.num_points,
        seed         = args.seed,
        output_path  = args.output,
        box_low      = args.box_low,
        box_high     = args.box_high,
        min_radius   = args.min_radius,
        batch_size   = args.batch_size,
        rtol         = args.rtol,
        atol         = args.atol,
        log_every    = args.log_every,
        eps          = args.eps,
        gd_max_iter  = args.gd_max_iter,
        gd_lr        = args.gd_lr,
        gd_fd_step   = args.gd_fd_step,
        gd_noise_std = args.gd_noise_std,
    )


if __name__ == "__main__":
    main()
