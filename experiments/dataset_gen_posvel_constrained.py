"""Dataset generation: lambda_m(t=0) constraint + soft position/velocity magnitude penalty.

GD minimises the composite loss at each iteration:

    loss_i = lambda_m_i(t=0)
             + POS_VEL_ALPHA * relu( ||r_i(t=0)|| - POS_VEL_THRESHOLD )
             + POS_VEL_ALPHA * relu( ||v_i(t=0)|| - POS_VEL_THRESHOLD )

POS_VEL_ALPHA < 1 so the position/velocity terms are weighted less than lambda_m.
Acceptance criterion is still lambda_m(t=0) <= tau only (pos/vel are soft steering).

Free variables:  yf[3:6]  (terminal velocity) + costate_f[0:6]  (rv-costates)
Fixed variables: yf[0:3]  (terminal position),  yf[6] (mass),    costate_f[6]=0
"""
import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(1, str(_root / "experiments"))
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

# ── lambda_m threshold bounds ─────────────────────────────────────────────────
LAMBDA_M_THRESH_LOW  = 0.2
LAMBDA_M_THRESH_HIGH = 0.9

# ── Soft pos/vel magnitude penalty ───────────────────────────────────────────
POS_VEL_THRESHOLD = 1.8    # AU (pos) / AU·TU⁻¹ (vel) — shared threshold
POS_VEL_ALPHA     = 0.1    # penalty coefficient; lambda_m has implicit coeff 1.0

# ── Hard rejection threshold for pos/vel at t=0 ──────────────────────────────
PV_HARD_REJECT    = 2.7    # samples with ||r(t=0)|| or ||v(t=0)|| > this are dropped

# ── Gradient-descent hyper-parameters ────────────────────────────────────────
GD_MAX_ITER      = 25
GD_LR            = 1e-1
GD_FD_STEP       = 1e-4
GD_NOISE_STD     = 1e-4
GD_MAX_GRAD_NORM = 5.0    # per-sample gradient clip; prevents runaway trajectory explosion


# ─────────────────────────────────────────────────────────────────────────────
# Batched gradient-descent routine
# ─────────────────────────────────────────────────────────────────────────────

def run_gradient_descent_batched(
    batch_integrator,
    yf_np, cf_np, thresholds, rng,
    max_iter      = GD_MAX_ITER,
    lr            = GD_LR,
    fd_h          = GD_FD_STEP,
    noise_std     = GD_NOISE_STD,
    max_grad_norm = GD_MAX_GRAD_NORM,
    pv_threshold  = POS_VEL_THRESHOLD,
    pv_alpha      = POS_VEL_ALPHA,
    pv_hard_reject= PV_HARD_REJECT,
):
    """Batched GD minimising a composite loss:

        loss = relu(lambda_m(t=0) - tau)
               + pv_alpha * relu(||r(t=0)|| - pv_threshold)
               + pv_alpha * relu(||v(t=0)|| - pv_threshold)

    Using relu for lambda_m means its gradient contribution drops to zero once
    it reaches tau, so GD will not keep pushing it down while fixing r/v.

    Convergence (sample frozen) requires ALL three conditions:
        lambda_m(t=0) <= tau
        ||r(t=0)||    <= pv_hard_reject
        ||v(t=0)||    <= pv_hard_reject

    Each parameter j: +h/-h perturbations for all B samples are stacked into
    one 2B-sample batch call (9 such calls per GD step).
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

    def _eval(p):
        """Run batch integration; return (states, costates, ok, loss, lam_m, r_norm, v_norm)."""
        yf_, cf_ = _reconstruct(p)
        s, c, ok = batch_integrator(jnp.asarray(yf_), jnp.asarray(cf_))
        s_np  = np.asarray(jax.device_get(s),  dtype=np.float32)
        c_np  = np.asarray(jax.device_get(c),  dtype=np.float32)
        ok_np = np.asarray(jax.device_get(ok), dtype=bool)
        lam_m  = c_np[:, 0, 6].astype(np.float64)
        r_norm = np.linalg.norm(s_np[:, 0, :3],  axis=-1).astype(np.float64)
        v_norm = np.linalg.norm(s_np[:, 0, 3:6], axis=-1).astype(np.float64)
        # relu(lambda_m - tau): gradient is 0 once lambda_m reaches its threshold
        lam_pen = np.maximum(0.0, lam_m - thresholds)
        r_pen   = np.maximum(0.0, r_norm - pv_threshold)
        v_pen   = np.maximum(0.0, v_norm - pv_threshold)
        loss    = lam_pen + pv_alpha * r_pen + pv_alpha * v_pen
        return s_np, c_np, ok_np, loss, lam_m, r_norm, v_norm

    _, _, ok, _, lam_m, r_norm, v_norm = _eval(params)
    permanently_failed = ~np.isfinite(lam_m) | ~np.isfinite(r_norm) | ~np.isfinite(v_norm)
    if np.any(permanently_failed):
        print(f"  Dropping {int(np.sum(permanently_failed))} samples with non-finite initial values")
    needs_gd = ((lam_m > thresholds) | (r_norm > pv_hard_reject) | (v_norm > pv_hard_reject)) & ok & ~permanently_failed
    n_lam = int(np.sum((lam_m > thresholds) & ok & ~permanently_failed))
    n_pv  = int(np.sum(((r_norm > pv_hard_reject) | (v_norm > pv_hard_reject)) & ok & ~permanently_failed))
    print(
        f"  Batch GD start: {int(np.sum(needs_gd))}/{B} need GD  "
        f"(lam_m={n_lam}, r/v={n_pv})  "
        f"lam_m=[{lam_m.min():.4f}, {lam_m.max():.4f}]  "
        f"tau=[{thresholds.min():.4f}, {thresholds.max():.4f}]  "
        f"r_norm=[{r_norm.min():.3f}, {r_norm.max():.3f}]  "
        f"v_norm=[{v_norm.min():.3f}, {v_norm.max():.3f}]"
    )

    for it in range(max_iter):
        if not np.any(needs_gd):
            print(f"  All {B} samples converged before iter {it + 1}")
            break

        # ── Central-difference gradient of composite loss ─────────────────────
        # Only integrate samples that still need GD – skip converged and dropped.
        active_idx = np.where(needs_gd)[0]  # indices into [0, B)
        A = len(active_idx)
        active_params    = params[active_idx]
        active_thresholds = thresholds[active_idx]
        active_yf_pos    = yf_pos[active_idx]
        active_yf_mass   = yf_mass[active_idx]

        def _reconstruct_active(p):
            yf_ = np.concatenate([active_yf_pos, p[:, :3], active_yf_mass], axis=1).astype(np.float32)
            cf_ = np.concatenate([p[:, 3:9], np.zeros((len(p), 1), dtype=np.float32)], axis=1)
            return yf_, cf_

        grad = np.zeros((B, n_params))
        for j in range(n_params):
            p_p = active_params.copy()
            p_p[:, j] += fd_h
            p_m = active_params.copy()
            p_m[:, j] -= fd_h
            yf_p, cf_p = _reconstruct_active(p_p)
            yf_m, cf_m = _reconstruct_active(p_m)
            s2b, c2b, _ = batch_integrator(
                jnp.asarray(np.vstack([yf_p, yf_m])),
                jnp.asarray(np.vstack([cf_p, cf_m])),
            )
            s2b_np = np.asarray(jax.device_get(s2b), dtype=np.float32)
            c2b_np = np.asarray(jax.device_get(c2b), dtype=np.float32)
            lam2b  = c2b_np[:, 0, 6].astype(np.float64)
            thresholds_2a = np.concatenate([active_thresholds, active_thresholds])
            lam_pen2 = np.maximum(0.0, lam2b - thresholds_2a)
            r_pen2 = np.maximum(
                0.0, np.linalg.norm(s2b_np[:, 0, :3],  axis=-1).astype(np.float64) - pv_threshold
            )
            v_pen2 = np.maximum(
                0.0, np.linalg.norm(s2b_np[:, 0, 3:6], axis=-1).astype(np.float64) - pv_threshold
            )
            loss2a = lam_pen2 + pv_alpha * r_pen2 + pv_alpha * v_pen2
            grad[active_idx, j] = (loss2a[:A] - loss2a[A:]) / (2.0 * fd_h)
        # grad rows for permanently_failed remain 0.0 (initialised above)

        noise = rng.normal(0.0, noise_std, size=(B, n_params))
        # Drop samples whose gradient is non-finite – numerical singularity, hopeless.
        bad_grad = ~np.all(np.isfinite(grad), axis=1)
        newly_failed = bad_grad & needs_gd & ~permanently_failed
        if np.any(newly_failed):
            print(f"  Dropping {int(np.sum(newly_failed))} samples with non-finite gradient at iter {it + 1}")
        permanently_failed |= newly_failed
        needs_gd &= ~permanently_failed
        # Per-sample gradient clipping: scale down any sample whose gradient
        # norm exceeds max_grad_norm to prevent trajectory explosion.
        grad_norms_col = np.linalg.norm(grad, axis=1, keepdims=True)  # (B, 1)
        clip_scale     = np.minimum(1.0, max_grad_norm / (grad_norms_col + 1e-8))
        grad_clipped   = grad * clip_scale
        # Freeze already-converged samples
        params[needs_gd] -= lr * grad_clipped[needs_gd] + noise[needs_gd]

        _, _, ok, loss, lam_m, r_norm, v_norm = _eval(params)
        bad = ~np.isfinite(lam_m) | ~np.isfinite(r_norm) | ~np.isfinite(v_norm)
        newly_failed = bad & ~permanently_failed
        if np.any(newly_failed):
            print(f"  Dropping {int(np.sum(newly_failed))} samples with non-finite values at iter {it + 1}")
        permanently_failed |= newly_failed
        needs_gd = (
            (lam_m > thresholds) | (r_norm > pv_hard_reject) | (v_norm > pv_hard_reject) | ~ok
        ) & ~permanently_failed

        n_conv     = int(np.sum(~needs_gd & ~permanently_failed))
        active     = needs_gd  # alias for clarity
        grad_norms = np.linalg.norm(grad, axis=1)
        clipped    = int(np.sum((grad_norms > max_grad_norm) & active))
        if np.any(active):
            print(
                f"  GD iter {it + 1:3d}/{max_iter}: converged={n_conv}/{B}  dropped={int(np.sum(permanently_failed))}  "
                f"lam_m(active)=[{lam_m[active].min():.4f}, {lam_m[active].max():.4f}]  "
                f"r_norm=[{r_norm[active].min():.3f}, {r_norm[active].max():.3f}]  "
                f"v_norm=[{v_norm[active].min():.3f}, {v_norm[active].max():.3f}]  "
                f"|grad|=[{grad_norms[active].min():.3e}, {grad_norms[active].max():.3e}]  "
                f"clipped={clipped}/{int(np.sum(active))}"
            )
        else:
            n_dropped = int(np.sum(permanently_failed))
            print(f"  GD iter {it + 1:3d}/{max_iter}: all remaining converged  dropped={n_dropped}")

    states_out, costates_out, ok_out, _, lam_m_final, r_norm_final, v_norm_final = _eval(params)
    converged = (
        (lam_m_final <= thresholds)
        & (r_norm_final <= pv_hard_reject)
        & (v_norm_final <= pv_hard_reject)
        & ok_out
    )
    print(f"  Batch GD done: {int(np.sum(converged))}/{B} converged")
    return states_out, costates_out, converged


# ─────────────────────────────────────────────────────────────────────────────
# Dataset generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset_posvel_constrained(
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
    gd_max_iter     = GD_MAX_ITER,
    gd_lr           = GD_LR,
    gd_fd_step      = GD_FD_STEP,
    gd_noise_std    = GD_NOISE_STD,
    gd_max_grad_norm= GD_MAX_GRAD_NORM,
    pv_threshold    = POS_VEL_THRESHOLD,
    pv_alpha        = POS_VEL_ALPHA,
    pv_hard_reject  = PV_HARD_REJECT,
):
    _, jdy, norm = make_dynamics(eps=eps)
    mu  = norm["mu"]
    rng = np.random.default_rng(seed)

    _, batch_integrator = make_backward_batch_integrator(
        jdy=jdy, t_final=norm["t_f"],
        n_points=num_points, rtol=rtol, atol=atol,
    )

    chunk_dir = output_path.parent / output_path.stem
    chunk_dir.mkdir(parents=True, exist_ok=True)

    time_grid = np.linspace(0.0, np.float32(norm["t_f"]), num_points, dtype=np.float32)
    np.savez_compressed(
        chunk_dir / "metadata.npz",
        num_samples          = np.int64(num_samples),
        num_points           = np.int64(num_points),
        chunk_size           = np.int64(CHUNK_SAVE_SIZE),
        time_grid            = time_grid,
        t_final              = np.float32(norm["t_f"]),
        au                   = np.float32(norm["AU"]),
        tu                   = np.float32(norm["TU"]),
        state_box_low        = np.float32(box_low),
        state_box_high       = np.float32(box_high),
        min_radius           = np.float32(min_radius),
        seed                 = np.int64(seed),
        sampling_method      = "backward_lambda_m_posvel_constrained_gd",
        mass_final_low       = np.float32(MASS_FINAL_LOW),
        mass_final_high      = np.float32(MASS_FINAL_HIGH),
        eps                  = np.float32(eps),
        lambda_m_thresh_low  = np.float32(LAMBDA_M_THRESH_LOW),
        lambda_m_thresh_high = np.float32(LAMBDA_M_THRESH_HIGH),
        pv_threshold         = np.float32(pv_threshold),
        pv_alpha             = np.float32(pv_alpha),
        pv_hard_reject       = np.float32(pv_hard_reject),
        gd_max_iter          = np.int64(gd_max_iter),
        gd_lr                = np.float32(gd_lr),
        gd_fd_step           = np.float32(gd_fd_step),
        gd_noise_std         = np.float32(gd_noise_std),
        gd_max_grad_norm     = np.float32(gd_max_grad_norm),
    )

    chunk_buffers = create_chunk_buffers(CHUNK_SAVE_SIZE, num_points)
    chunk_index = 0
    accepted    = 0
    gd_count    = 0
    skip_count  = 0

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
        lam_m_batch = costates_np[:, 0, 6]
        thresholds  = rng.uniform(
            LAMBDA_M_THRESH_LOW, LAMBDA_M_THRESH_HIGH, size=batch_size
        ).astype(np.float64)
        needs_gd_mask = lam_m_batch > thresholds

        # Final trajectory arrays (updated in-place for GD samples)
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
                max_grad_norm    = gd_max_grad_norm,
                pv_threshold     = pv_threshold,
                pv_alpha         = pv_alpha,
                pv_hard_reject   = pv_hard_reject,
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
            r_norm_i = float(np.linalg.norm(final_states[i, 0, :3]))
            v_norm_i = float(np.linalg.norm(final_states[i, 0, 3:6]))
            if r_norm_i > pv_hard_reject or v_norm_i > pv_hard_reject:
                print(
                    f"  hard-reject sample {i}: "
                    f"r_norm={r_norm_i:.3f}  v_norm={v_norm_i:.3f}  "
                    f"(threshold={pv_hard_reject})"
                )
                skip_count += 1
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
        f"| GD: {gd_count}, skipped: {skip_count}, {pct:.1f}% needed GD"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate Earth-Mars min-fuel trajectories with lambda_m(t=0) constraint "
            "and soft position/velocity magnitude penalty in the GD objective. "
            f"lambda_m threshold: Uniform({LAMBDA_M_THRESH_LOW}, {LAMBDA_M_THRESH_HIGH}).  "
            f"Pos/vel penalty: {POS_VEL_ALPHA} * relu(norm - {POS_VEL_THRESHOLD})."
        )
    )
    parser.add_argument("--num-samples",   type=int,   default=2**20)
    parser.add_argument("--num-points",    type=int,   default=32)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "earth_mars_minfuel_posvel_constrained_32pts.npz",
    )
    parser.add_argument("--box-low",       type=float, default=-3.0)
    parser.add_argument("--box-high",      type=float, default=3.0)
    parser.add_argument("--min-radius",    type=float, default=1e-1)
    parser.add_argument("--batch-size",    type=int,   default=64)
    parser.add_argument("--rtol",          type=float, default=1e-7)
    parser.add_argument("--atol",          type=float, default=1e-9)
    parser.add_argument("--log-every",     type=int,   default=1)
    parser.add_argument("--eps",           type=float, default=1e-4)
    parser.add_argument("--gd-max-iter",   type=int,   default=GD_MAX_ITER)
    parser.add_argument("--gd-lr",         type=float, default=GD_LR)
    parser.add_argument("--gd-fd-step",    type=float, default=GD_FD_STEP)
    parser.add_argument("--gd-noise-std",      type=float, default=GD_NOISE_STD)
    parser.add_argument("--gd-max-grad-norm",  type=float, default=GD_MAX_GRAD_NORM,
                        help="Per-sample gradient clip norm (prevents trajectory explosion)")
    parser.add_argument("--pv-threshold",  type=float, default=POS_VEL_THRESHOLD,
                        help="Magnitude threshold for position and velocity penalty")
    parser.add_argument("--pv-alpha",      type=float, default=POS_VEL_ALPHA,
                        help="Penalty coefficient for pos/vel (should be < 1.0)")
    parser.add_argument("--pv-hard-reject", type=float, default=PV_HARD_REJECT,
                        help="Hard-reject samples where ||r(t=0)|| or ||v(t=0)|| exceeds this")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_points < 2:
        raise ValueError("num_points must be at least 2")
    if args.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if args.box_low >= args.box_high:
        raise ValueError("box_low must be smaller than box_high")
    generate_dataset_posvel_constrained(
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
        gd_noise_std    = args.gd_noise_std,
        gd_max_grad_norm= args.gd_max_grad_norm,
        pv_threshold    = args.pv_threshold,
        pv_alpha      = args.pv_alpha,
        pv_hard_reject= args.pv_hard_reject,
    )


if __name__ == "__main__":
    main()
