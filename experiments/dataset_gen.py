"""Dataset generation for min-fuel indirect trajectories."""
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
from diffrax import RESULTS, diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController
from equinox import EquinoxRuntimeError

from core import (
    STATE_DIM, COSTATE_DIM, AUGMENTED_DIM,
    COSTATE_NORM_MIN, build_normalization, make_dynamics, build_rhs, build_reverse_rhs,
)

CHUNK_SAVE_SIZE = 2**12


def orthonormal_basis_perp_to_r(r_vec):
    r_hat = r_vec / np.linalg.norm(r_vec)
    if abs(r_hat[0]) < 0.9:
        aux = np.array([1.0, 0.0, 0.0])
    else:
        aux = np.array([0.0, 1.0, 0.0])
    u1 = np.cross(r_hat, aux)
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(r_hat, u1)
    u2 /= np.linalg.norm(u2)
    return u1, u2


# Per-component uniform sampling bounds for costates: [λ_rx, λ_ry, λ_rz, λ_vx, λ_vy, λ_vz, λ_m]
COSTATE_LOW  = np.array([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, 0.0], dtype=np.float32)
COSTATE_HIGH = np.array([ 2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0], dtype=np.float32)

# Normalized mass range for the terminal state.
# Mass is normalized by the reference spacecraft mass M; full tank = 1.0.
# The final mass is sampled in [MASS_FINAL_LOW, 1.0]; backward integration
# then adds mass as the maneuver unwinds toward t = 0.
MASS_FINAL_LOW  = 0.2
MASS_FINAL_HIGH = 1.0


def sample_costate(rng, low=COSTATE_LOW, high=COSTATE_HIGH):
    costate = rng.uniform(low, high).astype(np.float32)
    return jnp.asarray(costate)


def sample_final_costate(rng):
    """Sample final costate with lambda_m = 0 (Pontryagin transversality condition).

    lambda_r and lambda_v are sampled freely from [-2, 2]^3 each.
    lambda_m at the terminal time must be 0 by the minimum principle when
    terminal mass is free.
    """
    lam_rv = rng.uniform(
        np.array([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0], dtype=np.float32),
        np.array([ 2.0,  2.0,  2.0,  2.0,  2.0,  2.0], dtype=np.float32),
    ).astype(np.float32)
    costate = np.concatenate([lam_rv, [np.float32(0.0)]]).astype(np.float32)
    return jnp.asarray(costate)


def sample_initial_state(rng, low, high, min_radius, mu, m0_normalized=1.0):
    """Sample position uniformly, compute orbital velocity via vis-viva.

    Mass starts at m0_normalized (=1.0 after normalization).
    """
    while True:
        pos = rng.uniform(low, high, size=3).astype(np.float32)
        r_mag = np.linalg.norm(pos)
        if r_mag > min_radius:
            break

    a = rng.uniform(r_mag / 2.0 + 1e-8, r_mag * 2.0)
    v_sq = mu * (2.0 / r_mag - 1.0 / a)
    v_mag = np.sqrt(max(v_sq, 0.0)).astype(np.float32)

    u1, u2 = orthonormal_basis_perp_to_r(pos)
    theta = rng.uniform(0, 2 * np.pi)
    v_dir = (np.cos(theta) * u1 + np.sin(theta) * u2).astype(np.float32)
    vel = v_mag * v_dir

    state = np.concatenate([pos, vel, [np.float32(m0_normalized)]]).astype(np.float32)
    return jnp.asarray(state)


def sample_initial_state_batch(rng, batch_size, low, high, min_radius, mu, m0_normalized=1.0):
    states = [
        sample_initial_state(rng, low, high, min_radius, mu, m0_normalized)
        for _ in range(batch_size)
    ]
    return jnp.stack(states, axis=0)


def sample_final_state(rng, low, high, min_radius, mu):
    """Sample a terminal (t = t_f) state with a randomly drawn normalized mass.

    Position and velocity are sampled via the same orbital vis-viva approach
    as sample_initial_state, but the mass component is drawn uniformly from
    [MASS_FINAL_LOW, MASS_FINAL_HIGH] to represent a spacecraft that has
    consumed propellant during the maneuver.
    """
    while True:
        pos = rng.uniform(low, high, size=3).astype(np.float32)
        r_mag = np.linalg.norm(pos)
        if r_mag > min_radius:
            break

    a = rng.uniform(r_mag / 2.0 + 1e-8, r_mag * 2.0)
    v_sq = mu * (2.0 / r_mag - 1.0 / a)
    v_mag = np.sqrt(max(v_sq, 0.0)).astype(np.float32)

    u1, u2 = orthonormal_basis_perp_to_r(pos)
    theta = rng.uniform(0, 2 * np.pi)
    v_dir = (np.cos(theta) * u1 + np.sin(theta) * u2).astype(np.float32)
    vel = v_mag * v_dir

    m_f = np.float32(rng.uniform(MASS_FINAL_LOW, MASS_FINAL_HIGH))
    state = np.concatenate([pos, vel, [m_f]]).astype(np.float32)
    return jnp.asarray(state)


def sample_final_state_batch(rng, batch_size, low, high, min_radius, mu):
    states = [
        sample_final_state(rng, low, high, min_radius, mu)
        for _ in range(batch_size)
    ]
    return jnp.stack(states, axis=0)


def sample_costate_batch(rng, batch_size, low=COSTATE_LOW, high=COSTATE_HIGH):
    costates = [sample_costate(rng, low, high) for _ in range(batch_size)]
    return jnp.stack(costates, axis=0)


def sample_final_costate_batch(rng, batch_size):
    costates = [sample_final_costate(rng) for _ in range(batch_size)]
    return jnp.stack(costates, axis=0)


def integrate_trajectory(jdy, y0, costate, t_final, n_points, rtol, atol):
    ts = jnp.linspace(0.0, t_final, n_points)
    augmented_y0 = jnp.concatenate(
        [jnp.asarray(y0, dtype=jnp.float32), jnp.asarray(costate, dtype=jnp.float32)],
        axis=0,
    )
    sol = diffeqsolve(
        ODETerm(build_rhs(jdy)),
        Dopri5(),
        t0=0.0,
        t1=float(t_final),
        dt0=None,
        y0=augmented_y0,
        args=None,
        stepsize_controller=PIDController(rtol=rtol, atol=atol),
        saveat=SaveAt(ts=ts),
    )
    augmented_ys = jnp.asarray(sol.ys, dtype=jnp.float32)
    return (
        ts,
        augmented_ys[:, :STATE_DIM],
        augmented_ys[:, STATE_DIM:STATE_DIM + COSTATE_DIM],
    )


def make_batch_integrator(jdy, t_final, n_points, rtol, atol):
    ts = jnp.linspace(0.0, t_final, n_points)
    term = ODETerm(build_rhs(jdy))
    solver = Dopri5()
    controller = PIDController(rtol=rtol, atol=atol)
    saveat = SaveAt(ts=ts)

    def single(y0, costate):
        augmented_y0 = jnp.concatenate([y0, costate], axis=0)
        sol = diffeqsolve(
            term, solver,
            t0=0.0, t1=float(t_final), dt0=None,
            y0=augmented_y0, args=None,
            stepsize_controller=controller,
            saveat=saveat, throw=False,
        )
        augmented_ys = jnp.asarray(sol.ys, dtype=jnp.float32)
        solve_succeeded = jnp.asarray(sol.result == RESULTS.successful, dtype=bool)
        return (
            augmented_ys[:, :STATE_DIM],
            augmented_ys[:, STATE_DIM:STATE_DIM + COSTATE_DIM],
            solve_succeeded,
        )

    return ts, jax.jit(jax.vmap(single))


def make_backward_batch_integrator(jdy, t_final, n_points, rtol, atol):
    """Batch integrator for backward (time-reversed) trajectories.

    Starting from final state y_f = y(t_f) and final costate lambda_f,
    integrates the negated ODE dz/ds = -f(z) from s=0 to s=t_f.  This maps
    s -> z(s) = y(t_f - s), so reversing sol.ys recovers y(t) in the usual
    forward-time order  t in [0, t_f].  Returned trajectories match the
    forward-integrator layout exactly:
      states[0]  = state at t=0  (backward-propagated)
      states[-1] = state at t=t_f (= the sampled final state y_f)
    """
    ts = jnp.linspace(0.0, float(t_final), n_points)  # ascending; SaveAt requires this
    term = ODETerm(build_reverse_rhs(jdy))
    solver = Dopri5()
    controller = PIDController(rtol=rtol, atol=atol)
    saveat = SaveAt(ts=ts)

    def single(yf, costate_f):
        augmented_yf = jnp.concatenate([yf, costate_f], axis=0)
        sol = diffeqsolve(
            term, solver,
            t0=0.0, t1=float(t_final), dt0=None,
            y0=augmented_yf, args=None,
            stepsize_controller=controller,
            saveat=saveat, throw=False,
        )
        # sol.ys[i] = z(ts[i]) = y(t_f - ts[i]).
        # Reverse so index 0 -> t=0 and index -1 -> t=t_f.
        augmented_ys = jnp.asarray(sol.ys[::-1], dtype=jnp.float32)
        solve_succeeded = jnp.asarray(sol.result == RESULTS.successful, dtype=bool)
        return (
            augmented_ys[:, :STATE_DIM],
            augmented_ys[:, STATE_DIM:STATE_DIM + COSTATE_DIM],
            solve_succeeded,
        )

    return ts, jax.jit(jax.vmap(single))


def create_chunk_buffers(chunk_size, num_points):
    return {
        "states": np.empty((chunk_size, num_points, STATE_DIM), dtype=np.float32),
        "costates": np.empty((chunk_size, num_points, COSTATE_DIM), dtype=np.float32),
        "initial_states": np.empty((chunk_size, STATE_DIM), dtype=np.float32),
        "final_states": np.empty((chunk_size, STATE_DIM), dtype=np.float32),
        "sampled_costates": np.empty((chunk_size, COSTATE_DIM), dtype=np.float32),
        "size": 0,
    }


def flush_chunk(chunk_dir, chunk_buffers, chunk_index):
    chunk_size = chunk_buffers["size"]
    if chunk_size == 0:
        return chunk_index

    chunk_path = chunk_dir / f"chunk_{chunk_index:06d}.npz"
    np.savez_compressed(
        chunk_path,
        states=chunk_buffers["states"][:chunk_size],
        costates=chunk_buffers["costates"][:chunk_size],
        initial_states=chunk_buffers["initial_states"][:chunk_size],
        final_states=chunk_buffers["final_states"][:chunk_size],
        sampled_costates=chunk_buffers["sampled_costates"][:chunk_size],
    )
    print(f"saved chunk {chunk_index} to {chunk_path}")
    chunk_buffers["size"] = 0
    return chunk_index + 1


def is_step_limit_error(exc):
    return "maximum number of solver steps" in str(exc).lower()


def generate_dataset(
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
):
    _, jdy, norm = make_dynamics(eps=eps)
    mu = norm["mu"]
    rng = np.random.default_rng(seed)
    # Backward integrator: sample final state/costate, integrate reversed ODE,
    # then flip to forward-time order so the dataset looks identical to before.
    _, batch_integrator = make_backward_batch_integrator(
        jdy=jdy,
        t_final=norm["t_f"],
        n_points=num_points,
        rtol=rtol,
        atol=atol,
    )
    chunk_dir = output_path.parent / output_path.stem
    chunk_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = chunk_dir / "metadata.npz"
    time_grid = np.linspace(0.0, np.float32(norm["t_f"]), num_points, dtype=np.float32)
    np.savez_compressed(
        metadata_path,
        num_samples=np.int64(num_samples),
        num_points=np.int64(num_points),
        chunk_size=np.int64(CHUNK_SAVE_SIZE),
        time_grid=time_grid,
        t_final=np.float32(norm["t_f"]),
        au=np.float32(norm["AU"]),
        tu=np.float32(norm["TU"]),
        state_box_low=np.float32(box_low),
        state_box_high=np.float32(box_high),
        min_radius=np.float32(min_radius),
        seed=np.int64(seed),
        sampling_method="orbital_vis_viva_backward",
        mass_final_low=np.float32(MASS_FINAL_LOW),
        mass_final_high=np.float32(MASS_FINAL_HIGH),
        eps=np.float32(eps),
    )

    chunk_buffers = create_chunk_buffers(CHUNK_SAVE_SIZE, num_points)
    chunk_index = 0

    for batch_start in range(0, num_samples, batch_size):
        current_batch_size = min(batch_size, num_samples - batch_start)
        batch_end = batch_start + current_batch_size
        while True:
            # Sample terminal states: position/velocity via vis-viva, mass ~ Uniform(0.2, 1.0)
            yf_batch = sample_final_state_batch(
                rng=rng,
                batch_size=current_batch_size,
                low=box_low,
                high=box_high,
                min_radius=min_radius,
                mu=mu,
            )
            # Sample terminal costates: lambda_r, lambda_v free; lambda_m = 0
            costate_batch = sample_final_costate_batch(rng, current_batch_size)
            try:
                trajectory_batch, costate_trajectory_batch, solve_succeeded = batch_integrator(
                    yf_batch, costate_batch,
                )
            except EquinoxRuntimeError as exc:
                if not is_step_limit_error(exc):
                    raise
                print(f"batched solve hit step limit for {batch_start}:{batch_end}; resampling")
                continue

            solve_succeeded_np = np.asarray(jax.device_get(solve_succeeded), dtype=bool)
            if not np.all(solve_succeeded_np):
                print(f"batched solve failed for {batch_start}:{batch_end}; resampling")
                del solve_succeeded_np, trajectory_batch, costate_trajectory_batch, yf_batch, costate_batch
                continue

            state_np_batch = np.asarray(jax.device_get(trajectory_batch), dtype=np.float32)
            costate_np_batch = np.asarray(jax.device_get(costate_trajectory_batch), dtype=np.float32)

            for sample_idx in range(current_batch_size):
                write_index = chunk_buffers["size"]
                chunk_buffers["states"][write_index] = state_np_batch[sample_idx]
                chunk_buffers["costates"][write_index] = costate_np_batch[sample_idx]
                # After reversal: states[0] = backward-propagated state at t=0
                chunk_buffers["initial_states"][write_index] = state_np_batch[sample_idx, 0]
                # states[-1] = the directly sampled terminal state at t=t_f
                chunk_buffers["final_states"][write_index] = state_np_batch[sample_idx, -1]
                # costates[-1] = sampled terminal costate (lambda_m = 0)
                chunk_buffers["sampled_costates"][write_index] = costate_np_batch[sample_idx, -1]
                chunk_buffers["size"] += 1
                if chunk_buffers["size"] == CHUNK_SAVE_SIZE:
                    chunk_index = flush_chunk(chunk_dir, chunk_buffers, chunk_index)

            del state_np_batch, costate_np_batch
            del trajectory_batch, costate_trajectory_batch, solve_succeeded, solve_succeeded_np
            del yf_batch, costate_batch
            break

        generated_count = batch_end
        if log_every > 0 and (generated_count % log_every == 0 or generated_count == num_samples):
            print(f"generated {generated_count}/{num_samples} trajectories")

    chunk_index = flush_chunk(chunk_dir, chunk_buffers, chunk_index)
    print(f"saved chunked dataset to {chunk_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate Earth-Mars min-fuel trajectory samples. "
            "14-dim augmented state: 7 state (r,v,m) + 7 costate (lambda_r, lambda_v, lambda_m)."
        )
    )
    parser.add_argument("--num-samples", type=int, default=2**25)
    parser.add_argument("--num-points", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "earth_mars_minfuel_dataset_32pts.npz",
    )
    parser.add_argument("--box-low", type=float, default=-3.0)
    parser.add_argument("--box-high", type=float, default=3.0)
    parser.add_argument("--min-radius", type=float, default=1e-1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rtol", type=float, default=1e-7)
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--eps", type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_points < 2:
        raise ValueError("num_points must be at least 2")
    if args.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if args.box_low >= args.box_high:
        raise ValueError("box_low must be smaller than box_high")
    generate_dataset(
        num_samples=args.num_samples,
        num_points=args.num_points,
        seed=args.seed,
        output_path=args.output,
        box_low=args.box_low,
        box_high=args.box_high,
        min_radius=args.min_radius,
        batch_size=args.batch_size,
        rtol=args.rtol,
        atol=args.atol,
        log_every=args.log_every,
        eps=args.eps,
    )


if __name__ == "__main__":
    main()
