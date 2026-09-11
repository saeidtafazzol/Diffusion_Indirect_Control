"""Unit tests for the unit-testable pieces of src/train.py.

train.py's `main()` is a CLI training loop (argparse + real dataset dir), so
it isn't exercised directly here. Instead we test the pure/composable helpers:
ChunkedEarthMarsDataset (against a tiny synthetic dataset dir), make_train_step,
make_segment_arc_integrator, and compute_segment_arcs.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from core import STATE_DIM, COSTATE_DIM
from jax_ddpm_scheduler import JaxDDPMScheduler
from policy import IndiffCtrlPolicy
from train import ChunkedEarthMarsDataset, compute_segment_arcs, make_segment_arc_integrator, make_train_step
from transformer_diffusion_model import DiffusionTransformer, DiffusionTransformerConfig

SEQ_LEN = 5


def _write_synthetic_dataset(dataset_dir, num_samples=12, chunk_size=5):
    rng = np.random.default_rng(0)
    time_grid = np.linspace(0.0, 1.0, SEQ_LEN, dtype=np.float32)
    np.savez(
        dataset_dir / "metadata.npz",
        num_samples=num_samples, num_points=SEQ_LEN, chunk_size=chunk_size,
        time_grid=time_grid, t_final=1.0, state_box_low=-1.0, state_box_high=1.0,
        min_radius=0.1,
    )
    remaining = num_samples
    chunk_idx = 0
    while remaining > 0:
        n = min(chunk_size, remaining)
        states = rng.normal(size=(n, SEQ_LEN, STATE_DIM)).astype(np.float32)
        states[:, :, 6] = 1.0
        costates = rng.normal(size=(n, SEQ_LEN, COSTATE_DIM)).astype(np.float32)
        initial_states = states[:, 0, :]
        final_states = states[:, -1, :]
        sampled_costates = costates[:, 0, :]
        np.savez(
            dataset_dir / f"chunk_{chunk_idx:03d}.npz",
            states=states, costates=costates, initial_states=initial_states,
            final_states=final_states, sampled_costates=sampled_costates,
        )
        remaining -= n
        chunk_idx += 1
    return time_grid


@pytest.fixture
def synthetic_dataset(tmp_path):
    _write_synthetic_dataset(tmp_path)
    return ChunkedEarthMarsDataset(tmp_path)


def test_chunked_dataset_metadata_parsed(synthetic_dataset):
    ds = synthetic_dataset
    assert ds.num_samples == 12
    assert ds.num_points == SEQ_LEN
    assert ds.time_grid.shape == (SEQ_LEN,)
    assert ds.chunk_sizes == [5, 5, 2]


def test_chunked_dataset_raises_on_empty_dir(tmp_path):
    # A valid metadata.npz but zero chunk_*.npz files: constructor reads
    # metadata before checking for chunks, so metadata must parse cleanly.
    time_grid = np.linspace(0.0, 1.0, SEQ_LEN, dtype=np.float32)
    np.savez(
        tmp_path / "metadata.npz",
        num_samples=0, num_points=SEQ_LEN, chunk_size=5,
        time_grid=time_grid, t_final=1.0, state_box_low=-1.0, state_box_high=1.0,
        min_radius=0.1,
    )
    with pytest.raises(ValueError):
        ChunkedEarthMarsDataset(tmp_path)


def test_chunked_dataset_sample_batch_shapes(synthetic_dataset):
    rng = np.random.default_rng(1)
    batch = synthetic_dataset.sample_batch(rng, batch_size=4)
    assert batch["states"].shape == (4, SEQ_LEN, STATE_DIM)
    assert batch["costates"].shape == (4, SEQ_LEN, COSTATE_DIM)
    assert batch["initial_states"].shape == (4, STATE_DIM)
    assert batch["final_states"].shape == (4, STATE_DIM)


def test_chunked_dataset_sample_batch_is_reproducible_with_seeded_rng(synthetic_dataset):
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    batch_a = synthetic_dataset.sample_batch(rng_a, batch_size=3)
    batch_b = synthetic_dataset.sample_batch(rng_b, batch_size=3)
    np.testing.assert_array_equal(batch_a["states"], batch_b["states"])


# ── make_train_step ───────────────────────────────────────────────────────────

@pytest.fixture
def small_policy():
    time_grid = np.linspace(0.0, 1.0, SEQ_LEN, dtype=np.float32)
    model = DiffusionTransformer(DiffusionTransformerConfig(
        seq_len=SEQ_LEN, state_dim=STATE_DIM, costate_dim=COSTATE_DIM,
        embd_dim=16, num_layers=1, num_heads=2, mlp_ratio=2,
        p_drop_embd=0.0, p_drop_attn=0.0,
    ))
    scheduler = JaxDDPMScheduler(num_train_timesteps=20)
    return IndiffCtrlPolicy(model=model, noise_scheduler=scheduler, time_grid=time_grid)


def test_make_train_step_decreases_loss_over_iterations(small_policy):
    policy = small_policy
    key = jax.random.PRNGKey(0)
    params = policy.init_params(key, batch_size=4)
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(params)
    train_step = make_train_step(policy, optimizer)

    rng = np.random.default_rng(0)
    states = rng.normal(size=(4, SEQ_LEN, STATE_DIM)).astype(np.float32)
    states[:, :, 6] = 1.0
    costates = rng.normal(size=(4, SEQ_LEN, COSTATE_DIM)).astype(np.float32)
    batch = {"states": states, "costates": costates}

    losses = []
    for i in range(15):
        key, noise_key, dropout_key = jax.random.split(key, 3)
        prepared = policy.prepare_training_batch(batch, noise_key)
        params, opt_state, loss = train_step(params, opt_state, prepared, dropout_key)
        losses.append(float(loss))

    assert all(np.isfinite(losses))
    # Overfitting a fixed batch should push loss down substantially.
    assert losses[-1] < losses[0]


# ── make_segment_arc_integrator / compute_segment_arcs ───────────────────────

_SAFE_STATE = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
# lambda_v = 0 is a genuine 0/0 singularity in the optimal thrust direction
# (alpha = -lambda_v / |lambda_v|), which blows up to inf and fails the
# adaptive-step integrator; keep it small-but-nonzero instead.
_SAFE_COSTATE = np.array([0, 0, 0, 1e-3, 1e-3, 1e-3, 0.0], dtype=np.float32)


def test_segment_arc_integrator_shapes(small_policy):
    integrator = make_segment_arc_integrator(small_policy, arc_points=4, rtol=1e-6, atol=1e-8)
    batch = 3
    y0 = jnp.broadcast_to(jnp.asarray(_SAFE_STATE), (batch, STATE_DIM))
    costate = jnp.broadcast_to(jnp.asarray(_SAFE_COSTATE), (batch, COSTATE_DIM))
    dt = jnp.full((batch,), 0.01)
    arcs, success = integrator(y0, costate, dt)
    assert arcs.shape == (batch, 4, STATE_DIM)
    assert success.shape == (batch,)


def test_compute_segment_arcs_matches_policy_segment_dt(small_policy):
    integrator = make_segment_arc_integrator(small_policy, arc_points=3, rtol=1e-6, atol=1e-8)
    states = np.broadcast_to(_SAFE_STATE, (SEQ_LEN, STATE_DIM)).copy()
    costates = np.broadcast_to(_SAFE_COSTATE, (SEQ_LEN, COSTATE_DIM)).copy()
    arcs, success = compute_segment_arcs(small_policy, integrator, states, costates)
    assert arcs.shape == (SEQ_LEN - 1, 3, STATE_DIM)
    assert success.shape == (SEQ_LEN - 1,)
    assert np.all(success)  # benign zero-ish dynamics should integrate successfully
