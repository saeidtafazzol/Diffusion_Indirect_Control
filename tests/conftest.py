"""Shared pytest fixtures for the src/ test suite.

src/ modules use flat imports (``from core import ...``) rather than a
package-relative style, so this conftest inserts src/ onto sys.path exactly
like experiments/*.py do, before any test module imports from it.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

CHECKPOINT_PATH = ROOT / "checkpoints" / "indiff_ctrl_latest.msgpack"
KNOWN_TRAJECTORIES_PATH = Path(__file__).resolve().parent / "fixtures" / "known_convergent_trajectories.json"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: exercises the real checkpoint / IPOPT solves (seconds-to-minutes per test)"
    )


@pytest.fixture(scope="session")
def dynamics_bundle():
    """(dy, jdy, norm) from core.make_dynamics, compiled once for the whole session."""
    from core import make_dynamics

    return make_dynamics(eps=1e-4, compile_jax=True)


@pytest.fixture(scope="session")
def dynamics_stm_bundle():
    """(dy, jdy, jjac, norm) from core.make_dynamics_stm, compiled once for the whole session."""
    from core import make_dynamics_stm

    return make_dynamics_stm(eps=1e-4)


@pytest.fixture(scope="session")
def norm():
    from core import build_normalization

    return build_normalization()


@pytest.fixture(scope="session")
def known_trajectories():
    with open(KNOWN_TRAJECTORIES_PATH) as f:
        data = json.load(f)
    for traj in data["trajectories"]:
        traj["initial_state"] = np.asarray(traj["initial_state"], dtype=np.float64)
        traj["final_state"] = np.asarray(traj["final_state"], dtype=np.float64)
    return data


@pytest.fixture(scope="session")
def checkpoint_bundle():
    """(params, policy, time_grid, norm) loaded from the real trained checkpoint.

    Skips dependent tests if the checkpoint file is not present, rather than
    failing the whole run in environments where the (450MB) weights aren't
    checked out.
    """
    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"checkpoint not found at {CHECKPOINT_PATH}")

    import jax
    import optax
    from flax import serialization
    from flax.serialization import msgpack_restore

    from core import STATE_DIM, COSTATE_DIM, make_dynamics
    from jax_ddpm_scheduler import JaxDDPMScheduler
    from policy import IndiffCtrlPolicy
    from transformer_diffusion_model import DiffusionTransformer, DiffusionTransformerConfig

    raw_bytes = CHECKPOINT_PATH.read_bytes()
    raw = msgpack_restore(raw_bytes)
    tg = np.asarray(raw["time_grid"], dtype=np.float32).ravel()
    sc = raw.get("scheduler_config", {})

    def _s(v):
        return v.decode() if isinstance(v, bytes) else v

    sc = {_s(k): (_s(v) if isinstance(v, bytes) else v) for k, v in sc.items()}
    seq_len = len(tg)

    def _sc(k, d):
        v = sc.get(k, d)
        return d if v is None else v

    noise_scheduler = JaxDDPMScheduler(
        num_train_timesteps=int(_sc("num_train_timesteps", 5000)),
        beta_schedule=str(_sc("beta_schedule", "squaredcos_cap_v2")),
        prediction_type=str(_sc("prediction_type", "epsilon")),
        clip_sample=bool(_sc("clip_sample", True)),
        clip_sample_range=float(_sc("clip_sample_range", 5.0)),
        noise_scale=float(_sc("noise_scale", 2.0)),
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
    sch_tpl = {
        "num_train_timesteps": 5000, "beta_schedule": "squaredcos_cap_v2",
        "prediction_type": "epsilon", "clip_sample": True,
        "clip_sample_range": 5.0, "noise_scale": 2.0,
    }
    state_tpl = {
        "params": params_tpl, "opt_state": opt_tpl,
        "step": np.int32(0), "time_grid": np.zeros(seq_len, np.float32),
        "scheduler_config": sch_tpl,
    }
    loaded = serialization.from_bytes(state_tpl, raw_bytes)
    params = loaded["params"]

    _, _, dyn_norm = make_dynamics(eps=1e-4, compile_jax=False)
    return params, policy, tg, dyn_norm
