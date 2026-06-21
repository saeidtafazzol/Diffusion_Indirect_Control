from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from diffrax import Dopri5, ODETerm, PIDController, RESULTS, SaveAt, diffeqsolve
from flax import struct

from core import COSTATE_DIM, STATE_DIM, build_rhs, make_dynamics
from jax_ddpm_scheduler import JaxDDPMScheduler
from transformer_diffusion_model import (
	DiffusionTransformer,
	default_costate_known_mask,
	default_state_known_mask,
)


@struct.dataclass
class PreparedTrainingBatch:
	noisy_states: jnp.ndarray
	noisy_costates: jnp.ndarray
	target_states: jnp.ndarray
	target_costates: jnp.ndarray
	state_known_mask: jnp.ndarray
	costate_known_mask: jnp.ndarray
	timesteps: jnp.ndarray


class IndiffCtrlPolicy:
	def __init__(
		self,
		model: DiffusionTransformer,
		noise_scheduler: JaxDDPMScheduler,
		time_grid: np.ndarray,
		segment_rtol: float = 1e-7,
		segment_atol: float = 1e-9,
		num_inference_steps: Optional[int] = None,
		eps: float = 1e-4,
	):
		self.model = model
		self.noise_scheduler = noise_scheduler
		self.time_grid = np.asarray(time_grid, dtype=np.float32)
		if self.time_grid.ndim != 1 or self.time_grid.shape[0] < 2:
			raise ValueError("time_grid must be a 1D array with at least two entries")

		_, self.jdy, self.norm = make_dynamics(eps=eps)
		self.segment_dt = jnp.asarray(np.diff(self.time_grid), dtype=jnp.float32)
		self.segment_integrator = self._make_segment_integrator(segment_rtol, segment_atol)

		if num_inference_steps is None:
			num_inference_steps = noise_scheduler.config.num_train_timesteps
		self.num_inference_steps = num_inference_steps

	def init_params(self, rng_key: jax.Array, batch_size: int) -> Any:
		cfg = self.model.config
		inputs = {
			"noisy_states": jnp.zeros((batch_size, cfg.seq_len, cfg.state_dim), dtype=jnp.float32),
			"noisy_costates": jnp.zeros((batch_size, cfg.seq_len, cfg.costate_dim), dtype=jnp.float32),
			"diffusion_steps": jnp.zeros((batch_size,), dtype=jnp.int32),
			"train": False,
		}
		if cfg.use_integration_signals:
			inputs["integrated_states"] = jnp.zeros(
				(batch_size, cfg.seq_len - 1, cfg.state_dim), dtype=jnp.float32
			)
			inputs["integrated_costates"] = jnp.zeros(
				(batch_size, cfg.seq_len - 1, cfg.costate_dim), dtype=jnp.float32
			)
			inputs["integration_failed"] = jnp.zeros((batch_size, cfg.seq_len - 1), dtype=bool)
		variables = self.model.init(rng_key, **inputs)
		return variables["params"]

	def _make_segment_integrator(self, rtol: float, atol: float):
		term = ODETerm(build_rhs(self.jdy))
		solver = Dopri5()
		controller = PIDController(rtol=rtol, atol=atol)
		saveat = SaveAt(t1=True)

		def single(y0, costate, dt):
			augmented_y0 = jnp.concatenate([y0, costate], axis=0)
			sol = diffeqsolve(
				term, solver,
				t0=0.0, t1=dt, dt0=None,
				y0=augmented_y0, args=None,
				stepsize_controller=controller,
				saveat=saveat, throw=False,
			)
			solve_succeeded = jnp.asarray(sol.result == RESULTS.successful, dtype=bool)
			terminal = jnp.asarray(sol.ys[0], dtype=jnp.float32)
			return terminal[:STATE_DIM], terminal[STATE_DIM:STATE_DIM + COSTATE_DIM], solve_succeeded

		return jax.jit(jax.vmap(single))

	def _validate_mask(self, mask: jnp.ndarray, batch_size: int, seq_len: int, name: str) -> jnp.ndarray:
		mask = jnp.asarray(mask, dtype=bool)
		if "state" in name and "costate" not in name:
			valid_shapes = [(batch_size, seq_len), (batch_size, seq_len, STATE_DIM)]
		else:
			valid_shapes = [(batch_size, seq_len), (batch_size, seq_len, COSTATE_DIM)]
		if mask.shape not in valid_shapes:
			raise ValueError(f"{name} must have shape in {valid_shapes}, got {mask.shape}")
		return mask

	def _resolve_known_masks(
		self,
		batch_size: int,
		seq_len: int,
		state_known_mask: Optional[jnp.ndarray] = None,
		costate_known_mask: Optional[jnp.ndarray] = None,
	) -> Tuple[jnp.ndarray, jnp.ndarray]:
		if state_known_mask is None:
			state_known_mask = default_state_known_mask(batch_size, seq_len)
		else:
			state_known_mask = self._validate_mask(state_known_mask, batch_size, seq_len, "state_known_mask")

		if costate_known_mask is None:
			costate_known_mask = default_costate_known_mask(batch_size, seq_len)
		else:
			costate_known_mask = self._validate_mask(costate_known_mask, batch_size, seq_len, "costate_known_mask")

		return state_known_mask, costate_known_mask

	def _combine_trajectory(self, states: jnp.ndarray, costates: jnp.ndarray) -> jnp.ndarray:
		return jnp.concatenate([states, costates], axis=-1)

	def _split_trajectory(self, trajectory: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
		return trajectory[..., :STATE_DIM], trajectory[..., STATE_DIM:STATE_DIM + COSTATE_DIM]

	def _build_condition_mask(self, state_known_mask: jnp.ndarray, costate_known_mask: jnp.ndarray) -> jnp.ndarray:
		if state_known_mask.ndim == 3:  # (B, T, STATE_DIM) per-component
			state_mask = state_known_mask
		else:  # (B, T) → broadcast to (B, T, STATE_DIM)
			state_mask = jnp.broadcast_to(state_known_mask[..., None], state_known_mask.shape + (STATE_DIM,))
		if costate_known_mask.ndim == 3:  # (B, T, COSTATE_DIM) per-component
			costate_mask = costate_known_mask
		else:  # (B, T) → broadcast to (B, T, COSTATE_DIM)
			costate_mask = jnp.broadcast_to(costate_known_mask[..., None], costate_known_mask.shape + (COSTATE_DIM,))
		return self._combine_trajectory(state_mask, costate_mask)

	def compute_integration_signals(
		self,
		states: jnp.ndarray,
		costates: jnp.ndarray,
	) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
		"""Integrate each segment forward and return next-point predictions.

		Args:
			states: (B, T, STATE_DIM=7) position/velocity/mass
			costates: (B, T, COSTATE_DIM=7) full costate including lambda_m
		"""
		batch_size, seq_len, state_dim = states.shape
		if seq_len != self.model.config.seq_len:
			raise ValueError(f"Expected {self.model.config.seq_len} states, got {seq_len}")
		if state_dim != STATE_DIM:
			raise ValueError(f"Expected state_dim={STATE_DIM}, got {state_dim}")
		if costates.shape != (batch_size, seq_len, COSTATE_DIM):
			raise ValueError(
				f"Expected costates shape {(batch_size, seq_len, COSTATE_DIM)}, got {costates.shape}"
			)

		segment_count = seq_len - 1

		segment_y0 = states[:, :-1, :]        # (B, seg, 7)
		segment_costates = costates[:, :-1, :] # (B, seg, 7)
		segment_dt = jnp.broadcast_to(self.segment_dt[None, :], (batch_size, segment_count))

		flat_y0 = segment_y0.reshape((-1, STATE_DIM))
		flat_costates = segment_costates.reshape((-1, COSTATE_DIM))
		flat_dt = segment_dt.reshape((-1,))

		int_state_flat, int_costate_flat, success_flat = self.segment_integrator(
			flat_y0, flat_costates, flat_dt,
		)
		integrated_states = int_state_flat.reshape((batch_size, segment_count, STATE_DIM))
		integrated_costates = int_costate_flat.reshape((batch_size, segment_count, COSTATE_DIM))
		integration_failed = ~success_flat.reshape((batch_size, segment_count))

		integrated_states = jnp.where(integration_failed[..., None], 0.0, integrated_states)
		integrated_costates = jnp.where(integration_failed[..., None], 0.0, integrated_costates)

		return (
			jax.lax.stop_gradient(integrated_states),
			jax.lax.stop_gradient(integrated_costates),
			jax.lax.stop_gradient(integration_failed),
		)

	def prepare_training_batch(
		self,
		batch: Dict[str, np.ndarray],
		rng_key: jax.Array,
		state_known_mask: Optional[jnp.ndarray] = None,
		costate_known_mask: Optional[jnp.ndarray] = None,
	) -> PreparedTrainingBatch:
		states = jnp.asarray(batch["states"], dtype=jnp.float32)
		costates = jnp.asarray(batch["costates"], dtype=jnp.float32)
		batch_size, seq_len, _ = states.shape
		state_known_mask, costate_known_mask = self._resolve_known_masks(
			batch_size=batch_size,
			seq_len=seq_len,
			state_known_mask=state_known_mask,
			costate_known_mask=costate_known_mask,
		)

		clean_trajectory = self._combine_trajectory(states, costates)
		condition_mask = self._build_condition_mask(state_known_mask, costate_known_mask)
		rng_key, noise_key, timestep_key = jax.random.split(rng_key, 3)
		noise = self.noise_scheduler.sample_noise(noise_key, clean_trajectory.shape, dtype=jnp.float32)
		timesteps = jax.random.randint(
			timestep_key,
			shape=(batch_size,),
			minval=0,
			maxval=self.noise_scheduler.config.num_train_timesteps,
			dtype=jnp.int32,
		)
		noisy_trajectory = self.noise_scheduler.add_noise(clean_trajectory, noise, timesteps)
		noisy_trajectory = jnp.where(condition_mask, clean_trajectory, noisy_trajectory)

		prediction_type = self.noise_scheduler.config.prediction_type
		if prediction_type == "epsilon":
			target = noise
		elif prediction_type == "sample":
			target = clean_trajectory
		else:
			raise ValueError(f"Unsupported prediction type {prediction_type}")

		noisy_states, noisy_costates = self._split_trajectory(noisy_trajectory)
		target_states, target_costates = self._split_trajectory(target)

		return PreparedTrainingBatch(
			noisy_states=noisy_states,
			noisy_costates=noisy_costates,
			target_states=target_states,
			target_costates=target_costates,
			state_known_mask=state_known_mask,
			costate_known_mask=costate_known_mask,
			timesteps=timesteps,
		)

	def compute_loss_from_prepared(
		self,
		params: Any,
		prepared_batch: PreparedTrainingBatch,
		dropout_key: jax.Array,
	) -> jnp.ndarray:
		integrated_states = None
		integrated_costates = None
		integration_failed = None
		if self.model.config.use_integration_signals:
			integrated_states, integrated_costates, integration_failed = self.compute_integration_signals(
				prepared_batch.noisy_states,
				prepared_batch.noisy_costates,
			)

		outputs = self.model.apply(
			{"params": params},
			noisy_states=prepared_batch.noisy_states,
			noisy_costates=prepared_batch.noisy_costates,
			diffusion_steps=prepared_batch.timesteps,
			state_known_mask=prepared_batch.state_known_mask,
			costate_known_mask=prepared_batch.costate_known_mask,
			integrated_states=integrated_states,
			integrated_costates=integrated_costates,
			integration_failed=integration_failed,
			train=True,
			rngs={"dropout": dropout_key},
		)

		state_mask = (~prepared_batch.state_known_mask).astype(jnp.float32)
		if state_mask.ndim == 2:  # (B, T) → expand to (B, T, STATE_DIM)
			state_mask = state_mask[..., None]
		# costate_known_mask may be (B, T) or per-component (B, T, COSTATE_DIM).
		if prepared_batch.costate_known_mask.ndim == 3:
			costate_mask = (~prepared_batch.costate_known_mask).astype(jnp.float32)
		else:
			costate_mask = (~prepared_batch.costate_known_mask)[..., None].astype(jnp.float32)
		state_sqerr = (outputs["state_eps"] - prepared_batch.target_states) ** 2
		costate_sqerr = (outputs["costate_eps"] - prepared_batch.target_costates) ** 2

		state_loss = jnp.sum(state_sqerr * state_mask)
		costate_loss = jnp.sum(costate_sqerr * costate_mask)
		denom = jnp.sum(state_mask) + jnp.sum(costate_mask)
		return (state_loss + costate_loss) / jnp.maximum(denom, 1.0)

	def compute_loss(
		self,
		params: Any,
		batch: Dict[str, np.ndarray],
		rng_key: jax.Array,
		dropout_key: jax.Array,
		state_known_mask: Optional[jnp.ndarray] = None,
		costate_known_mask: Optional[jnp.ndarray] = None,
	) -> jnp.ndarray:
		prepared_batch = self.prepare_training_batch(
			batch=batch,
			rng_key=rng_key,
			state_known_mask=state_known_mask,
			costate_known_mask=costate_known_mask,
		)
		return self.compute_loss_from_prepared(params, prepared_batch, dropout_key)

	def predict_trajectory(
		self,
		params: Any,
		rng_key: jax.Array,
		initial_states: np.ndarray,
		final_states: np.ndarray,
		state_known_mask: Optional[jnp.ndarray] = None,
		costate_known_mask: Optional[jnp.ndarray] = None,
		known_costates: Optional[np.ndarray] = None,
		num_inference_steps: Optional[int] = None,
		**scheduler_step_kwargs,
	) -> Dict[str, jnp.ndarray]:
		initial_states = np.asarray(initial_states, dtype=np.float32)
		final_states = np.asarray(final_states, dtype=np.float32)
		if initial_states.shape != final_states.shape:
			raise ValueError(
				f"initial_states and final_states must have the same shape, "
				f"got {initial_states.shape} and {final_states.shape}"
			)
		if initial_states.ndim != 2 or initial_states.shape[1] != STATE_DIM:
			raise ValueError(f"Expected (B, {STATE_DIM}), got {initial_states.shape}")

		batch_size = initial_states.shape[0]
		cfg = self.model.config
		state_known_mask, costate_known_mask = self._resolve_known_masks(
			batch_size=batch_size,
			seq_len=cfg.seq_len,
			state_known_mask=state_known_mask,
			costate_known_mask=costate_known_mask,
		)

		condition_states = np.zeros((batch_size, cfg.seq_len, STATE_DIM), dtype=np.float32)
		condition_costates = np.zeros((batch_size, cfg.seq_len, COSTATE_DIM), dtype=np.float32)
		condition_states[:, 0, :] = initial_states
		condition_states[:, -1, :] = final_states
		if known_costates is not None:
			known_costates = np.asarray(known_costates, dtype=np.float32)
			if known_costates.shape != (batch_size, cfg.seq_len, COSTATE_DIM):
				raise ValueError(
					f"known_costates must have shape {(batch_size, cfg.seq_len, COSTATE_DIM)}, "
					f"got {known_costates.shape}"
				)
			condition_costates = known_costates

		condition_trajectory = jnp.asarray(
			np.concatenate([condition_states, condition_costates], axis=-1),
			dtype=jnp.float32,
		)
		condition_mask = self._build_condition_mask(state_known_mask, costate_known_mask)
		rng_key, sample_key = jax.random.split(rng_key)
		trajectory = self.noise_scheduler.sample_noise(sample_key, condition_trajectory.shape, dtype=jnp.float32)

		if num_inference_steps is None:
			num_inference_steps = self.num_inference_steps
		self.noise_scheduler.set_timesteps(num_inference_steps)

		for timestep in self.noise_scheduler.timesteps:
			trajectory = jnp.where(condition_mask, condition_trajectory, trajectory)
			states = trajectory[..., :STATE_DIM]
			costates = trajectory[..., STATE_DIM:STATE_DIM + COSTATE_DIM]

			integrated_states = None
			integrated_costates = None
			integration_failed = None
			if self.model.config.use_integration_signals:
				integrated_states, integrated_costates, integration_failed = self.compute_integration_signals(
					states, costates,
				)

			outputs = self.model.apply(
				{"params": params},
				noisy_states=states,
				noisy_costates=costates,
				diffusion_steps=jnp.full((batch_size,), int(timestep), dtype=jnp.int32),
				state_known_mask=state_known_mask,
				costate_known_mask=costate_known_mask,
				integrated_states=integrated_states,
				integrated_costates=integrated_costates,
				integration_failed=integration_failed,
				train=False,
			)

			model_output = self._combine_trajectory(outputs["state_eps"], outputs["costate_eps"])
			rng_key, step_key = jax.random.split(rng_key)
			step = self.noise_scheduler.step(
				model_output=model_output,
				timestep=int(timestep),
				sample=trajectory,
				rng_key=step_key,
				**scheduler_step_kwargs,
			)
			trajectory = step.prev_sample

		trajectory = jnp.where(condition_mask, condition_trajectory, trajectory)
		predicted_states = trajectory[..., :STATE_DIM]
		predicted_costates = trajectory[..., STATE_DIM:STATE_DIM + COSTATE_DIM]

		integrated_states = None
		integrated_costates = None
		integration_failed = None
		if self.model.config.use_integration_signals:
			integrated_states, integrated_costates, integration_failed = self.compute_integration_signals(
				predicted_states, predicted_costates,
			)

		return {
			"states": predicted_states,
			"costates": predicted_costates,
			"state_known_mask": state_known_mask,
			"costate_known_mask": costate_known_mask,
			"integrated_states": integrated_states,
			"integrated_costates": integrated_costates,
			"integration_failed": integration_failed,
		}
