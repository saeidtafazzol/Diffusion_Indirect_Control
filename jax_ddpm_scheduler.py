from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np


def _betas_for_alpha_bar(
	num_diffusion_timesteps: int,
	max_beta: float = 0.999,
) -> np.ndarray:
	def alpha_bar(time_step: float) -> float:
		return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

	betas = []
	for step in range(num_diffusion_timesteps):
		t1 = step / num_diffusion_timesteps
		t2 = (step + 1) / num_diffusion_timesteps
		betas.append(min(1.0 - alpha_bar(t2) / alpha_bar(t1), max_beta))
	return np.asarray(betas, dtype=np.float32)


@dataclass(frozen=True)
class DDPMStepOutput:
	prev_sample: jnp.ndarray
	pred_original_sample: jnp.ndarray


class JaxDDPMScheduler:
	def __init__(
		self,
		num_train_timesteps: int = 1000,
		beta_start: float = 1e-4,
		beta_end: float = 2e-2,
		beta_schedule: str = "linear",
		prediction_type: str = "epsilon",
		variance_type: str = "fixed_small",
		clip_sample: bool = False,
		clip_sample_range: float = 3.0,
		noise_scale: float = 1.0,
	):
		if prediction_type not in {"epsilon", "sample"}:
			raise ValueError(f"Unsupported prediction_type {prediction_type}")
		if variance_type != "fixed_small":
			raise ValueError(f"Unsupported variance_type {variance_type}")
		if noise_scale <= 0.0:
			raise ValueError(f"noise_scale must be positive, got {noise_scale}")

		self.config = SimpleNamespace(
			num_train_timesteps=num_train_timesteps,
			beta_start=beta_start,
			beta_end=beta_end,
			beta_schedule=beta_schedule,
			prediction_type=prediction_type,
			variance_type=variance_type,
			clip_sample=clip_sample,
			clip_sample_range=clip_sample_range,
			noise_scale=noise_scale,
		)

		if beta_schedule == "linear":
			betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
		elif beta_schedule == "scaled_linear":
			betas = np.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=np.float32) ** 2
		elif beta_schedule == "squaredcos_cap_v2":
			betas = _betas_for_alpha_bar(num_train_timesteps)
		else:
			raise ValueError(f"Unsupported beta_schedule {beta_schedule}")

		self.betas = jnp.asarray(betas, dtype=jnp.float32)
		self.alphas = 1.0 - self.betas
		self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
		self.one = jnp.array(1.0, dtype=jnp.float32)
		self.timesteps = np.arange(num_train_timesteps - 1, -1, -1, dtype=np.int32)
		self._prev_timestep = {int(t): int(prev) for t, prev in zip(self.timesteps[:-1], self.timesteps[1:])}
		self._prev_timestep[int(self.timesteps[-1])] = -1

	def sample_noise(
		self,
		rng_key: jax.Array,
		shape: tuple[int, ...],
		dtype: jnp.dtype = jnp.float32,
	) -> jnp.ndarray:
		return self.config.noise_scale * jax.random.normal(rng_key, shape, dtype=dtype)

	def set_timesteps(self, num_inference_steps: int, power: float = 1.0) -> None:
		"""Set inference timesteps.

		Args:
			num_inference_steps: Number of denoising steps.
			power: Controls step density near the fully-denoised end (t = 0).
				``power=1.0`` gives uniform spacing.  Increasing power concentrates
				more steps near t ≈ 0 (the final, clean trajectory).  For example
				``power=2.0`` squares the uniform grid so spacing grows toward t_max.
		"""
		if num_inference_steps < 1 or num_inference_steps > self.config.num_train_timesteps:
			raise ValueError(
				f"num_inference_steps must be in [1, {self.config.num_train_timesteps}], got {num_inference_steps}"
			)
		if power <= 0.0:
			raise ValueError(f"power must be positive, got {power}")
		T = self.config.num_train_timesteps
		# linspace(0, (T-1)^(1/power), N)**power — dense near 0, higher power = more concentrated
		timesteps = np.linspace(0.0, (T - 1) ** (1.0 / power), num_inference_steps, dtype=np.float32) ** power
		timesteps = np.round(timesteps)[::-1].astype(np.int32)
		# deduplicate while preserving descending order
		_, idx = np.unique(-timesteps, return_index=True)
		timesteps = timesteps[np.sort(idx)]
		self.timesteps = timesteps
		self._prev_timestep = {int(t): int(prev) for t, prev in zip(timesteps[:-1], timesteps[1:])}
		self._prev_timestep[int(timesteps[-1])] = -1

	def add_noise(
		self,
		original_samples: jnp.ndarray,
		noise: jnp.ndarray,
		timesteps: jnp.ndarray,
	) -> jnp.ndarray:
		timesteps = jnp.asarray(timesteps, dtype=jnp.int32)
		sqrt_alpha_prod = jnp.sqrt(self.alphas_cumprod[timesteps]).reshape((-1, 1, 1))
		sqrt_one_minus_alpha_prod = jnp.sqrt(1.0 - self.alphas_cumprod[timesteps]).reshape((-1, 1, 1))
		return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

	def _get_variance(self, timestep: int, prev_timestep: int) -> jnp.ndarray:
		alpha_prod_t = self.alphas_cumprod[timestep]
		alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.one
		current_beta_t = 1.0 - alpha_prod_t / alpha_prod_t_prev
		variance = (1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t) * current_beta_t
		return jnp.clip(variance, a_min=1e-20)

	def step(
		self,
		model_output: jnp.ndarray,
		timestep: int,
		sample: jnp.ndarray,
		rng_key: Optional[jax.Array] = None,
		variance_noise: Optional[jnp.ndarray] = None,
	) -> DDPMStepOutput:
		timestep = int(timestep)
		prev_timestep = self._prev_timestep[timestep]

		alpha_prod_t = self.alphas_cumprod[timestep]
		alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.one
		beta_prod_t = 1.0 - alpha_prod_t
		beta_prod_t_prev = 1.0 - alpha_prod_t_prev
		current_alpha_t = alpha_prod_t / alpha_prod_t_prev
		current_beta_t = 1.0 - current_alpha_t

		if self.config.prediction_type == "epsilon":
			pred_original_sample = (sample - jnp.sqrt(beta_prod_t) * model_output) / jnp.sqrt(alpha_prod_t)
		else:
			pred_original_sample = model_output

		if self.config.clip_sample:
			pred_original_sample = jnp.clip(
				pred_original_sample,
				-self.config.clip_sample_range,
				self.config.clip_sample_range,
			)

		pred_original_sample_coeff = jnp.sqrt(alpha_prod_t_prev) * current_beta_t / beta_prod_t
		current_sample_coeff = jnp.sqrt(current_alpha_t) * beta_prod_t_prev / beta_prod_t
		prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

		if timestep > 0:
			variance = jnp.sqrt(self._get_variance(timestep, prev_timestep))
			if variance_noise is None:
				if rng_key is None:
					raise ValueError("step requires rng_key or variance_noise when timestep > 0")
				variance_noise = self.sample_noise(rng_key, sample.shape, dtype=sample.dtype)
			prev_sample = prev_sample + variance * variance_noise

		return DDPMStepOutput(
			prev_sample=prev_sample,
			pred_original_sample=pred_original_sample,
		)