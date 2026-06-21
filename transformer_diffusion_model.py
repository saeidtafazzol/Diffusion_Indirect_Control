from dataclasses import dataclass

import jax.numpy as jnp
import flax.linen as nn


def sinusoidal_time_embedding(timesteps, dim, max_period=10000.0):
	timesteps = jnp.asarray(timesteps, dtype=jnp.float32).reshape(-1, 1)
	half_dim = dim // 2
	if half_dim == 0:
		return jnp.zeros((timesteps.shape[0], dim), dtype=jnp.float32)
	frequency_exponents = jnp.arange(half_dim, dtype=jnp.float32)
	frequency_exponents = frequency_exponents / jnp.maximum(half_dim - 1, 1)
	frequencies = jnp.exp(-jnp.log(max_period) * frequency_exponents)
	angles = timesteps * frequencies[None, :]
	embedding = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
	if dim % 2 == 1:
		embedding = jnp.pad(embedding, ((0, 0), (0, 1)))
	return embedding


def default_state_known_mask(batch_size, seq_len):
	mask = jnp.zeros((batch_size, seq_len), dtype=bool)
	mask = mask.at[:, 0].set(True)
	mask = mask.at[:, -1].set(True)
	return mask


def default_costate_known_mask(batch_size, seq_len):
	return jnp.zeros((batch_size, seq_len), dtype=bool)


def default_state_known_mask_no_final_mass(batch_size, seq_len, state_dim=7):
	"""Per-component state mask that pins the full initial state at t=0 and only
	position + velocity at t=t_f, leaving the terminal mass m(t_f) free.

	For the min-fuel problem the terminal mass is a free variable determined by
	the optimal control; it must NOT be prescribed as a boundary condition.
	Returns shape (batch_size, seq_len, state_dim).
	"""
	mask = jnp.zeros((batch_size, seq_len, state_dim), dtype=bool)
	mask = mask.at[:, 0, :].set(True)        # full initial state
	mask = mask.at[:, -1, :state_dim - 1].set(True)  # r_f + v_f, NOT m_f
	return mask


def final_lambda_m_costate_mask(batch_size, seq_len, costate_dim=7):
	"""Per-component costate mask marking lambda_m at the final time step as known (= 0).

	lambda_m is the last costate component (index costate_dim-1).  It is zero
	at t = t_f by the Pontryagin transversality condition when terminal mass is
	free.  Pass this mask as costate_known_mask to enforce the constraint during
	training and inference.

	Returns shape (batch_size, seq_len, costate_dim) with True only at
	[:, -1, costate_dim-1].
	"""
	mask = jnp.zeros((batch_size, seq_len, costate_dim), dtype=bool)
	return mask.at[:, -1, costate_dim - 1].set(True)


@dataclass(frozen=True)
class DiffusionTransformerConfig:
	seq_len: int = 32
	state_dim: int = 7
	costate_dim: int = 7
	embd_dim: int = 256
	num_layers: int = 8
	num_heads: int = 4
	mlp_ratio: int = 4
	p_drop_embd: float = 0.1
	p_drop_attn: float = 0.1
	use_integration_signals: bool = False
	zero_known_state_eps: bool = True


class TimeEmbedding(nn.Module):
	embd_dim: int

	@nn.compact
	def __call__(self, diffusion_steps):
		return sinusoidal_time_embedding(diffusion_steps, self.embd_dim)


class FeedForward(nn.Module):
	embd_dim: int
	mlp_ratio: int
	p_drop_embd: float

	@nn.compact
	def __call__(self, x, train):
		hidden_dim = self.embd_dim * self.mlp_ratio
		x = nn.Dense(hidden_dim, name="fc1")(x)
		x = nn.gelu(x, approximate=False)
		x = nn.Dense(self.embd_dim, name="fc2")(x)
		x = nn.Dropout(rate=self.p_drop_embd)(x, deterministic=not train)
		return x


class TransformerBlock(nn.Module):
	embd_dim: int
	num_heads: int
	mlp_ratio: int
	p_drop_embd: float
	p_drop_attn: float

	@nn.compact
	def __call__(self, x, train):
		attn_input = nn.LayerNorm(name="ln_attn")(x)
		attn_output = nn.MultiHeadDotProductAttention(
			num_heads=self.num_heads,
			qkv_features=self.embd_dim,
			out_features=self.embd_dim,
			dropout_rate=self.p_drop_attn,
			name="self_attn",
		)(attn_input, attn_input, deterministic=not train)
		attn_output = nn.Dropout(rate=self.p_drop_embd)(attn_output, deterministic=not train)
		x = x + attn_output
		ff_input = nn.LayerNorm(name="ln_ff")(x)
		x = x + FeedForward(
			embd_dim=self.embd_dim,
			mlp_ratio=self.mlp_ratio,
			p_drop_embd=self.p_drop_embd,
			name="ff",
		)(ff_input, train=train)
		return x


class IntegrationSignalEmbedding(nn.Module):
	config: DiffusionTransformerConfig

	@nn.compact
	def __call__(
		self,
		position_embedding,
		integrated_states,
		integrated_costates,
		integration_failed,
	):
		cfg = self.config
		batch_size = position_embedding.shape[0]
		segment_count = cfg.seq_len - 1
		if integrated_states is None:
			integrated_states = jnp.zeros((batch_size, segment_count, cfg.state_dim), dtype=jnp.float32)
		if integrated_costates is None:
			integrated_costates = jnp.zeros((batch_size, segment_count, cfg.costate_dim), dtype=jnp.float32)
		if integration_failed is None:
			integration_failed = jnp.zeros((batch_size, segment_count), dtype=bool)

		segment_position_embedding = position_embedding[:, 1:, :]

		integrated_state_tokens = nn.Dense(cfg.embd_dim, name="integrated_state_embed")(integrated_states)
		integrated_costate_tokens = nn.Dense(cfg.embd_dim, name="integrated_costate_embed")(integrated_costates)
		integrated_state_tokens = integrated_state_tokens + segment_position_embedding
		integrated_costate_tokens = integrated_costate_tokens + segment_position_embedding

		integrated_state_failure_token = self.param(
			"integrated_state_failure_token",
			nn.initializers.normal(stddev=0.02),
			(1, 1, cfg.embd_dim),
		)
		integrated_costate_failure_token = self.param(
			"integrated_costate_failure_token",
			nn.initializers.normal(stddev=0.02),
			(1, 1, cfg.embd_dim),
		)

		broadcast_state_failure_token = jnp.broadcast_to(
			integrated_state_failure_token,
			(batch_size, segment_count, cfg.embd_dim),
		) + segment_position_embedding
		broadcast_costate_failure_token = jnp.broadcast_to(
			integrated_costate_failure_token,
			(batch_size, segment_count, cfg.embd_dim),
		) + segment_position_embedding

		integrated_state_tokens = jnp.where(
			integration_failed[..., None],
			broadcast_state_failure_token,
			integrated_state_tokens,
		)
		integrated_costate_tokens = jnp.where(
			integration_failed[..., None],
			broadcast_costate_failure_token,
			integrated_costate_tokens,
		)
		return integrated_state_tokens, integrated_costate_tokens


class DiffusionTransformer(nn.Module):
	config: DiffusionTransformerConfig

	@nn.compact
	def __call__(
		self,
		noisy_states,
		noisy_costates,
		diffusion_steps,
		state_known_mask=None,
		costate_known_mask=None,
		integrated_states=None,
		integrated_costates=None,
		integration_failed=None,
		train=False,
	):
		cfg = self.config
		batch_size, seq_len, state_dim = noisy_states.shape
		if seq_len != cfg.seq_len:
			raise ValueError(f"Expected seq_len={cfg.seq_len}, got {seq_len}")
		if state_dim != cfg.state_dim:
			raise ValueError(f"Expected state_dim={cfg.state_dim}, got {state_dim}")
		if noisy_costates.shape != (batch_size, cfg.seq_len, cfg.costate_dim):
			raise ValueError(
				f"Expected noisy_costates shape {(batch_size, cfg.seq_len, cfg.costate_dim)}, got {noisy_costates.shape}"
			)

		if state_known_mask is None:
			state_known_mask_raw = default_state_known_mask(batch_size, cfg.seq_len)
		else:
			state_known_mask_raw = jnp.asarray(state_known_mask, dtype=bool)
		# Expand to per-component (B, T, state_dim) — mirrors costate handling.
		if state_known_mask_raw.ndim == 2:  # (B, T) → broadcast all dims
			state_known_mask_per_comp = jnp.broadcast_to(
				state_known_mask_raw[..., None],
				(batch_size, cfg.seq_len, cfg.state_dim),
			)
		else:  # (B, T, state_dim) per-component
			state_known_mask_per_comp = state_known_mask_raw
		# Per-timestep signal (any component known) used for the embedding lookup.
		state_known_mask_seq = jnp.any(state_known_mask_per_comp, axis=-1)  # (B, T)

		if costate_known_mask is None:
			costate_known_mask_raw = default_costate_known_mask(batch_size, cfg.seq_len)
		else:
			costate_known_mask_raw = jnp.asarray(costate_known_mask, dtype=bool)

		# Expand to per-component (B, T, costate_dim) for output masking.
		if costate_known_mask_raw.ndim == 2:  # (B, T)
			costate_known_mask_per_comp = jnp.broadcast_to(
				costate_known_mask_raw[..., None],
				(batch_size, cfg.seq_len, cfg.costate_dim),
			)
		else:  # (B, T, costate_dim)
			costate_known_mask_per_comp = costate_known_mask_raw
		# For the embed lookup we need a per-timestep bool: True if ANY component known.
		costate_known_mask_seq = jnp.any(costate_known_mask_per_comp, axis=-1)  # (B, T)

		position_embedding = self.param(
			"position_embedding",
			nn.initializers.normal(stddev=0.02),
			(1, cfg.seq_len, cfg.embd_dim),
		)
		position_embedding = jnp.broadcast_to(position_embedding, (batch_size, cfg.seq_len, cfg.embd_dim))

		state_tokens = nn.Dense(cfg.embd_dim, name="state_embed")(noisy_states)
		costate_tokens = nn.Dense(cfg.embd_dim, name="costate_embed")(noisy_costates)
		state_tokens = state_tokens + position_embedding
		costate_tokens = costate_tokens + position_embedding

		known_state_tokens = nn.Embed(num_embeddings=2, features=cfg.embd_dim, name="state_known_embed")(
			state_known_mask_seq.astype(jnp.int32)
		)
		state_tokens = state_tokens + known_state_tokens

		known_costate_tokens = nn.Embed(num_embeddings=2, features=cfg.embd_dim, name="costate_known_embed")(
			costate_known_mask_seq.astype(jnp.int32)
		)
		costate_tokens = costate_tokens + known_costate_tokens

		diffusion_token = TimeEmbedding(embd_dim=cfg.embd_dim, name="time_embed")(diffusion_steps)
		diffusion_token = diffusion_token[:, None, :]

		token_streams = [diffusion_token, state_tokens, costate_tokens]

		if cfg.use_integration_signals:
			integrated_state_tokens, integrated_costate_tokens = IntegrationSignalEmbedding(
				config=cfg,
				name="integration_signals",
			)(
				position_embedding=position_embedding,
				integrated_states=integrated_states,
				integrated_costates=integrated_costates,
				integration_failed=integration_failed,
			)
			token_streams.extend([integrated_state_tokens, integrated_costate_tokens])

		tokens = jnp.concatenate(token_streams, axis=1)

		for layer_idx in range(cfg.num_layers):
			tokens = TransformerBlock(
				embd_dim=cfg.embd_dim,
				num_heads=cfg.num_heads,
				mlp_ratio=cfg.mlp_ratio,
				p_drop_embd=cfg.p_drop_embd,
				p_drop_attn=cfg.p_drop_attn,
				name=f"block_{layer_idx}",
			)(tokens, train=train)

		tokens = nn.LayerNorm(name="final_ln")(tokens)
		state_start = 1
		costate_start = state_start + cfg.seq_len
		state_hidden = tokens[:, state_start:costate_start, :]
		costate_hidden = tokens[:, costate_start:costate_start + cfg.seq_len, :]
		state_eps = nn.Dense(cfg.state_dim, name="state_eps_head")(state_hidden)
		costate_eps = nn.Dense(cfg.costate_dim, name="costate_eps_head")(costate_hidden)

		if cfg.zero_known_state_eps:
			# Per-component mask: zero prediction only for each known state component.
			state_eps = jnp.where(state_known_mask_per_comp, 0.0, state_eps)
		# Per-component mask: zero prediction only for each known costate component.
		costate_eps = jnp.where(costate_known_mask_per_comp, 0.0, costate_eps)

		return {
			"state_eps": state_eps,
			"costate_eps": costate_eps,
			"hidden": tokens,
			"state_known_mask": state_known_mask_raw,
			"costate_known_mask": costate_known_mask_raw,
		}
