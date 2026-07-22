
import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(1, str(_root / "experiments"))
del _root
import argparse
import json
import math
import time
import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from diffrax import Dopri5, ODETerm, PIDController, RESULTS, SaveAt, diffeqsolve
from flax import serialization

from core import COSTATE_DIM, STATE_DIM, build_rhs
from dataset_gen import integrate_trajectory, sample_costate, sample_initial_state
from policy import IndiffCtrlPolicy
from jax_ddpm_scheduler import JaxDDPMScheduler
from transformer_diffusion_model import DiffusionTransformer, DiffusionTransformerConfig, final_lambda_m_costate_mask, default_state_known_mask_no_final_mass


class ChunkedEarthMarsDataset:
	def __init__(self, dataset_dir: Path):
		self.dataset_dir = Path(dataset_dir)
		self.metadata = np.load(self.dataset_dir / "metadata.npz")
		self.chunk_paths = sorted(self.dataset_dir.glob("chunk_*.npz"))
		if not self.chunk_paths:
			raise ValueError(f"No chunk files found in {self.dataset_dir}")

		self.num_samples = int(self.metadata["num_samples"].item())
		self.num_points = int(self.metadata["num_points"].item())
		self.chunk_size = int(self.metadata["chunk_size"].item())
		self.time_grid = self.metadata["time_grid"].astype(np.float32)
		self.t_final = float(self.metadata["t_final"].item())
		self.state_box_low = float(self.metadata["state_box_low"].item())
		self.state_box_high = float(self.metadata["state_box_high"].item())
		self.min_radius = float(self.metadata["min_radius"].item())
		self.chunk_sizes = self._build_chunk_sizes()
		self.chunk_probs = np.asarray(self.chunk_sizes, dtype=np.float64)
		self.chunk_probs = self.chunk_probs / self.chunk_probs.sum()
		self._cache_index = None
		self._cache = None

	def _build_chunk_sizes(self):
		chunk_sizes = []
		remaining = self.num_samples
		for _ in self.chunk_paths:
			size = min(self.chunk_size, remaining)
			chunk_sizes.append(size)
			remaining -= size
		return chunk_sizes

	def _load_chunk(self, chunk_index: int):
		if self._cache_index != chunk_index:
			self._cache = np.load(self.chunk_paths[chunk_index])
			self._cache_index = chunk_index
		return self._cache

	def sample_batch(self, rng: np.random.Generator, batch_size: int):
		chunk_index = int(rng.choice(len(self.chunk_paths), p=self.chunk_probs))
		chunk = self._load_chunk(chunk_index)
		local_indices = rng.integers(0, self.chunk_sizes[chunk_index], size=batch_size)
		return {
			"states": chunk["states"][local_indices].astype(np.float32),
			"costates": chunk["costates"][local_indices].astype(np.float32),
			"initial_states": chunk["initial_states"][local_indices].astype(np.float32),
			"final_states": chunk["final_states"][local_indices].astype(np.float32),
			"sampled_costates": chunk["sampled_costates"][local_indices].astype(np.float32),
		}


def make_train_step(policy: IndiffCtrlPolicy, optimizer: optax.GradientTransformation):
	def loss_fn(params, prepared_batch, dropout_key):
		return policy.compute_loss_from_prepared(params, prepared_batch, dropout_key)

	@jax.jit
	def train_step(params, opt_state, prepared_batch, dropout_key):
		loss, grads = jax.value_and_grad(loss_fn)(params, prepared_batch, dropout_key)
		updates, opt_state = optimizer.update(grads, opt_state, params)
		params = optax.apply_updates(params, updates)
		return params, opt_state, loss

	return train_step


def make_segment_arc_integrator(policy: IndiffCtrlPolicy, arc_points: int, rtol: float, atol: float):
	segment_term = ODETerm(build_rhs(policy.jdy))
	segment_solver = Dopri5()
	segment_controller = PIDController(rtol=rtol, atol=atol)
	segment_ts_base = jnp.linspace(0.0, 1.0, arc_points, dtype=jnp.float32)

	def single(y0, costate, dt):
		augmented_y0 = jnp.concatenate([y0, costate], axis=0)
		ts = segment_ts_base * dt
		sol = diffeqsolve(
			segment_term, segment_solver,
			t0=0.0, t1=dt, dt0=None,
			y0=augmented_y0, args=None,
			stepsize_controller=segment_controller,
			saveat=SaveAt(ts=ts), throw=False,
			max_steps=512,
		)
		success = jnp.asarray(sol.result == RESULTS.successful, dtype=bool)
		return jnp.asarray(sol.ys, dtype=jnp.float32)[:, :STATE_DIM], success

	return jax.jit(jax.vmap(single))


def generate_truth_trajectory(
	policy: IndiffCtrlPolicy,
	dataset: ChunkedEarthMarsDataset,
	rng: np.random.Generator,
	rtol: float,
	atol: float,
):
	initial_state = sample_initial_state(
		rng=rng,
		low=dataset.state_box_low,
		high=dataset.state_box_high,
		min_radius=dataset.min_radius,
		mu=policy.norm["mu"],
		m0_normalized=1.0,
	)
	sampled_costate = sample_costate(rng)
	ts, ys, costates = integrate_trajectory(
		jdy=policy.jdy,
		y0=initial_state,
		costate=sampled_costate,
		t_final=dataset.t_final,
		n_points=dataset.num_points,
		rtol=rtol,
		atol=atol,
	)
	states = np.asarray(ys, dtype=np.float32)[:, :STATE_DIM]
	costates = np.asarray(costates, dtype=np.float32)
	return {
		"time_grid": np.asarray(ts, dtype=np.float32),
		"states": states,
		"costates": costates,
		"initial_state": np.asarray(initial_state, dtype=np.float32)[:STATE_DIM],
		"final_state": states[-1],
		"sampled_costate": np.asarray(sampled_costate, dtype=np.float32),
	}


def compute_segment_arcs(
	policy: IndiffCtrlPolicy,
	segment_arc_integrator,
	states: np.ndarray,
	costates: np.ndarray,
):
	segment_count = states.shape[0] - 1
	y0 = states[:-1].astype(np.float32)
	dt = np.asarray(policy.segment_dt[:segment_count], dtype=np.float32)
	arcs, success = segment_arc_integrator(
		jnp.asarray(y0, dtype=jnp.float32),
		jnp.asarray(costates[:-1], dtype=jnp.float32),
		jnp.asarray(dt, dtype=jnp.float32),
	)
	return np.asarray(arcs, dtype=np.float32), np.asarray(success, dtype=bool)


def maybe_render_eval_plot(
	output_path: Path,
	truth: dict,
	predicted_states: np.ndarray,
	pointwise_error: np.ndarray,
	segment_arcs: np.ndarray,
	arc_success: np.ndarray,
):
	try:
		matplotlib = importlib.import_module("matplotlib")
		matplotlib.use("Agg")
		plt = importlib.import_module("matplotlib.pyplot")
	except Exception as exc:
		print(f"skipping eval plot {output_path.name}: {exc}")
		return

	truth_states = truth["states"]
	xyz_abs_max = float(max(np.max(np.abs(truth_states[:, :3])), np.max(np.abs(predicted_states[:, :3])), 1e-3))
	max_error = float(max(pointwise_error.max(), 1e-6))

	fig = plt.figure(figsize=(12, 5))
	ax_traj = fig.add_subplot(1, 2, 1, projection="3d")
	ax_err = fig.add_subplot(1, 2, 2)

	ax_traj.plot(
		truth_states[:, 0], truth_states[:, 1], truth_states[:, 2],
		linestyle="--", linewidth=2.0, color="0.45", label="ground truth",
	)
	ax_traj.plot(
		predicted_states[:, 0], predicted_states[:, 1], predicted_states[:, 2],
		marker="o", markersize=3, linewidth=2.0, color="tab:blue", label="predicted",
	)

	for seg_idx, arc in enumerate(segment_arcs):
		if arc_success[seg_idx]:
			ax_traj.plot(arc[:, 0], arc[:, 1], arc[:, 2], color="tab:orange", alpha=0.45, linewidth=1.0)
		else:
			segment = predicted_states[seg_idx:seg_idx + 2]
			ax_traj.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="tab:red", linestyle=":", linewidth=1.0)

	ax_traj.scatter(*truth["initial_state"][:3], color="green", s=30, label="start/end")
	ax_traj.scatter(*truth["final_state"][:3], color="red", s=30)
	ax_traj.set_xlim(-xyz_abs_max, xyz_abs_max)
	ax_traj.set_ylim(-xyz_abs_max, xyz_abs_max)
	ax_traj.set_zlim(-xyz_abs_max, xyz_abs_max)
	ax_traj.set_xlabel("x")
	ax_traj.set_ylabel("y")
	ax_traj.set_zlabel("z")
	ax_traj.set_title("Generated rollout evaluation")
	ax_traj.legend(loc="upper right")

	ax_err.plot(pointwise_error, color="tab:purple", linewidth=2.0)
	ax_err.set_ylim(0.0, max_error * 1.05)
	ax_err.set_xlabel("trajectory index")
	ax_err.set_ylabel("state L2 error")
	ax_err.set_title(f"mean error={pointwise_error.mean():.4e} | max error={pointwise_error.max():.4e}")
	ax_err.grid(True, alpha=0.3)

	fig.tight_layout()
	fig.savefig(output_path, dpi=160, bbox_inches="tight")
	plt.close(fig)


def evaluate_on_generated_paths(
	policy: IndiffCtrlPolicy,
	params,
	dataset: ChunkedEarthMarsDataset,
	np_rng: np.random.Generator,
	rng_key: jax.Array,
	output_dir: Path,
	step: int,
	num_cases: int,
	num_inference_steps: int,
	generation_rtol: float,
	generation_atol: float,
	segment_arc_integrator,
	save_plots: bool,
	state_known_mask=None,
	costate_known_mask=None,
):
	step_dir = output_dir / f"step_{step:06d}"
	step_dir.mkdir(parents=True, exist_ok=True)
	rows = []
	case_keys = jax.random.split(rng_key, num_cases)

	for case_index in range(num_cases):
		truth = generate_truth_trajectory(
			policy=policy,
			dataset=dataset,
			rng=np_rng,
			rtol=generation_rtol,
			atol=generation_atol,
		)
		prediction = policy.predict_trajectory(
			params=params,
			rng_key=case_keys[case_index],
			initial_states=truth["initial_state"][None, :],
			final_states=truth["final_state"][None, :],
			num_inference_steps=num_inference_steps,
			state_known_mask=state_known_mask,
			costate_known_mask=costate_known_mask,
		)
		predicted_states = np.asarray(prediction["states"], dtype=np.float32)[0]
		predicted_costates = np.asarray(prediction["costates"], dtype=np.float32)[0]
		pointwise_error = np.linalg.norm(predicted_states - truth["states"], axis=1)
		interior_pointwise_error = pointwise_error[1:-1] if pointwise_error.shape[0] > 2 else pointwise_error
		endpoint_error = float(np.linalg.norm(predicted_states[-1] - truth["final_state"]))
		start_error = float(np.linalg.norm(predicted_states[0] - truth["initial_state"]))
		integration_failed = prediction.get("integration_failed")
		integration_failed = np.asarray(integration_failed, dtype=bool)[0] if integration_failed is not None else np.zeros((dataset.num_points - 1,), dtype=bool)
		segment_arcs, arc_success = compute_segment_arcs(
			policy, segment_arc_integrator, predicted_states, predicted_costates,
		)
		segment_residual = np.linalg.norm(segment_arcs[:, -1, :] - predicted_states[1:], axis=1)
		segment_residual = np.where(arc_success, segment_residual, np.nan)
		mean_segment_residual = float(np.nanmean(segment_residual)) if np.any(arc_success) else float("nan")
		max_segment_residual = float(np.nanmax(segment_residual)) if np.any(arc_success) else float("nan")

		case_path = step_dir / f"case_{case_index:02d}.npz"
		np.savez_compressed(
			case_path,
			truth_states=truth["states"],
			truth_costates=truth["costates"],
			predicted_states=predicted_states,
			predicted_costates=predicted_costates,
			pointwise_error=pointwise_error,
			interior_pointwise_error=interior_pointwise_error,
			segment_arcs=segment_arcs,
			arc_success=arc_success,
			segment_residual=segment_residual,
			integration_failed=integration_failed,
		)

		if save_plots:
			maybe_render_eval_plot(
				output_path=step_dir / f"case_{case_index:02d}.png",
				truth=truth,
				predicted_states=predicted_states,
				pointwise_error=pointwise_error,
				segment_arcs=segment_arcs,
				arc_success=arc_success,
			)

		rows.append({
			"case_index": case_index,
			"mean_state_l2_error": float(pointwise_error.mean()),
			"max_state_l2_error": float(pointwise_error.max()),
			"mean_interior_state_l2_error": float(interior_pointwise_error.mean()),
			"max_interior_state_l2_error": float(interior_pointwise_error.max()),
			"start_state_l2_error": start_error,
			"end_state_l2_error": endpoint_error,
			"mean_segment_residual": mean_segment_residual,
			"max_segment_residual": max_segment_residual,
			"integration_failure_count": int(np.count_nonzero(integration_failed)),
			"arc_failure_count": int(np.count_nonzero(~arc_success)),
		})

	summary = {
		"step": step,
		"num_cases": num_cases,
		"mean_state_l2_error": float(np.mean([r["mean_state_l2_error"] for r in rows])),
		"max_state_l2_error": float(np.max([r["max_state_l2_error"] for r in rows])),
		"mean_interior_state_l2_error": float(np.mean([r["mean_interior_state_l2_error"] for r in rows])),
		"max_interior_state_l2_error": float(np.max([r["max_interior_state_l2_error"] for r in rows])),
		"mean_end_state_l2_error": float(np.mean([r["end_state_l2_error"] for r in rows])),
		"mean_segment_residual": float(np.nanmean([r["mean_segment_residual"] for r in rows])),
		"max_segment_residual": float(np.nanmax([r["max_segment_residual"] for r in rows])),
		"total_integration_failures": int(np.sum([r["integration_failure_count"] for r in rows])),
		"rows": rows,
	}

	with open(step_dir / "summary.json", "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)

	return summary


def parse_args():
	parser = argparse.ArgumentParser(description="Train IndiffCtrl diffusion policy on chunked Earth-Mars dataset.")
	parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).resolve().parent / "earth_mars_minfuel_posvel_constrained_32pts")
	parser.add_argument("--batch-size", type=int, default=256)
	parser.add_argument("--epochs", type=float, default=30.0)
	parser.add_argument("--learning-rate", type=float, default=1e-4)
	parser.add_argument("--final-learning-rate", type=float, default=1e-5)
	parser.add_argument("--lr-decay-epochs", type=float, default=24)
	parser.add_argument("--weight-decay", type=float, default=1e-6)
	parser.add_argument("--log-every", type=int, default=10)
	parser.add_argument("--checkpoint-every", type=int, default=1000)
	parser.add_argument("--eval-every", type=int, default=1000)
	parser.add_argument("--eval-num-cases", type=int, default=3)
	parser.add_argument("--eval-generation-rtol", type=float, default=1e-7)
	parser.add_argument("--eval-generation-atol", type=float, default=1e-9)
	parser.add_argument("--eval-segment-arc-points", type=int, default=16)
	parser.add_argument("--eval-save-plots", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--eval-output-dir", type=Path, default=None)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--num-train-timesteps", type=int, default=5000)
	parser.add_argument("--num-inference-steps", type=int, default=100)
	parser.add_argument("--prediction-type", type=str, default="epsilon", choices=["epsilon", "sample"])
	parser.add_argument("--beta-schedule", type=str, default="squaredcos_cap_v2")
	parser.add_argument("--clip-sample", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--clip-sample-range", type=float, default=5.0)
	parser.add_argument("--noise-scale", type=float, default=2.0)
	parser.add_argument("--embd-dim", type=int, default=512)
	parser.add_argument("--num-layers", type=int, default=12)
	parser.add_argument("--num-heads", type=int, default=4)
	parser.add_argument("--mlp-ratio", type=int, default=4)
	parser.add_argument("--p-drop-embd", type=float, default=0.1)
	parser.add_argument("--p-drop-attn", type=float, default=0.1)
	parser.add_argument("--segment-rtol", type=float, default=1e-7)
	parser.add_argument("--segment-atol", type=float, default=1e-9)
	parser.add_argument("--eps", type=float, default=1e-4)
	parser.add_argument(
		"--checkpoint-path", type=Path,
		default=Path(__file__).resolve().parent / "checkpoints" / "indiff_ctrl_latest.msgpack",
	)
	return parser.parse_args()


def main():
	args = parse_args()
	dataset = ChunkedEarthMarsDataset(args.dataset_dir)
	steps_per_epoch = math.ceil(dataset.num_samples / args.batch_size)
	total_steps = max(1, math.ceil(args.epochs * steps_per_epoch))
	lr_decay_epochs = args.epochs if args.lr_decay_epochs is None else args.lr_decay_epochs
	if lr_decay_epochs <= 0.0:
		raise ValueError("lr_decay_epochs must be positive")
	lr_decay_steps = max(1, math.ceil(lr_decay_epochs * steps_per_epoch))
	learning_rate_schedule = optax.linear_schedule(
		init_value=args.learning_rate,
		end_value=args.final_learning_rate,
		transition_steps=lr_decay_steps,
	)

	model_config = DiffusionTransformerConfig(
		seq_len=dataset.num_points,
		state_dim=STATE_DIM,
		costate_dim=COSTATE_DIM,
		embd_dim=args.embd_dim,
		num_layers=args.num_layers,
		num_heads=args.num_heads,
		mlp_ratio=args.mlp_ratio,
		p_drop_embd=args.p_drop_embd,
		p_drop_attn=args.p_drop_attn,
		use_integration_signals=True,
		zero_known_state_eps=True,
	)
	model = DiffusionTransformer(config=model_config)
	noise_scheduler = JaxDDPMScheduler(
		num_train_timesteps=args.num_train_timesteps,
		beta_schedule=args.beta_schedule,
		prediction_type=args.prediction_type,
		clip_sample=args.clip_sample,
		clip_sample_range=args.clip_sample_range,
		noise_scale=args.noise_scale,
	)
	policy = IndiffCtrlPolicy(
		model=model,
		noise_scheduler=noise_scheduler,
		time_grid=dataset.time_grid,
		segment_rtol=args.segment_rtol,
		segment_atol=args.segment_atol,
		num_inference_steps=args.num_inference_steps,
		eps=args.eps,
	)

	# Condition λ_m(t_f) = 0 — Pontryagin transversality condition
	# State mask: pin full initial state + only r_f,v_f at t_f (terminal mass is FREE)
	train_state_known_mask  = default_state_known_mask_no_final_mass(args.batch_size, dataset.num_points)
	eval_state_known_mask   = default_state_known_mask_no_final_mass(1, dataset.num_points)
	train_costate_known_mask = final_lambda_m_costate_mask(args.batch_size, dataset.num_points)
	eval_costate_known_mask  = final_lambda_m_costate_mask(1, dataset.num_points)

	key = jax.random.PRNGKey(args.seed)
	params = policy.init_params(key, batch_size=args.batch_size)
	optimizer = optax.adamw(learning_rate=learning_rate_schedule, weight_decay=args.weight_decay)
	opt_state = optimizer.init(params)

	# ── Resume from checkpoint if one exists ─────────────────────────────────
	start_step = 0
	if args.checkpoint_path.exists():
		print(f"Resuming from {args.checkpoint_path} …")
		target = {
			"params":    params,
			"opt_state": opt_state,
			"step":      np.int32(0),
			"time_grid": dataset.time_grid,
			"scheduler_config": {
				"num_train_timesteps": args.num_train_timesteps,
				"beta_schedule":       args.beta_schedule,
				"prediction_type":     args.prediction_type,
				"clip_sample":         args.clip_sample,
				"clip_sample_range":   args.clip_sample_range,
				"noise_scale":         args.noise_scale,
			},
		}
		loaded    = serialization.from_bytes(target, args.checkpoint_path.read_bytes())
		params    = loaded["params"]
		opt_state = loaded["opt_state"]
		start_step = int(loaded["step"])
		print(f"  Resumed at step {start_step}/{total_steps} "
			  f"(epoch {start_step / steps_per_epoch:.2f}/{args.epochs:.2f})")
	else:
		print("No checkpoint found — starting from scratch.")
	train_step = make_train_step(policy, optimizer)
	segment_arc_integrator = make_segment_arc_integrator(
		policy, args.eval_segment_arc_points, args.segment_rtol, args.segment_atol,
	)

	np_rng = np.random.default_rng(args.seed)
	eval_np_rng = np.random.default_rng(args.seed + 1)
	eval_key = jax.random.PRNGKey(args.seed + 1)
	args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
	eval_output_dir = args.eval_output_dir or (args.checkpoint_path.parent / "generated_eval")
	eval_output_dir.mkdir(parents=True, exist_ok=True)
	start_time = time.time()

	for step in range(start_step + 1, total_steps + 1):
		batch = dataset.sample_batch(np_rng, args.batch_size)
		key, noise_key, dropout_key = jax.random.split(key, 3)
		prepared_batch = policy.prepare_training_batch(batch, noise_key, state_known_mask=train_state_known_mask, costate_known_mask=train_costate_known_mask)
		params, opt_state, loss = train_step(params, opt_state, prepared_batch, dropout_key)

		if step % args.log_every == 0 or step == 1:
			elapsed = time.time() - start_time
			current_epoch = step / steps_per_epoch
			current_lr = float(learning_rate_schedule(step - 1))
			print(
				f"step={step}/{total_steps} epoch={current_epoch:.4f}/{args.epochs:.4f} "
				f"loss={float(loss):.6e} lr={current_lr:.6e} elapsed={elapsed:.1f}s"
			)

		if args.eval_every > 0 and (step % args.eval_every == 0 or step == total_steps):
			eval_start = time.time()
			eval_key, rollout_key = jax.random.split(eval_key)
			summary = evaluate_on_generated_paths(
				policy=policy,
				params=params,
				dataset=dataset,
				np_rng=eval_np_rng,
				rng_key=rollout_key,
				output_dir=eval_output_dir,
				step=step,
				num_cases=args.eval_num_cases,
				num_inference_steps=args.num_inference_steps,
				generation_rtol=args.eval_generation_rtol,
				generation_atol=args.eval_generation_atol,
				segment_arc_integrator=segment_arc_integrator,
				save_plots=args.eval_save_plots,
				state_known_mask=eval_state_known_mask,
				costate_known_mask=eval_costate_known_mask,
			)
			eval_elapsed = time.time() - eval_start
			print(
				"[eval] "
				f"step={step} "
				f"cases={summary['num_cases']} "
				f"mean_interior_state_l2={summary['mean_interior_state_l2_error']:.6e} "
				f"max_interior_state_l2={summary['max_interior_state_l2_error']:.6e} "
				f"mean_segment_residual={summary['mean_segment_residual']:.6e} "
				f"integration_failures={summary['total_integration_failures']} "
				f"elapsed={eval_elapsed:.1f}s"
			)

		if step % args.checkpoint_every == 0 or step == total_steps:
			payload = {
				"params": params,
				"opt_state": opt_state,
				"step": np.int32(step),
				"time_grid": dataset.time_grid,
				"scheduler_config": {
					"num_train_timesteps": args.num_train_timesteps,
					"beta_schedule": args.beta_schedule,
					"prediction_type": args.prediction_type,
					"clip_sample": args.clip_sample,
					"clip_sample_range": args.clip_sample_range,
					"noise_scale": args.noise_scale,
				},
			}
			with open(args.checkpoint_path, "wb") as f:
				f.write(serialization.to_bytes(payload))
			print(f"saved checkpoint to {args.checkpoint_path}")


if __name__ == "__main__":
	main()
