#!/usr/bin/env python
"""
Test a JEDI-Linear model on jet datasets (hls4ml, top, QG, or jetclass),
profiling performance and generating metrics.
"""
import os
import sys

# ─── make project root importable ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)
# ─────────────────────────────────────────────────────────────────────────────

import time
import argparse
import logging
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

# import model builders
from models.JEDI_Linear import (
	build_jedi_linear_classifier,
	build_jedi_linear_small,
	build_jedi_linear_medium,
	build_jedi_linear_large,
	build_jedi_linear_matched,
	build_jedi_linear_matched_16p16f,
	build_jedi_linear_matched_32p16f,
	build_jedi_linear_matched_64p16f,
	build_jedi_linear_matched_16p3f,
	build_jedi_linear_matched_64p3f
)


def profile_gpu_memory_during_inference(model: tf.keras.Model, input_data: np.ndarray) -> tuple[float, float]:
	"""Profile GPU memory usage during inference."""
	logging.info("Starting GPU memory profiling")
	try:
		tf.config.experimental.reset_memory_stats('GPU:0')
	except Exception:
		logging.warning("GPU memory stats not available; skipping.")
		return 0.0, 0.0

	@tf.function
	def infer(x):
		return model(x, training=False)

	_ = infer(input_data[:1])
	_ = infer(input_data)
	mem = tf.config.experimental.get_memory_info('GPU:0')
	curr = mem['current'] / (1024**2)
	peak = mem['peak'] / (1024**2)
	logging.info("GPU memory profiling done: current=%.1f MB, peak=%.1f MB", curr, peak)
	return curr, peak


def get_flops(model):
	"""Calculate FLOPs for the model."""
	logging.info("Starting FLOPs calculation")
	input_shape = model.input_shape
	concrete_shape = tuple([1] + list(input_shape[1:]))
	from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

	inp = tf.TensorSpec(concrete_shape, tf.float32)
	func = tf.function(model).get_concrete_function(inp)
	frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(func)
	new_graph = tf.Graph()
	with new_graph.as_default():
		tf.compat.v1.import_graph_def(graph_def, name='')
		run_meta = tf.compat.v1.RunMetadata()
		opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
		prof = tf.compat.v1.profiler.profile(graph=new_graph, run_meta=run_meta, cmd='op', options=opts)
		flops = prof.total_float_ops
	logging.info("FLOPs calculation done: %d FLOPs", flops)
	return flops


def apply_sorting(x, sort_by):
	"""Apply sorting to input data (note: JEDI-Linear is permutation-invariant)."""
	logging.info("Starting sorting by '%s'", sort_by)
	if sort_by == "none":
		logging.info("No sorting applied (JEDI-Linear is permutation-invariant)")
		return x
	elif sort_by == "pt":
		key = x[:, :, 0]
	elif sort_by == "eta":
		key = x[:, :, 1]
	elif sort_by == "phi":
		key = x[:, :, 2]
	elif sort_by == "delta_R":
		key = np.sqrt(x[:, :, 1] ** 2 + x[:, :, 2] ** 2)
	elif sort_by == "kt":
		key = x[:, :, 0] * np.sqrt(x[:, :, 1] ** 2 + x[:, :, 2] ** 2)
	else:
		return x
	idx = np.argsort(key, axis=1)[:, ::-1]
	sorted_x = np.take_along_axis(x, idx[:, :, None], axis=1)
	logging.info("Sorting done; data shape: %s", sorted_x.shape)
	return sorted_x


def load_test_data(dataset, data_dir, num_particles):
	"""Load test data for the specified dataset."""
	logging.info("Loading test data for '%s'", dataset)
	if dataset == "hls4ml":
		x_test = np.load(os.path.join(data_dir, f"x_val_robust_{num_particles}const_ptetaphi.npy"))
		y_test = np.load(os.path.join(data_dir, f"y_val_robust_{num_particles}const_ptetaphi.npy"))
	elif dataset == "top":
		x_test = np.load(os.path.join(data_dir, "test/features.npy"))
		y_test = np.load(os.path.join(data_dir, "test/labels.npy"))
	elif dataset == "jetclass":
		x_test = np.load(os.path.join(data_dir, "test/features.npy"))
		y_test = np.load(os.path.join(data_dir, "test/labels.npy"))
		x_test = x_test.transpose(0, 2, 1)
	else:  # QG
		x_test = np.load(os.path.join(data_dir, "test/features.npy"))
		y_test = np.load(os.path.join(data_dir, "test/labels.npy"))
	logging.info("Loaded test arrays: x=%s, y=%s", x_test.shape, y_test.shape)
	return x_test, y_test


def main():
	parser = argparse.ArgumentParser(
		description="Test JEDI-Linear model",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Available model presets:
  small             : Small model (~1-2K params) for quick testing
  medium            : Medium model (~50K params) with balanced performance
  large             : Large model (~100K+ params) for maximum accuracy

  matched           : Generic paper-matched architecture
  matched_16p16f    : 16 particles, 16 features (78.3% acc, 72ns FPGA)
  matched_32p16f    : 32 particles, 16 features (81.4% acc, 79ns FPGA)
  matched_64p16f    : 64 particles, 16 features (82.4% acc, 93ns FPGA)
  matched_16p3f     : 16 particles, 3 features (73.6% acc, 75ns FPGA)
  matched_64p3f     : 64 particles, 3 features (81.8% acc, 78ns FPGA)

Examples:
  python scripts/test_jedi_linear.py --dataset hls4ml --data_dir data/hls4ml --save_dir results --preset medium
  python scripts/test_jedi_linear.py --dataset jetclass --data_dir data/jetclass --save_dir results --preset matched_64p16f
		"""
	)

	# Required arguments
	parser.add_argument("--dataset", choices=["hls4ml", "top", "jetclass", "QG"], required=True)
	parser.add_argument("--data_dir", required=True)
	parser.add_argument("--save_dir", required=True)

	# Optional arguments
	parser.add_argument("--sort_by", choices=["pt", "eta", "phi", "delta_R", "kt", "none"], default="none",
	                   help="Particle sorting (note: JEDI-Linear is permutation-invariant)")
	parser.add_argument("--batch_size", type=int, default=4096)
	parser.add_argument("--preset", choices=["small", "medium", "large", "matched",
	                                          "matched_16p16f", "matched_32p16f", "matched_64p16f",
	                                          "matched_16p3f", "matched_64p3f"], default="medium")
	parser.add_argument("--weights", help="Path to weights .h5 file (defaults to save_dir/best.weights.h5)")
	parser.add_argument("--aggregation", choices=["mean", "max"], default="mean",
	                   help="Aggregation method for final pooling")

	args = parser.parse_args()

	# Logging setup
	os.makedirs(args.save_dir, exist_ok=True)
	log_path = os.path.join(args.save_dir, "test_jedi_linear.log")
	logging.basicConfig(filename=log_path, filemode="w", level=logging.INFO,
	                   format="%(asctime)s %(levelname)s %(message)s")
	cwd = os.getcwd()
	print(f"Running in directory: {cwd}")
	logging.info("Running in directory: %s", cwd)
	logging.info("Arguments: %s", args)

	# Dataset defaults for num_particles and output_dim
	if args.dataset == "jetclass":
		num_particles = 150
		output_dim = 10
		feature_dim = 3
	elif args.dataset == "top":
		num_particles = 200
		output_dim = 1
		feature_dim = 3
	elif args.dataset == "QG":
		num_particles = 150
		output_dim = 1
		feature_dim = 3
	else:  # hls4ml
		num_particles = 150
		output_dim = 5
		feature_dim = 3

	# Load data
	x_test, y_test = load_test_data(args.dataset, args.data_dir, num_particles)
	x_test = apply_sorting(x_test, args.sort_by)

	# Build model based on preset
	logging.info("Building JEDI-Linear model with preset: %s", args.preset)
	if args.preset == "small":
		model = build_jedi_linear_small(
			num_particles=num_particles,
			feature_dim=feature_dim,
			output_dim=output_dim
		)
	elif args.preset == "medium":
		model = build_jedi_linear_medium(
			num_particles=num_particles,
			feature_dim=feature_dim,
			output_dim=output_dim
		)
	elif args.preset == "large":
		model = build_jedi_linear_large(
			num_particles=num_particles,
			feature_dim=feature_dim,
			output_dim=output_dim
		)
	elif args.preset == "matched":
		model = build_jedi_linear_matched(
			num_particles=num_particles,
			feature_dim=feature_dim,
			output_dim=output_dim
		)
	elif args.preset == "matched_16p16f":
		model = build_jedi_linear_matched_16p16f(output_dim=output_dim)
		num_particles = 16
		feature_dim = 16
	elif args.preset == "matched_32p16f":
		model = build_jedi_linear_matched_32p16f(output_dim=output_dim)
		num_particles = 32
		feature_dim = 16
	elif args.preset == "matched_64p16f":
		model = build_jedi_linear_matched_64p16f(output_dim=output_dim)
		num_particles = 64
		feature_dim = 16
	elif args.preset == "matched_16p3f":
		model = build_jedi_linear_matched_16p3f(output_dim=output_dim)
		num_particles = 16
		feature_dim = 3
	elif args.preset == "matched_64p3f":
		model = build_jedi_linear_matched_64p3f(output_dim=output_dim)
		num_particles = 64
		feature_dim = 3

	model.summary(print_fn=lambda s: logging.info(s))
	logging.info("Preset: %s", args.preset)
	logging.info("Model parameters: num_particles=%d, feature_dim=%d, output_dim=%d",
	            num_particles, feature_dim, output_dim)

	# Load weights
	default_weights = os.path.join(args.save_dir, "model.weights.h5")
	ckpt_weights = os.path.join(args.save_dir, "best.weights.h5")
	weights_path = args.weights or (default_weights if os.path.isfile(default_weights) else ckpt_weights)

	logging.info("Loading weights from %s", weights_path)
	try:
		model.load_weights(weights_path)
		logging.info("Weights loaded successfully via load_weights.")
	except Exception as e:
		logging.warning("load_weights failed: %s; retrying with skip_mismatch=True", e)
		try:
			model.load_weights(weights_path, skip_mismatch=True)
			logging.info("Weights loaded with skip_mismatch=True (some variables may be skipped).")
		except Exception as e2:
			logging.error("Failed to load weights from %s: %s", weights_path, e2)
			raise

	# FLOPs calculation
	flops = get_flops(model)
	logging.info("FLOPs per inference: %d", flops)
	logging.info("MACs per inference: %d", flops // 2)

	# Timing inference
	logging.info("Warming up and timing inference (20 runs)")
	_ = model.predict(x_test[:args.batch_size], batch_size=args.batch_size)
	times = []
	for _ in range(20):
		t0 = time.perf_counter()
		_ = model.predict(x_test[:args.batch_size], batch_size=args.batch_size)
		times.append(time.perf_counter() - t0)
	avg_ns = np.mean(times) / args.batch_size * 1e9
	logging.info("Avg inference time/event: %.2f ns", avg_ns)

	# GPU memory profiling
	curr_mb, peak_mb = profile_gpu_memory_during_inference(model, x_test[:args.batch_size])
	logging.info("GPU memory current: %.1f MB, peak: %.1f MB", curr_mb, peak_mb)

	# Full inference
	logging.info("Running full prediction on test set")
	preds = model.predict(x_test, batch_size=args.batch_size)
	logging.info("Predictions shape: %s", preds.shape)

	# Compute metrics
	logging.info("Computing metrics for dataset '%s'", args.dataset)
	if args.dataset in ("top", "QG"):
		acc = accuracy_score(y_test, (preds.ravel() > 0.5).astype(int))
		auc_m = roc_auc_score(y_test, preds.ravel())
		logging.info("Test Accuracy: %.4f, ROC AUC: %.4f", acc, auc_m)
	else:
		acc = accuracy_score(np.argmax(y_test, 1), np.argmax(preds, 1))
		auc_m = roc_auc_score(y_test, preds, average="macro", multi_class="ovo")
		logging.info("Test Accuracy: %.4f, ROC AUC: %.4f", acc, auc_m)

	# ROC curves
	if args.dataset == "hls4ml":
		labels = ["q", "g", "W", "Z", "t"]
	elif args.dataset == "top":
		labels = ["qcd", "top"]
	elif args.dataset == "QG":
		labels = ["Gluon", "Quark"]
	else:  # jetclass
		labels = [
			"label_QCD", "label_Hbb", "label_Hcc", "label_Hgg", "label_H4q",
			"label_Hqql", "label_Zqq", "label_Wqq", "label_Tbqq", "label_Tbl"
		]

	plt.figure(figsize=(6, 6))
	one_over_fpr = {}
	for i, lab in enumerate(labels):
		if args.dataset in ("top", "QG"):
			fpr, tpr, _ = roc_curve(y_test, preds.ravel())
		else:
			fpr, tpr, _ = roc_curve(y_test[:, i], preds[:, i])
		roc_val = auc(fpr, tpr)
		plt.plot(fpr, tpr, label=f"{lab} (AUC={roc_val:.2f})")
		if np.max(tpr) >= 0.8:
			fpr_t = np.interp(0.8, tpr, fpr)
			one_over_fpr[lab] = 1.0 / fpr_t if fpr_t > 0 else np.nan
			plt.plot(fpr_t, 0.8, "o")

	plt.plot([0, 1], [0, 1], "k--")
	plt.xlabel("FPR")
	plt.ylabel("TPR")
	plt.title("ROC curves - JEDI-Linear")
	plt.legend(loc="lower right")
	plt.tight_layout()
	plt.savefig(os.path.join(args.save_dir, "roc_curves_jedi_linear.png"))
	plt.close()

	for lab, val in one_over_fpr.items():
		logging.info("1/FPR@0.8 for %s: %.3f", lab, val)
	if one_over_fpr:
		logging.info("Avg 1/FPR@0.8: %.3f", np.nanmean(list(one_over_fpr.values())))

	# Background rejection (for multi-class datasets)
	if args.dataset not in ("top", "QG"):
		logging.info("Computing background rejection metrics")
		rej_vals = []
		for i, lab in enumerate(labels[1:], start=1):
			mask_bg = (
				((y_test[:, 0] == 1) | (y_test[:, 1] == 1) | (y_test[:, i] == 1))
				if args.dataset != "jetclass"
				else np.ones_like(y_test[:, 0], dtype=bool)
			)
			if args.dataset == "jetclass":
				bin_y = (y_test[mask_bg, i] == i).astype(int)
				bin_s = preds[mask_bg, i]
			else:
				bin_y = (y_test[mask_bg, i] == 1).astype(int)
				bin_s = preds[mask_bg, i]

			fpr, tpr, _ = roc_curve(bin_y, bin_s)
			idx = np.argmin(np.abs(tpr - 0.8))
			rej = 1.0 / fpr[idx] if fpr[idx] > 0 else np.inf
			logging.info("Bg rejection@0.8 %s: %.3f", lab, rej)
			rej_vals.append(rej)
		logging.info("Avg bg rejection@0.8: %.3f", np.nanmean(rej_vals))

	logging.info("Testing complete!")
	print(f"Test Accuracy: {acc:.4f}, ROC AUC: {auc_m:.4f}")
	print(f"Results saved to {args.save_dir}")
	print(f"See {log_path} for detailed logs")


if __name__ == "__main__":
	main()
