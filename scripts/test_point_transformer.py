#!/usr/bin/env python
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

from models.PointTransformerV3TF import build_ptv3_jet_classifier


def profile_gpu_memory_during_inference(model: tf.keras.Model, input_data: np.ndarray) -> tuple[float, float]:
	logging.info("Starting GPU memory profiling")
	try:
		tf.config.experimental.reset_memory_stats('GPU:0')
	except Exception:
		logging.warning("GPU memory stats not available; skipping.")
		return 0.0, 0.0
	@tf.function
	def infer(x):
		return model(x, training=False)
	_ = infer(input_data[:1]); _ = infer(input_data)
	mem = tf.config.experimental.get_memory_info('GPU:0')
	curr = mem['current']/(1024**2)
	peak = mem['peak']/(1024**2)
	logging.info("GPU memory profiling done: current=%.1f MB, peak=%.1f MB", curr, peak)
	return curr, peak


def get_flops(model):
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
	logging.info("Starting sorting by '%s'", sort_by)
	if sort_by == "pt":
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
	logging.info("Loading test data for '%s'", dataset)
	if dataset == "hls4ml":
		x_test = np.load(os.path.join(data_dir, f"x_val_robust_{num_particles}const_ptetaphi.npy"))
		y_test = np.load(os.path.join(data_dir, f"y_val_robust_{num_particles}const_ptetaphi.npy"))
	elif dataset == "top":
		top_dir = os.path.join(data_dir, "TopTagging", str(num_particles), "test")
		x_test = np.load(os.path.join(top_dir, "features.npy"))
		y_test = np.load(os.path.join(top_dir, "labels.npy"))
	elif dataset == "jetclass":
		x_test = np.load(os.path.join(data_dir, "JetClass/kinematics/test/features.npy"))
		y_test = np.load(os.path.join(data_dir, "JetClass/kinematics/test/labels.npy"))
		x_test = x_test.transpose(0, 2, 1)
	else:  # QG
		x_test = np.load(os.path.join(data_dir, "QuarkGluon/test/features.npy"))
		y_test = np.load(os.path.join(data_dir, "QuarkGluon/test/labels.npy"))
	logging.info("Loaded test arrays: x=%s, y=%s", x_test.shape, y_test.shape)
	return x_test, y_test


def select_preset(model_size):
	presets = {
			"small":  dict(enc_dims=[16], enc_layers=[1], enc_heads=[4], enc_strides=[2], enc_patch_sizes=[25], cpe_k=8, use_rpe=False),
    		"medium": dict(enc_dims=[12, 24, 32], enc_layers=[1, 1, 1], enc_heads=[4, 4, 4], enc_strides=[2, 2], enc_patch_sizes=[25, 25, 25], cpe_k=8, use_rpe=False),
    		"large":  dict(enc_dims=[16, 24, 32], enc_layers=[1, 1, 1], enc_heads=[4, 4, 4], enc_strides=[2, 2], enc_patch_sizes=[25, 25, 25], cpe_k=8, use_rpe=False),
    	}
	return presets[model_size]


def main():
	parser = argparse.ArgumentParser(description="Test PointTransformerV3TF model")
	parser.add_argument("--dataset", choices=["hls4ml","top","jetclass","QG"], required=True)
	parser.add_argument("--data_dir", required=True)
	parser.add_argument("--save_dir", required=True)
	parser.add_argument("--sort_by", choices=["pt","eta","phi","delta_R","kt"], default="pt")
	parser.add_argument("--batch_size", type=int, default=4096)
	parser.add_argument("--model_size", choices=["small", "medium", "large"], default="small")
	parser.add_argument("--disable_pool", action="store_true", help="Disable GeometricPooling between stages")
	parser.add_argument("--use_rpe", action="store_true", help="Enable RPE regardless of preset")
	parser.add_argument("--grid_size", type=float, default=0.05, help="GeometricCPE grid size (coarser -> smaller grid)")
	parser.add_argument("--weights", help="Path to weights .h5 file (defaults to save_dir/best.weights.h5)")
	args = parser.parse_args()

	# Logging
	os.makedirs(args.save_dir, exist_ok=True)
	log_path = os.path.join(args.save_dir, "test_ptv3.log")
	logging.basicConfig(filename=log_path, filemode="w", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
	cwd = os.getcwd()
	print(f"Running in directory: {cwd}")
	logging.info("Running in directory: %s", cwd)

	# Dataset defaults for num_particles and output_dim
	if args.dataset == "jetclass":
		num_particles = 150; output_dim = 10
	elif args.dataset == "top":
		num_particles = 200; output_dim = 1
	elif args.dataset == "QG":
		num_particles = 150; output_dim = 1
	else:
		num_particles = 150; output_dim = 5

	# Load data
	x_test, y_test = load_test_data(args.dataset, args.data_dir, num_particles)
	x_test = apply_sorting(x_test, args.sort_by)

	# Build model from preset
	cfg = select_preset(args.model_size)
	enc_dims = cfg["enc_dims"]
	enc_layers = cfg["enc_layers"]
	enc_heads = cfg["enc_heads"]
	enc_strides = cfg["enc_strides"]
	enc_patch_sizes = cfg["enc_patch_sizes"]
	cpe_k = cfg["cpe_k"]
	use_rpe = args.use_rpe or cfg["use_rpe"]

	model = build_ptv3_jet_classifier(
		num_particles=num_particles,
		output_dim=output_dim,
		enc_dims=enc_dims,
		enc_layers=enc_layers,
		enc_heads=enc_heads,
		enc_patch_sizes=enc_patch_sizes,
		enc_strides=enc_strides,
		cpe_k=cpe_k,
		grid_size=args.grid_size,
		use_rpe=use_rpe,
		use_pool=(not args.disable_pool),
		dropout=0.0,
		aggregation="max",
	)
	model.summary(print_fn=lambda s: logging.info(s))
	logging.info("Preset: %s", args.model_size)
	logging.info("Hyperparams: dims=%s layers=%s heads=%s strides=%s patch=%s", enc_dims, enc_layers, enc_heads, enc_strides, enc_patch_sizes)

	# Load weights
	weights_path = args.weights or os.path.join(args.save_dir, "best.weights.h5")
	logging.info("Loading weights from %s", weights_path)
	model.load_weights(weights_path)
	logging.info("Weights loaded.")

	# FLOPs and timing
	flops = get_flops(model)
	logging.info("FLOPs per inference: %d", flops)
	logging.info("MACs per inference: %d", flops // 2)

	logging.info("Warming up and timing inference (20 runs)")
	_ = model.predict(x_test[:args.batch_size], batch_size=args.batch_size)
	times = []
	for _ in range(20):
		t0 = time.perf_counter()
		_ = model.predict(x_test[:args.batch_size], batch_size=args.batch_size)
		times.append(time.perf_counter() - t0)
	avg_ns = np.mean(times) / args.batch_size * 1e9
	logging.info("Avg inference time/event: %.2f ns", avg_ns)

	curr_mb, peak_mb = profile_gpu_memory_during_inference(model, x_test[:args.batch_size])
	logging.info("GPU memory current: %.1f MB, peak: %.1f MB", curr_mb, peak_mb)

	# Inference
	logging.info("Running full prediction")
	preds = model.predict(x_test, batch_size=args.batch_size)
	logging.info("Predictions shape: %s", preds.shape)

	# Metrics
	logging.info("Computing metrics for dataset '%s'", args.dataset)
	if args.dataset in ("top", "QG"):
		acc = accuracy_score(y_test, (preds.ravel() > 0.5).astype(int))
		auc_m = roc_auc_score(y_test, preds.ravel())
		logging.info("Test Accuracy: %.4f, ROC AUC: %.4f", acc, auc_m)
	else:
		acc = accuracy_score(np.argmax(y_test, 1), np.argmax(preds, 1))
		auc_m = roc_auc_score(y_test, preds, average="macro", multi_class="ovo")
		logging.info("Test Accuracy: %.4f, ROC AUC: %.4f", acc, auc_m)

	# ROC curves (optional)
	if args.dataset == "hls4ml":
		labels = ["q", "g", "W", "Z", "t"]
	elif args.dataset == "top":
		labels = ["qcd", "top"]
	elif args.dataset == "QG":
		labels = ["Gluon", "Quark"]
	else:
		labels = [f"label_{i}" for i in range(preds.shape[1])]

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
	plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC curves")
	plt.legend(loc="lower right"); plt.tight_layout()
	plt.savefig(os.path.join(args.save_dir, "roc_curves_ptv3.png"))
	plt.close()
	for lab, val in one_over_fpr.items():
		logging.info("1/FPR@0.8 for %s: %.3f", lab, val)
	if one_over_fpr:
		logging.info("Avg 1/FPR@0.8: %.3f", np.nanmean(list(one_over_fpr.values())))


if __name__ == "__main__":
	main()


