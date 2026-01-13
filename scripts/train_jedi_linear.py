#!/usr/bin/env python
"""
Train a JEDI-Linear model on jet datasets (hls4ml, top, QG, or jetclass),
profiling performance and generating ROC curves.
"""
import os
import sys

# ─── make the parent directory (project root) importable ─────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

import time
import argparse
import logging
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
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


# ---------------------------
# FLOPs computation
# ---------------------------
def get_flops(model, input_shape):
	from tensorflow.python.framework.convert_to_constants import (
		convert_variables_to_constants_v2_as_graph,
	)

	spec_x = tf.TensorSpec(input_shape, tf.float32)

	@tf.function
	def model_fn(x):
		return model(x)

	concrete = model_fn.get_concrete_function(spec_x)
	frozen, graph_def = convert_variables_to_constants_v2_as_graph(concrete)
	with tf.Graph().as_default() as g:
		tf.compat.v1.import_graph_def(graph_def, name="")
		run_meta = tf.compat.v1.RunMetadata()
		opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
		prof = tf.compat.v1.profiler.profile(
			graph=g, run_meta=run_meta, cmd="op", options=opts
		)
		return prof.total_float_ops


# ---------------------------
# GPU memory profiling
# ---------------------------
def profile_gpu_memory_during_inference(model, input_data):
	tf.config.experimental.reset_memory_stats("GPU:0")

	@tf.function
	def infer(x):
		return model(x, training=False)

	_ = infer(input_data[:1])
	_ = infer(input_data)

	mem = tf.config.experimental.get_memory_info("GPU:0")
	return mem["current"] / (1024**2), mem["peak"] / (1024**2)


# ---------------------------
# Sorting helper
# ---------------------------
def apply_sorting(x, sort_by):
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
	return np.take_along_axis(x, idx[:, :, None], axis=1)


# ---------------------------
# Testing / Profiling
# ---------------------------
def run_testing(model, dataset, data_dir, save_dir, sort_by, batch_size, num_particles):
	logging.info("Starting testing phase...")

	# load test set
	if dataset == "hls4ml":
		x_test = np.load(
			os.path.join(data_dir, f"x_val_robust_{num_particles}const_ptetaphi.npy")
		)
		y_test = np.load(
			os.path.join(data_dir, f"y_val_robust_{num_particles}const_ptetaphi.npy")
		)
	else:  # jetclass, top, or QG
		x_test = np.load(os.path.join(data_dir, "test/features.npy"))
		y_test = np.load(os.path.join(data_dir, "test/labels.npy"))
	logging.info(
		"Loaded TEST arrays for %s: %s, %s", dataset, x_test.shape, y_test.shape
	)

	if dataset == "jetclass":
		x_test = x_test.transpose(0, 2, 1)

	# sorting for test
	x_test = apply_sorting(x_test, sort_by)
	logging.info("Applied '%s' sorting to TEST set", sort_by)

	# flops & macs
	num_p, feat_d = x_test.shape[1], x_test.shape[2]
	flops = get_flops(model, (1, num_p, feat_d))
	macs = flops // 2
	logging.info("FLOPs per inference: %d", flops)
	logging.info("MACs per inference: %d", macs)

	# timing
	_ = model.predict(x_test[:batch_size], batch_size=batch_size)
	times = []
	for _ in range(20):
		t0 = time.perf_counter()
		_ = model.predict(x_test[:batch_size], batch_size=batch_size)
		times.append(time.perf_counter() - t0)
	avg_ns = np.mean(times) / batch_size * 1e9
	logging.info("Avg inference time/event: %.2f ns", avg_ns)

	# GPU memory
	curr, peak = profile_gpu_memory_during_inference(model, x_test[:batch_size])
	logging.info("GPU memory current: %.1f MB, peak: %.1f MB", curr, peak)

	# metrics
	preds = model.predict(x_test, batch_size=batch_size)
	if dataset == "top" or dataset == "QG":
		acc = accuracy_score(y_test, (preds.ravel() > 0.5).astype(int))
		auc_m = roc_auc_score(y_test, preds.ravel())
	else:
		acc = accuracy_score(np.argmax(y_test, 1), np.argmax(preds, 1))
		auc_m = roc_auc_score(y_test, preds, average="macro", multi_class="ovo")
	logging.info("Test Accuracy: %.4f, ROC AUC: %.4f", acc, auc_m)

	# ROC curves and labels
	if dataset == "hls4ml":
		labels = ["q", "g", "W", "Z", "t"]
	elif dataset == "top":
		labels = ["qcd", "top"]
	elif dataset == "QG":  # gluon is 0 and quark is 1.
		labels = ["Gluon", "Quark"]
	else:
		labels = [
			"label_QCD",
			"label_Hbb",
			"label_Hcc",
			"label_Hgg",
			"label_H4q",
			"label_Hqql",
			"label_Zqq",
			"label_Wqq",
			"label_Tbqq",
			"label_Tbl",
		]

	plt.figure(figsize=(6, 6))
	one_over_fpr = {}
	for i, lab in enumerate(labels):
		if dataset == "top" or dataset == "QG":
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
	plt.title("ROC curves")
	plt.legend(loc="lower right")
	plt.tight_layout()
	plt.savefig(os.path.join(save_dir, "roc_curves.png"))
	plt.close()

	for lab, val in one_over_fpr.items():
		logging.info("1/FPR@0.8 for %s: %.3f", lab, val)
	logging.info("Avg 1/FPR@0.8: %.3f", np.nanmean(list(one_over_fpr.values())))

	# background rejection combined
	if dataset != "top" and dataset != "QG":
		rej_vals = []
		for i, lab in enumerate(labels[1:], start=1):
			mask_bg = (
				((y_test[:, 0] == 1) | (y_test[:, 1] == 1) | (y_test[:, i] == 1))
				if dataset != "jetclass"
				else np.ones_like(y_test[:, 0], dtype=bool)
			)
			if dataset == "jetclass":
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


# ---------------------------
# Argument parsing and main
# ---------------------------
def parse_args():
	p = argparse.ArgumentParser(
		description="Train JEDI-Linear on jet data",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Available model presets:
  small             : Small model (~1-2K params) for quick testing
  medium            : Medium model (~50K params) with balanced performance
  large             : Large model (~100K+ params) for maximum accuracy

  matched           : Generic paper-matched architecture (uses num_particles/feature_dim args)
  matched_16p16f    : 16 particles, 16 features (78.3% acc, 72ns FPGA)
  matched_32p16f    : 32 particles, 16 features (81.4% acc, 79ns FPGA)
  matched_64p16f    : 64 particles, 16 features (82.4% acc, 93ns FPGA)
  matched_16p3f     : 16 particles, 3 features (73.6% acc, 75ns FPGA)
  matched_64p3f     : 64 particles, 3 features (81.8% acc, 78ns FPGA)

  custom            : Custom configuration (use hyperparameter flags)

Examples:
  python scripts/train_jedi_linear.py --data_dir data/hls4ml --save_dir results --preset medium
  python scripts/train_jedi_linear.py --data_dir data/hls4ml --save_dir results --preset matched_64p16f
  python scripts/train_jedi_linear.py --data_dir data/jetclass --save_dir results --dataset jetclass --preset large
		"""
	)

	# Data arguments
	p.add_argument("--data_dir", required=True, help="Directory containing training data")
	p.add_argument("--save_dir", required=True, help="Directory to save results")
	p.add_argument(
		"--dataset", choices=["hls4ml", "top", "QG", "jetclass"], default="hls4ml",
		help="Dataset to use for training"
	)
	p.add_argument(
		"--sort_by",
		choices=["pt", "eta", "phi", "delta_R", "kt", "none"],
		default="kt",
		help="Particle sorting strategy (JEDI-Linear is permutation-invariant, so this mainly affects comparison)"
	)
	p.add_argument("--batch_size", type=int, default=4096, help="Batch size for training")
	p.add_argument("--val_split", type=float, default=0.2, help="Validation split fraction")

	# Model selection
	p.add_argument(
		"--preset",
		choices=["small", "medium", "large", "matched",
		        "matched_16p16f", "matched_32p16f", "matched_64p16f",
		        "matched_16p3f", "matched_64p3f", "custom"],
		default="medium",
		help="Model size preset"
	)

	# Model hyperparameters (for custom preset)
	p.add_argument("--embedding_dim", type=int, default=16, help="Embedding dimension")
	p.add_argument("--num_blocks", type=int, default=2, help="Number of JEDI-Linear blocks")
	p.add_argument("--token_hidden", type=int, default=None, help="Hidden units for global interaction")
	p.add_argument("--channel_hidden", type=int, default=None, help="Hidden units for channel mixing")
	p.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
	p.add_argument("--aggregation", choices=["mean", "max"], default="mean", help="Global pooling method")
	p.add_argument("--head_hidden", type=int, nargs="+", default=[64, 32], help="Classification head hidden dimensions")

	return p.parse_args()


def main():
	args = parse_args()

	# pick num_particles & output_dim based on dataset
	if args.dataset == "jetclass":
		num_particles = 150
		output_dim = 10
		loss_fn = "categorical_crossentropy"
		feature_dim = 3  # pt, eta, phi
	elif args.dataset == "top":
		num_particles = 200
		output_dim = 1
		loss_fn = "binary_crossentropy"
		feature_dim = 3
	elif args.dataset == "QG":
		num_particles = 150
		output_dim = 1
		loss_fn = "binary_crossentropy"
		feature_dim = 3
	else:  # hls4ml
		num_particles = 150
		output_dim = 5
		loss_fn = "categorical_crossentropy"
		feature_dim = 3

	# prepare save directory with trial numbering
	save_dir = os.path.join(args.save_dir, str(num_particles), args.sort_by, args.preset)
	trial = 0
	while True:
		cand = os.path.join(save_dir, f"trial-{trial}")
		time.sleep(random.randint(1, 4))
		if not os.path.isdir(cand):
			save_dir = cand
			break
		trial += 1
	os.makedirs(save_dir, exist_ok=True)

	logging.basicConfig(
		filename=os.path.join(save_dir, "train.log"),
		filemode="w",
		level=logging.INFO,
		format="%(asctime)s %(levelname)s %(message)s",
	)
	logging.info("Args: %s", args)

	# load train/val data
	if args.dataset == "hls4ml":
		x = np.load(
			os.path.join(
				args.data_dir, f"x_train_robust_{num_particles}const_ptetaphi.npy"
			)
		)
		y = np.load(
			os.path.join(
				args.data_dir, f"y_train_robust_{num_particles}const_ptetaphi.npy"
			)
		)
		x_train, x_val, y_train, y_val = train_test_split(
			x, y, test_size=args.val_split, random_state=42
		)
	else:  # jetclass, top, or QG
		x_train = np.load(os.path.join(args.data_dir, "train/features.npy"))
		y_train = np.load(os.path.join(args.data_dir, "train/labels.npy"))
		x_val = np.load(os.path.join(args.data_dir, "val/features.npy"))
		y_val = np.load(os.path.join(args.data_dir, "val/labels.npy"))

	if args.dataset == "jetclass":
		x_train = x_train.transpose(0, 2, 1)
		x_val = x_val.transpose(0, 2, 1)

	logging.info(
		"Loaded train x=%s y=%s, val x=%s y=%s",
		x_train.shape,
		y_train.shape,
		x_val.shape,
		y_val.shape,
	)

	# apply sorting
	x_train = apply_sorting(x_train, args.sort_by)
	x_val = apply_sorting(x_val, args.sort_by)

	# build model based on preset
	if args.preset == "small":
		logging.info("Building SMALL JEDI-Linear model")
		model = build_jedi_linear_small(
			num_particles=num_particles,
			feature_dim=feature_dim,
			output_dim=output_dim
		)
	elif args.preset == "medium":
		logging.info("Building MEDIUM JEDI-Linear model")
		model = build_jedi_linear_medium(
			num_particles=num_particles,
			feature_dim=feature_dim,
			output_dim=output_dim
		)
	elif args.preset == "large":
		logging.info("Building LARGE JEDI-Linear model")
		model = build_jedi_linear_large(
			num_particles=num_particles,
			feature_dim=feature_dim,
			output_dim=output_dim
		)
	elif args.preset == "matched":
		logging.info("Building MATCHED JEDI-Linear model (generic paper architecture)")
		model = build_jedi_linear_matched(
			num_particles=num_particles,
			feature_dim=feature_dim,
			output_dim=output_dim
		)
	elif args.preset == "matched_16p16f":
		logging.info("Building MATCHED 16p16f JEDI-Linear model (16 particles, 16 features)")
		model = build_jedi_linear_matched_16p16f(output_dim=output_dim)
		# Note: This overrides num_particles and feature_dim
		num_particles = 16
		feature_dim = 16
	elif args.preset == "matched_32p16f":
		logging.info("Building MATCHED 32p16f JEDI-Linear model (32 particles, 16 features)")
		model = build_jedi_linear_matched_32p16f(output_dim=output_dim)
		num_particles = 32
		feature_dim = 16
	elif args.preset == "matched_64p16f":
		logging.info("Building MATCHED 64p16f JEDI-Linear model (64 particles, 16 features)")
		model = build_jedi_linear_matched_64p16f(output_dim=output_dim)
		num_particles = 64
		feature_dim = 16
	elif args.preset == "matched_16p3f":
		logging.info("Building MATCHED 16p3f JEDI-Linear model (16 particles, 3 features)")
		model = build_jedi_linear_matched_16p3f(output_dim=output_dim)
		num_particles = 16
		feature_dim = 3
	elif args.preset == "matched_64p3f":
		logging.info("Building MATCHED 64p3f JEDI-Linear model (64 particles, 3 features)")
		model = build_jedi_linear_matched_64p3f(output_dim=output_dim)
		num_particles = 64
		feature_dim = 3
	else:  # custom
		logging.info("Building CUSTOM JEDI-Linear model")
		model = build_jedi_linear_classifier(
			num_particles=num_particles,
			feature_dim=feature_dim,
			embedding_dim=args.embedding_dim,
			num_blocks=args.num_blocks,
			token_hidden=args.token_hidden,
			channel_hidden=args.channel_hidden,
			output_dim=output_dim,
			aggregation=args.aggregation,
			dropout_rate=args.dropout,
			head_hidden_dims=args.head_hidden,
		)

	# compile model
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
		loss=loss_fn,
		metrics=["accuracy"]
	)
	model.summary(print_fn=lambda l: logging.info(l))
	logging.info("Total params: %d", model.count_params())

	# callbacks
	ckpt = ModelCheckpoint(
		os.path.join(save_dir, "best.weights.h5"),
		monitor="val_loss",
		save_best_only=True,
		verbose=1,
	)
	early = EarlyStopping(
		monitor="val_loss", patience=40, restore_best_weights=True, verbose=1
	)

	# training schedule: progressively increase batch size
	schedule = [
		(128, 200),
		(256, 200),
		(512, 200),
		(1024, 200),
		(2048, 200),
		(4096, 400),
	]

	ce = 0
	histories = []
	for bs, ep in schedule:
		logging.info("Training with batch_size=%d for %d epochs", bs, ep)
		tf.keras.backend.set_value(model.optimizer.lr, 1e-3)
		hist = model.fit(
			x_train,
			y_train,
			validation_data=(x_val, y_val),
			initial_epoch=ce,
			epochs=ce + ep,
			batch_size=bs,
			callbacks=[ckpt, early],
			verbose=1,
		)
		histories.append(hist)
		ce += ep

	# save weights and metrics
	model.save_weights(os.path.join(save_dir, "model.weights.h5"))
	train_loss = np.concatenate([h.history["loss"] for h in histories])
	val_loss = np.concatenate([h.history["val_loss"] for h in histories])
	train_acc = np.concatenate([h.history["accuracy"] for h in histories])
	val_acc = np.concatenate([h.history["val_accuracy"] for h in histories])
	np.save(os.path.join(save_dir, "train_loss.npy"), train_loss)
	np.save(os.path.join(save_dir, "val_loss.npy"), val_loss)
	np.save(os.path.join(save_dir, "train_accuracy.npy"), train_acc)
	np.save(os.path.join(save_dir, "val_accuracy.npy"), val_acc)

	# plot loss and accuracy
	plt.figure()
	plt.plot(train_loss, label="Train Loss")
	plt.plot(val_loss, label="Val Loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(save_dir, "loss_curve.png"))
	plt.close()

	plt.figure()
	plt.plot(train_acc, label="Train Acc")
	plt.plot(val_acc, label="Val Acc")
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(save_dir, "accuracy_curve.png"))
	plt.close()

	# final testing
	run_testing(
		model,
		args.dataset,
		args.data_dir,
		save_dir,
		args.sort_by,
		args.batch_size,
		num_particles,
	)


if __name__ == "__main__":
	main()
