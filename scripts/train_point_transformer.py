#!/usr/bin/env python
"""
Train a PointTransformerV3-like TensorFlow model on jet datasets (hls4ml, top, QG, or jetclass),
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
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

# import model builders
from models.PointTransformerV3TF import build_ptv3_jet_classifier, build_jedi_ptv3_hybrid
from models.PointTransformer_serialized import build_ptv3_serialized_jet_classifier


# ---------------------------
# FLOPs computation (no mask)
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
# GPU memory profiling (no mask)
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
# Patch size utilities
# ---------------------------
def compute_stage_lengths(num_particles, enc_strides):
		"""
		Length before each stage (after previous downsamples applied).
		For stages S, and strides defined between stages: len(enc_strides) == S-1
		"""
		lengths = []
		current = num_particles
		for i in range(len(enc_strides) + 1):
				lengths.append(current)
				if i < len(enc_strides):
						stride = enc_strides[i]
						current = int(math.ceil(current / float(stride)))
		return lengths


def choose_divisible_patch_sizes(stage_lengths, preferred=[64, 32, 16, 8, 4, 2, 1]):
		patch_sizes = []
		for L in stage_lengths:
				ps = 1
				for cand in preferred:
						if L >= cand and (L % cand == 0):
								ps = cand
								break
				patch_sizes.append(ps)
		return patch_sizes


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
		p = argparse.ArgumentParser(description="Train PointTransformerV3TF on jet data")
		p.add_argument("--data_dir", required=True)
		p.add_argument("--save_dir", required=True)
		p.add_argument(
				"--dataset", choices=["hls4ml", "top", "QG", "jetclass"], default="hls4ml"
		)
		p.add_argument(
				"--sort_by",
				choices=["pt", "eta", "phi", "delta_R", "kt", "cluster"],
				default="kt",
		)
		p.add_argument("--batch_size", type=int, default=4096)
		p.add_argument("--val_split", type=float, default=0.2)

		# Model hyperparameters
		p.add_argument("--enc_dims", type=int, nargs="+", default=[12, 24, 32])
		p.add_argument("--enc_layers", type=int, nargs="+", default=[1, 1, 1])
		p.add_argument("--enc_heads", type=int, nargs="+", default=[4, 4, 4])
		p.add_argument("--enc_patch_sizes", type=int, nargs="+", default=[2, 2, 2])
		p.add_argument("--enc_strides", type=int, nargs="+", default=[2, 2])
		p.add_argument("--cpe_k", type=int, default=8)
		p.add_argument("--grid_size", type=float, default=0.05, help="GeometricCPE grid size (coarser -> smaller grid)")
		p.add_argument("--use_rpe", action="store_true")
		p.add_argument("--disable_pool", action="store_true", help="Disable GeometricPooling between stages")
		p.add_argument("--dropout", type=float, default=0.0)
		p.add_argument("--aggregation", choices=["mean", "max"], default="max")
		p.add_argument("--model_size", choices=["small", "small_2layer_no_downsamp", "small_2layer_2_downsamp", "matched", "medium", "large"], default="small")
		p.add_argument('--use_serialized_model', action='store_true', help='Use the serialized version of the PointTransformer model')
		p.add_argument('--serialize_by', choices=['morton','pt','kt'], default='morton', help='Serialization strategy when using the serialized model')
		p.add_argument('--use_jedi_hybrid', action='store_true', help='Use JEDI-PTv3 Hybrid (O(N) global interaction instead of attention)')
		p.add_argument('--disable_cpe', action='store_true', help='Disable CPE in JEDI hybrid (for pure JEDI-style permutation invariance)')
		p.add_argument("--ffn_activation", choices=["relu", "gelu", "swish", "silu", "tanh"], default="gelu", help="Activation function for feed-forward network (relu is fastest, gelu is default)")
		p.add_argument("--jit_compile", action="store_true", help="Enable XLA JIT compilation for faster training (5-15%% speedup on modern GPUs)")
		p.add_argument("--use_flash_attention", action="store_true", help="Enable Flash Attention for faster and more memory-efficient attention (requires TensorFlow 2.11+ and compatible GPU)")
		return p.parse_args()


def main():
		args = parse_args()

		# pick num_particles & output_dim
		if args.dataset == "jetclass":
				num_particles = 150
				output_dim = 10
				loss_fn = "categorical_crossentropy"
		elif args.dataset == "top":
				num_particles = 200
				output_dim = 1
				loss_fn = "binary_crossentropy"
		elif args.dataset == "QG":
				num_particles = 150
				output_dim = 1
				loss_fn = "binary_crossentropy"
		else:  # hls4ml
				num_particles = 150
				output_dim = 5
				loss_fn = "categorical_crossentropy"

		# prepare save directory
		save_dir = os.path.join(args.save_dir, str(num_particles), args.sort_by)
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

		# load train/val
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

		# select preset
		presets = {
			"small":  dict(enc_dims=[16], enc_layers=[1], enc_heads=[4], enc_strides=[2], enc_patch_sizes=[25], cpe_k=8, use_rpe=False),
			"small_2layer_no_downsamp": dict(enc_dims=[16, 16], enc_layers=[1, 1], enc_heads=[4, 4], enc_strides=[1, 1], enc_patch_sizes=[25, 25], cpe_k=8, use_rpe=False),
			"small_2layer_2_downsamp": dict(enc_dims=[16, 16], enc_layers=[1, 1], enc_heads=[4, 4], enc_strides=[2], enc_patch_sizes=[25, 25], cpe_k=8, use_rpe=False),
    		"matched": dict(enc_dims=[12, 16], enc_layers=[1, 1], enc_heads=[4, 4], enc_strides=[2], enc_patch_sizes=[25, 25], cpe_k=8, use_rpe=False),
    		"medium": dict(enc_dims=[12, 24, 32], enc_layers=[1, 1, 1], enc_heads=[4, 4, 4], enc_strides=[2, 2], enc_patch_sizes=[25, 25, 25], cpe_k=8, use_rpe=False),
    		"large":  dict(enc_dims=[16, 24, 32], enc_layers=[1, 1, 1], enc_heads=[4, 4, 4], enc_strides=[2, 2], enc_patch_sizes=[25, 25, 25], cpe_k=8, use_rpe=False),
    	}
		cfg = presets[args.model_size]
		enc_dims = cfg["enc_dims"]
		enc_layers = cfg["enc_layers"]
		enc_heads = cfg["enc_heads"]
		enc_strides = cfg["enc_strides"]
		enc_patch_sizes = cfg["enc_patch_sizes"]
		cpe_k = cfg["cpe_k"] if args.cpe_k is None else args.cpe_k
		use_rpe = args.use_rpe or cfg["use_rpe"]

		# build and compile model
		logging.info("CPE type: %s, CPE enabled: %s", args.cpe_type, not args.disable_cpe)
		logging.info("Flash Attention enabled: %s", args.use_flash_attention)
		if args.use_jedi_hybrid:
			logging.info("Building JEDI-PTv3 Hybrid model (O(N) global interaction)")
			model = build_jedi_ptv3_hybrid(
				num_particles=num_particles,
				output_dim=output_dim,
				enc_dims=enc_dims,
				enc_layers=enc_layers,
				enc_strides=enc_strides,
				cpe_k=cpe_k,
				grid_size=args.grid_size,
				use_pool=(not args.disable_pool),
				use_cpe=(not args.disable_cpe),
				cpe_type=args.cpe_type,
				dropout=args.dropout,
				aggregation=args.aggregation,
				ffn_activation=args.ffn_activation,
			)
		elif args.use_serialized_model:
			model = build_ptv3_serialized_jet_classifier(
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
				dropout=args.dropout,
				aggregation=args.aggregation,
				serialize_by=args.serialize_by,
			)
		else:
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
				use_cpe=(not args.disable_cpe),
				cpe_type=args.cpe_type,
				use_pool=(not args.disable_pool),
				dropout=args.dropout,
				aggregation=args.aggregation,
				ffn_activation=args.ffn_activation,
				use_flash_attention=args.use_flash_attention,
			)
		model.compile(
				optimizer=tf.keras.optimizers.Adam(),
				loss=loss_fn,
				metrics=["accuracy"],
				jit_compile=args.jit_compile,
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


