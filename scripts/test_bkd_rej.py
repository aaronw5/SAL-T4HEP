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
from tensorflow.keras.models import load_model
import fastjet as fj
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

# import your custom layer/classes
from models.Linformer import (
    AggregationLayer,
    AttentionConvLayer,
    DynamicTanh,
    ClusteredLinformerAttention,
    LinformerTransformerBlock
)
from models.Transformer import (
    StandardMultiHeadAttention,
    StandardTransformerBlock,
    build_standard_transformer_classifier
)

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
						# Only append to rejection values for non-light-flavor backgrounds
						if lab not in ["q", "g"]:
							rej_vals.append(rej)
				logging.info("Avg bg rejection@0.8: %.3f", np.nanmean(rej_vals))

def profile_gpu_memory_during_inference(model: tf.keras.Model, input_data: np.ndarray) -> tuple[float, float]:
    logging.info("Starting GPU memory profiling")
    tf.config.experimental.reset_memory_stats('GPU:0')
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
    # Use correct context manager for new graphs
    new_graph = tf.Graph()
    with new_graph.as_default():
        tf.compat.v1.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        prof = tf.compat.v1.profiler.profile(graph=new_graph, run_meta=run_meta, cmd='op', options=opts)
        flops = prof.total_float_ops
    logging.info("FLOPs calculation done: %d FLOPs", flops)
    return flops


def sort_events_by_cluster(x, R, batch_size):
    logging.info("Starting cluster-based sorting with R=%.2f, batch_size=%d", R, batch_size)
    n_events, n_particles, _ = x.shape
    sorted_x = np.zeros_like(x)
    jet_def = fj.JetDefinition(fj.antikt_algorithm, R)
    for i0 in range(0, n_events, batch_size):
        for bi, ev in enumerate(x[i0:i0+batch_size]):
            pts, etas, phis = ev[:,0], ev[:,1], ev[:,2]
            px = pts * np.cos(phis); py = pts * np.sin(phis)
            pz = pts * np.sinh(etas); E = pts * np.cosh(etas)
            ps = [fj.PseudoJet(px[j], py[j], pz[j], E[j]) for j in range(len(pts))]
            for pj in ps:
                pj.set_user_index(ps.index(pj))
            seq = fj.ClusterSequence(ps, jet_def)
            jets = seq.inclusive_jets()
            jets.sort(key=lambda J: J.perp(), reverse=True)
            idxs = [c.user_index() for J in jets for c in J.constituents()]
            remain = [j for j in range(len(pts)) if j not in idxs]
            idxs.extend(remain)
            sorted_x[i0+bi] = ev[idxs]
    logging.info("Cluster-based sorting done")
    return sorted_x


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/j-jepa-vol/l1-jet-id/data/jetid/processed")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--sort_by", choices=["pt","eta","phi","delta_R","kt","cluster"], default="pt")
    parser.add_argument("--cluster_R", type=float, default=0.4)
    parser.add_argument("--cluster_batch_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--test_model", help="Path to the saved full model")
    args = parser.parse_args()
    args.dataset = "hls4ml"

    # Setup logging
    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.save_dir, "bg_rej.log"), filemode='w',
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    cwd = os.getcwd()
    print(f"Running in directory: {cwd}")
    logging.info("Running in directory: %s", cwd)

    # Model loading
    model_path = args.test_model or os.path.join(args.save_dir, "best.weights.h5")
    logging.info("Loading model from %s", model_path)
    model = load_model(model_path, custom_objects={
        'AggregationLayer': AggregationLayer,
        'AttentionConvLayer': AttentionConvLayer,
        'DynamicTanh': DynamicTanh,
        'ClusteredLinformerAttention': ClusteredLinformerAttention,
        'LinformerTransformerBlock': LinformerTransformerBlock,
        'StandardMultiHeadAttention': StandardMultiHeadAttention,
        'StandardTransformerBlock': StandardTransformerBlock,
    })
    logging.info("Model loaded successfully.")
    model.summary(print_fn=lambda s: logging.info(s))
    logging.info("Total parameters: %d", model.count_params())

    # # FLOPs & timing
    # flops = get_flops(model)
    # logging.info("MACs per inference: %d", flops // 2)

    # logging.info("Starting inference timing (20 runs)")
    # _ = model.predict(x[:args.batch_size], batch_size=args.batch_size)
    # times = []
    # for _ in range(20):
    #     t0 = time.perf_counter()
    #     _ = model.predict(x[:args.batch_size], batch_size=args.batch_size)
    #     times.append(time.perf_counter() - t0)
    # avg_ns = np.mean(times) / args.batch_size * 1e9
    # logging.info("Inference timing done: avg %.2f ns/event", avg_ns)

    # curr, peak = profile_gpu_memory_during_inference(model, x[:args.batch_size])

    # # Predictions
    # logging.info("Starting full dataset prediction")
    # preds = model.predict(x, batch_size=args.batch_size)
    # logging.info("Prediction done: preds shape=%s", preds.shape)

    # # Metrics
    # logging.info("Starting metric computation for '%s'", args.dataset)
    # if args.dataset in ('top', 'quark_gluon'):
    #     scores = preds.ravel()
    #     acc = accuracy_score(y, (scores > 0.5).astype(int))
    #     auc_m = roc_auc_score(y, scores)
    #     logging.info("Accuracy = %.4f, AUC = %.4f", acc, auc_m)
    #     fpr_vals, tpr_vals, _ = roc_curve(y, scores)
    #     for thresh in (0.5, 0.3):
    #         idx = np.abs(tpr_vals - thresh).argmin()
    #         fpr_t = fpr_vals[idx]
    #         rej = 1.0/fpr_t if fpr_t>0 else np.inf
    #         logging.info("Rejection at %d%% TPR: %.3f", int(thresh*100), rej)
    # elif args.dataset == 'jetclass':
    #     overall_auc = roc_auc_score(y, preds, average='macro', multi_class='ovo')
    #     true_lbl = np.argmax(y, axis=1)
    #     pred_lbl = np.argmax(preds, axis=1)
    #     acc = accuracy_score(true_lbl, pred_lbl)
    #     logging.info("Overall ROC AUC = %.4f, Accuracy = %.4f", overall_auc, acc)
    #     rejections = []
    #     for i in range(preds.shape[1]):
    #         lab = f'label_{i}'
    #         fpr_i, tpr_i, _ = roc_curve(y[:,i], preds[:,i])
    #         auc_i = auc(fpr_i, tpr_i)
    #         logging.info("ROC AUC for %s: %.4f", lab, auc_i)
    #         if i > 0:
    #             idx = np.abs(tpr_i - 0.5).argmin()
    #             fpr_t = fpr_i[idx]
    #             rej = 1.0/fpr_t if fpr_t>0 else np.inf
    #             rejections.append(rej)
    #             logging.info("Rejection at 50%% TPR for %s: %.3f", lab, rej)
    #     logging.info("Avg rejection across classes: %.3f", np.nanmean(rejections))
    # else:
    #     labels = ['q','g','W','Z','t'] if args.dataset=='hls4ml' else ['q','g']
    #     for lab_i, lab in enumerate(labels):
    #         fpr_vals, tpr_vals, _ = roc_curve(y[:,lab_i], preds[:,lab_i])
    #         auc_val = auc(fpr_vals, tpr_vals)
    #         logging.info("ROC AUC for %s: %.4f", lab, auc_val)
    # logging.info("Metric computation done")

    # # Accuracy vs. number of particles
    # logging.info("Starting accuracy vs. number of particles bins computation")
    # num_parts = np.sum(x[:,:,0] != 0, axis=1)
    # edges = np.linspace(0, npart, 5, dtype=int)
    # results = []
    # for i in range(4):
    #     low, high = edges[i], edges[i+1]
    #     mask = (num_parts >= low) & (num_parts < (high if i<3 else high+1))
    #     if mask.sum() == 0:
    #         acc_bin = np.nan
    #     elif args.dataset in ('top','quark_gluon'):
    #         scr = preds[mask].ravel()
    #         lbl = y[mask]
    #         acc_bin = accuracy_score(lbl, (scr>0.5).astype(int))
    #     else:
    #         acc_bin = accuracy_score(np.argmax(y[mask], axis=1), np.argmax(preds[mask], axis=1))
    #     results.append((low, high, mask.sum(), acc_bin))
    #     logging.info("Bin %d [%d-%d]: acc=%.4f over %d events", i+1, low, high, acc_bin, mask.sum())
    # logging.info("Accuracy vs. number of particles bins done")

    # # Print and log textual table
    # header = f"{'Bin':>3}   {'Range':>7}   {'#Evts':>6}   {'Accuracy':>8}"    
    # logging.info(header)
    # print("\nAccuracy vs. number of particles:")
    # print(header)
    # for idx, (low, high, cnt, acc_bin) in enumerate(results, start=1):
    #     line = f"{idx:>3}   [{low:3d}-{high:3d}]   {cnt:6d}   {acc_bin:8.3f}"
    #     logging.info(line)
    #     print(line)

    # Execute run_testing with necessary arguments
    run_testing(
        model=model,
        dataset=args.dataset,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        sort_by=args.sort_by,
        batch_size=args.batch_size,
        num_particles=model.input_shape[1]
    )

if __name__ == "__main__":
    main()
