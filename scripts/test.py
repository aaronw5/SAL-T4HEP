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
import glob

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import fastjet as fj
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

# import your custom layer/classes for reconstruction
from models.Linformer import (
    AggregationLayer,
    AttentionConvLayer,
    DynamicTanh,
    ClusteredLinformerAttention,
    LinformerTransformerBlock,
)
from models.Transformer import StandardTransformerBlock


def profile_gpu_memory_during_inference(
    model: tf.keras.Model,
    input_data: np.ndarray,
) -> tuple[float, float]:
    """
    Runs one forward pass in @tf.function and returns
    (current_gpu_mb, peak_gpu_mb) allocated during that call.
    """
    # reset stats so we get a fresh peak measurement
    tf.config.experimental.reset_memory_stats("GPU:0")

    @tf.function
    def infer(x):
        return model(x, training=False)

    # warm-up to allocate buffers
    _ = infer(input_data[:1])
    # actual profiling
    _ = infer(input_data)

    mem = tf.config.experimental.get_memory_info("GPU:0")
    current_mb = mem["current"] / (1024**2)
    peak_mb = mem["peak"] / (1024**2)
    return current_mb, peak_mb


def get_flops(model, input_shape):
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2_as_graph,
    )

    inp = tf.TensorSpec(input_shape, tf.float32)
    func = tf.function(model).get_concrete_function(inp)
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(func)
    with tf.Graph().as_default() as g:
        tf.compat.v1.import_graph_def(graph_def, name="")
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(
            graph=g, run_meta=run_meta, cmd="op", options=opts
        )
        return flops.total_float_ops


def sort_events_by_cluster(x, R, batch_size):
    n_events, n_particles, _ = x.shape
    sorted_x = np.zeros_like(x)
    jet_def = fj.JetDefinition(fj.antikt_algorithm, R)
    for i0 in range(0, n_events, batch_size):
        batch = x[i0 : i0 + batch_size]
        for bi, ev in enumerate(batch):
            pts, etas, phis = ev[:, 0], ev[:, 1], ev[:, 2]
            px = pts * np.cos(phis)
            py = pts * np.sin(phis)
            pz = pts * np.sinh(etas)
            E = pts * np.cosh(etas)
            ps = [fj.PseudoJet(px[j], py[j], pz[j], E[j]) for j in range(len(pts))]
            for j, pj in enumerate(ps):
                pj.set_user_index(j)
            seq = fj.ClusterSequence(ps, jet_def)
            jets = seq.inclusive_jets()
            jets.sort(key=lambda J: J.perp(), reverse=True)
            idxs = [c.user_index() for J in jets for c in J.constituents()]
            remain = [j for j in range(len(pts)) if j not in idxs]
            idxs.extend(remain)
            sorted_x[i0 + bi] = ev[idxs]
    return sorted_x


def apply_sorting(x, sort_by, R, batch_size):
    if sort_by in ("pt", "eta", "phi", "delta_R", "kt"):
        if sort_by == "pt":
            key = x[:, :, 0]
        elif sort_by == "eta":
            key = x[:, :, 1]
        elif sort_by == "phi":
            key = x[:, :, 2]
        elif sort_by == "delta_R":
            key = np.sqrt(x[:, :, 1] ** 2 + x[:, :, 2] ** 2)
        else:
            key = x[:, :, 0] * np.sqrt(x[:, :, 1] ** 2 + x[:, :, 2] ** 2)
        idx = np.argsort(key, axis=1)[:, ::-1]
        return np.take_along_axis(x, idx[:, :, None], axis=1)
    else:
        return sort_events_by_cluster(x, R, batch_size)


def process_directory(
    data_dir, save_dir, sort_by, cluster_R=0.4, cluster_batch_size=1024, batch_size=4096
):
    """Process a single directory with the given parameters"""
    # determine model path
    model_path = os.path.join(save_dir, "best.weights.h5")

    # Set up logging
    logging.basicConfig(
        filename=os.path.join(save_dir, "train.log"),
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.info("Loading model from %s", model_path)

    # load full model with custom objects
    custom_objs = {
        "AggregationLayer": AggregationLayer,
        "AttentionConvLayer": AttentionConvLayer,
        "DynamicTanh": DynamicTanh,
        "ClusteredLinformerAttention": ClusteredLinformerAttention,
        "LinformerTransformerBlock": LinformerTransformerBlock,
        "StandardTransformerBlock": StandardTransformerBlock,
    }
    model = load_model(model_path, custom_objects=custom_objs)
    logging.info("Model loaded successfully.")

    # Log model summary and total parameters
    model.summary(print_fn=lambda s: logging.info(s))
    total_params = model.count_params()
    logging.info("Total parameters: %d", total_params)

    # Load and sort data
    num_particles = model.input_shape[1]
    feat_dim = model.input_shape[2]
    x_file = f"x_val_robust_{num_particles}const_ptetaphi.npy"
    y_file = f"y_val_robust_{num_particles}const_ptetaphi.npy"
    x = np.load(os.path.join(data_dir, x_file))
    y = np.load(os.path.join(data_dir, y_file))
    logging.info("Loaded TEST arrays: %s, %s", x_file, y_file)

    x = apply_sorting(x, sort_by, cluster_R, cluster_batch_size)
    logging.info("Applied '%s' sorting to TEST set", sort_by)

    # FLOPs & timing
    flops = get_flops(model, [1, num_particles, feat_dim])
    macs = flops // 2
    logging.info("FLOPs per inference: %d", flops)
    logging.info("MACs per inference: %d", macs)

    # inference timing
    _ = model.predict(x[:batch_size], batch_size=batch_size)
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = model.predict(x[:batch_size], batch_size=batch_size)
        times.append(time.perf_counter() - t0)
    avg_ns = np.mean(times) / batch_size * 1e9
    logging.info("Avg inference time / event: %.2f ns", avg_ns)

    # Profile GPU memory usage
    current_gpu_mb, peak_gpu_mb = profile_gpu_memory_during_inference(
        model, x[:batch_size]
    )
    logging.info(
        "GPU memory — current: %.1f MB, peak: %.1f MB", current_gpu_mb, peak_gpu_mb
    )

    # Predictions & metrics
    preds = model.predict(x, batch_size=batch_size)
    acc = accuracy_score(np.argmax(y, 1), np.argmax(preds, 1))
    auc_m = roc_auc_score(y, preds, average="macro", multi_class="ovo")
    logging.info("Test Accuracy: %.4f, ROC AUC: %.4f", acc, auc_m)

    # ROC curves + background rejection
    class_labels = ["g", "q", "W", "Z", "t"]
    plt.figure(figsize=(6, 6))
    one_over_fpr = {}
    for i, label in enumerate(class_labels):
        fpr_vals, tpr_vals, _ = roc_curve(y[:, i], preds[:, i])
        roc_auc_val = auc(fpr_vals, tpr_vals)
        logging.info("ROC AUC for %s: %.4f", label, roc_auc_val)
        plt.plot(fpr_vals, tpr_vals, label=f"{label} (AUC={roc_auc_val:.2f})")
        if np.max(tpr_vals) >= 0.8:
            fpr_t = np.interp(0.8, tpr_vals, fpr_vals)
            one_over_fpr[label] = 1.0 / fpr_t if fpr_t > 0 else np.nan
            plt.plot(fpr_t, 0.8, "o")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(save_dir, "roc_curves.png")
    plt.savefig(roc_path)
    plt.close()

    for label, val in one_over_fpr.items():
        logging.info("1/FPR @0.8 TPR for %s: %.3f", label, val)
    avg_one_over = np.nanmean(list(one_over_fpr.values()))
    logging.info("Avg 1/FPR @0.8 TPR: %.3f", avg_one_over)

    # Updated background rejection: background = g AND q for all classes
    rej_list = []
    for i, label in enumerate(class_labels[1:], start=1):
        mask = (y[:, 0] == 1) | (y[:, 1] == 1) | (y[:, i] == 1)
        bin_y = (y[mask, i] == 1).astype(int)
        bin_s = preds[mask, i]
        fpr_vals, tpr_vals, _ = roc_curve(bin_y, bin_s)
        idx = np.argmin(np.abs(tpr_vals - 0.8))
        rej = 1.0 / fpr_vals[idx] if fpr_vals[idx] > 0 else np.inf
        logging.info("Background rejection @0.8 for %s: %.3f", label, rej)
        rej_list.append(rej)
    logging.info("Avg background rejection @0.8: %.3f", np.nanmean(rej_list))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument(
        "--root_dir",
        required=True,
        help="Root directory containing all experiment directories",
    )
    p.add_argument("--cluster_R", type=float, default=0.4)
    p.add_argument("--cluster_batch_size", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=4096)
    args = p.parse_args()

    # Find all valid directories
    valid_dirs = []
    for root, dirs, files in os.walk(args.root_dir):
        if "loss_curve.png" in files and "train.log" in files:
            valid_dirs.append(root)

    # Process each valid directory
    for save_dir in valid_dirs:
        # Extract sort_by from the path
        path_parts = save_dir.split(os.sep)
        sort_by = None
        for part in path_parts:
            if part in ["pt", "eta", "phi", "delta_R", "kt", "cluster"]:
                sort_by = part
                break

        if sort_by is None:
            logging.warning(f"Could not determine sort_by from path: {save_dir}")
            continue

        print(f"Processing directory: {save_dir}")
        print(f"Using sort_by: {sort_by}")

        try:
            process_directory(
                args.data_dir,
                save_dir,
                sort_by,
                args.cluster_R,
                args.cluster_batch_size,
                args.batch_size,
            )
        except Exception as e:
            print(f"Error processing {save_dir}: {str(e)}")
            continue


if __name__ == "__main__":
    main()
