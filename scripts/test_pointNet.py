#!/usr/bin/env python
"""
Test a trained PointNet classifier on the chosen dataset,
reporting metrics, FLOPs, timing, and GPU memory.
"""
import os
import sys
import time
import argparse
import logging
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

# ─── make the parent directory (project root) importable ─────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.pointNet import build_pointnet_classifier


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


def main():
    parser = argparse.ArgumentParser(description="Test PointNet model")
    parser.add_argument("--dataset", choices=["hls4ml","top","jetclass","QG"], required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--sort_by", choices=["pt","eta","phi","delta_R","kt"], default="pt")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weights", help="Path to weights .h5 file (defaults to save_dir/best.weights.h5)")
    args = parser.parse_args()

    # Logging
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, "test_pointnet.log")
    logging.basicConfig(filename=log_path, filemode="w", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(f"Running in directory: {os.getcwd()}")

    # Dataset defaults
    if args.dataset == "jetclass":
        num_particles = 150; output_dim = 10
    elif args.dataset == "top":
        num_particles = 200; output_dim = 1
    elif args.dataset == "QG":
        num_particles = 150; output_dim = 1
    else:
        num_particles = 150; output_dim = 5

    # Load
    x_test, y_test = load_test_data(args.dataset, args.data_dir, num_particles)
    x_test = apply_sorting(x_test, args.sort_by)

    # Build and load weights
    model = build_pointnet_classifier(
        num_particles=num_particles,
        feature_dim=x_test.shape[2],
        output_dim=output_dim,
        dropout_rate=args.dropout,
    )
    weights_path = args.weights or os.path.join(args.save_dir, "best.weights.h5")
    logging.info("Loading weights from %s", weights_path)
    model.load_weights(weights_path)
    logging.info("Weights loaded.")
    model.summary(print_fn=lambda s: logging.info(s))

    # FLOPs, timing, memory
    flops = get_flops(model)
    logging.info("FLOPs per inference: %d (MACs ~ %d)", flops, flops // 2)
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
    preds = model.predict(x_test, batch_size=args.batch_size)

    # Metrics
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
        labels = ["q","g","W","Z","t"]
    elif args.dataset == "top":
        labels = ["qcd","top"]
    elif args.dataset == "QG":
        labels = ["Gluon","Quark"]
    else:
        labels = [f"label_{i}" for i in range(preds.shape[1])]

    plt.figure(figsize=(6, 6))
    for i, lab in enumerate(labels):
        if args.dataset in ("top","QG"):
            fpr, tpr, _ = roc_curve(y_test, preds.ravel())
        else:
            fpr, tpr, _ = roc_curve(y_test[:, i], preds[:, i])
        roc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{lab} (AUC={roc_val:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC curves (PointNet)")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "roc_curves_pointnet.png"))
    plt.close()

    if args.dataset != "top" and args.dataset != "QG":
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


if __name__ == "__main__":
    main()


