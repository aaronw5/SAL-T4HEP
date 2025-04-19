#!/usr/bin/env python
import os
import sys

# ─── make the parent directory (project root) importable ─────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ─────────────────────────────────────────────────────────────────────────────

import time
import argparse
import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

# import your model definitions
from models.Linformer import (
    AggregationLayer,
    ClusteredLinformerAttention,
    LinformerTransformerBlock,
    build_linformer_transformer_classifier
)

def get_flops(model, input_shape):
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
    input_tensor = tf.TensorSpec(input_shape, tf.float32)
    concrete_func = tf.function(model).get_concrete_function(input_tensor)
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.compat.v1.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(
            graph=graph, run_meta=run_meta, cmd='op', options=opts
        )
        return flops.total_float_ops

def parse_args():
    p = argparse.ArgumentParser(description="Train a (clustered) Linformer on jet data")
    p.add_argument("--data_dir",       required=True,
                   help="Directory containing x_train_*.npy and y_train_*.npy")
    p.add_argument("--save_dir",       required=True,
                   help="Base directory for saving weights, plots, logs, metrics")
    p.add_argument("--cluster_E",      action="store_true",
                   help="If set, use clustered projection for keys")
    p.add_argument("--cluster_F",      action="store_true",
                   help="If set, use clustered projection for values")
    p.add_argument("--batch_size",     type=int,   default=4096)
    p.add_argument("--num_epochs",     type=int,   default=1000)
    p.add_argument("--d_model",        type=int,   default=16)
    p.add_argument("--d_ff",           type=int,   default=16)
    p.add_argument("--output_dim",     type=int,   default=16)
    p.add_argument("--num_heads",      type=int,   default=4)
    p.add_argument("--proj_dim",       type=int,   default=4,
                   help="Linformer projection dimension (ignored if clustered)")
    p.add_argument("--val_split",      type=float, default=0.2,
                   help="Fraction of training data to hold out for validation")
    p.add_argument("--num_particles",  type=int,   required=True,
                   help="Number of particles in each event (replaces hard‑coded value)")
    p.add_argument("--sort_by",
                   choices=["pt","eta","phi","delta_R"],
                   default="pt",
                   help="How to sort particles before training: pt, eta, phi, or ΔR=√(η²+φ²)")
    return p.parse_args()

def main():
    args = parse_args()

    # create a subdirectory under save_dir based on num_particles and sort_by
    save_dir = os.path.join(args.save_dir,str(args.num_particles),args.sort_by)
    os.makedirs(save_dir, exist_ok=True)

    # setup logging
    logging.basicConfig(
        filename=os.path.join(save_dir, "train.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.info("Starting training with args: %s", args)

    # 1) load data using num_particles in filename
    x_path = os.path.join(
        args.data_dir,
        f"x_train_robust_{args.num_particles}const_ptetaphi.npy"
    )
    y_path = os.path.join(
        args.data_dir,
        f"y_train_robust_{args.num_particles}const_ptetaphi.npy"
    )
    x = np.load(x_path)
    y = np.load(y_path)
    logging.info("Loaded x shape %s, y shape %s", x.shape, y.shape)

    # 2) sort each event according to args.sort_by
    #    last‐dim order is [pt, eta, phi]
    if args.sort_by == "pt":
        key = x[:, :, 0]
    elif args.sort_by == "eta":
        key = x[:, :, 1]
    elif args.sort_by == "phi":
        key = x[:, :, 2]
    else:  # delta_R = sqrt(eta^2 + phi^2)
        key = np.sqrt(x[:, :, 1]**2 + x[:, :, 2]**2)

    sort_idx = np.argsort(key, axis=1)[:, ::-1]  # descending
    x = np.take_along_axis(x, sort_idx[:, :, None], axis=1)

    # 3) train/validation split
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=args.val_split, random_state=42, shuffle=True
    )
    logging.info("Split into train %s and val %s", x_train.shape, x_val.shape)

    # 4) build model
    num_particles, feature_dim = x_train.shape[1], x_train.shape[2]
    model = build_linformer_transformer_classifier(
        num_particles,
        feature_dim,
        d_model=args.d_model,
        d_ff=args.d_ff,
        output_dim=args.output_dim,
        num_heads=args.num_heads,
        proj_dim=args.proj_dim,
        cluster_E=args.cluster_E,
        cluster_F=args.cluster_F
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary(print_fn=lambda s: logging.info(s))

    # 5) train
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.num_epochs,
        batch_size=args.batch_size,
        verbose=1
    )

    # 6) save weights
    weights_file = os.path.join(save_dir, "model.weights.h5")
    model.save_weights(weights_file)
    logging.info("Saved weights to %s", weights_file)

    # 7) save training & validation metrics as .npy files
    np.save(os.path.join(save_dir, "train_loss.npy"),
            np.array(history.history["loss"]))
    np.save(os.path.join(save_dir, "val_loss.npy"),
            np.array(history.history["val_loss"]))
    np.save(os.path.join(save_dir, "train_accuracy.npy"),
            np.array(history.history["accuracy"]))
    np.save(os.path.join(save_dir, "val_accuracy.npy"),
            np.array(history.history["val_accuracy"]))
    logging.info("Saved history metrics to .npy files")

    # 8) compute FLOPs
    sample_shape = [1, num_particles, feature_dim]
    flops = get_flops(model, sample_shape)
    logging.info("FLOPs per inference: %d", flops)
    print(f"FLOPs per inference: {flops}")

    # 9) inference timing on validation set
    _ = model(x_val[:args.batch_size])  # warmup
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = model(x_val[:args.batch_size])
        times.append(time.perf_counter() - t0)
    avg_ns = np.mean(np.array(times) / args.batch_size) * 1e9
    logging.info("Avg inference time per event (ns): %.3f", avg_ns)
    print(f"Avg inference time per event: {avg_ns:.3f} ns")

    # 10) validation accuracy
    preds = model.predict(x_val, batch_size=args.batch_size)
    val_acc = accuracy_score(np.argmax(y_val, 1), np.argmax(preds, 1))
    logging.info("Validation accuracy: %.4f", val_acc)
    print(f"Validation accuracy: {val_acc:.4f}")

    # 11) plot training curves
    plt.figure(figsize=(6,4))
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(history.history["accuracy"], label="train acc")
    plt.plot(history.history["val_accuracy"], label="val acc")
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("acc")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "acc_curve.png"))
    plt.close()
    logging.info("Saved training curves")

    # 12) ROC curves & 1/FPR@TPR=0.8
    class_labels = ["g","q","W","Z","t"]
    target_tpr = 0.8
    one_over_fpr = {}
    plt.figure(figsize=(6,6))
    for i, label in enumerate(class_labels):
        fpr_vals, tpr_vals, thresh = roc_curve(y_val[:, i], preds[:, i])
        roc_auc = auc(fpr_vals, tpr_vals)
        plt.plot(fpr_vals, tpr_vals, label=f"{label} (AUC={roc_auc:.2f})")
        if tpr_vals[-1] >= target_tpr:
            fpr_t = np.interp(target_tpr, tpr_vals, fpr_vals)
            one_over_fpr[label] = 1.0 / fpr_t if fpr_t > 0 else np.nan
            plt.plot(fpr_t, target_tpr, 'o')
    plt.plot([0,1], [0,1], "k--")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curves.png"))
    plt.close()

    for label, v in one_over_fpr.items():
        logging.info("1/FPR@TPR=0.8 for %s: %.3f", label, v)
    avg_inv = np.nanmean(list(one_over_fpr.values()))
    logging.info("Average 1/FPR: %.3f", avg_inv)

if __name__ == "__main__":
    main()
