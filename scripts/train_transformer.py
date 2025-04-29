#!/usr/bin/env python3
import os
import random
import sys
import time
import argparse
import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import fastjet as fj
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from models.Transformer import AggregationLayer, build_standard_transformer_classifier


# ----------------------------------------------------------------------------
def get_flops(model, input_shape):
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2_as_graph,
    )

    input_tensor = tf.TensorSpec(input_shape, tf.float32)
    concrete_func = tf.function(model).get_concrete_function(input_tensor)
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.compat.v1.import_graph_def(graph_def, name="")
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(
            graph=graph, run_meta=run_meta, cmd="op", options=opts
        )
        return flops.total_float_ops


# ----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Train a Standard Transformer on jet data with sorting and optional convolution"
    )
    p.add_argument(
        "--data_dir", required=True, help="Directory containing x and y .npy files"
    )
    p.add_argument("--save_dir", required=True, help="Base directory for outputs")
    p.add_argument(
        "--convolution", action="store_true", help="Use convolution on attention scores"
    )
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--num_epochs", type=int, default=1000)
    p.add_argument("--d_model", type=int, default=16)
    p.add_argument("--d_ff", type=int, default=16)
    p.add_argument("--output_dim", type=int, default=16)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--num_particles", type=int, required=True)
    p.add_argument(
        "--sort_by",
        choices=["pt", "eta", "phi", "delta_R", "kt", "cluster"],
        default="pt",
    )
    p.add_argument("--cluster_R", type=float, default=0.4)
    p.add_argument("--cluster_batch_size", type=int, default=1024)
    return p.parse_args()


# ----------------------------------------------------------------------------
def sort_events_by_cluster(x, R, batch_size):
    n_events, n_particles, _ = x.shape
    sorted_x = np.zeros_like(x)
    jet_def = fj.JetDefinition(fj.antikt_algorithm, R)
    for start in range(0, n_events, batch_size):
        end = min(start + batch_size, n_events)
        batch = x[start:end]
        for i, event in enumerate(batch):
            pts = event[:, 0]
            etas = event[:, 1]
            phis = event[:, 2]
            px = pts * np.cos(phis)
            py = pts * np.sin(phis)
            pz = pts * np.sinh(etas)
            E = pts * np.cosh(etas)
            ps = [
                fj.PseudoJet(float(px[j]), float(py[j]), float(pz[j]), float(E[j]))
                for j in range(n_particles)
            ]
            for j, pj in enumerate(ps):
                pj.set_user_index(j)
            seq = fj.ClusterSequence(ps, jet_def)
            jets = seq.inclusive_jets()
            jets.sort(key=lambda jet: jet.perp(), reverse=True)
            sorted_indices = []
            for jet in jets:
                for const in jet.constituents():
                    sorted_indices.append(const.user_index())
            remaining = [idx for idx in range(n_particles) if idx not in sorted_indices]
            sorted_indices.extend(remaining)
            sorted_x[start + i] = event[sorted_indices]
    return sorted_x


# ----------------------------------------------------------------------------
def main():
    args = parse_args()

    # create a subdirectory under save_dir based on num_particles and sort_by
    save_dir = os.path.join(args.save_dir, str(args.num_particles), args.sort_by)
    trial_num = 0
    while True:
        trial_dir = os.path.join(save_dir, f"trial-{trial_num}")
        time.sleep(random.randint(1, 4))
        # Check if directory doesn't exist
        if not os.path.isdir(trial_dir):
            save_dir = trial_dir
            break
        trial_num += 1
    os.makedirs(save_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(save_dir, "train.log"),
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.info("Args: %s", args)

    # Load or sort data
    x_filename = f"x_train_robust_{args.num_particles}const_ptetaphi.npy"
    y_filename = f"y_train_robust_{args.num_particles}const_ptetaphi.npy"
    x_path = os.path.join(args.data_dir, x_filename)
    sorted_path = x_path.replace(".npy", f"_sorted_{args.sort_by}.npy")

    if os.path.exists(sorted_path):
        x = np.load(sorted_path)
        logging.info("Loaded pre-sorted data from %s", sorted_path)
    else:
        x = np.load(x_path)
        logging.info("Loaded raw data from %s", x_path)
        if args.sort_by in ["pt", "eta", "phi", "delta_R", "kt"]:
            if args.sort_by == "pt":
                key = x[:, :, 0]
            elif args.sort_by == "eta":
                key = x[:, :, 1]
            elif args.sort_by == "phi":
                key = x[:, :, 2]
            elif args.sort_by == "delta_R":
                key = np.sqrt(x[:, :, 1] ** 2 + x[:, :, 2] ** 2)
            else:
                key = x[:, :, 0] * np.sqrt(x[:, :, 1] ** 2 + x[:, :, 2] ** 2)
            idx = np.argsort(key, axis=1)[:, ::-1]
            x = np.take_along_axis(x, idx[:, :, None], axis=1)
        else:
            x = sort_events_by_cluster(x, args.cluster_R, args.cluster_batch_size)
        np.save(sorted_path, x)
        logging.info("Saved sorted data to %s", sorted_path)

    y = np.load(os.path.join(args.data_dir, y_filename))
    logging.info("Loaded labels from %s", y_filename)

    # Split
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=args.val_split, random_state=42, shuffle=True
    )
    logging.info("Split into train %s and val %s", x_train.shape, x_val.shape)

    # Build standard transformer classifier
    model = build_standard_transformer_classifier(
        args.num_particles,
        x.shape[2],
        d_model=args.d_model,
        d_ff=args.d_ff,
        output_dim=args.output_dim,
        num_heads=args.num_heads,
        convolution=args.convolution,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    model.summary(print_fn=lambda line: logging.info(line))

    # Log parameter count
    total_params = model.count_params()
    logging.info("Total parameters: %d", total_params)
    print(f"Total parameters: {total_params}")

    # Callbacks
    ckpt = ModelCheckpoint(
        os.path.join(save_dir, "best.weights.h5"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )
    early_stop = EarlyStopping(
        monitor="val_loss", patience=40, restore_best_weights=True, verbose=1
    )
    callbacks = [early_stop, ckpt]

    # Training schedule
    # schedule = [
    #     (128, 200),
    #     (256, 200),
    #     (512, 200),
    #     (1024, 200),
    #     (2048, 200),
    #     (4096, 400),
    # ]

    schedule = [(2048, 1400)]

    current_epoch = 0
    all_histories = []

    # Phased training
    for bs, ne in schedule:
        tf.keras.backend.set_value(model.optimizer.lr, 1e-3)
        start = current_epoch
        stop = current_epoch + ne
        print(f"\n--- Training epochs {start}→{stop} with batch_size={bs} ---")
        hist = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            initial_epoch=start,
            epochs=stop,
            batch_size=bs,
            callbacks=callbacks,
            verbose=1,
        )
        all_histories.append(hist)
        current_epoch = stop

    # Save final weights
    model.save_weights(os.path.join(save_dir, "model.weights.h5"))
    logging.info("Saved weights to %s", save_dir)

    # Metrics arrays
    train_loss = np.concatenate([h.history["loss"] for h in all_histories])
    val_loss = np.concatenate([h.history["val_loss"] for h in all_histories])
    train_acc = np.concatenate([h.history["accuracy"] for h in all_histories])
    val_acc = np.concatenate([h.history["val_accuracy"] for h in all_histories])
    np.save(os.path.join(save_dir, "train_loss.npy"), train_loss)
    np.save(os.path.join(save_dir, "val_loss.npy"), val_loss)
    np.save(os.path.join(save_dir, "train_accuracy.npy"), train_acc)
    np.save(os.path.join(save_dir, "val_accuracy.npy"), val_acc)

    plt.figure()
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

    # plot accuracy curve
    plt.figure()
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"))
    plt.close()
    # FLOPs and MACs
    flops = get_flops(model, [1, args.num_particles, x.shape[2]])
    macs = flops // 2
    logging.info("FLOPs per inference: %d", flops)
    logging.info("MACs per inference: %d", macs)
    print(f"FLOPs per inference: {flops}")
    print(f"MACs per inference: {macs}")

    # Inference timing
    _ = model(x_val[: args.batch_size])
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = model(x_val[: args.batch_size])
        times.append(time.perf_counter() - t0)
    avg_ns = np.mean(np.array(times) / args.batch_size) * 1e9
    logging.info("Avg inference time per event (ns): %.3f", avg_ns)
    print(f"Avg inference time per event: {avg_ns:.3f} ns")

    # Validation metrics
    preds = model.predict(x_val, batch_size=args.batch_size)
    val_acc = accuracy_score(np.argmax(y_val, 1), np.argmax(preds, 1))
    overall_auc = roc_auc_score(y_val, preds, average="macro", multi_class="ovo")
    logging.info("Validation Accuracy: %.4f, ROC AUC: %.4f", val_acc, overall_auc)
    print(f"Val Acc: {val_acc:.4f}, ROC AUC: {overall_auc:.4f}")

    # ROC curves plotting
    class_labels = ["g", "q", "W", "Z", "t"]
    plt.figure(figsize=(6, 6))
    for i, label in enumerate(class_labels):
        fpr_vals, tpr_vals, _ = roc_curve(y_val[:, i], preds[:, i])
        roc_auc_val = auc(fpr_vals, tpr_vals)
        plt.plot(fpr_vals, tpr_vals, label=f"{label} (AUC={roc_auc_val:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.title("ROC curves")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curves.png"))
    plt.close()

    # Background rejection @0.8 TPR
    rej_list = []
    scores = preds[:, 1:] / (preds[:, :1] + preds[:, 1:])
    scores = np.concatenate([preds[:, :1], scores], axis=1)
    for i, label in enumerate(class_labels[1:], start=1):
        mask = (y_val[:, 0] == 1) | (y_val[:, i] == 1)
        bin_y = (y_val[mask, i] == 1).astype(int)
        bin_s = scores[mask, i]
        fpr_vals, tpr_vals, _ = roc_curve(bin_y, bin_s)
        idx = np.argmin(np.abs(tpr_vals - 0.8))
        rej = 1.0 / fpr_vals[idx] if fpr_vals[idx] > 0 else np.inf
        logging.info("Background rejection @0.8 for %s: %.3f", label, rej)
        rej_list.append(rej)
    avg_rej = np.nanmean(rej_list)
    logging.info("Average background rejection @0.8 TPR: %.3f", avg_rej)


if __name__ == "__main__":
    main()
