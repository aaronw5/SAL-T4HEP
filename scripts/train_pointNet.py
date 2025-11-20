#!/usr/bin/env python
"""
Train a PointNet classifier on jet datasets (hls4ml, top, QG, or jetclass),
profile performance, and generate metrics/plots.
"""
import os
import sys
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

# ─── make the parent directory (project root) importable ─────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.pointNet import build_pointnet_classifier


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
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
    except Exception:
        return 0.0, 0.0

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
        labels = [f"label_{i}" for i in range(preds.shape[1])]

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
    plt.title("ROC curves (PointNet)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curves_pointnet.png"))
    plt.close()

    for lab, val in one_over_fpr.items():
        logging.info("1/FPR@0.8 for %s: %.3f", lab, val)
    if one_over_fpr:
        logging.info("Avg 1/FPR@0.8: %.3f", np.nanmean(list(one_over_fpr.values())))


# ---------------------------
# Argument parsing and main
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train PointNet on jet data")
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
    p.add_argument("--dropout", type=float, default=0.3)
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

    # build and compile model
    model = build_pointnet_classifier(
        num_particles=num_particles,
        feature_dim=x_train.shape[2],
        output_dim=output_dim,
        dropout_rate=args.dropout,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=loss_fn,
        metrics=["accuracy"],
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

    # training schedule (kept conservative)
    schedule = [
        (512, 50)
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


