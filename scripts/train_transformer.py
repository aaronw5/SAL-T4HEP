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
    elif dataset == "top":
        x_test = np.load(
            os.path.join(data_dir, "TopTagging/temp_feats/test_features.npy")
        )
        y_test = np.load(
            os.path.join(data_dir, "TopTagging/temp_labels/test_labels.npy")
        )
    else:  # jetclass
        x_test = np.load(os.path.join(data_dir, "test/features.npy"))
        y_test = np.load(os.path.join(data_dir, "test/labels.npy"))
    logging.info(
        "Loaded TEST arrays for %s: %s, %s", dataset, x_test.shape, y_test.shape
    )

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
    if dataset == "top":
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
        if dataset == "top":
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
    if dataset != "top":
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


# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Train a Standard Transformer on jet data with sorting and optional convolution"
    )
    p.add_argument(
        "--data_dir", required=True, help="Directory containing x and y .npy files"
    )
    p.add_argument("--dataset", choices=["hls4ml", "top", "jetclass"], default="hls4ml")
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
def main():
    args = parse_args()

    # pick num_particles & output_dim
    if args.dataset == "jetclass":
        num_particles = 150
        args.output_dim = 10
        loss_fn = "categorical_crossentropy"
    elif args.dataset == "top":
        num_particles = 150
        args.output_dim = 1
        loss_fn = "binary_crossentropy"
    else:  # hls4ml
        num_particles = args.num_particles
        args.output_dim = 5
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
    elif args.dataset == "top":
        x_train = np.load(
            os.path.join(args.data_dir, "TopTagging/temp_feats/train_features.npy")
        )
        y_train = np.load(
            os.path.join(args.data_dir, "TopTagging/temp_labels/train_labels.npy")
        )
        x_val = np.load(
            os.path.join(args.data_dir, "TopTagging/temp_feats/val_features.npy")
        )
        y_val = np.load(
            os.path.join(args.data_dir, "TopTagging/temp_labels/val_labels.npy")
        )
    else:  # jetclass
        x_train = np.load(os.path.join(args.data_dir, "train/features.npy"))
        y_train = np.load(os.path.join(args.data_dir, "train/labels.npy"))
        x_val = np.load(os.path.join(args.data_dir, "val/features.npy"))
        y_val = np.load(os.path.join(args.data_dir, "val/labels.npy"))

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

    # training schedule
    # schedule = [
    #     (128, 200),
    #     (256, 200),
    #     (512, 200),
    #     (1024, 200),
    #     (2048, 200),
    #     (4096, 400),
    # ]
    # reduced training schedule due to OOM
    schedule = [
        (128, 200),
        (256, 200),
        (512, 200),
        (1024, 200),
        (2048, 600),
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
        args.num_particles,
    )


if __name__ == "__main__":
    main()
