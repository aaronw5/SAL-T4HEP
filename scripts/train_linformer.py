#!/usr/bin/env python
import os
import sys

# ─── make the parent directory (project root) importable ─────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ─────────────────────────────────────────────────────────────────────────────

import time
import argparse
import logging
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import fastjet as fj
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score

import matplotlib.pyplot as plt

# import your model definitions
from models.Linformer import (
    AggregationLayer,
    ClusteredLinformerAttention,
    LinformerTransformerBlock,
    build_linformer_transformer_classifier,
)
import tensorflow as tf
import numpy as np

def get_flops(model, input_shapes):
    """
    Computes FLOPs for a 2-input model.

    Args:
      model: tf.keras.Model expecting [features, mask]
      input_shapes: [
         (1, seq_len, feat_dim),  # features shape
         (1, seq_len)             # mask shape
      ]
    """
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2_as_graph,
    )

    # 1) build specs for each input
    spec_x   = tf.TensorSpec(input_shapes[0], tf.float32)
    spec_mask= tf.TensorSpec(input_shapes[1], tf.bool)       # or float32 if you cast mask to float

    # 2) wrap model call
    @tf.function
    def model_fn(x, mask):
        return model([x, mask])

    # 3) get concrete function
    concrete = model_fn.get_concrete_function(spec_x, spec_mask)

    # 4) freeze and profile
    frozen, graph_def = convert_variables_to_constants_v2_as_graph(concrete)
    with tf.Graph().as_default() as g:
        tf.compat.v1.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts     = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        prof     = tf.compat.v1.profiler.profile(
            graph=g, run_meta=run_meta, cmd='op', options=opts
        )
        return prof.total_float_ops


def profile_gpu_memory_during_inference(
    model: tf.keras.Model,
    input_data: np.ndarray,
    mask_data: np.ndarray,
) -> tuple[float, float]:
    """
    Profiles GPU memory for a 2-input model.
    """
    # reset memory stats
    tf.config.experimental.reset_memory_stats('GPU:0')

    @tf.function
    def infer(x, mask):
        return model([x, mask], training=False)

    # warmup
    _ = infer(input_data[:1], mask_data[:1])
    # actual
    _ = infer(input_data,    mask_data)

    mem = tf.config.experimental.get_memory_info('GPU:0')
    return mem['current']/(1024**2), mem['peak']/(1024**2)



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
        
def run_testing(
    model,
    data_dir,
    save_dir,
    sort_by,
    batch_size,
    cluster_R=0.4,
    cluster_batch_size=1024,
):
    """Run testing on the trained model"""
    logging.info("Starting testing phase...")

    # Load and sort test data
    print(model.input_shape)
    num_particles = model.input_shape[0][1]
    feat_dim = model.input_shape[0][2]
    x_file = f"x_val_robust_{num_particles}const_ptetaphi.npy"
    y_file = f"y_val_robust_{num_particles}const_ptetaphi.npy"
    x = np.load(os.path.join(data_dir, x_file))
    y = np.load(os.path.join(data_dir, y_file))
    logging.info("Loaded TEST arrays: %s, %s", x_file, y_file)

    x = apply_sorting(x, sort_by, cluster_R, cluster_batch_size)
    mask = np.any(x != 0.0, axis=-1)  # (N, num_particles)
    logging.info("Applied '%s' sorting to TEST set", sort_by)

    # FLOPs & timing
    flops = get_flops(
        model,
        [
            (1, num_particles, feat_dim),
            (1, num_particles),
        ]
    )
    macs = flops // 2
    logging.info("FLOPs per inference: %d", flops)
    logging.info("MACs per inference: %d", macs)

    # inference timing
    _ = model.predict([x[:batch_size], mask[:batch_size]], batch_size=batch_size)
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = model.predict([x[:batch_size], mask[:batch_size]], batch_size=batch_size)
        times.append(time.perf_counter() - t0)
    avg_ns = np.mean(times) / batch_size * 1e9
    logging.info("Avg inference time / event: %.2f ns", avg_ns)

    # Profile GPU memory usage
    current_mb, peak_mb = profile_gpu_memory_during_inference(
        model,
        x[:batch_size],
        mask[:batch_size]
    )
    logging.info(
        "GPU memory — current: %.1f MB, peak: %.1f MB", current_mb, peak_mb
    )

    # final preds & metrics
    preds = model.predict([x, mask], batch_size=batch_size)
    acc = accuracy_score(np.argmax(y,1), np.argmax(preds,1))
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


def parse_args():
    p = argparse.ArgumentParser(
        description="Train a Linformer on jet data with sorting reuse"
    )
    p.add_argument(
        "--data_dir", required=True, help="Directory containing x and y .npy files"
    )
    p.add_argument("--save_dir", required=True, help="Base directory for outputs")
    p.add_argument(
        "--cluster_E", action="store_true", help="Use clustered projection for keys"
    )
    p.add_argument(
        "--cluster_F", action="store_true", help="Use clustered projection for values"
    )
    p.add_argument(
        "--share_EF",
        action="store_true",
        help="Share E and F projection matrices (E=F)",
    )
    p.add_argument(
        "--convolution", action="store_true", help="Use convolution on attention scores"
    )
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--num_epochs", type=int, default=1000)
    p.add_argument("--d_model", type=int, default=16)
    p.add_argument("--d_ff", type=int, default=16)
    p.add_argument("--output_dim", type=int, default=16)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument(
        "--proj_dim", type=int, default=4, help="Linformer projection dimension"
    )
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--num_particles", type=int, required=True)
    p.add_argument(
        "--sort_by",
        choices=["pt", "eta", "phi", "delta_R", "kt", "cluster"],
        default="pt",
        help="How to sort particles",
    )
    p.add_argument("--cluster_R", type=float, default=0.4)
    p.add_argument("--cluster_batch_size", type=int, default=1024)
    return p.parse_args()


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

    x_filename = f"x_train_robust_{args.num_particles}const_ptetaphi.npy"
    y_filename = f"y_train_robust_{args.num_particles}const_ptetaphi.npy"
    if args.sort_by == "cluster":
        args.data_dir = args.data_dir.replace("processed", "sorted_by_cluster")
        logging.info("Loaded pre-sorted data from %s", args.data_dir)

    x_path = os.path.join(args.data_dir, x_filename)
    y_path = os.path.join(args.data_dir, y_filename)
    x = np.load(x_path)
    y = np.load(y_path)
    logging.info("Loaded labels from %s", y_path)

    if args.sort_by == "pt":
        key = x[:, :, 0]
    elif args.sort_by == "eta":
        key = x[:, :, 1]
    elif args.sort_by == "phi":
        key = x[:, :, 2]
    elif args.sort_by == "delta_R":
        key = np.sqrt(x[:, :, 1] ** 2 + x[:, :, 2] ** 2)
    elif args.sort_by == "kt":
        key = x[:, :, 0] * np.sqrt(x[:, :, 1] ** 2 + x[:, :, 2] ** 2)
        idx = np.argsort(key, axis=1)[:, ::-1]
        x = np.take_along_axis(x, idx[:, :, None], axis=1)

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=args.val_split, random_state=42, shuffle=True
    )
    logging.info("Split into train %s and val %s", x_train.shape, x_val.shape)

    model = build_linformer_transformer_classifier(
        args.num_particles,
        x.shape[2],
        d_model=args.d_model,
        d_ff=args.d_ff,
        output_dim=args.output_dim,
        num_heads=args.num_heads,
        proj_dim=args.proj_dim,
        cluster_E=args.cluster_E,
        cluster_F=args.cluster_F,
        share_EF=args.share_EF,
        convolution=args.convolution,
        conv_filter_heights=[1, 3, 5],
        vertical_stride=1,
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

    ckpt = ModelCheckpoint(
        os.path.join(save_dir, "best.weights.h5"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )
    early_stop = EarlyStopping(
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

    current_epoch = 0
    all_histories = []
    callbacks = [ckpt]

    mask_train = tf.reduce_any(x_train != 0.0, axis=-1)  # (n,150) bool
    mask_val = tf.reduce_any(x_val != 0.0, axis=-1)
    for bs, ne in schedule:
        tf.keras.backend.set_value(model.optimizer.lr, 1e-3)
        start = current_epoch
        stop = current_epoch + ne
        print(f"\n--- Training epochs {start}→{stop} with batch_size={bs} ---")
        hist = model.fit(
            [x_train, mask_train],  # <-- features + mask
            y_train,
            validation_data=([x_val, mask_val], y_val),  # <-- features + mask
            initial_epoch=start,
            epochs=stop,
            batch_size=bs,
            callbacks=callbacks,
            verbose=1,
        )
        all_histories.append(hist)
        current_epoch = stop

    # save final weights
    weights_file = os.path.join(save_dir, "model.weights.h5")
    model.save_weights(weights_file)
    logging.info("Saved weights to %s", weights_file)

    # concatenate and save training metrics across all phases
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
    

    run_testing(
        model=model,
        data_dir=args.data_dir,
        save_dir=save_dir,
        sort_by=args.sort_by,
        batch_size=args.batch_size,
        cluster_R=args.cluster_R,
        cluster_batch_size=args.cluster_batch_size,
    )


if __name__ == "__main__":
    main()
