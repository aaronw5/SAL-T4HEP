#!/usr/bin/env python3
"""
Evaluate a trained PyTorch HEPT classifier on hls4ml jet tagging.
"""
import argparse
import logging
import os
import sys
import time

import matplotlib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, auc, roc_auc_score, roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.HEPT import HEPTClassifier


DEFAULT_LABELS = ["q", "g", "W", "Z", "t"]


def apply_sorting(x: np.ndarray, sort_by: str) -> np.ndarray:
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


def count_flops_with_profiler(model: torch.nn.Module, input_tensor: torch.Tensor, device: torch.device):
    import torch.autograd.profiler as profiler

    model.eval()
    try:
        with profiler.profile(
            record_shapes=True,
            with_flops=True,
            use_cuda=(device.type == "cuda"),
        ) as prof:
            with torch.no_grad():
                _ = model(input_tensor[:1])
        total_flops = 0
        for event in prof.key_averages():
            total_flops += int(getattr(event, "flops", 0) or 0)
        if total_flops > 0:
            return total_flops, sum(p.numel() for p in model.parameters())
    except Exception as exc:
        logging.warning("PyTorch profiler FLOPs failed: %s", exc)

    try:
        from thop import profile as thop_profile

        flops, params = thop_profile(model, inputs=(input_tensor[:1],), verbose=False)
        return int(flops), int(params)
    except Exception as exc:
        logging.warning("thop FLOPs failed: %s", exc)
        return None, None


def profile_gpu_memory(model: torch.nn.Module, input_tensor: torch.Tensor, device: torch.device):
    if device.type != "cuda":
        return 0.0, 0.0
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = model(input_tensor[:1])
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = model(input_tensor)
        current_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        return current_mb, peak_mb
    except Exception as exc:
        logging.warning("GPU memory profiling failed: %s", exc)
        return 0.0, 0.0


def load_checkpoint(path: str, device: torch.device):
    return torch.load(path, map_location=device)


def evaluate_run(
    data_dir: str,
    save_dir: str,
    num_particles: int,
    sort_by_override: str | None,
    batch_size: int,
    checkpoint_name: str,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "test_hept.log")
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(save_dir, checkpoint_name)
    checkpoint = load_checkpoint(ckpt_path, device)
    model_config = checkpoint["model_config"]
    sort_by = sort_by_override or checkpoint.get("sort_by", "kt")
    class_labels = checkpoint.get("class_labels", DEFAULT_LABELS)

    x_test = np.load(os.path.join(data_dir, f"x_val_robust_{num_particles}const_ptetaphi.npy"))
    y_test = np.load(os.path.join(data_dir, f"y_val_robust_{num_particles}const_ptetaphi.npy"))
    x_test = apply_sorting(x_test, sort_by)
    logging.info("Loaded test arrays x=%s y=%s sort_by=%s", x_test.shape, y_test.shape, sort_by)

    model = HEPTClassifier(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    x_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)

    flops, _ = count_flops_with_profiler(model, x_tensor[: min(len(x_tensor), batch_size)], device)
    macs = flops // 2 if flops is not None else None

    with torch.no_grad():
        _ = model(x_tensor[: min(len(x_tensor), batch_size)])
    times = []
    repeats = 20
    for _ in range(repeats):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x_tensor[: min(len(x_tensor), batch_size)])
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    avg_ns = np.mean(times) / min(len(x_tensor), batch_size) * 1e9
    current_gpu_mb, peak_gpu_mb = profile_gpu_memory(
        model, x_tensor[: min(len(x_tensor), batch_size)], device
    )

    preds_list = []
    with torch.no_grad():
        for start in range(0, len(x_tensor), batch_size):
            logits = model(x_tensor[start : start + batch_size])
            preds_list.append(torch.softmax(logits, dim=1).cpu().numpy())
    preds = np.vstack(preds_list)

    acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(preds, axis=1))
    auc_m = roc_auc_score(y_test, preds, average="macro", multi_class="ovo")
    logging.info("Accuracy %.4f ROC-AUC %.4f", acc, auc_m)

    plt.figure(figsize=(8, 8))
    one_over_fpr = {}
    for i, label in enumerate(class_labels):
        fpr_vals, tpr_vals, _ = roc_curve(y_test[:, i], preds[:, i])
        roc_auc_val = auc(fpr_vals, tpr_vals)
        plt.plot(fpr_vals, tpr_vals, label=f"{label} (AUC={roc_auc_val:.3f})", linewidth=2)
        if np.max(tpr_vals) >= 0.8:
            fpr_t = np.interp(0.8, tpr_vals, fpr_vals)
            one_over_fpr[label] = 1.0 / fpr_t if fpr_t > 0 else np.nan
            plt.plot(fpr_t, 0.8, "o", markersize=6)
        logging.info("Class %s AUC %.4f", label, roc_auc_val)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"HEPT ROC Curves ({num_particles} particles, {sort_by})")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curves.png"), dpi=150)
    plt.close()

    rejections = []
    signal_labels = class_labels[2:]
    for class_index, label in enumerate(signal_labels, start=2):
        mask = (y_test[:, 0] == 1) | (y_test[:, 1] == 1) | (y_test[:, class_index] == 1)
        binary_y = (y_test[mask, class_index] == 1).astype(int)
        binary_score = preds[mask, class_index]
        fpr_vals, tpr_vals, _ = roc_curve(binary_y, binary_score)
        idx = np.argmin(np.abs(tpr_vals - 0.8))
        rejection = 1.0 / fpr_vals[idx] if fpr_vals[idx] > 0 else np.inf
        rejections.append(rejection)
        logging.info("Background rejection @0.8 for %s: %.3f", label, rejection)
    avg_one_over = np.nanmean(list(one_over_fpr.values())) if one_over_fpr else float("nan")
    avg_bkg_rej = np.nanmean(rejections) if rejections else float("nan")

    train_loss_path = os.path.join(save_dir, "train_loss.npy")
    val_loss_path = os.path.join(save_dir, "val_loss.npy")
    train_acc_path = os.path.join(save_dir, "train_accuracy.npy")
    val_acc_path = os.path.join(save_dir, "val_accuracy.npy")
    if all(os.path.exists(path) for path in [train_loss_path, val_loss_path]):
        train_loss = np.load(train_loss_path)
        val_loss = np.load(val_loss_path)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_loss, label="Train")
        plt.plot(val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        if all(os.path.exists(path) for path in [train_acc_path, val_acc_path]):
            train_acc = np.load(train_acc_path)
            val_acc = np.load(val_acc_path)
            plt.subplot(1, 2, 2)
            plt.plot(train_acc, label="Train")
            plt.plot(val_acc, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("Accuracy")
            plt.grid(True, alpha=0.3)
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "test_loss_curves.png"), dpi=150)
        plt.close()

    results_path = os.path.join(save_dir, "RESULTS.txt")
    with open(results_path, "w", encoding="utf-8") as handle:
        handle.write(f"HEPT Results ({num_particles} particles, {sort_by} sorting) - hls4ml test set\n")
        handle.write("=" * 70 + "\n\n")
        handle.write("Model Architecture:\n")
        handle.write(f"  Total Parameters: {total_params:,}\n")
        if flops is not None:
            handle.write(f"  FLOPs per inference: {flops:,}\n")
            handle.write(f"  MACs per inference: {macs:,}\n")
        handle.write(f"  Inference time per event: {avg_ns:.2f} ns ({avg_ns / 1000.0:.2f} us)\n")
        if current_gpu_mb > 0:
            handle.write(f"  GPU memory (current/peak): {current_gpu_mb:.1f} MB / {peak_gpu_mb:.1f} MB\n")
        handle.write("\nClassification Performance:\n")
        handle.write(f"  Test Accuracy: {acc:.4f} ({acc * 100:.2f}%)\n")
        handle.write(f"  Test ROC AUC (macro): {auc_m:.4f}\n")
        handle.write(f"  Avg 1/FPR@0.8 TPR: {avg_one_over:.3f}\n")
        handle.write("\nPer-class 1/FPR@0.8 TPR:\n")
        for label in class_labels:
            if label in one_over_fpr:
                handle.write(f"  {label}: {one_over_fpr[label]:.3f}\n")
        handle.write("\nBackground Rejection @0.8 TPR (q+g as background):\n")
        for label, rejection in zip(signal_labels, rejections):
            handle.write(f"  {label}: {rejection:.3f}\n")
        handle.write(f"  Average: {avg_bkg_rej:.3f}\n")

    print(f"Checkpoint: {ckpt_path}")
    print(f"Test accuracy: {acc:.4f}")
    print(f"Test ROC AUC: {auc_m:.4f}")
    if flops is not None:
        print(f"FLOPs per inference: {flops:,}")
    print(f"Inference time/event: {avg_ns:.2f} ns")
    if current_gpu_mb > 0:
        print(f"GPU memory current/peak: {current_gpu_mb:.1f} MB / {peak_gpu_mb:.1f} MB")
    print(f"Average 1/FPR@0.8: {avg_one_over:.3f}")
    print(f"Average background rejection@0.8: {avg_bkg_rej:.3f}")
    print(f"Saved summary to: {results_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test HEPT on hls4ml jet tagging")
    parser.add_argument("--data_dir", required=True, help="Directory containing hls4ml .npy files")
    parser.add_argument("--save_dir", required=True, help="Directory containing HEPT checkpoints")
    parser.add_argument("--num_particles", type=int, default=150, help="Number of jet constituents")
    parser.add_argument("--sort_by", choices=["pt", "eta", "phi", "delta_R", "kt"], default=None)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--checkpoint_name", default="best_model.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_run(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        num_particles=args.num_particles,
        sort_by_override=args.sort_by,
        batch_size=args.batch_size,
        checkpoint_name=args.checkpoint_name,
    )


if __name__ == "__main__":
    main()
