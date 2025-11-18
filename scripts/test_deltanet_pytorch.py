#!/usr/bin/env python
"""
Test a PyTorch DeltaNet model on jet datasets,
generating comprehensive metrics: accuracy, ROC curves, FLOPs, timing, etc.
Adapted from test.py for DeltaNet models.
"""
import os
import sys
import time
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# make project root importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.deltanetorignal import DeltaNet


class DeltaNetClassifier(nn.Module):
    """DeltaNet-based classifier for jet tagging (must match training script)"""
    
    def __init__(
        self,
        num_particles,
        feature_dim,
        d_model=16,
        num_heads=4,
        num_layers=1,
        output_dim=5,
        conv_size=4,
        use_short_conv=True,
        dropout=0.1
    ):
        super().__init__()
        
        self.num_particles = num_particles
        self.feature_dim = feature_dim
        self.d_model = d_model
        
        # Input embedding
        self.input_proj = nn.Linear(feature_dim, d_model)
        
        # DeltaNet layers
        self.layers = nn.ModuleList([
            DeltaNet(
                mode='chunk',
                d_model=d_model,
                hidden_size=d_model,
                num_heads=num_heads,
                use_beta=True,
                use_gate=False,
                use_short_conv=use_short_conv,
                conv_size=conv_size,
                qk_activation='silu',
                qk_norm='l2'
            )
            for _ in range(num_layers)
        ])
        
        # Aggregation and classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x: (batch, num_particles, feature_dim)
        x = self.input_proj(x)  # (batch, num_particles, d_model)
        x = self.relu(x)
        
        # Apply DeltaNet layers
        for layer in self.layers:
            residual = x
            x, _, _ = layer(x)
            x = x + residual  # residual connection
        
        # Max pooling aggregation
        x = torch.max(x, dim=1)[0]  # (batch, d_model)
        
        # Classification head
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def apply_sorting(x, sort_by):
    """Sort particles according to specified method"""
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


def count_flops_with_profiler(model, input_tensor, device):
    """
    Measure ACTUAL FLOPs using PyTorch profiler.
    This counts real operations, not estimates.
    """
    import torch.autograd.profiler as profiler
    
    model.eval()
    
    try:
        # Run profiler
        with profiler.profile(record_shapes=True, use_cuda=(device.type == 'cuda')) as prof:
            with torch.no_grad():
                _ = model(input_tensor[:1])  # Single sample
        
        # Sum up FLOPs from all operations
        total_flops = 0
        for evt in prof.key_averages():
            # Count FLOPs from matrix multiplications and convolutions
            if 'mul' in evt.key.lower() or 'addmm' in evt.key.lower() or 'mm' in evt.key.lower():
                # These are the actual compute operations
                total_flops += evt.flops
        
        # If profiler doesn't give FLOPs, try alternative method
        if total_flops == 0:
            logging.warning("PyTorch profiler didn't capture FLOPs. Trying thop library...")
            try:
                from thop import profile as thop_profile
                input_sample = input_tensor[:1].clone()
                flops, params = thop_profile(model, inputs=(input_sample,), verbose=False)
                return int(flops), int(params)
            except ImportError:
                logging.warning("thop library not available. FLOPs cannot be measured accurately.")
                return None, None
            except Exception as e:
                logging.warning(f"thop profiling failed: {e}")
                return None, None
        
        return int(total_flops), sum(p.numel() for p in model.parameters())
    
    except Exception as e:
        logging.error(f"FLOPs profiling failed: {e}")
        return None, None


def profile_gpu_memory(model, input_tensor, device):
    """Profile GPU memory usage during inference"""
    if device.type == 'cpu':
        return 0.0, 0.0
    
    try:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        
        # Warm-up
        with torch.no_grad():
            _ = model(input_tensor[:1])
        
        torch.cuda.reset_peak_memory_stats(device)
        
        # Actual profiling
        with torch.no_grad():
            _ = model(input_tensor)
        
        current_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        
        return current_mb, peak_mb
    except Exception as e:
        logging.warning(f"Could not profile GPU memory: {e}")
        return 0.0, 0.0


def process_directory(data_dir, save_dir, sort_by, num_particles, batch_size=4096):
    """Process a single directory with the given parameters"""
    
    # Setup logging to append to existing train.log
    log_file = os.path.join(save_dir, "train.log")
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    logging.info("=" * 70)
    logging.info("Starting testing phase for DeltaNet PyTorch model")
    logging.info("=" * 70)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)
    
    # DeltaNet requires bfloat16 precision
    dtype = torch.bfloat16
    logging.info("Using dtype: %s (required by DeltaNet)", dtype)
    
    # Load test data
    x_file = f"x_val_robust_{num_particles}const_ptetaphi.npy"
    y_file = f"y_val_robust_{num_particles}const_ptetaphi.npy"
    x_test = np.load(os.path.join(data_dir, x_file))
    y_test = np.load(os.path.join(data_dir, y_file))
    logging.info("Loaded TEST arrays: %s, %s", x_file, y_file)
    logging.info("Test data shapes: x=%s, y=%s", x_test.shape, y_test.shape)
    
    # Apply sorting
    x_test = apply_sorting(x_test, sort_by)
    logging.info("Applied '%s' sorting to TEST set", sort_by)
    
    # Build model architecture
    feature_dim = x_test.shape[2]
    output_dim = y_test.shape[1]
    
    model = DeltaNetClassifier(
        num_particles=num_particles,
        feature_dim=feature_dim,
        d_model=16,
        num_heads=4,
        num_layers=1,
        output_dim=output_dim,
        conv_size=4,
        use_short_conv=True,
        dropout=0.1
    ).to(device).to(dtype)  # Convert model to bfloat16
    
    # Load trained weights
    model_path = os.path.join(save_dir, "best_model.pt")
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        logger.removeHandler(file_handler)
        file_handler.close()
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logging.info("Loaded model from %s", model_path)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logging.info("Total parameters: %d", total_params)
    
    # FLOPs measurement using PyTorch profiler
    print(f"\n  Measuring FLOPs with PyTorch profiler...")
    x_test_tensor = torch.tensor(x_test, dtype=dtype).to(device)
    flops, params_check = count_flops_with_profiler(model, x_test_tensor[:100], device)
    
    if flops is not None:
        macs = flops // 2
        logging.info("FLOPs per inference (measured): %d", flops)
        logging.info("MACs per inference (measured): %d", macs)
        print(f"  FLOPs per inference: {flops:,}")
        print(f"  MACs per inference: {macs:,}")
    else:
        logging.warning("Could not measure FLOPs accurately. Skipping FLOPs reporting.")
        print("  FLOPs measurement not available")
        macs = None
    
    # Inference timing (reuse x_test_tensor from FLOPs measurement)
    
    # Warm-up
    with torch.no_grad():
        _ = model(x_test_tensor[:batch_size])
    
    # Time inference
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x_test_tensor[:batch_size])
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    
    avg_time_per_event = np.mean(times) / batch_size
    avg_ns = avg_time_per_event * 1e9
    logging.info("Avg inference time / event: %.2f ns", avg_ns)
    print(f"  Avg inference time/event: {avg_ns:.2f} ns")
    
    # GPU memory profiling
    current_gpu_mb, peak_gpu_mb = profile_gpu_memory(model, x_test_tensor[:batch_size], device)
    if current_gpu_mb > 0:
        logging.info("GPU memory — current: %.1f MB, peak: %.1f MB", current_gpu_mb, peak_gpu_mb)
        print(f"  GPU memory current: {current_gpu_mb:.1f} MB, peak: {peak_gpu_mb:.1f} MB")
    
    # Predictions
    print(f"  Running inference on full test set ({len(x_test)} samples)...")
    all_preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            batch = x_test_tensor[i:i+batch_size]
            preds = model(batch)
            # Convert to float32 for softmax and numpy
            preds_fp32 = preds.float()
            all_preds.append(torch.softmax(preds_fp32, dim=1).cpu().numpy())
    
    preds = np.vstack(all_preds)
    
    # Metrics
    acc = accuracy_score(np.argmax(y_test, 1), np.argmax(preds, 1))
    auc_m = roc_auc_score(y_test, preds, average="macro", multi_class="ovo")
    logging.info("Test Accuracy: %.4f, ROC AUC: %.4f", acc, auc_m)
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  Test ROC AUC: {auc_m:.4f}")
    
    # Plot training loss curves (if available)
    train_loss_file = os.path.join(save_dir, "train_loss.npy")
    val_loss_file = os.path.join(save_dir, "val_loss.npy")
    
    if os.path.exists(train_loss_file) and os.path.exists(val_loss_file):
        train_loss = np.load(train_loss_file)
        val_loss = np.load(val_loss_file)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_loss, label='Training Loss', linewidth=2)
        plt.plot(val_loss, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training History - Loss', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Plot accuracy if available
        train_acc_file = os.path.join(save_dir, "train_accuracy.npy")
        val_acc_file = os.path.join(save_dir, "val_accuracy.npy")
        if os.path.exists(train_acc_file) and os.path.exists(val_acc_file):
            train_acc = np.load(train_acc_file)
            val_acc = np.load(val_acc_file)
            
            plt.subplot(1, 2, 2)
            plt.plot(train_acc, label='Training Accuracy', linewidth=2)
            plt.plot(val_acc, label='Validation Accuracy', linewidth=2)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Accuracy', fontsize=12)
            plt.title('Training History - Accuracy', fontsize=14)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        loss_curve_path = os.path.join(save_dir, "test_loss_curves.png")
        plt.savefig(loss_curve_path, dpi=150)
        plt.close()
        logging.info("Saved loss curves to %s", loss_curve_path)
        print(f"  Saved loss curves to: test_loss_curves.png")
    
    # ROC curves
    class_labels = ["g", "q", "W", "Z", "t"]
    plt.figure(figsize=(8, 8))
    one_over_fpr = {}
    
    for i, label in enumerate(class_labels):
        fpr_vals, tpr_vals, _ = roc_curve(y_test[:, i], preds[:, i])
        roc_auc_val = auc(fpr_vals, tpr_vals)
        logging.info("ROC AUC for %s: %.4f", label, roc_auc_val)
        plt.plot(fpr_vals, tpr_vals, label=f"{label} (AUC={roc_auc_val:.3f})", linewidth=2)
        
        if np.max(tpr_vals) >= 0.8:
            fpr_t = np.interp(0.8, tpr_vals, fpr_vals)
            one_over_fpr[label] = 1.0 / fpr_t if fpr_t > 0 else np.nan
            plt.plot(fpr_t, 0.8, "o", markersize=8)
    
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"DeltaNet ROC Curves ({num_particles} particles, {sort_by})", fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(save_dir, "roc_curves.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    logging.info("Saved ROC curves to %s", roc_path)
    
    for label, val in one_over_fpr.items():
        logging.info("1/FPR @0.8 TPR for %s: %.3f", label, val)
    avg_one_over = np.nanmean(list(one_over_fpr.values()))
    logging.info("Avg 1/FPR @0.8 TPR: %.3f", avg_one_over)
    print(f"  Avg 1/FPR@0.8: {avg_one_over:.3f}")
    
    # Background rejection (g+q as background for W, Z, t)
    print(f"  Background Rejection (g+q as background):")
    rej_list = []
    for i, label in enumerate(class_labels[2:], start=2):  # W, Z, t
        mask = (y_test[:, 0] == 1) | (y_test[:, 1] == 1) | (y_test[:, i] == 1)
        bin_y = (y_test[mask, i] == 1).astype(int)
        bin_s = preds[mask, i]
        fpr_vals, tpr_vals, _ = roc_curve(bin_y, bin_s)
        idx = np.argmin(np.abs(tpr_vals - 0.8))
        rej = 1.0 / fpr_vals[idx] if fpr_vals[idx] > 0 else np.inf
        logging.info("Background rejection @0.8 for %s: %.3f", label, rej)
        print(f"    {label}: {rej:.3f}")
        rej_list.append(rej)
    avg_bkg_rej = np.nanmean(rej_list)
    logging.info("Avg background rejection @0.8: %.3f", avg_bkg_rej)
    print(f"    Average: {avg_bkg_rej:.3f}")
    
    # Save results summary with ALL measured metrics
    results_file = os.path.join(save_dir, "RESULTS.txt")
    with open(results_file, "w") as f:
        f.write(f"DeltaNet PyTorch Results ({num_particles} particles, {sort_by} sorting) - hls4ml test set\n")
        f.write("=" * 70 + "\n\n")
        
        # Model architecture
        f.write("Model Architecture:\n")
        f.write(f"  Total Parameters: {total_params:,}\n")
        if flops is not None:
            f.write(f"  FLOPs per inference: {flops:,} (measured via PyTorch profiler)\n")
            f.write(f"  MACs per inference: {macs:,}\n")
        f.write(f"  Inference time per event: {avg_ns:.2f} ns ({avg_ns/1000:.2f} µs)\n")
        if current_gpu_mb > 0:
            f.write(f"  GPU memory (current/peak): {current_gpu_mb:.1f} MB / {peak_gpu_mb:.1f} MB\n")
        
        # Classification performance
        f.write(f"\nClassification Performance:\n")
        f.write(f"  Test Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")
        f.write(f"  Test ROC AUC (macro): {auc_m:.4f}\n")
        f.write(f"  Avg 1/FPR@0.8 TPR: {avg_one_over:.3f}\n")
        
        # Per-class metrics
        f.write(f"\nPer-class 1/FPR@0.8 TPR:\n")
        for label, val in one_over_fpr.items():
            f.write(f"  {label}: {val:.3f}\n")
        
        # Background rejection
        f.write(f"\nBackground Rejection @0.8 TPR (g+q as background):\n")
        for i, label in enumerate(class_labels[2:], start=2):
            f.write(f"  {label}: {rej_list[i-2]:.3f}\n")
        f.write(f"  Average: {avg_bkg_rej:.3f}\n")
        
        # Methodology note
        f.write(f"\n" + "=" * 70 + "\n")
        f.write("Methodology:\n")
        f.write("- All metrics calculated using sklearn.metrics (accuracy_score, roc_curve, auc, roc_auc_score)\n")
        if flops is not None:
            f.write("- FLOPs measured using PyTorch autograd profiler (actual operations counted)\n")
        f.write("- Inference time: average of 20 runs on batch_size=4096 with CUDA synchronization\n")
        f.write("- GPU memory measured via torch.cuda.memory_allocated() and max_memory_allocated()\n")
        f.write(f"- Test set: 260,000 samples from x_val_robust_{num_particles}const_ptetaphi.npy\n")
    
    logging.info("Saved results to %s", results_file)
    logging.info("Testing complete!")
    print(f"  Results saved to: {results_file}")
    
    # Remove handler
    logger.removeHandler(file_handler)
    file_handler.close()


def main():
    p = argparse.ArgumentParser(description="Test PyTorch DeltaNet model")
    p.add_argument("--data_dir", required=True, help="Path to data directory")
    p.add_argument("--save_dir", required=True, help="Path to model directory")
    p.add_argument("--sort_by", choices=["pt", "eta", "phi", "delta_R", "kt"], default="kt")
    p.add_argument("--num_particles", type=int, default=150, help="Number of particles")
    p.add_argument("--batch_size", type=int, default=4096, help="Batch size for inference")
    args = p.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Testing DeltaNet PyTorch Model")
    print(f"{'='*70}")
    print(f"Model directory: {args.save_dir}")
    print(f"Data directory: {args.data_dir}")
    print(f"Sort by: {args.sort_by}")
    print(f"Num particles: {args.num_particles}\n")
    
    process_directory(
        args.data_dir,
        args.save_dir,
        args.sort_by,
        args.num_particles,
        args.batch_size
    )
    
    print(f"\n{'='*70}")
    print("Testing complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

