#!/usr/bin/env python3
"""
Train a PyTorch HEPT classifier for hls4ml jet tagging.

The attention core is adapted from the HEPT implementation in /j-jepa-vol/HEPT-Zihan
and wrapped to follow SAL-T4HEP's hls4ml training workflow.
"""
import argparse
import logging
import os
import random
import sys
import time

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.HEPT import HEPTClassifier


CLASS_LABELS = ["q", "g", "W", "Z", "t"]


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


def one_hot_to_index(y: np.ndarray) -> np.ndarray:
    return np.argmax(y, axis=1).astype(np.int64)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch_idx: int,
    num_epochs: int,
    log_interval: int,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    num_batches = len(loader)
    for batch_idx, (batch_x, batch_y) in enumerate(loader, start=1):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss detected at epoch={epoch_idx + 1} batch={batch_idx}: {loss.item()}"
            )
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)
        total_correct += (logits.argmax(dim=1) == batch_y).sum().item()
        total_examples += batch_x.size(0)
        if batch_idx == 1 or batch_idx % max(1, log_interval) == 0 or batch_idx == num_batches:
            running_loss = total_loss / total_examples
            running_acc = total_correct / total_examples
            print(
                f"[Epoch {epoch_idx + 1:03d}/{num_epochs}] "
                f"batch {batch_idx:04d}/{num_batches:04d} "
                f"running_loss={running_loss:.5f} running_acc={running_acc:.4f}",
                flush=True,
            )
    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        total_loss += loss.item() * batch_x.size(0)
        total_correct += (logits.argmax(dim=1) == batch_y).sum().item()
        total_examples += batch_x.size(0)
    return total_loss / total_examples, total_correct / total_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HEPT on hls4ml jet tagging")
    parser.add_argument("--data_dir", required=True, help="Directory containing hls4ml .npy files")
    parser.add_argument("--save_dir", required=True, help="Base output directory")
    parser.add_argument("--num_particles", type=int, default=150, help="Number of jet constituents")
    parser.add_argument("--sort_by", choices=["pt", "eta", "phi", "delta_R", "kt"], default="kt")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--block_size", type=int, default=8)
    parser.add_argument("--n_hashes", type=int, default=4)
    parser.add_argument("--num_regions", type=int, default=16)
    parser.add_argument("--num_w_per_dist", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--aggregation", choices=["max", "mean"], default="max")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--log_interval",
        type=int,
        default=20,
        help="Print running train metrics every N batches",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    save_root = os.path.join(args.save_dir, str(args.num_particles), args.sort_by)
    trial = 0
    while True:
        candidate = os.path.join(save_root, f"trial-{trial}")
        time.sleep(random.randint(1, 2))
        if not os.path.isdir(candidate):
            save_dir = candidate
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    x = np.load(os.path.join(args.data_dir, f"x_train_robust_{args.num_particles}const_ptetaphi.npy"))
    y = np.load(os.path.join(args.data_dir, f"y_train_robust_{args.num_particles}const_ptetaphi.npy"))
    non_finite_x = np.size(x) - np.isfinite(x).sum()
    non_finite_y = np.size(y) - np.isfinite(y).sum()
    if non_finite_x or non_finite_y:
        logging.warning(
            "Detected non-finite values in dataset: x=%d y=%d. Replacing with finite values.",
            int(non_finite_x),
            int(non_finite_y),
        )
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=args.val_split, random_state=args.seed
    )
    logging.info(
        "Loaded arrays: x_train=%s y_train=%s x_val=%s y_val=%s",
        x_train.shape,
        y_train.shape,
        x_val.shape,
        y_val.shape,
    )

    x_train = apply_sorting(x_train, args.sort_by)
    x_val = apply_sorting(x_val, args.sort_by)
    y_train_idx = one_hot_to_index(y_train)
    y_val_idx = one_hot_to_index(y_val)

    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train_idx, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val_idx, dtype=torch.long),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = HEPTClassifier(
        num_particles=args.num_particles,
        feature_dim=x_train.shape[2],
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        output_dim=len(CLASS_LABELS),
        block_size=args.block_size,
        n_hashes=args.n_hashes,
        num_regions=args.num_regions,
        num_w_per_dist=args.num_w_per_dist,
        dropout=args.dropout,
        aggregation=args.aggregation,
    ).to(device)
    logging.info("Model: %s", model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Total params: %d", total_params)
    logging.info("Trainable params: %d", trainable_params)
    print(f"Model built with {trainable_params:,} trainable parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, args.patience // 3),
    )

    train_loss_hist = []
    val_loss_hist = []
    train_acc_hist = []
    val_acc_hist = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None

    for epoch in range(args.num_epochs):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch_idx=epoch,
            num_epochs=args.num_epochs,
            log_interval=args.log_interval,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        logging.info(
            "Epoch %03d train_loss=%.5f train_acc=%.4f val_loss=%.5f val_acc=%.4f lr=%.3e time=%.2fs",
            epoch + 1,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            current_lr,
            elapsed,
        )
        print(
            f"Epoch {epoch + 1:03d}/{args.num_epochs} "
            f"train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model_state_dict": best_state,
                    "model_config": {
                        "num_particles": args.num_particles,
                        "feature_dim": x_train.shape[2],
                        "hidden_dim": args.hidden_dim,
                        "num_heads": args.num_heads,
                        "num_layers": args.num_layers,
                        "output_dim": len(CLASS_LABELS),
                        "block_size": args.block_size,
                        "n_hashes": args.n_hashes,
                        "num_regions": args.num_regions,
                        "num_w_per_dist": args.num_w_per_dist,
                        "dropout": args.dropout,
                        "aggregation": args.aggregation,
                    },
                    "sort_by": args.sort_by,
                    "class_labels": CLASS_LABELS,
                    "best_epoch": epoch + 1,
                },
                os.path.join(save_dir, "best_model.pt"),
            )
        elif epoch - best_epoch >= args.patience:
            logging.info("Early stopping triggered at epoch %d", epoch + 1)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "num_particles": args.num_particles,
                "feature_dim": x_train.shape[2],
                "hidden_dim": args.hidden_dim,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "output_dim": len(CLASS_LABELS),
                "block_size": args.block_size,
                "n_hashes": args.n_hashes,
                "num_regions": args.num_regions,
                "num_w_per_dist": args.num_w_per_dist,
                "dropout": args.dropout,
                "aggregation": args.aggregation,
            },
            "sort_by": args.sort_by,
            "class_labels": CLASS_LABELS,
            "best_epoch": best_epoch + 1,
        },
        os.path.join(save_dir, "final_model.pt"),
    )

    np.save(os.path.join(save_dir, "train_loss.npy"), np.asarray(train_loss_hist, dtype=np.float32))
    np.save(os.path.join(save_dir, "val_loss.npy"), np.asarray(val_loss_hist, dtype=np.float32))
    np.save(os.path.join(save_dir, "train_accuracy.npy"), np.asarray(train_acc_hist, dtype=np.float32))
    np.save(os.path.join(save_dir, "val_accuracy.npy"), np.asarray(val_acc_hist, dtype=np.float32))

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_hist, label="Train")
    plt.plot(val_loss_hist, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_hist, label="Train")
    plt.plot(val_acc_hist, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()

    logging.info("Best validation loss %.5f at epoch %d", best_val_loss, best_epoch + 1)
    logging.info("Training complete. Outputs saved to %s", save_dir)
    print(f"Best validation loss: {best_val_loss:.5f} at epoch {best_epoch + 1}")
    print(f"Training outputs saved to: {save_dir}")
    print(
        "Evaluate with: "
        f"python {os.path.join(PROJECT_ROOT, 'scripts', 'test_hept.py')} "
        f"--data_dir {args.data_dir} --save_dir {save_dir} --num_particles {args.num_particles}"
    )


if __name__ == "__main__":
    main()
