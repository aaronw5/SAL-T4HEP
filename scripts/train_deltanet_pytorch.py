#!/usr/bin/env python
"""
Train a PyTorch DeltaNet transformer on jet datasets (hls4ml),
following the Linformer training pipeline structure.
"""
import os
import sys
import time
import argparse
import logging
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
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
    """DeltaNet-based classifier for jet tagging"""
    
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


def train_epoch(model, dataloader, criterion, optimizer, device, dtype):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        # Convert outputs to float32 for loss calculation
        outputs_fp32 = outputs.float()
        loss = criterion(outputs_fp32, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_x.size(0)
        _, predicted = outputs_fp32.max(1)
        _, labels = batch_y.max(1)
        correct += predicted.eq(labels).sum().item()
        total += batch_x.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device, dtype):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            # Convert outputs to float32 for loss calculation
            outputs_fp32 = outputs.float()
            loss = criterion(outputs_fp32, batch_y)
            
            total_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs_fp32.max(1)
            _, labels = batch_y.max(1)
            correct += predicted.eq(labels).sum().item()
            total += batch_x.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def parse_args():
    p = argparse.ArgumentParser(description="Train a PyTorch DeltaNet on jet data")
    p.add_argument("--data_dir", required=True, help="Path to data directory")
    p.add_argument("--save_dir", required=True, help="Path to save results")
    p.add_argument("--dataset", choices=["hls4ml", "top", "QG", "jetclass"], default="hls4ml")
    p.add_argument("--sort_by", choices=["pt", "eta", "phi", "delta_R", "kt"], default="kt")
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--num_particles", type=int, default=150, help="Number of particles")
    p.add_argument("--d_model", type=int, default=16, help="Model dimension")
    p.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    p.add_argument("--num_layers", type=int, default=1, help="Number of DeltaNet layers")
    p.add_argument("--conv_size", type=int, default=4, help="Short convolution kernel size")
    p.add_argument("--use_short_conv", action="store_true", default=True, help="Use short convolution")
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    return p.parse_args()


def main():
    args = parse_args()
    
    # Dataset specifics
    if args.dataset == "hls4ml":
        num_particles = args.num_particles
        output_dim = 5
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented yet")
    
    # Setup save directory
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
    
    # Setup logging
    logging.basicConfig(
        filename=os.path.join(save_dir, "train.log"),
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.info("Args: %s", args)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)
    if torch.cuda.is_available():
        logging.info("CUDA device: %s", torch.cuda.get_device_name(0))
    
    # DeltaNet requires bfloat16 precision
    dtype = torch.bfloat16
    logging.info("Using dtype: %s (required by DeltaNet)", dtype)
    
    # Load data
    logging.info("Loading data...")
    if args.dataset == "hls4ml":
        x = np.load(os.path.join(args.data_dir, f"x_train_robust_{num_particles}const_ptetaphi.npy"))
        y = np.load(os.path.join(args.data_dir, f"y_train_robust_{num_particles}const_ptetaphi.npy"))
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=args.val_split, random_state=42)
    
    logging.info("Loaded train x=%s y=%s, val x=%s y=%s", x_train.shape, y_train.shape, x_val.shape, y_val.shape)
    
    # Apply sorting
    x_train = apply_sorting(x_train, args.sort_by)
    x_val = apply_sorting(x_val, args.sort_by)
    logging.info("Applied '%s' sorting", args.sort_by)
    
    # Convert to PyTorch tensors (bfloat16 for DeltaNet)
    x_train_tensor = torch.tensor(x_train, dtype=dtype)
    y_train_tensor = torch.FloatTensor(y_train)  # Keep labels as float32
    x_val_tensor = torch.tensor(x_val, dtype=dtype)
    y_val_tensor = torch.FloatTensor(y_val)  # Keep labels as float32
    
    # Build model
    model = DeltaNetClassifier(
        num_particles=num_particles,
        feature_dim=x_train.shape[2],
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        output_dim=output_dim,
        conv_size=args.conv_size,
        use_short_conv=args.use_short_conv,
        dropout=args.dropout
    ).to(device).to(dtype)  # Convert model to bfloat16
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Total params: %d", total_params)
    logging.info("Trainable params: %d", trainable_params)
    print(f"Model built with {trainable_params:,} parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training schedule (matching Linformer)
    schedule = [
        (128, 200),
        (256, 200),
        (512, 200),
        (1024, 200),
        (2048, 600),
    ]
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_loss = float('inf')
    current_epoch = 0
    
    # Training loop
    for batch_size, num_epochs in schedule:
        logging.info(f"Training with batch_size={batch_size} for {num_epochs} epochs")
        print(f"\nTraining with batch_size={batch_size}, epochs={current_epoch} to {current_epoch + num_epochs}")
        
        # Create dataloaders
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        
        # Reset learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-3
        
        patience = 40
        patience_counter = 0
        
        for epoch in range(num_epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, dtype)
            val_loss, val_acc = validate(model, val_loader, criterion, device, dtype)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            if (epoch + 1) % 50 == 0 or epoch == num_epochs - 1:
                print(f"  Epoch {current_epoch + epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
                logging.info(f"Epoch {current_epoch + epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {current_epoch + epoch + 1}")
                print(f"  Early stopping at epoch {current_epoch + epoch + 1}")
                break
        
        current_epoch += num_epochs
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pt"))
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt")))
    logging.info("Loaded best model weights")
    
    # Save training history
    np.save(os.path.join(save_dir, "train_loss.npy"), np.array(train_losses))
    np.save(os.path.join(save_dir, "val_loss.npy"), np.array(val_losses))
    np.save(os.path.join(save_dir, "train_accuracy.npy"), np.array(train_accs))
    np.save(os.path.join(save_dir, "val_accuracy.npy"), np.array(val_accs))
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curves")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curves")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=150)
    plt.close()
    
    logging.info("Training complete!")
    print(f"\nTraining complete! Results saved to {save_dir}")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

