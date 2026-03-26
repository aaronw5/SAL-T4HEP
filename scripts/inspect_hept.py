#!/usr/bin/env python3
"""
Inspect HEPT parameter count and FLOPs for hls4ml-sized inputs.

The default configuration is tuned to sit in the same rough compute range as the
other SAL-T4HEP baselines: about 5.9K params and 1.17M FLOPs on (1, 150, 3)
using PyTorch profiler on this machine.
"""
import argparse
import os
import sys

import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.HEPT import HEPTClassifier


def count_flops_with_profiler(model: torch.nn.Module, input_tensor: torch.Tensor):
    try:
        from torch.profiler import profile, ProfilerActivity

        total_flops = 0
        with profile(
            activities=[ProfilerActivity.CPU],
            with_flops=True,
            record_shapes=True,
        ) as prof:
            with torch.no_grad():
                _ = model(input_tensor)

        for event in prof.key_averages():
            total_flops += int(getattr(event, "flops", 0) or 0)

        if total_flops > 0:
            return total_flops, "torch.profiler"
    except Exception as exc:
        print(f"[warn] torch.profiler FLOPs failed: {exc}")

    try:
        from thop import profile as thop_profile

        flops, _ = thop_profile(model, inputs=(input_tensor,), verbose=False)
        return int(flops), "thop"
    except Exception as exc:
        print(f"[warn] thop FLOPs failed: {exc}")
        return None, "unavailable"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect HEPT size and FLOPs")
    parser.add_argument("--num_particles", type=int, default=150, help="Sequence length")
    parser.add_argument("--feature_dim", type=int, default=3)
    parser.add_argument("--output_dim", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--block_size", type=int, default=8)
    parser.add_argument("--n_hashes", type=int, default=4)
    parser.add_argument("--num_regions", type=int, default=16)
    parser.add_argument("--num_w_per_dist", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--aggregation", choices=["max", "mean"], default="max")
    parser.add_argument("--batch_size", type=int, default=1, help="Dummy batch size for profiling")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    device = torch.device(args.device)
    model = HEPTClassifier(
        num_particles=args.num_particles,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        output_dim=args.output_dim,
        block_size=args.block_size,
        n_hashes=args.n_hashes,
        num_regions=args.num_regions,
        num_w_per_dist=args.num_w_per_dist,
        dropout=args.dropout,
        aggregation=args.aggregation,
    ).to(device)
    model.eval()

    input_tensor = torch.randn(
        args.batch_size,
        args.num_particles,
        args.feature_dim,
        device=device,
    )
    flops, flops_source = count_flops_with_profiler(model, input_tensor)
    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=== HEPT Inspection ===")
    print(f"num_particles = {args.num_particles}, feature_dim = {args.feature_dim}")
    print(f"hidden_dim    = {args.hidden_dim}")
    print(f"num_heads     = {args.num_heads}")
    print(f"num_layers    = {args.num_layers}")
    print(f"block_size    = {args.block_size}")
    print(f"n_hashes      = {args.n_hashes}")
    print(f"num_regions   = {args.num_regions}")
    print(f"num_w_per_dist= {args.num_w_per_dist}")
    print(f"aggregation   = {args.aggregation}")
    print(f"device        = {device.type}")
    print("---------------------------------------")
    print(f"Total params  = {params:,}")
    print(f"Trainable     = {trainable_params:,}")
    if flops is not None:
        print(f"FLOPs (1 x)   = {flops:,}")
        print(f"MACs (approx) = {flops // 2:,}")
        print(f"FLOP source   = {flops_source}")
    else:
        print("FLOPs (1 x)   = unavailable")
        print(f"FLOP source   = {flops_source}")


if __name__ == "__main__":
    main()
