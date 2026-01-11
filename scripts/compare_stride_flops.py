#!/usr/bin/env python
"""
Compare theoretical FLOPs for different stride configurations.
Shows hardware-agnostic computational cost.
"""
import os
import sys
import tensorflow as tf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import from inspect_point_transformer
from inspect_point_transformer import estimate_flops_from_runtime
from models.PointTransformerV3TF import build_ptv3_jet_classifier


def test_config(name, enc_strides):
    print(f"\n{'='*70}")
    print(f"Configuration: {name} (enc_strides={enc_strides})")
    print('='*70)

    model = build_ptv3_jet_classifier(
        num_particles=150,
        output_dim=5,
        enc_dims=[16, 16],
        enc_layers=[1, 1],
        enc_heads=[4, 4],
        enc_patch_sizes=[25, 25],
        enc_strides=enc_strides,
        cpe_k=8,
        grid_size=0.2,
        use_rpe=False,
        use_pool=True,
        dropout=0.0,
        aggregation="max",
    )

    # Create fresh input for each model
    concrete_input = tf.random.normal((1, 150, 3))

    print("\nTheoretical FLOP breakdown:")
    print("-" * 70)

    # Reset TensorFlow graph state
    tf.keras.backend.clear_session()

    # Rebuild model to ensure clean state
    model = build_ptv3_jet_classifier(
        num_particles=150,
        output_dim=5,
        enc_dims=[16, 16],
        enc_layers=[1, 1],
        enc_heads=[4, 4],
        enc_patch_sizes=[25, 25],
        enc_strides=enc_strides,
        cpe_k=8,
        grid_size=0.2,
        use_rpe=False,
        use_pool=True,
        dropout=0.0,
        aggregation="max",
    )

    total_flops = estimate_flops_from_runtime(model, concrete_input)

    print("-" * 70)
    print(f"TOTAL THEORETICAL FLOPs: {total_flops:,}")
    print(f"Total params: {model.count_params():,}")
    print('='*70)

    return total_flops


if __name__ == "__main__":
    flops_stride1 = test_config("No downsampling (stride=1)", enc_strides=[1])
    flops_stride2 = test_config("2× downsampling (stride=2)", enc_strides=[2])
    flops_stride4 = test_config("4× downsampling (stride=4)", enc_strides=[4])

    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    print(f"{'Config':<30} {'FLOPs':>15} {'vs stride=2':>15}")
    print("-"*70)
    print(f"{'stride=1 (no downsample)':<30} {flops_stride1:>15,} {flops_stride1/flops_stride2:>14.2f}x")
    print(f"{'stride=2 (2× downsample)':<30} {flops_stride2:>15,} {flops_stride2/flops_stride2:>14.2f}x")
    print(f"{'stride=4 (4× downsample)':<30} {flops_stride4:>15,} {flops_stride4/flops_stride2:>14.2f}x")
    print("="*70)
    print("\nThese FLOPs are hardware-agnostic (same for GPU, FPGA, CPU)")
    print("="*70)
