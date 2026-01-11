#!/usr/bin/env python
"""
Simple check: just count parameters and look at actual tensor shapes during inference.
"""
import os
import sys
import tensorflow as tf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.PointTransformerV3TF import build_ptv3_jet_classifier


def test_stride(stride_value):
    print(f"\n{'='*60}")
    print(f"Testing stride={stride_value}")
    print('='*60)

    model = build_ptv3_jet_classifier(
        num_particles=150,
        output_dim=5,
        enc_dims=[16, 16],
        enc_layers=[1, 1],
        enc_heads=[4, 4],
        enc_patch_sizes=[25, 25],
        enc_strides=[stride_value],
        cpe_k=8,
        grid_size=0.2,
        use_rpe=False,
        use_pool=True,
        dropout=0.0,
        aggregation="max",
    )

    # Create dummy input
    dummy_input = tf.random.normal((2, 150, 3))  # batch_size=2 for variety

    # Run inference
    print(f"\nInput shape: {dummy_input.shape}")
    output = model(dummy_input, training=False)
    print(f"Output shape: {output.shape}")

    # Show model summary
    print(f"\nTotal params: {model.count_params():,}")

    # Look at layer output shapes
    print("\nLayer output shapes:")
    for layer in model.layers:
        if hasattr(layer, 'output'):
            try:
                if isinstance(layer.output, list):
                    shapes = [str(o.shape) for o in layer.output]
                    print(f"  {layer.name:30s}: {shapes}")
                else:
                    print(f"  {layer.name:30s}: {layer.output.shape}")
            except:
                pass


if __name__ == "__main__":
    test_stride(1)
    test_stride(2)
    test_stride(4)
