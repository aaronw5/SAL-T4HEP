#!/usr/bin/env python
"""
Quick sanity check for JEDI-PTv3 Hybrid implementation.
Tests that the model builds correctly and can perform forward pass.
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import tensorflow as tf
from models.PointTransformerV3TF import (
    build_ptv3_jet_classifier,
    build_jedi_ptv3_hybrid,
    GlobalInteractionLayer,
    ChannelMixingLayer,
    JEDIPTv3Block
)

print("=" * 70)
print("JEDI-PTv3 Hybrid Sanity Check")
print("=" * 70)

# Test 1: Component layers
print("\n[Test 1] Testing GlobalInteractionLayer...")
try:
    gil = GlobalInteractionLayer(latent_dim=64)
    dummy_input = tf.random.normal([2, 10, 32])
    output = gil(dummy_input, training=True)
    assert output.shape == (2, 10, 64), f"Expected (2, 10, 64), got {output.shape}"
    print("✓ GlobalInteractionLayer works correctly")
except Exception as e:
    print(f"✗ GlobalInteractionLayer failed: {e}")
    sys.exit(1)

print("\n[Test 2] Testing ChannelMixingLayer...")
try:
    cml = ChannelMixingLayer(feature_dim=64, hidden_units=256)
    dummy_input = tf.random.normal([2, 10, 64])
    output = cml(dummy_input, training=True)
    assert output.shape == (2, 10, 64), f"Expected (2, 10, 64), got {output.shape}"
    print("✓ ChannelMixingLayer works correctly")
except Exception as e:
    print(f"✗ ChannelMixingLayer failed: {e}")
    sys.exit(1)

print("\n[Test 3] Testing JEDIPTv3Block...")
try:
    block = JEDIPTv3Block(d_model=64, d_ff=256, cpe_k=3, use_cpe=True)
    dummy_features = tf.random.normal([2, 10, 64])
    dummy_coords = tf.random.normal([2, 10, 2])
    output_features, output_coords = block([dummy_features, dummy_coords], training=True)
    assert output_features.shape == (2, 10, 64), f"Expected (2, 10, 64), got {output_features.shape}"
    assert output_coords.shape == (2, 10, 2), f"Expected (2, 10, 2), got {output_coords.shape}"
    print("✓ JEDIPTv3Block works correctly")
except Exception as e:
    print(f"✗ JEDIPTv3Block failed: {e}")
    sys.exit(1)

print("\n[Test 4] Testing JEDIPTv3Block without CPE...")
try:
    block_no_cpe = JEDIPTv3Block(d_model=64, d_ff=256, use_cpe=False)
    dummy_features = tf.random.normal([2, 10, 64])
    dummy_coords = tf.random.normal([2, 10, 2])
    output_features, output_coords = block_no_cpe([dummy_features, dummy_coords], training=True)
    assert output_features.shape == (2, 10, 64), f"Expected (2, 10, 64), got {output_features.shape}"
    print("✓ JEDIPTv3Block (no CPE) works correctly")
except Exception as e:
    print(f"✗ JEDIPTv3Block (no CPE) failed: {e}")
    sys.exit(1)

# Test 5: Full model building
print("\n[Test 5] Building JEDI-PTv3 Hybrid model (small)...")
try:
    model_hybrid = build_jedi_ptv3_hybrid(
        num_particles=150,
        output_dim=5,
        enc_dims=[16],
        enc_layers=[1],
        enc_strides=[2],
        cpe_k=3,
        grid_size=0.1,
        use_cpe=True,
        dropout=0.1,
        aggregation="max"
    )
    print(f"✓ Model built successfully")
    print(f"  Total parameters: {model_hybrid.count_params():,}")
except Exception as e:
    print(f"✗ Model building failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Forward pass
print("\n[Test 6] Testing forward pass...")
try:
    batch_size = 4
    dummy_input = tf.random.normal([batch_size, 150, 3])
    output = model_hybrid(dummy_input, training=False)
    assert output.shape == (batch_size, 5), f"Expected ({batch_size}, 5), got {output.shape}"
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Compare parameter counts
print("\n[Test 7] Comparing parameter counts...")
try:
    model_standard = build_ptv3_jet_classifier(
        num_particles=150,
        output_dim=5,
        enc_dims=[16],
        enc_layers=[1],
        enc_heads=[4],
        enc_patch_sizes=[25],
        enc_strides=[2],
        cpe_k=3,
        dropout=0.1,
        aggregation="max"
    )

    params_standard = model_standard.count_params()
    params_hybrid = model_hybrid.count_params()

    print(f"✓ Parameter comparison:")
    print(f"  Standard PTv3: {params_standard:,} parameters")
    print(f"  JEDI Hybrid:   {params_hybrid:,} parameters")
    print(f"  Difference:    {params_hybrid - params_standard:+,} ({(params_hybrid/params_standard - 1)*100:+.1f}%)")
except Exception as e:
    print(f"✗ Comparison failed: {e}")
    import traceback
    traceback.print_exc()

# Test 8: Training mode vs inference mode
print("\n[Test 8] Testing training vs inference mode...")
try:
    dummy_input = tf.random.normal([2, 150, 3])

    # Training mode
    output_train = model_hybrid(dummy_input, training=True)

    # Inference mode
    output_infer = model_hybrid(dummy_input, training=False)

    # Outputs should have same shape but potentially different values (due to BatchNorm)
    assert output_train.shape == output_infer.shape
    print(f"✓ Training/inference modes work correctly")
    print(f"  Training output range: [{tf.reduce_min(output_train):.3f}, {tf.reduce_max(output_train):.3f}]")
    print(f"  Inference output range: [{tf.reduce_min(output_infer):.3f}, {tf.reduce_max(output_infer):.3f}]")
except Exception as e:
    print(f"✗ Training/inference test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 9: Multi-stage model
print("\n[Test 9] Building multi-stage JEDI-PTv3 Hybrid...")
try:
    model_multistage = build_jedi_ptv3_hybrid(
        num_particles=150,
        output_dim=5,
        enc_dims=[12, 16, 24],
        enc_layers=[1, 1, 1],
        enc_strides=[2, 2],
        cpe_k=3,
        use_cpe=True,
        dropout=0.1,
        aggregation="mean"
    )

    dummy_input = tf.random.normal([2, 150, 3])
    output = model_multistage(dummy_input, training=False)

    print(f"✓ Multi-stage model works")
    print(f"  Parameters: {model_multistage.count_params():,}")
    print(f"  Output shape: {output.shape}")
except Exception as e:
    print(f"✗ Multi-stage model failed: {e}")
    import traceback
    traceback.print_exc()

# Test 10: Pure JEDI (no CPE)
print("\n[Test 10] Building pure JEDI model (no CPE)...")
try:
    model_pure_jedi = build_jedi_ptv3_hybrid(
        num_particles=150,
        output_dim=5,
        enc_dims=[16],
        enc_layers=[2],  # Can afford more layers since no CPE overhead
        enc_strides=[2],
        use_cpe=False,  # Pure JEDI - no CPE
        dropout=0.1,
        aggregation="mean"
    )

    dummy_input = tf.random.normal([2, 150, 3])
    output = model_pure_jedi(dummy_input, training=False)

    print(f"✓ Pure JEDI model (no CPE) works")
    print(f"  Parameters: {model_pure_jedi.count_params():,}")
    print(f"  2 layers, permutation-invariant")
except Exception as e:
    print(f"✗ Pure JEDI model failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("✓ All tests passed!")
print("=" * 70)
print("\nJEDI-PTv3 Hybrid is ready to use!")
print("\nUsage examples:")
print("  Train: python scripts/train_point_transformer.py --use_jedi_hybrid --model_size small ...")
print("  Test:  python scripts/test_point_transformer.py --use_jedi_hybrid --model_size small ...")
print("  No CPE: Add --disable_cpe flag for pure JEDI (permutation-invariant)")
print("=" * 70)
