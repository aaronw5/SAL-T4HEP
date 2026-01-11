#!/usr/bin/env python
"""
Check what actually happens to tensor shapes with different stride values.
"""
import os
import sys
import numpy as np
import tensorflow as tf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.PointTransformerV3TF import build_ptv3_jet_classifier


def trace_shapes(model, dummy_input):
    """Run a forward pass and print shapes at each layer."""
    print("\n=== Forward Pass Shape Trace ===")
    print(f"Input shape: {dummy_input.shape}")

    # Use functional API to extract intermediate outputs
    # Find the layers we care about
    embedding_layer = None
    block0_layer = None
    pooling_layer = None
    block1_layer = None

    for layer in model.layers:
        if layer.name == 'dense':
            embedding_layer = layer
        elif layer.name == 'p_tv3_block':
            block0_layer = layer
        elif 'geometric_pooling' in layer.name:
            pooling_layer = layer
        elif layer.name == 'p_tv3_block_1':
            block1_layer = layer

    # Build intermediate models
    if embedding_layer:
        embed_model = tf.keras.Model(inputs=model.input, outputs=embedding_layer.output)
        x_embed = embed_model(dummy_input, training=False)
        print(f"After embedding: {x_embed.shape}")

    if block0_layer:
        # Find the output of block 0
        block0_output = None
        for node in block0_layer._outbound_nodes:
            if node.outbound_layer.name == pooling_layer.name if pooling_layer else False:
                block0_output = node.output_tensors
                break

        if block0_output is None:
            # Just run the full model and extract from layer outputs
            _ = model(dummy_input, training=False)
            # Get block 0 output from its internal state
            for layer in model.layers:
                if layer.name == 'p_tv3_block':
                    # Try to get cached output
                    pass

    # Simpler approach: use a custom callback to capture shapes
    shapes_captured = {}

    class ShapeCapture(tf.keras.layers.Layer):
        def __init__(self, name, **kwargs):
            super().__init__(name=name, **kwargs)

        def call(self, inputs):
            shapes_captured[self.name] = [x.shape for x in (inputs if isinstance(inputs, list) else [inputs])]
            return inputs

    # Actually, let's just run the model normally and inspect layer outputs
    _ = model(dummy_input, training=False)

    # Manually extract intermediate outputs by building submodels
    coords_input = dummy_input[..., :2]

    # After block 0
    if block0_layer:
        try:
            # Get the actual outputs by looking at the layer's output
            print(f"After Block 0: output exists")
        except:
            pass

    # Try a different approach: use model.get_layer and build partial models
    try:
        # Get pooling layer
        pool_layer = model.get_layer('geometric_pooling')

        # Find what feeds into pooling
        # This is complex, let's just print layer connectivity
        print("\nLayer connectivity:")
        for i, layer in enumerate(model.layers):
            if 'PTv3Block' in str(type(layer)) or 'GeometricPooling' in str(type(layer)):
                print(f"  {layer.name}: {layer.output_shape if hasattr(layer, 'output_shape') else 'N/A'}")
    except Exception as e:
        print(f"Could not trace: {e}")

    # Final output
    output = model(dummy_input, training=False)
    print(f"\nFinal output: {output.shape}")


def main():
    print("\n" + "="*60)
    print("Testing stride=1")
    print("="*60)

    model_stride1 = build_ptv3_jet_classifier(
        num_particles=150,
        output_dim=5,
        enc_dims=[16, 16],
        enc_layers=[1, 1],
        enc_heads=[4, 4],
        enc_patch_sizes=[25, 25],
        enc_strides=[1],  # No downsampling
        cpe_k=8,
        grid_size=0.2,
        use_rpe=False,
        use_pool=True,
        dropout=0.0,
        aggregation="max",
    )

    # Create dummy input (batch_size=1, num_particles=150, features=3)
    dummy_input = tf.random.normal((1, 150, 3))

    trace_shapes(model_stride1, dummy_input)

    print("\n" + "="*60)
    print("Testing stride=2")
    print("="*60)

    model_stride2 = build_ptv3_jet_classifier(
        num_particles=150,
        output_dim=5,
        enc_dims=[16, 16],
        enc_layers=[1, 1],
        enc_heads=[4, 4],
        enc_patch_sizes=[25, 25],
        enc_strides=[2],  # 2x downsampling
        cpe_k=8,
        grid_size=0.2,
        use_rpe=False,
        use_pool=True,
        dropout=0.0,
        aggregation="max",
    )

    trace_shapes(model_stride2, dummy_input)

    print("\n" + "="*60)
    print("Testing stride=4")
    print("="*60)

    model_stride4 = build_ptv3_jet_classifier(
        num_particles=150,
        output_dim=5,
        enc_dims=[16, 16],
        enc_layers=[1, 1],
        enc_heads=[4, 4],
        enc_patch_sizes=[25, 25],
        enc_strides=[4],  # 4x downsampling
        cpe_k=8,
        grid_size=0.2,
        use_rpe=False,
        use_pool=True,
        dropout=0.0,
        aggregation="max",
    )

    trace_shapes(model_stride4, dummy_input)


if __name__ == "__main__":
    main()
