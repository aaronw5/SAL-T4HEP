#!/usr/bin/env python
"""
Initialize a JEDI-Linear model and report parameter count and FLOPs.
Use this to quickly size models and pick hyperparameters comparable to SAL-T.
"""
import os
import sys
import argparse
import tensorflow as tf

# ─── make the parent directory (project root) importable ─────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.JEDI_Linear import (
    build_jedi_linear_classifier,
    build_jedi_linear_small,
    build_jedi_linear_medium,
    build_jedi_linear_large
)


def get_flops(model, input_shape):
    """
    Estimate FLOPs by profiling the model with TensorFlow's profiler.
    """
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2_as_graph,
    )

    spec_x = tf.TensorSpec(input_shape, tf.float32)

    @tf.function
    def model_fn(x):
        return model(x, training=False)

    concrete = model_fn.get_concrete_function(spec_x)
    _, graph_def = convert_variables_to_constants_v2_as_graph(concrete)

    with tf.Graph().as_default() as g:
        tf.compat.v1.import_graph_def(graph_def, name="")
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        prof = tf.compat.v1.profiler.profile(
            graph=g, run_meta=run_meta, cmd="op", options=opts
        )
        flops = prof.total_float_ops

    return flops


def estimate_flops_manual(model, input_shape):
    """
    Manually estimate FLOPs based on layer structure.
    """
    batch_size = input_shape[0]
    num_particles = input_shape[1]
    feature_dim = input_shape[2]

    total_flops = 0

    # Track current shape through the model
    current_shape = list(input_shape)

    for layer in model.layers:
        layer_name = layer.name
        layer_type = type(layer).__name__

        if isinstance(layer, tf.keras.layers.Dense):
            # Dense layer: FLOPs = 2 * input_units * output_units * batch * sequence
            input_units = layer.input_shape[-1]
            output_units = layer.units
            if len(layer.input_shape) == 3:  # [B, N, C]
                flops = 2 * batch_size * num_particles * input_units * output_units
            else:  # [B, C]
                flops = 2 * batch_size * input_units * output_units
            total_flops += flops
            print(f"  {layer_name:40s} Dense: {input_units:5d} -> {output_units:5d}, FLOPs: {flops:,}")

        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            # BatchNorm: ~1 FLOP per element (normalization + scale/shift)
            if hasattr(layer, 'input_shape') and layer.input_shape is not None:
                elements = 1
                for dim in layer.input_shape[1:]:
                    if dim is not None:
                        elements *= dim
                flops = batch_size * elements
                total_flops += flops

        elif 'TokenMixing' in layer_name or 'ChannelMixing' in layer_name:
            # These are handled by their internal Dense and BatchNorm layers
            print(f"  {layer_name:40s} (composite layer)")

        elif 'JEDILinearBlock' in layer_name or 'jedi_block' in layer_name:
            print(f"  {layer_name:40s} (composite block)")

    return total_flops


def report_model_structure(model):
    """
    Print detailed structure of JEDI-Linear blocks.
    """
    print("\n" + "="*60)
    print("=== JEDI-Linear Block Breakdown ===")
    print("="*60)

    blocks = [l for l in model.layers if 'jedi_block' in l.name]
    print(f"Number of JEDI blocks: {len(blocks)}")

    total_block_params = 0
    for idx, block in enumerate(blocks):
        block_params = block.count_params()
        total_block_params += block_params

        # Get subcomponents
        token_mixing_params = block.token_mixing.count_params()
        channel_mixing_params = block.channel_mixing.count_params()

        print(f"\n- Block {idx}: {block.name}")
        print(f"  Total params: {block_params:,}")
        print(f"    Token mixing:   {token_mixing_params:,}")
        print(f"    Channel mixing: {channel_mixing_params:,}")

    print(f"\nTotal params in JEDI blocks: {total_block_params:,}")

    # Report other components
    print("\n=== Other Components ===")
    embedding_params = 0
    head_params = 0

    for layer in model.layers:
        if 'embedding' in layer.name or 'input_norm' in layer.name:
            params = layer.count_params()
            embedding_params += params
            print(f"  {layer.name:40s}: {params:,} params")
        elif 'head' in layer.name or 'output' in layer.name:
            params = layer.count_params()
            head_params += params
            print(f"  {layer.name:40s}: {params:,} params")

    print(f"\nTotal embedding params: {embedding_params:,}")
    print(f"Total head params: {head_params:,}")


def parse_args():
    p = argparse.ArgumentParser(description="Inspect JEDI-Linear model size and FLOPs")

    # Data shape
    p.add_argument("--num_particles", type=int, default=150, help="Number of particles (sequence length)")
    p.add_argument("--feature_dim", type=int, default=3, help="Feature dimension (3 for pt, eta, phi)")

    # Model hyperparameters
    p.add_argument("--embedding_dim", type=int, default=16, help="Embedding dimension")
    p.add_argument("--num_blocks", type=int, default=2, help="Number of JEDI-Linear blocks")
    p.add_argument("--token_hidden", type=int, default=None, help="Hidden units for token mixing (default: num_particles)")
    p.add_argument("--channel_hidden", type=int, default=None, help="Hidden units for channel mixing (default: embedding_dim * 4)")
    p.add_argument("--output_dim", type=int, default=5, help="Number of output classes")
    p.add_argument("--aggregation", choices=["mean", "max"], default="mean", help="Aggregation method")
    p.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    p.add_argument("--head_hidden", type=int, nargs="+", default=[64, 32], help="Classification head hidden dimensions")

    # Presets
    p.add_argument("--preset", choices=["small", "medium", "large", "custom"], default="custom",
                   help="Use preset model size (overrides other hyperparameters)")

    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "="*60)
    print("=== JEDI-Linear Model Inspection ===")
    print("="*60)

    # Build model based on preset or custom config
    if args.preset == "small":
        print("Using SMALL preset")
        model = build_jedi_linear_small(
            num_particles=args.num_particles,
            feature_dim=args.feature_dim,
            output_dim=args.output_dim
        )
    elif args.preset == "medium":
        print("Using MEDIUM preset")
        model = build_jedi_linear_medium(
            num_particles=args.num_particles,
            feature_dim=args.feature_dim,
            output_dim=args.output_dim
        )
    elif args.preset == "large":
        print("Using LARGE preset")
        model = build_jedi_linear_large(
            num_particles=args.num_particles,
            feature_dim=args.feature_dim,
            output_dim=args.output_dim
        )
    else:
        print("Using CUSTOM configuration")
        model = build_jedi_linear_classifier(
            num_particles=args.num_particles,
            feature_dim=args.feature_dim,
            embedding_dim=args.embedding_dim,
            num_blocks=args.num_blocks,
            token_hidden=args.token_hidden,
            channel_hidden=args.channel_hidden,
            output_dim=args.output_dim,
            aggregation=args.aggregation,
            dropout_rate=args.dropout,
            head_hidden_dims=args.head_hidden,
        )

    # Count parameters
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])

    print(f"\nModel: {model.name}")
    print(f"Input shape: ({args.num_particles}, {args.feature_dim})")
    print(f"Output classes: {args.output_dim}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Estimate FLOPs
    print("\n" + "="*60)
    print("=== FLOP Estimation ===")
    print("="*60)

    input_shape = (1, args.num_particles, args.feature_dim)

    try:
        flops = get_flops(model, input_shape)
        macs = flops // 2
        print(f"FLOPs (TF profiler): {flops:,}")
        print(f"MACs  (TF profiler): {macs:,}")
    except Exception as e:
        print(f"TF profiler failed: {e}")
        print("Falling back to manual estimation...")

    print("\nManual FLOP breakdown:")
    manual_flops = estimate_flops_manual(model, input_shape)
    manual_macs = manual_flops // 2
    print(f"\nFLOPs (manual est.): {manual_flops:,}")
    print(f"MACs  (manual est.): {manual_macs:,}")

    # Model structure
    report_model_structure(model)

    # Model summary
    print("\n" + "="*60)
    print("=== Model Summary ===")
    print("="*60)
    model.summary()

    print("\n" + "="*60)
    print("=== Comparison to Other Models ===")
    print("="*60)
    print("For reference:")
    print("  SAL-T (Transformer):     ~10K-15K params")
    print("  PointNet:                ~20K-30K params")
    print("  PointTransformerV3:      ~15K-50K params (depends on config)")
    print("  JEDI-Linear (this):      ~{:,} params".format(total_params))


if __name__ == "__main__":
    main()
