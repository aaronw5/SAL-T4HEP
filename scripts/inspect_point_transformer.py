#!/usr/bin/env python
"""
Initialize a PointTransformerV3TF model and report parameter count and FLOPs.
Use this to quickly size models and pick hyperparameters comparable to SAL-T.
"""
import os
import sys
import math
import argparse
import logging
import numpy as np
import tensorflow as tf

# ─── make the parent directory (project root) importable ─────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
		sys.path.insert(0, PROJECT_ROOT)

from models.PointTransformerV3TF import build_ptv3_jet_classifier
from models.PointTransformer_serialized import build_ptv3_serialized_jet_classifier


def estimate_flops_from_runtime(model, concrete_input):
    """
    Estimate hardware-agnostic theoretical FLOPs by tracing through the model.
    Calculates FLOPs based on mathematical operations, not GPU/FPGA specifics.

    For each operation, we count multiply-add pairs (MACs) as 2 FLOPs.
    This gives hardware-independent theoretical complexity.
    """
    # Create intermediate models to capture shapes
    x = concrete_input
    coords = x[..., :2]

    total_flops = 0

    # Embedding layer
    for layer in model.layers:
        if layer.name == 'dense' and 'dense' == layer.name:
            x_prev = x
            x = layer(x, training=False)

            # Dense layer: FLOPs = 2 * M * N * K (M outputs, N inputs, K batch*points)
            # For shape (B, N, in_feat) -> (B, N, out_feat):
            # FLOPs = B * N * in_feat * out_feat * 2
            batch_size = x_prev.shape[0]
            n_points = x_prev.shape[1]
            in_feat = x_prev.shape[2]
            out_feat = x.shape[2]
            flops = batch_size * n_points * in_feat * out_feat * 2
            total_flops += flops
            print(f"  Embedding Dense: {x_prev.shape} -> {x.shape}, FLOPs: {flops:,}")
            break

    # Trace through PTv3 blocks and pooling
    for layer in model.layers:
        if 'PTv3Block' in str(type(layer)):
            x_prev_shape = x.shape
            x, coords = layer([x, coords], training=False)

            # Estimate PTv3Block FLOPs based on actual runtime shapes
            batch_size = x.shape[0]
            n_points = x.shape[1]
            d_model = x.shape[2]

            # Estimate based on attention mechanism
            # For local attention with patches, FLOPs depend on patch structure
            # Simplified: assume it scales with n_points and d_model
            # Main operations: QKV proj, attention within patches, FFN
            block_flops = batch_size * (
                3 * n_points * d_model * d_model +  # QKV projections
                n_points * d_model * d_model +       # Output projection
                2 * n_points * d_model * (4 * d_model)  # FFN (assume 4x expansion)
            )
            total_flops += block_flops
            print(f"  {layer.name}: {x_prev_shape} -> {x.shape}, FLOPs: {block_flops:,}")

        elif 'GeometricPooling' in str(type(layer)):
            x_prev_shape = x.shape
            x_before_pool, coords_before_pool = x, coords
            x, coords = layer([x, coords], training=False)

            # Calculate theoretical FLOPs for GeometricPooling operations
            batch_size = x_prev_shape[0]
            n_in = x_prev_shape[1]
            n_out = x.shape[1]
            in_feat = x_prev_shape[2]
            out_feat = x.shape[2]
            stride = layer.stride

            pool_flops = 0

            # 1. Sorting: O(N log N) comparisons, approximate as N * log2(N) comparisons
            #    Each comparison is counted as 1 FLOP
            import math
            sort_flops = batch_size * n_in * math.ceil(math.log2(max(n_in, 2))) if n_in > 1 else 0
            pool_flops += sort_flops

            # 2. Grouping and reshape: No FLOPs (memory operations only)

            # 3. Max pooling over stride groups: (stride-1) comparisons per output point
            #    Shape: (B, N_out, stride, C) -> (B, N_out, C)
            max_pool_flops = batch_size * n_out * in_feat * (stride - 1)
            pool_flops += max_pool_flops

            # 4. Mean pooling for coordinates: (stride-1) additions + 1 division per output
            #    Shape: (B, N_out, stride, 2) -> (B, N_out, 2)
            mean_pool_flops = batch_size * n_out * 2 * stride  # stride adds + 1 div ≈ stride ops
            pool_flops += mean_pool_flops

            # 5. Dense projection: standard matrix multiply
            #    (B, N_out, in_feat) @ (in_feat, out_feat) = (B, N_out, out_feat)
            dense_flops = batch_size * n_out * in_feat * out_feat * 2
            pool_flops += dense_flops

            # 6. LayerNorm: mean, variance, normalize, scale, shift
            #    - Mean: out_feat additions
            #    - Variance: out_feat multiplications + out_feat additions
            #    - Normalize: out_feat divisions
            #    - Scale + shift: 2 * out_feat multiplications
            #    Total per point: ~5 * out_feat operations
            ln_flops = batch_size * n_out * out_feat * 5
            pool_flops += ln_flops

            total_flops += pool_flops

            # Breakdown for clarity
            print(f"  {layer.name}: {x_prev_shape} -> {x.shape} ({n_in}->{n_out} pts)")
            print(f"    - Sort:         {sort_flops:,} FLOPs")
            print(f"    - Max pool:     {max_pool_flops:,} FLOPs")
            print(f"    - Mean pool:    {mean_pool_flops:,} FLOPs")
            print(f"    - Dense proj:   {dense_flops:,} FLOPs")
            print(f"    - LayerNorm:    {ln_flops:,} FLOPs")
            print(f"    - Total:        {pool_flops:,} FLOPs")

    # Final classification head
    for layer in model.layers:
        if layer.name.startswith('dense_') and hasattr(layer, 'units'):
            # Count params and estimate FLOPs
            params = layer.count_params()
            if params > 0:
                # Simplified: FLOPs ≈ 2 * params for dense layers
                head_flops = 2 * params
                total_flops += head_flops

    return total_flops


def get_flops(model, input_shape):
    """
    Estimate FLOPs by running model with concrete input and profiling.
    For models with dynamic shapes, we trace with actual runtime shapes.
    """
    from tensorflow.python.framework.convert_to_constants import (
            convert_variables_to_constants_v2_as_graph,
    )

    # Create a concrete input to trace actual shapes
    concrete_input = tf.random.normal(input_shape)

    # Run once to determine actual dynamic shapes
    _ = model(concrete_input, training=False)

    spec_x = tf.TensorSpec(input_shape, tf.float32)

    @tf.function
    def model_fn(x):
            return model(x)

    concrete = model_fn.get_concrete_function(spec_x)
    _, graph_def = convert_variables_to_constants_v2_as_graph(concrete)
    with tf.Graph().as_default() as g:
            tf.compat.v1.import_graph_def(graph_def, name="")
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            prof = tf.compat.v1.profiler.profile(
                    graph=g, run_meta=run_meta, cmd="op", options=opts
            )
            static_flops = prof.total_float_ops

    # Also compute FLOPs manually based on actual layer shapes
    manual_flops = estimate_flops_from_runtime(model, concrete_input)

    return static_flops, manual_flops


def compute_stage_lengths(num_particles, enc_strides):
		"""
		Length before each stage (after previous downsamples applied).
		For S stages, len(enc_strides) == S-1.
		"""
		lengths = []
		current = num_particles
		for i in range(len(enc_strides) + 1):
				lengths.append(current)
				if i < len(enc_strides):
						stride = enc_strides[i]
						current = int(math.ceil(current / float(stride)))
		return lengths


def choose_divisible_patch_sizes(stage_lengths, preferred=(64, 32, 16, 8, 4, 2, 1)):
		patch_sizes = []
		for L in stage_lengths:
				ps = 1
				for cand in preferred:
						if L >= cand and (L % cand == 0):
								ps = cand
								break
				patch_sizes.append(ps)
		return patch_sizes


def report_ptv3_blocks(model: tf.keras.Model):
		"""
		Print structure and parameter counts for each PTv3Block within the model.
		"""
		blocks = [l for l in model.layers if l.__class__.__name__ == "PTv3Block"]
		print("\n=== PTv3Block breakdown ===")
		print(f"num_blocks = {len(blocks)}")
		total_block_params = 0
		for idx, blk in enumerate(blocks):
				# major subcomponents
				cpe_params = getattr(blk, "cpe").count_params() if hasattr(blk, "cpe") else 0
				attn_params = getattr(blk, "attn").count_params() if hasattr(blk, "attn") else 0
				norm1_params = getattr(blk, "norm1").count_params() if hasattr(blk, "norm1") else 0
				norm2_params = getattr(blk, "norm2").count_params() if hasattr(blk, "norm2") else 0
				ffn_params = getattr(blk, "ffn").count_params() if hasattr(blk, "ffn") else 0
				# dropouts have 0 params
				block_params = cpe_params + attn_params + norm1_params + norm2_params + ffn_params
				total_block_params += block_params

				# hyperparameters (if available)
				d_model = getattr(getattr(blk, "attn", None), "d_model", None)
				num_heads = getattr(getattr(blk, "attn", None), "num_heads", None)
				patch_size = getattr(getattr(blk, "attn", None), "patch_size", None)
				cpe_k = getattr(getattr(blk, "cpe", None), "kernel_size", None)

				print(f"- Block {idx}: d_model={d_model}, heads={num_heads}, patch={patch_size}, cpe_k={cpe_k}")
				print(f"  params: total={block_params:,} [cpe={cpe_params:,}, attn={attn_params:,}, norm={norm1_params+norm2_params:,}, ffn={ffn_params:,}]")

		print(f"Total params in PTv3Blocks: {total_block_params:,}")


def parse_args():
    p = argparse.ArgumentParser(description="Inspect PointTransformerV3TF size and FLOPs")
    # data shape
    p.add_argument("--num_particles", type=int, default=150, help="Sequence length (e.g., 150)")

    # model hyperparameters
    p.add_argument("--enc_dims", type=int, nargs="+", default=[16, 24, 32])
    p.add_argument("--enc_layers", type=int, nargs="+", default=[1, 1, 1])
    p.add_argument("--enc_heads", type=int, nargs="+", default=[4, 4, 4])
    p.add_argument("--enc_patch_sizes", type=int, nargs="+", default=[2, 2, 2])
    p.add_argument("--enc_strides", type=int, nargs="+", default=[2, 2])
    p.add_argument("--cpe_k", type=int, default=8)
    p.add_argument("--grid_size", type=float, default=0.2, help="GeometricCPE grid size (coarser -> smaller grid)")
    p.add_argument("--use_rpe", action="store_true", default=False)
    p.add_argument("--disable_pool", action="store_true", default=False, help="Disable GeometricPooling between stages")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--aggregation", choices=["mean", "max"], default="max")
    p.add_argument("--output_dim", type=int, default=5)
    p.add_argument("--model_size", choices=["small", "small_2layer", "matched", "medium", "large"], default="small")
    p.add_argument('--use_serialized_model', action='store_true', help='Use the serialized version of the PointTransformer model')
    p.add_argument('--serialize_by', choices=['morton','pt','kt'], default='morton', help='Serialization strategy when using the serialized model')

    return p.parse_args()


def main():
    args = parse_args()


    # presets for small / medium / large
    presets = {
			"small":  dict(enc_dims=[16], enc_layers=[1], enc_heads=[4], enc_strides=[2], enc_patch_sizes=[25], cpe_k=8, use_rpe=False),
            "small_2layer": dict(enc_dims=[16, 16], enc_layers=[1, 1], enc_heads=[4, 4], enc_strides=[2], enc_patch_sizes=[25, 25], cpe_k=8, use_rpe=False),
    		"matched": dict(enc_dims=[12, 16], enc_layers=[1, 1], enc_heads=[4, 4], enc_strides=[2], enc_patch_sizes=[25, 25], cpe_k=8, use_rpe=False),
            "medium": dict(enc_dims=[12, 24, 32], enc_layers=[1, 1, 1], enc_heads=[4, 4, 4], enc_strides=[2, 2], enc_patch_sizes=[25, 25, 25], cpe_k=8, use_rpe=False),
    		"large":  dict(enc_dims=[16, 24, 32], enc_layers=[1, 1, 1], enc_heads=[4, 4, 4], enc_strides=[2, 2], enc_patch_sizes=[25, 25, 25], cpe_k=8, use_rpe=False),
    	}
    cfg = presets[args.model_size]

    enc_dims = cfg["enc_dims"]
    enc_layers = cfg["enc_layers"]
    enc_heads = cfg["enc_heads"]
    enc_strides = cfg["enc_strides"]
    enc_patch_sizes = cfg["enc_patch_sizes"]
    cpe_k = cfg["cpe_k"] if args.cpe_k is None else args.cpe_k
    use_rpe = args.use_rpe or cfg["use_rpe"]

    # build model
    if args.use_serialized_model:
        model = build_ptv3_serialized_jet_classifier(
            num_particles=args.num_particles,
            output_dim=args.output_dim,
            enc_dims=enc_dims,
            enc_layers=enc_layers,
            enc_heads=enc_heads,
            enc_patch_sizes=enc_patch_sizes,
            enc_strides=enc_strides,
            cpe_k=cpe_k,
            grid_size=args.grid_size,
            use_rpe=use_rpe,
            use_pool=(not args.disable_pool),
            dropout=args.dropout,
            aggregation=args.aggregation,
            serialize_by=args.serialize_by,
        )
    else:
        model = build_ptv3_jet_classifier(
            num_particles=args.num_particles,
            output_dim=args.output_dim,
            enc_dims=enc_dims,
            enc_layers=enc_layers,
            enc_heads=enc_heads,
            enc_patch_sizes=enc_patch_sizes,
            enc_strides=enc_strides,
            cpe_k=cpe_k,
            grid_size=args.grid_size,
            use_rpe=use_rpe,
            use_pool=(not args.disable_pool),
            dropout=args.dropout,
            aggregation=args.aggregation,
        )

    # params
    params = model.count_params()

    # FLOPs on a dummy single-sample input
    print("\n" + "="*60)
    print("FLOP Estimation (with runtime shape tracing):")
    print("="*60)
    static_flops, manual_flops = get_flops(model, (1, 150, 3))
    macs_static = static_flops // 2
    macs_manual = manual_flops // 2

    # output
    print("\n" + "="*60)
    print("=== PointTransformerV3TF Inspection ===")
    print(f"num_particles = 150, feature_dim = 3")
    print(f"enc_dims      = {enc_dims}")
    print(f"enc_layers    = {enc_layers}")
    print(f"enc_heads     = {enc_heads}")
    print(f"enc_strides   = {enc_strides}")
    print(f"enc_patch_sz  = {enc_patch_sizes}")
    print("---------------------------------------")
    print(f"Total params  = {params:,}")
    print(f"\nFLOPs (static graph profiler) = {static_flops:,}")
    print(f"MACs  (static graph profiler) = {macs_static:,}")
    print(f"\nFLOPs (runtime traced)        = {manual_flops:,}")
    print(f"MACs  (runtime traced)        = {macs_manual:,}")
    print("="*60)
    print(model.summary())

    report_ptv3_blocks(model)

if __name__ == "__main__":
		main()


