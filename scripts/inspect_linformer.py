#!/usr/bin/env python
"""
Initialize a Linformer model and report parameter count and FLOPs.
Use this to experiment with configs and match FLOPs with SAL-T.
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

from models.Linformer import build_linformer_transformer_classifier
from models.LinformerBig import build_linformer_transformer_classifier_big


def get_flops(model, input_shape):
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2_as_graph,
    )

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
        return prof.total_float_ops


def parse_args():
    p = argparse.ArgumentParser(description="Inspect Linformer size and FLOPs")
    # data shape
    p.add_argument("--num_particles", type=int, default=150, help="Sequence length")
    p.add_argument("--feature_dim", type=int, default=3)
    p.add_argument("--output_dim", type=int, default=5)

    # model hyperparameters (aligned with train_linformer.py defaults)
    p.add_argument("--d_model", type=int, default=20)
    p.add_argument("--d_ff", type=int, default=16)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--proj_dim", type=int, default=4)
    p.add_argument("--cluster_E", action="store_true")
    p.add_argument("--cluster_F", action="store_true")
    p.add_argument("--share_EF", action="store_true")
    p.add_argument("--convolution", action="store_true", help="Use convolution on attention scores")
    p.add_argument("--conv_filter_heights", type=int, nargs="+", default=[1, 3, 5])
    p.add_argument("--vertical_stride", type=int, default=1)
    p.add_argument("--num_layers", type=int, default=1, help="Use LinformerBig when > 1")
    p.add_argument("--print_summary", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()

    # build model
    if args.num_layers > 1:
        model = build_linformer_transformer_classifier_big(
            args.num_particles,
            args.feature_dim,
            d_model=args.d_model,
            d_ff=args.d_ff,
            output_dim=args.output_dim,
            num_heads=args.num_heads,
            proj_dim=args.proj_dim,
            cluster_E=args.cluster_E,
            cluster_F=args.cluster_F,
            share_EF=args.share_EF,
            convolution=args.convolution,
            conv_filter_heights=args.conv_filter_heights,
            vertical_stride=args.vertical_stride,
            num_layers=args.num_layers,
        )
    else:
        model = build_linformer_transformer_classifier(
            args.num_particles,
            args.feature_dim,
            d_model=args.d_model,
            d_ff=args.d_ff,
            output_dim=args.output_dim,
            num_heads=args.num_heads,
            proj_dim=args.proj_dim,
            cluster_E=args.cluster_E,
            cluster_F=args.cluster_F,
            share_EF=args.share_EF,
            convolution=args.convolution,
            conv_filter_heights=args.conv_filter_heights,
            vertical_stride=args.vertical_stride,
            shuffle_all=0,
            shuffle_234=0,
            shuffle_34=0,
        )

    # params
    params = model.count_params()

    # FLOPs on a dummy single-sample input
    flops = get_flops(model, (1, args.num_particles, args.feature_dim))
    macs = flops // 2

    # output
    print("=== Linformer Inspection ===")
    print(f"num_particles = {args.num_particles}, feature_dim = {args.feature_dim}")
    print(f"d_model       = {args.d_model}")
    print(f"d_ff          = {args.d_ff}")
    print(f"num_heads     = {args.num_heads}")
    print(f"proj_dim      = {args.proj_dim}")
    print(f"convolution   = {args.convolution}, filters = {args.conv_filter_heights}, vstride = {args.vertical_stride}")
    print(f"cluster_E     = {args.cluster_E}, cluster_F = {args.cluster_F}, share_EF = {args.share_EF}")
    print(f"num_layers    = {args.num_layers}")
    print("---------------------------------------")
    print(f"Total params  = {params:,}")
    print(f"FLOPs (1 x)   = {flops:,}")
    print(f"MACs (approx) = {macs:,}")
    if args.print_summary:
        print(model.summary())


if __name__ == "__main__":
    main()


