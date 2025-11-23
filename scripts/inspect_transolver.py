#!/usr/bin/env python
"""
Initialize a Transolver (PhysicsAttentionStructuredMesh2D) model and report parameter count and FLOPs.
Use this to quickly size models and pick hyperparameters comparable to the standard Transformer.
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

from models.transolver import build_transolver_classifier


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


def choose_hw(num_particles, target_ratio=5.0):
    """
    Choose (H, W) factors of num_particles with H >= W, minimizing |H/W - target_ratio|.
    For 150, this selects H=25, W=6 (close to ratio 4.17 and uses full 150 tokens).
    """
    best = (num_particles, 1)
    best_err = abs((num_particles / 1.0) - target_ratio)
    n = num_particles
    for w in range(1, int(n ** 0.5) + 1):
        if n % w == 0:
            h = n // w
            if h < w:
                h, w = w, h
            err = abs((h / float(w)) - target_ratio)
            if err < best_err:
                best = (h, w)
                best_err = err
    return best


def parse_args():
    p = argparse.ArgumentParser(description="Inspect Transolver (2D structured mesh attention) size and FLOPs")
    # data shape
    p.add_argument("--num_particles", type=int, default=150, help="Sequence length (e.g., 150)")
    p.add_argument("--feature_dim", type=int, default=3)
    p.add_argument("--H", type=int, default=25, help="Grid height; if 0, auto-choose")
    p.add_argument("--W", type=int, default=6, help="Grid width; if 0, auto-choose")

    # model hyperparameters (aligned with Transformer defaults)
    p.add_argument("--d_model", type=int, default=16)
    p.add_argument("--d_ff", type=int, default=16)  # unused but kept for parity
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--dim_head", type=int, default=16)
    p.add_argument("--slice_num", type=int, default=16)
    p.add_argument("--kernel", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--aggregation", choices=["mean", "max"], default="max")
    p.add_argument("--output_dim", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()

    # choose H, W if not provided
    H, W = (args.H, args.W)
    if H <= 0 or W <= 0:
        H, W = choose_hw(args.num_particles, target_ratio=5.0)

    # sanity
    assert H * W == args.num_particles, f"H*W must equal num_particles; got {H}*{W} != {args.num_particles}"

    # build model
    model = build_transolver_classifier(
        num_particles=args.num_particles,
        feature_dim=args.feature_dim,
        output_dim=args.output_dim,
        d_model=args.d_model,
        d_ff=args.d_ff,
        heads=args.num_heads,
        dim_head=args.dim_head,
        slice_num=args.slice_num,
        H=H,
        W=W,
        kernel=args.kernel,
        dropout=args.dropout,
        aggreg=args.aggregation,
    )

    # params
    params = model.count_params()

    # FLOPs on a dummy single-sample input
    flops = get_flops(model, (1, args.num_particles, args.feature_dim))
    macs = flops // 2

    # output
    print("=== Transolver Inspection ===")
    print(f"num_particles = {args.num_particles}, feature_dim = {args.feature_dim}")
    print(f"H, W          = {H}, {W}")
    print(f"d_model       = {args.d_model}")
    print(f"num_heads     = {args.num_heads}")
    print(f"dim_head      = {args.dim_head}")
    print(f"slice_num     = {args.slice_num}")
    print(f"kernel        = {args.kernel}")
    print("---------------------------------------")
    print(f"Total params  = {params:,}")
    print(f"FLOPs (1 x)   = {flops:,}")
    print(f"MACs (approx) = {macs:,}")
    print(model.summary())


if __name__ == "__main__":
    main()


