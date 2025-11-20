#!/usr/bin/env python
"""
Inspect a PointNet classifier: parameter count and FLOPs.
"""
import os
import sys
import argparse
import numpy as np
import tensorflow as tf

# ─── make the parent directory (project root) importable ─────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
		sys.path.insert(0, PROJECT_ROOT)

from models.pointNet import build_pointnet_classifier


def get_flops(model, input_shape):
		"""
		Compute FLOPs using TF v1 profiler on the frozen graph.
		"""
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
		p = argparse.ArgumentParser(description="Inspect PointNet size and FLOPs")
		p.add_argument("--num_particles", type=int, default=150, help="Sequence length (e.g., 150)")
		p.add_argument("--feature_dim", type=int, default=3, help="Input feature dim (e.g., 3 for pt,eta,phi)")
		p.add_argument("--output_dim", type=int, default=5)
		p.add_argument("--dropout", type=float, default=0.3)
		p.add_argument("--print_summary", action="store_true", default=True)
		return p.parse_args()


def main():
		args = parse_args()

		model = build_pointnet_classifier(
				num_particles=args.num_particles,
				feature_dim=args.feature_dim,
				output_dim=args.output_dim,
				dropout_rate=args.dropout,
		)

		params = model.count_params()
		flops = get_flops(model, (1, args.num_particles, args.feature_dim))
		macs = flops // 2

		print("=== PointNet Inspection ===")
		print(f"num_particles = {args.num_particles}, feature_dim = {args.feature_dim}")
		print("---------------------------------------")
		print(f"Total params  = {params:,}")
		print(f"FLOPs (1 x)   = {flops:,}")
		print(f"MACs (approx) = {macs:,}")
		if args.print_summary:
				model.summary()


if __name__ == "__main__":
		main()


