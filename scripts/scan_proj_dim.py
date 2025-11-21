#!/usr/bin/env python
"""
Scan proj_dim from 1 to 30 for Linformer and SAL-T (PyTorch, 1-layer),
printing params, FLOPs, time, and peak memory for each proj_dim.
"""
import os
import sys
import time
import csv
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

# ─── import HEPT-Zihan linformer/SAL-T attention ──────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HEPT_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "HEPT-Zihan", "src"))
if HEPT_SRC not in sys.path:
		sys.path.insert(0, HEPT_SRC)
from models.multiConvAttention import MultiheadLinearAttention  # type: ignore

# Use the exact same implementations and measurement routine as benchmark_pytorch_models.py
try:
	# Same-directory import
	import benchmark_pytorch_models as bm
	from benchmark_pytorch_models import Linformer1L as BmLinformer1L
	from benchmark_pytorch_models import SALT1L as BmSALT1L
	from benchmark_pytorch_models import FlashTransformer1L as BmFlashTransformer1L
	from benchmark_pytorch_models import measure as bm_measure
	from benchmark_pytorch_models import num_params as bm_num_params
except Exception:
	# Fallback to package-style import if executed from repo root
	from scripts import benchmark_pytorch_models as bm  # type: ignore
	from scripts.benchmark_pytorch_models import Linformer1L as BmLinformer1L  # type: ignore
	from scripts.benchmark_pytorch_models import SALT1L as BmSALT1L  # type: ignore
	from scripts.benchmark_pytorch_models import FlashTransformer1L as BmFlashTransformer1L  # type: ignore
	from scripts.benchmark_pytorch_models import measure as bm_measure  # type: ignore
	from scripts.benchmark_pytorch_models import num_params as bm_num_params  # type: ignore

class Linformer1L(nn.Module):
	def __init__(self, in_dim: int, d_model: int, num_heads: int, seq_len: int, proj_dim: int):
		super().__init__()
		self.embed = nn.Linear(in_dim, d_model, bias=False)
		self.attn = MultiheadLinearAttention(
			embed_dim=d_model,
			num_heads=num_heads,
			max_seq_len=seq_len,
			proj_dim=proj_dim,
			cluster_E=False,
			cluster_F=False,
			share_EF=False,
			convolution=False,
			self_attention=True,
		)
		self.norm1 = nn.LayerNorm(d_model)
		self.ffn = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
		self.norm2 = nn.LayerNorm(d_model)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.embed(x)
		q = x.transpose(0, 1)
		out, _ = self.attn(q, None, None, need_weights=False)
		out = out.transpose(0, 1)
		y = self.norm1(x + out)
		y = self.norm2(y + self.ffn(y))
		return y


class SALT1L(nn.Module):
	def __init__(self, in_dim: int, d_model: int, num_heads: int, seq_len: int, proj_dim: int, conv_filter_heights=None, vertical_stride: int = 1,
							 cluster_E: bool = True, cluster_F: bool = True, share_EF: bool = False):
		super().__init__()
		if conv_filter_heights is None:
			conv_filter_heights = [1, 3, 5]
		self.embed = nn.Linear(in_dim, d_model, bias=False)
		self.attn = MultiheadLinearAttention(
			embed_dim=d_model,
			num_heads=num_heads,
			max_seq_len=seq_len,
			proj_dim=proj_dim,
			cluster_E=cluster_E,
			cluster_F=cluster_F,
			share_EF=share_EF,
			convolution=True,
			conv_filter_heights=conv_filter_heights,
			vertical_stride=vertical_stride,
			self_attention=True,
		)
		self.norm1 = nn.LayerNorm(d_model)
		self.ffn = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
		self.norm2 = nn.LayerNorm(d_model)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.embed(x)
		q = x.transpose(0, 1)
		out, _ = self.attn(q, None, None, need_weights=False)
		out = out.transpose(0, 1)
		y = self.norm1(x + out)
		y = self.norm2(y + self.ffn(y))
		return y


def num_params(model: nn.Module) -> int:
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure(model: nn.Module, x: torch.Tensor, device: str = "cuda"):
	model.eval().to(device)
	x = x.to(device)
	# warmup
	for _ in range(5):
		with torch.inference_mode():
			_ = model(x)
	if torch.cuda.is_available():
		torch.cuda.synchronize()
	# timing
	times = []
	for _ in range(20):
		t0 = time.perf_counter()
		with torch.inference_mode():
			_ = model(x)
		if torch.cuda.is_available():
			torch.cuda.synchronize()
		times.append(time.perf_counter() - t0)
	avg_ns = (sum(times) / len(times)) / x.size(0) * 1e9
	# peak mem
	if torch.cuda.is_available():
		torch.cuda.reset_peak_memory_stats()
		with torch.inference_mode():
			_ = model(x)
		if torch.cuda.is_available():
			torch.cuda.synchronize()
		peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
	else:
		peak_mb = 0.0
	# FLOPs
	flops = None
	try:
		# Suppress Kineto profiler console logs
		import os as _os
		from contextlib import redirect_stderr as _redirect_stderr
		with open(_os.devnull, "w") as _devnull, _redirect_stderr(_devnull):
			with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_flops=True) as prof:
				with torch.inference_mode():
					_ = model(x)
		total = 0
		for evt in prof.key_averages():
			evt_flops = getattr(evt, "flops", None)
			if isinstance(evt_flops, (int, float)):
				total += int(evt_flops)
		flops = total if total > 0 else None
	except Exception:
		flops = None
	return avg_ns, peak_mb, flops


def main():
	parser = argparse.ArgumentParser(description="Scan proj_dim for Linformer/SAL-T (1-layer)")
	parser.add_argument("--model", choices=["linformer", "salt", "both"], default="both")
	parser.add_argument("--seq_len", type=int, default=150)
	parser.add_argument("--in_dim", type=int, default=3)
	parser.add_argument("--d_model", type=int, default=16)
	parser.add_argument("--num_heads", type=int, default=4)
	parser.add_argument("--batch_size", type=int, default=512)
	parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--conv_filter_heights", type=int, nargs="+", default=[1, 3, 5])
	parser.add_argument("--vertical_stride", type=int, default=1)
	parser.add_argument("--out_csv", type=str, default="scan_proj_dim_results.csv")
	parser.add_argument("--use_flash", action="store_true", help="Use Flash SDP for the one-time Transformer comparison")
	args = parser.parse_args()

	B, T, Fin, D, H = args.batch_size, args.seq_len, args.in_dim, args.d_model, args.num_heads
	x = torch.randn(B, T, Fin)

	header_cols = ["proj_dim"]
	if args.model in ("linformer", "both"):
		header_cols += ["linf_params", "linf_flops", "linf_ns", "linf_peakMB"]
	if args.model in ("salt", "both"):
		header_cols += ["salt_params", "salt_flops", "salt_ns", "salt_peakMB"]
	# Always include flash columns for a one-time comparison row
	header_cols += ["flash_params", "flash_flops", "flash_ns", "flash_peakMB"]
	rows = []

	for proj_dim in range(1, 31):
		row = [proj_dim]
		if args.model in ("linformer", "both"):
			linf = BmLinformer1L(Fin, D, H, seq_len=T, proj_dim=proj_dim)
			lp = bm_num_params(linf)
			lns, lmb, lf = bm_measure(linf, x, device=args.device)
			row += [lp, (lf if lf is not None else "NA"), float(f"{lns:.2f}"), float(f"{lmb:.1f}")]
		else:
			# placeholders if not selected
			row += ["NA", "NA", "NA", "NA"]
		if args.model in ("salt", "both"):
			salt = BmSALT1L(Fin, D, H, seq_len=T, proj_dim=proj_dim, conv_filter_heights=args.conv_filter_heights, vertical_stride=args.vertical_stride)
			sp = bm_num_params(salt)
			sns, smb, sf = bm_measure(salt, x, device=args.device)
			row += [sp, (sf if sf is not None else "NA"), float(f"{sns:.2f}"), float(f"{smb:.1f}")]
		else:
			row += ["NA", "NA", "NA", "NA"]
		# placeholders for flash columns on sweep rows
		row += ["NA", "NA", "NA", "NA"]
		rows.append(row)

	# One-time Flash Transformer comparison
	flash = BmFlashTransformer1L(Fin, D, H, use_flash=args.use_flash)
	fp = bm_num_params(flash)
	fns, fmb, ff = bm_measure(flash, x, device=args.device)
	flash_row = ["flash"]
	# place NA for linformer/salt groups (8 columns if both), then flash metrics
	if args.model in ("linformer", "both"):
		flash_row += ["NA", "NA", "NA", "NA"]
	if args.model in ("salt", "both"):
		flash_row += ["NA", "NA", "NA", "NA"]
	flash_row += [fp, (ff if ff is not None else "NA"), float(f"{fns:.2f}"), float(f"{fmb:.1f}")]
	rows.append(flash_row)

	# write CSV
	with open(args.out_csv, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(header_cols)
		for r in rows:
			writer.writerow(r)


if __name__ == "__main__":
	main()


