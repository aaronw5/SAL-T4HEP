#!/usr/bin/env python
"""
Benchmark PyTorch one-layer models: Linformer (PyTorch), SAL-T (PyTorch), and a standard Transformer using Flash SDP.
Reports: params, FLOPs (if available via profiler), inference time, peak GPU memory.
"""
import os
import sys
import time
import argparse
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

# ─── import HEPT-Zihan linformer/SAL-T attention ──────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HEPT_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "HEPT-Zihan", "src"))
if HEPT_SRC not in sys.path:
		sys.path.insert(0, HEPT_SRC)
try:
		from models.multiConvAttention import MultiheadLinearAttention  # type: ignore
except Exception as e:
		print("Failed to import MultiheadLinearAttention from HEPT-Zihan:", e)
		raise


# ─── Model blocks ─────────────────────────────────────────────────────────────
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
		# x: [B,T,D]
		x = self.embed(x)
		q = x.transpose(0, 1)  # [T,B,D] for module API
		out, _ = self.attn(q, None, None, need_weights=False)
		out = out.transpose(0, 1)  # [B,T,D]
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


class FlashTransformer1L(nn.Module):
	def __init__(self, in_dim: int, d_model: int, num_heads: int, use_flash: bool = True):
		super().__init__()
		self.d_model = d_model
		self.num_heads = num_heads
		self.head_dim = d_model // num_heads
		assert self.head_dim * num_heads == d_model, "d_model must be divisible by heads"
		self.use_flash = use_flash
		self.embed = nn.Linear(in_dim, d_model, bias=False)
		self.wq = nn.Linear(d_model, d_model, bias=False)
		self.wk = nn.Linear(d_model, d_model, bias=False)
		self.wv = nn.Linear(d_model, d_model, bias=False)
		self.wo = nn.Linear(d_model, d_model, bias=False)
		self.norm1 = nn.LayerNorm(d_model)
		self.ffn = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
		self.norm2 = nn.LayerNorm(d_model)

	def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
		B, T, D = x.shape
		return x.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B,H,T,Dh]

	def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
		B, H, T, Dh = x.shape
		return x.permute(0, 2, 1, 3).contiguous().view(B, T, H * Dh)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# project input features to model width
		x = self.embed(x)
		q = self._split_heads(self.wq(x))
		k = self._split_heads(self.wk(x))
		v = self._split_heads(self.wv(x))
		# scaled dot-product attention (optionally using flash kernels)
		try:
			ctx = torch.backends.cuda.sdp_kernel(
				enable_flash=self.use_flash,
				enable_mem_efficient=self.use_flash,
				enable_math=not self.use_flash,
			)
		except Exception:
			class _Noop:
				def __enter__(self_): return None
				def __exit__(self_, exc_type, exc, tb): return False
			ctx = _Noop()
		with ctx:
			out = F.scaled_dot_product_attention(
				q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
			)  # [B,H,T,Dh]
		out = self._merge_heads(out)
		out = self.wo(out)
		y = self.norm1(x + out)
		y = self.norm2(y + self.ffn(y))
		return y


# ─── Utils ────────────────────────────────────────────────────────────────────
def num_params(model: nn.Module) -> int:
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


@contextlib.contextmanager
def gpu_peak_mem():
	if torch.cuda.is_available():
		torch.cuda.reset_peak_memory_stats()
	yield
	if torch.cuda.is_available():
		peak = torch.cuda.max_memory_allocated() / (1024**2)
	else:
		peak = 0.0
	return


def measure(model: nn.Module, x: torch.Tensor, warmup: int = 5, iters: int = 20, device: str = "cuda"):
	model.eval()
	model.to(device)
	x = x.to(device)
	# warmup
	for _ in range(warmup):
		with torch.inference_mode():
			_ = model(x)
	if torch.cuda.is_available():
		torch.cuda.synchronize()
	# timing
	times = []
	for _ in range(iters):
		t0 = time.perf_counter()
		with torch.inference_mode():
			_ = model(x)
		if torch.cuda.is_available():
			torch.cuda.synchronize()
		times.append(time.perf_counter() - t0)
	avg_per_item_ns = (sum(times) / len(times)) / x.size(0) * 1e9
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
	# FLOPs via profiler (best effort)
	total_flops = None
	try:
		# Suppress Kineto profiler console logs (ActivityProfilerController.cpp)
		import os as _os
		from contextlib import redirect_stderr as _redirect_stderr
		with open(_os.devnull, "w") as _devnull, _redirect_stderr(_devnull):
			with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_flops=True) as prof:
				with torch.inference_mode():
					_ = model(x)
		events = prof.key_averages()
		flops = 0
		for evt in events:
			evt_flops = getattr(evt, "flops", None)
			if isinstance(evt_flops, (int, float)):
				flops += int(evt_flops)
		total_flops = flops if flops > 0 else None
	except Exception:
		total_flops = None
	return avg_per_item_ns, peak_mb, total_flops


def main():
	parser = argparse.ArgumentParser(description="PyTorch Linformer / SAL-T / Flash-Transformer benchmarking (1-layer)")
	parser.add_argument("--seq_len", type=int, default=150)
	parser.add_argument("--in_dim", type=int, default=3, help="Input feature dim (e.g., 3 for pt,eta,phi)")
	parser.add_argument("--d_model", type=int, default=16)
	parser.add_argument("--num_heads", type=int, default=4)
	parser.add_argument("--proj_dim", type=int, default=4, help="Linformer/SAL-T projection dim")
	parser.add_argument("--batch_size", type=int, default=512)
	parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--conv_filter_heights", type=int, nargs="+", default=[1, 3, 5])
	parser.add_argument("--vertical_stride", type=int, default=1)
	parser.add_argument("--use_flash", action="store_true", help="Use Flash SDP for Transformer")
	args = parser.parse_args()

	B, T, Fin, D, H = args.batch_size, args.seq_len, args.in_dim, args.d_model, args.num_heads
	x = torch.randn(B, T, Fin)

	models = {
		"linformer_1L": Linformer1L(Fin, D, H, seq_len=T, proj_dim=args.proj_dim),
		"salt_1L": SALT1L(Fin, D, H, seq_len=T, proj_dim=args.proj_dim, conv_filter_heights=args.conv_filter_heights, vertical_stride=args.vertical_stride),
		"flash_transformer_1L": FlashTransformer1L(Fin, D, H, use_flash=args.use_flash),
	}

	print("=== PyTorch 1-Layer Benchmarks ===")
	print(f"B={B}, T={T}, in_dim={Fin}, D={D}, heads={H}, proj_dim={args.proj_dim}, device={args.device}")
	for name, model in models.items():
		pcount = num_params(model)
		avg_ns, peak_mb, flops = measure(model, x, device=args.device)
		flops_str = f"{flops:,}" if flops is not None else "N/A"
		print(f"- {name}: params={pcount:,}, FLOPs={flops_str}, time={avg_ns:.2f} ns/seq, peak_mem={peak_mb:.1f} MB")


if __name__ == "__main__":
	main()


