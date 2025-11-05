#!/usr/bin/env python3
"""
Minimal, self-contained Transformer attention/MLP benchmark for CPU / CUDA / MPS.
This is a drop-in working version for Windows (CUDA) or macOS (MPS) or CPU-only.

Outputs:
  - bench_results.csv with latency (p50/p95), FLOPs, achieved TFLOPs, and peak CUDA memory.
  - traces/*.json Chrome traces from torch.profiler for each configuration.

Example runs:
  python main.py --devices cpu --modes attention mlp block --dtype fp32
  python main.py --devices cuda --dtype fp32 bf16 --sdpa-backend flash
  python main.py --devices mps  --dtype fp32 bf16

Optional libraries for FLOPs: fvcore, thop
  pip install fvcore thop
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.benchmark import Timer

# ---------------------- Utilities ----------------------

def set_seed(seed: int = 1234):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_ok(device: str) -> bool:
    if device == "cpu":
        return True
    if device == "cuda":
        return torch.cuda.is_available()
    if device == "mps":
        return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    return False


def synchronize(device: str):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, 'mps') and torch.backends.mps.is_available():
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def to_dtype(dtype_str: str):
    mapping = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return mapping[dtype_str]


# ---------------------- Models ----------------------

class AttentionOnly(nn.Module):
    def __init__(self, d_model: int, num_heads: int, head_dim: int, bias: bool = False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        assert d_model == num_heads * head_dim, "d_model must equal num_heads * head_dim"
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x, attn_impl: str = "auto"):
        B, S, D = x.shape
        H = self.num_heads
        Dh = self.head_dim
        q = self.q_proj(x).view(B, S, H, Dh).transpose(1, 2)  # [B,H,S,Dh]
        k = self.k_proj(x).view(B, S, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, S, H, Dh).transpose(1, 2)

        # Select SDPA backend if available; otherwise PyTorch will choose a fallback
        if attn_impl in {"math", "flash", "efficient"}:
            set_sdpa_backend(attn_impl)
        elif attn_impl == "auto":
            set_sdpa_backend(None)  # reset to defaults

        # scaled dot-product attention expects [B,H,S,Dh]
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(y)


class MLPOnly(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: int = 4, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * mlp_ratio, bias=bias)
        self.fc2 = nn.Linear(d_model * mlp_ratio, d_model, bias=bias)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, head_dim: int, mlp_ratio: int = 4, bias: bool = False):
        super().__init__()
        self.attn = AttentionOnly(d_model, num_heads, head_dim, bias)
        self.mlp = MLPOnly(d_model, mlp_ratio, bias=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_impl: str = "auto"):
        x = x + self.attn(self.ln1(x), attn_impl=attn_impl)
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------- SDPA backend control ----------------------

def set_sdpa_backend(backend: Optional[str]):
    """Try to switch SDPA backend when available. None resets to defaults."""
    # Only defined for CUDA; guard attributes to avoid crashes on CPU-only builds
    try:
        if not hasattr(torch.backends, 'cuda'):
            return
        if backend is None:
            # Reset to default heuristics.
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(True)
            if hasattr(torch.backends.cuda, 'enable_math_sdp'):
                torch.backends.cuda.enable_math_sdp(True)
            if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            return
        backend = backend.lower()
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(backend == "flash")
        if hasattr(torch.backends.cuda, 'enable_math_sdp'):
            torch.backends.cuda.enable_math_sdp(backend == "math")
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(backend == "efficient")
    except Exception:
        pass


# ---------------------- FLOPs estimation ----------------------

@dataclass
class FlopBreakdown:
    qkv: float
    attn_scores: float
    attn_apply: float
    proj_out: float
    mlp: float

    @property
    def total(self):
        return self.qkv + self.attn_scores + self.attn_apply + self.proj_out + self.mlp


def linear_gemm_flops(n: int, m: int, k: int) -> float:
    # GEMM ~ 2*n*m*k FLOPs (multiply+add)
    return 2.0 * n * m * k


def estimate_flops(mode: str, B: int, S: int, H: int, Dh: int, d_model: int, mlp_ratio: int = 4) -> FlopBreakdown:
    # QKV projections: 3 * [B*S, D] x [D, D]
    qkv = 3.0 * linear_gemm_flops(B * S, d_model, d_model)
    # Attention scores QK^T: per head: [B*H, S, Dh] x [B*H, Dh, S]
    attn_scores = linear_gemm_flops(B * H, S, Dh) * S  # = B*H*2*S*S*Dh
    # Attention apply AV: [B*H, S, S] x [B*H, S, Dh]
    attn_apply = linear_gemm_flops(B * H, S, S) * Dh  # = B*H*2*S*S*Dh
    # Output projection: [B*S, D] x [D, D]
    proj_out = linear_gemm_flops(B * S, d_model, d_model)
    # MLP: two linears with expansion r
    Dff = d_model * mlp_ratio
    mlp = 2.0 * linear_gemm_flops(B * S, d_model, Dff)  # fc1 + fc2

    if mode == "attention":
        mlp = 0.0
    elif mode == "mlp":
        qkv = attn_scores = attn_apply = proj_out = 0.0
    # else: full block

    return FlopBreakdown(qkv, attn_scores, attn_apply, proj_out, mlp)


# ---------------------- Benchmarking ----------------------

def run_latency(model, inputs, device: str, mode: str, attn_impl: str, warmup: int, min_run_time: float) -> Tuple[float, float]:
    """Manual timing loop to compute p50/p95; avoids relying on Measurement.stdev."""
    model.eval()
    fn = (lambda: model(inputs, attn_impl=attn_impl)) if mode != "mlp" and hasattr(model, 'forward') and 'attn_impl' in model.forward.__code__.co_varnames else (lambda: model(inputs))

    # Warmup
    for _ in range(warmup):
        _ = fn()
        synchronize(device)

    # Manual timing loop for robust percentiles
    # Choose iteration count based on min_run_time (roughly ~50 iters/sec) but at least 20
    iters = max(20, int(min_run_time * 50))
    samples_ms: List[float] = []

    for _ in range(iters):
        t0 = time.perf_counter()
        _ = fn()
        synchronize(device)
        t1 = time.perf_counter()
        samples_ms.append((t1 - t0) * 1e3)

    samples_ms.sort()
    n = len(samples_ms)
    p50 = samples_ms[n // 2]
    p95 = samples_ms[int(0.95 * (n - 1))]
    return p50, p95


def profile_once(model, inputs, device: str, trace_path: str, mode: str, attn_impl: str):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda" and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        if mode != "mlp":
            _ = model(inputs, attn_impl=attn_impl)
        else:
            _ = model(inputs)

    try:
        prof.export_chrome_trace(trace_path)
    except Exception as e:
        print(f"[warn] could not export trace: {e}")


def get_peak_mem_mb(device: str) -> Optional[float]:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        return peak / (1024 ** 2)
    return None


# ---------------------- Main ----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', nargs='+', default=['cpu'], help='cpu cuda mps')
    parser.add_argument('--modes', nargs='+', default=['attention', 'mlp', 'block'], choices=['attention', 'mlp', 'block'])
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 4, 8])
    parser.add_argument('--seq-lens', type=int, nargs='+', default=[128, 512, 2048])
    parser.add_argument('--num-heads', type=int, nargs='+', default=[8, 16])
    parser.add_argument('--head-dim', type=int, nargs='+', default=[64, 128])
    parser.add_argument('--mlp-ratio', type=int, default=4)
    parser.add_argument('--dtype', nargs='+', default=['fp32', 'bf16'], choices=['fp32', 'fp16', 'bf16'])
    parser.add_argument('--sdpa-backend', default='auto', choices=['auto', 'math', 'flash', 'efficient'])
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--min-run-time', type=float, default=0.3, help='seconds per Timer blocked_autorange')
    parser.add_argument('--csv', default='bench_results.csv')
    parser.add_argument('--trace-dir', default='traces')
    parser.add_argument('--cpu-threads', type=int, default=None, help='limit intraop threads on CPU')
    parser.add_argument('--flops-tool', default='analytic', choices=['analytic', 'fvcore', 'thop'])

    args = parser.parse_args()

    os.makedirs(args.trace_dir, exist_ok=True)

    if args.cpu_threads is not None:
        try:
            torch.set_num_threads(args.cpu_threads)
        except Exception:
            pass

    set_seed(1234)

    # Optional FLOPs tools
    fvcore_get_flops = None
    thop_profile = None
    if args.flops_tool == 'fvcore':
        try:
            from fvcore.nn import FlopCountAnalysis
            fvcore_get_flops = FlopCountAnalysis
        except Exception:
            print('[warn] fvcore not available; falling back to analytic')
            args.flops_tool = 'analytic'
    if args.flops_tool == 'thop':
        try:
            from thop import profile as thop_profile  # type: ignore
        except Exception:
            print('[warn] thop not available; falling back to analytic')
            args.flops_tool = 'analytic'

    # CSV header
    first_write = not os.path.exists(args.csv)
    csv_file = open(args.csv, 'a', newline='')
    writer = csv.writer(csv_file)
    if first_write:
        writer.writerow(['device', 'mode', 'dtype', 'B', 'S', 'H', 'Dh', 'Dmodel', 'sdpa_backend', 'p50_ms', 'p95_ms', 'FLOPs', 'achieved_TFLOPs', 'peak_mem_MB', 'trace'])

    for device in args.devices:
        if not device_ok(device):
            print(f"[skip] device {device} not available")
            continue

        for mode in args.modes:
            for dtype_str in args.dtype:
                dtype = to_dtype(dtype_str)

                for B in args.batch_sizes:
                    for S in args.seq_lens:
                        for H in args.num_heads:
                            for Dh in args.head_dim:
                                D = H * Dh

                                # Build module per mode
                                if mode == 'attention':
                                    module = AttentionOnly(D, H, Dh)
                                elif mode == 'mlp':
                                    module = MLPOnly(D, args.mlp_ratio)
                                else:
                                    module = TransformerBlock(D, H, Dh, mlp_ratio=args.mlp_ratio)

                                # Move & dtype
                                dev = torch.device(device)
                                module.to(dev, dtype=dtype)

                                # Inputs
                                x = torch.randn(B, S, D, device=dev, dtype=dtype)

                                # Correctness sanity vs fp32 CPU (small shape only)
                                if (B, S) == (1, 16):
                                    with torch.no_grad():
                                        try:
                                            module_cpu = module.to('cpu', dtype=torch.float32)
                                            x_cpu = x.to('cpu', dtype=torch.float32)
                                            if mode == 'mlp':
                                                ref = module_cpu(x_cpu)
                                                out = module(x)
                                            else:
                                                ref = module_cpu(x_cpu, attn_impl=args.sdpa_backend if args.sdpa_backend != 'auto' else 'math')
                                                out = module(x, attn_impl=args.sdpa_backend)
                                            atol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
                                            ok = torch.allclose(out.float().cpu(), ref.float(), atol=atol, rtol=1e-3)
                                            if not ok:
                                                print(f"[warn] correctness check failed for {device} {mode} {dtype_str} H={H} Dh={Dh}")
                                        except Exception as e:
                                            print(f"[warn] correctness check skipped: {e}")

                                # Reset CUDA peak mem stats before timing
                                if device == 'cuda' and torch.cuda.is_available():
                                    torch.cuda.reset_peak_memory_stats()

                                # Latency
                                p50, p95 = run_latency(module, x, device, mode, args.sdpa_backend, args.warmup, args.min_run_time)

                                # FLOPs
                                flop_bd = estimate_flops(mode, B, S, H, Dh, D, args.mlp_ratio)
                                total_flops = flop_bd.total

                                if args.flops_tool == 'fvcore' and fvcore_get_flops is not None:
                                    try:
                                        inputs = (x,)
                                        from fvcore.nn import FlopCountAnalysis
                                        fca = FlopCountAnalysis(module, inputs)
                                        total_flops = float(fca.total())
                                    except Exception as e:
                                        print(f"[warn] fvcore failed: {e}; using analytic")
                                        total_flops = flop_bd.total
                                elif args.flops_tool == 'thop' and thop_profile is not None:
                                    try:
                                        macs, _ = thop_profile(module, inputs=(x,), verbose=False)
                                        total_flops = float(macs) * 2.0  # 1 MAC ~ 2 FLOPs
                                    except Exception as e:
                                        print(f"[warn] thop failed: {e}; using analytic")
                                        total_flops = flop_bd.total

                                achieved_tflops = (total_flops / (p50 / 1e3)) / 1e12  # use p50

                                # Memory
                                peak_mem_mb = get_peak_mem_mb(device)

                                # Profile once per config
                                trace_name = f"trace_{device}_{mode}_{dtype_str}_B{B}_S{S}_H{H}_Dh{Dh}.json".replace('/', '-')
                                trace_path = os.path.join(args.trace_dir, trace_name)
                                try:
                                    profile_once(module, x, device, trace_path, mode, args.sdpa_backend)
                                except Exception as e:
                                    print(f"[warn] profiling failed: {e}")
                                    trace_path = ''

                                # Write CSV
                                writer.writerow([
                                    device, mode, dtype_str, B, S, H, Dh, D, args.sdpa_backend, f"{p50:.3f}", f"{p95:.3f}",
                                    f"{total_flops:.3e}", f"{achieved_tflops:.3f}", f"{peak_mem_mb if peak_mem_mb is not None else ''}", trace_path
                                ])
                                csv_file.flush()

    csv_file.close()
    print(f"Done. Results in {args.csv}. Traces in {args.trace_dir}/")


if __name__ == '__main__':
    main()
