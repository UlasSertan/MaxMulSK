#!/usr/bin/env python3
# bench/bench_python.py
# Single-threaded GEMM benchmark: NumPy vs PyTorch (float32)
# Run via bench/run_bench.sh or directly.

import os
import time

# Force single thread BEFORE importing numpy/torch
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["OPENBLAS_NUM_THREADS"]   = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"]    = "1"

import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# =============================================================================
# Sizes — mirror bench_compare.cpp
# =============================================================================
cases = [
    ( 8,   8,   8, "tiny"),
    (16,  16,  16, "tiny"),
    (32,  32,  32, "small"),
    (64,  64,  64, "small"),
    (128, 128, 128, "L2"),
    (256, 256, 256, "L2"),
    (512, 512, 512, "L3"),
    (1024, 1024, 1024, "L3"),
    (2048, 2048, 2048, "mem-bound"),
    (1024, 1020, 1024, "N=85x12 aligned"),
    (2048,  512,   64, "tall-skinny"),
    ( 512, 2048,   64, "wide-flat"),
    (  65,   65,   65, "all tails +1"),
    ( 513,  509,  513, "all tails mixed"),
    (1025, 1021, 1025, "all tails large"),
    ( 128, 1100,  128, "N > Nc_cache"),
    ( 512, 2048,  512, "N >> Nc_cache"),
]

def gflops(M, N, K, ms):
    return (2.0 * M * N * K) / (ms * 1e-3) / 1e9

def bench(fn, iters):
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1000 / iters  # avg ms

def iters_for(maxdim):
    if maxdim <=  64: return 10000
    if maxdim <= 128: return  2000
    if maxdim <= 256: return   500
    if maxdim <= 512: return    50
    if maxdim <= 1024: return   20
    if maxdim <= 2048: return    5
    return 3

# =============================================================================
# Header
# =============================================================================
W_SIZE, W_TAG, W_NUM = 22, 18, 12

print("=" * 80)
print("  Single-thread GEMM: NumPy vs PyTorch (float32)")
print(f"  numpy {np.__version__}  |  torch {torch.__version__}")
print("  OMP/MKL/OPENBLAS/VECLIB threads all pinned to 1")
print("=" * 80)
print()

header = (f"{'Size (MxKxN)':<{W_SIZE}}"
          f"{'Tag':<{W_TAG}}"
          f"{'NP ms':>{W_NUM}}"
          f"{'NP GFLOPS':>{W_NUM}}"
          f"{'PT ms':>{W_NUM}}"
          f"{'PT GFLOPS':>{W_NUM}}"
          f"{'MaxDiff':>{10}}")
print(header)
print("-" * (W_SIZE + W_TAG + W_NUM * 4 + 10))

for (M, K, N, tag) in cases:
    iters = iters_for(max(M, N, K))

    rng = np.random.default_rng(42)
    A_np = rng.random((M, K), dtype=np.float32)
    B_np = rng.random((K, N), dtype=np.float32)

    A_pt = torch.from_numpy(A_np)
    B_pt = torch.from_numpy(B_np)

    # NumPy
    ms_np = bench(lambda: np.matmul(A_np, B_np), iters)
    gf_np = gflops(M, N, K, ms_np)

    # PyTorch
    ms_pt = bench(lambda: torch.matmul(A_pt, B_pt), iters)
    gf_pt = gflops(M, N, K, ms_pt)

    # Correctness
    C_np = np.matmul(A_np, B_np)
    C_pt = torch.matmul(A_pt, B_pt).numpy()
    diff = float(np.max(np.abs(C_np - C_pt)))

    size_label = f"{M}x{K}x{N}"
    print(f"{size_label:<{W_SIZE}}"
          f"{tag:<{W_TAG}}"
          f"{ms_np:>{W_NUM}.2f}"
          f"{gf_np:>{W_NUM}.1f}"
          f"{ms_pt:>{W_NUM}.2f}"
          f"{gf_pt:>{W_NUM}.1f}"
          f"{diff:>{10}.5f}")

print()
print("  NP = NumPy (backed by Accelerate on macOS)")
print("  PT = PyTorch (backed by its own BLAS / Accelerate)")
print("  MaxDiff: NumPy vs PyTorch outputs")
