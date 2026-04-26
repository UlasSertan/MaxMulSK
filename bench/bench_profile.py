#!/opt/anaconda3/bin/python
# bench/bench_profile.py
# Single-library, single-size, N-iteration profiling driver for NumPy and
# PyTorch. Output format mirrors bench_profile.cpp / SMETest::profile() so
# profile.sh and profile_power.sh can parse it the same way.
#
# Usage:
#   bench_profile.py <numpy|pytorch> <M> <K> <N> <iters>

import os
import sys
import time

# Pin to single thread BEFORE importing numpy/torch.
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["OPENBLAS_NUM_THREADS"]   = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"]    = "1"

import numpy as np

if len(sys.argv) != 6:
    sys.exit(f"usage: {sys.argv[0]} <numpy|pytorch> M K N iters")

lib = sys.argv[1]
M, K, N, iters = (int(x) for x in sys.argv[2:6])

rng = np.random.default_rng(42)
A_np = rng.random((M, K), dtype=np.float32)
B_np = rng.random((K, N), dtype=np.float32)

if lib == "numpy":
    label = "numpy"
    matmul = lambda: np.matmul(A_np, B_np)
elif lib == "pytorch":
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    label = "pytorch"
    A_pt = torch.from_numpy(A_np)
    B_pt = torch.from_numpy(B_np)
    matmul = lambda: torch.matmul(A_pt, B_pt)
else:
    sys.exit(f"unknown library: {lib}  (expected numpy|pytorch)")

print("  Threads: 1")

matmul()  # warmup
t0 = time.perf_counter()
for _ in range(iters):
    matmul()
sec = time.perf_counter() - t0

gflops      = (2.0 * M * N * K * iters / 1e9) / sec
ms_per_iter = sec * 1000.0 / iters
print(f"  [{label}] {M}x{K}x{N} iters={iters} : {gflops:.1f} GFLOPS ({ms_per_iter:.2f} ms/iter)")
