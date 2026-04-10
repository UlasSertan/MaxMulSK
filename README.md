# MatrixLibrary

High-performance GEMM (General Matrix Multiply) for Apple Silicon, implemented in C++20.

Explores and benchmarks three levels of compute on ARM:

| Kernel | ISA | Strategy | GFLOPS (single-thread) |
|--------|-----|----------|------------------------|
| Scalar | — | Naive triple-loop reference | ~2 |
| NEON | ARMv8.4 | 8×12 micro-kernel, 4× K-unroll, in-register transpose | ~123 |
| NEON + OpenMP | ARMv8.4 | Above + multi-thread, dynamic scheduling | ~539 |
| SME | ARMv8.7 + SME/SME2 | 4×SVL outer-product micro-kernel using ZA accumulator tiles | WIP |

Both optimized kernels use cache-blocking (Mc=64, Kc=256, Nc=1020) and pack A/B into contiguous, kernel-friendly layouts before computing. Arbitrary matrix dimensions are handled correctly: non-aligned M/K/N tails are covered with zero impact on the aligned hot path. See [BENCHMARKS.md](BENCHMARKS.md) for full experimental results including power/energy analysis, cache behavior, instruction profiling, and micro-kernel tuning experiments.

## Single-Thread Comparison vs Industry Libraries

Benchmarked on Apple M4, single-threaded float32 GEMM, all thread counts pinned to 1. Full methodology and raw tables in `bench/COMPARISON.md`.

| Library | Backend | Peak GFLOPS | Notes |
|---------|---------|-------------|-------|
| Accelerate (`cblas_sgemm`) | AMX | ~1841 | Apple's undocumented matrix coprocessor |
| PyTorch 2.10 | AMX | ~1777 | Same hardware path as Accelerate |
| **Our NEON kernel** | NEON | **~123** | At theoretical NEON ceiling |
| NumPy 2.1 (macOS) | vecLib BLAS | ~115 | ~10% slower than ours |
| OpenBLAS 0.3.32 | NEON (aligned) / degrades | ~105–1673 | Erratic on non-aligned sizes |

**Key findings:**
- Accelerate and PyTorch use AMX — a dedicated matrix coprocessor, not a NEON implementation. The gap (~14×) is hardware, not algorithmic.
- Our kernel is **~10% faster than NumPy** on all large matrix sizes. NumPy's vecLib path incurs a row-major/column-major translation overhead that our natively row-major kernel avoids.
- Our kernel **beats OpenBLAS on all non-aligned sizes** (e.g. 1024×1020×1024, 513×509×513). OpenBLAS's tail handling degrades significantly outside its alignment assumptions; ours does not.
- We are within 2% of the M4's theoretical single-core NEON ceiling. Further NEON micro-optimization has diminishing returns — SME is the next step.

## Structure

```
MatrixLibrary/
├── common/
│   ├── utils.hpp              # Shared: fill_random, check_correctness, compute_gflops, scalar reference
│   └── Matrix.hpp             # Generic matrix class (addition, subtraction)
├── neon/
│   ├── GEMMKernels.hpp/.cpp   # NEON kernel + packing — complete
│   └── test_neon.hpp/.cpp     # Packing correctness + GEMM correctness vs scalar
├── sme/
│   ├── SME-GEMMKernels.hpp/.cpp  # SME/SVE kernel + packing — WIP, excluded from build
│   └── test_sme.hpp/.cpp         # SME tests — WIP, excluded from build
├── bench/
│   ├── bench_compare.cpp      # C++ single-thread comparison: NEON vs Accelerate vs OpenBLAS
│   ├── bench_python.py        # Python single-thread comparison: NumPy vs PyTorch
│   ├── run_bench.sh           # One-command build + run for both benchmarks
│   └── COMPARISON.md          # Full benchmark report with analysis
├── main.cpp                   # Test runner + benchmark harness
├── CMakeLists.txt
├── run.sh                     # One-command build + run for main binary
└── BENCHMARKS.md              # Detailed experimental results (power, cache, instructions)
```

## Requirements

- Apple M4 (SME/SME2 required for the SME kernel)
- LLVM/Clang via Homebrew (`/opt/homebrew/opt/llvm`)
- libomp (`/opt/homebrew/opt/libomp`)
- CMake 3.30+
- OpenBLAS (`brew install openblas`) — for `bench_compare` only
- Anaconda Python with `numpy` and `torch` — for `bench_python.py` only

## Build & Run

```bash
# Main binary (NEON tests + benchmark)
./run.sh

# Single-thread competitor comparison
./bench/run_bench.sh
```

## Output

Running `./run.sh` executes NEON unit tests (correctness vs scalar reference across all tail combinations), a speed sweep across aligned and edge-case sizes, and a 50-iteration stress test at 1024^3.

## Roadmap

- [x] Edge-case handling for NEON (non-divisible M/K/N)
- [x] OpenMP parallelism with correct thread structure and dynamic scheduling
- [x] Single-thread competitor benchmark (Accelerate, OpenBLAS, NumPy, PyTorch)
- [ ] Fix SME bugs (BUG-1 through BUG-6, see TODO.md)
- [ ] Re-enable SME in CMakeLists and main, run full correctness suite
- [ ] Benchmark SME vs NEON vs Accelerate
- [ ] Rust scheduling layer for heterogeneous work distribution (P-core vs E-core)
