# MatrixLibrary

High-performance GEMM (General Matrix Multiply) for Apple Silicon, implemented in C++20.

Explores and benchmarks three levels of compute on ARM:

| Kernel | ISA | Strategy | GFLOPS (single-thread) |
|--------|-----|----------|------------------------|
| Scalar | — | Naive triple-loop reference | ~2 |
| NEON | ARMv8.4 | 8×12 micro-kernel, 4× K-unroll, in-register transpose | ~123 |
| NEON + OpenMP | ARMv8.4 | Above + multi-thread, dynamic scheduling | ~539 |
| SME 4×1 | ARMv8.7 + SME/SME2 | 4×SVL outer-product micro-kernel, K_tile=2048 | ~1213 |
| SME 2×2 | ARMv8.7 + SME/SME2 | 2×2 SVL outer-product, interleaved B packing | ~1083 |

All optimized kernels use cache-blocking and pack A/B into contiguous, kernel-friendly layouts before computing. Arbitrary matrix dimensions are handled correctly: non-aligned M/K/N tails are covered with zero impact on the aligned hot path. The SME kernels use SVE predicated loads for tail handling and ZA tile accumulators for outer-product computation. Two SME micro-kernel layouts are implemented (4×1 and 2×2) for A/B testing; 4×1 wins on large matrices due to better B-reuse. See [BENCHMARKS.md](BENCHMARKS.md) for full experimental results including power/energy analysis, cache behavior, instruction profiling, and micro-kernel tuning experiments.

## Single-Thread Comparison vs Industry Libraries

Benchmarked on Apple M4, single-threaded float32 GEMM, all thread counts pinned to 1. Full methodology and raw tables in `bench/COMPARISON.md`.

| Library | Backend | Peak GFLOPS | Notes |
|---------|---------|-------------|-------|
| Accelerate (`cblas_sgemm`) | AMX | ~1813 | Apple's undocumented matrix coprocessor |
| PyTorch 2.10 | AMX | ~1815 | Same hardware path as Accelerate |
| **Our SME 4×1 kernel** | SME/ZA | **~1213** | 67% of AMX, ~10× faster than NEON |
| **Our SME 2×2 kernel** | SME/ZA | **~1083** | Alternative tile layout for comparison |
| Our NEON kernel | NEON | ~123 | At theoretical NEON ceiling |
| NumPy 2.1 (macOS) | vecLib BLAS | ~115 | ~10% slower than NEON kernel |
| OpenBLAS 0.3.32 | NEON (aligned) / degrades | ~105–1673 | Erratic on non-aligned sizes |

**Key findings:**
- **SME closes the gap with AMX to ~1.5×.** Our SME 4×1 kernel peaks at ~1213 GFLOPS vs Accelerate's ~1813 — a 67% ratio, up from the ~7% ratio with NEON alone.
- **SME is ~10× faster than NEON** on the same single core, confirming that the ZA accumulator is a fundamentally different compute tier.
- **4×1 beats 2×2 on large matrices** (~1213 vs ~1083 at 4096³) due to better B-panel reuse. 2×2 is slightly ahead at 1024³ (~1083 vs ~1071).
- Accelerate and PyTorch use AMX — a dedicated matrix coprocessor. The remaining ~1.5× gap is hardware, not algorithmic.
- NEON kernel remains ~10% faster than NumPy. OpenBLAS remains erratic on non-aligned sizes.
- **Known issue:** Accumulator precision degrades at 2048+ sizes (MaxDiff up to ~100). Pairwise summation is planned to fix this.

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
│   ├── SME-GEMMKernels.hpp/.cpp      # SME 4×1 kernel + packing — optimized
│   ├── SME-GEMMKernels2x2.hpp/.cpp   # SME 2×2 kernel + interleaved B packing
│   └── test_sme.hpp/.cpp             # Packing + correctness + benchmark + 4x1 vs 2x2 comparison
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
- [x] Fix SME bugs (BUG-1 through BUG-6 + butterfly permutation + software pipelining)
- [x] Re-enable SME in CMakeLists and main, run full correctness suite
- [x] Optimize SME kernel (K_tile=2048, 2×2 kernel variant, interleaved B packing)
- [x] Benchmark SME vs NEON vs Accelerate (single-thread: ~1213 GFLOPS peak)
- [ ] Fix accumulator precision at large sizes (pairwise summation)
- [ ] SME multi-threading
- [ ] Rust scheduling layer for heterogeneous work distribution (P-core vs E-core)
