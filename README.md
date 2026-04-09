# MatrixLibrary

High-performance GEMM (General Matrix Multiply) for Apple Silicon, implemented in C++20.

Explores and benchmarks three levels of compute on ARM:

| Kernel | ISA | Strategy | GFLOPS |
|--------|-----|----------|--------|
| Scalar | — | Naive triple-loop reference | ~2 |
| NEON | ARMv8.4 | 8×12 micro-kernel, 4× K-unroll, in-register transpose | ~120 |
| NEON + OpenMP | ARMv8.4 | Above + 8-thread parallelism | ~456 |
| SME | ARMv8.7 + SME/SME2 | 4×SVL outer-product micro-kernel using ZA accumulator tiles | WIP |

Both optimized kernels use cache-blocking (Mc=64, Kc=256) and pack A/B into contiguous, kernel-friendly layouts before computing. Arbitrary matrix dimensions are handled correctly: non-aligned M/K/N fall back to scalar for the tail regions with zero impact on the aligned hot path. See [BENCHMARKS.md](BENCHMARKS.md) for full experimental results including power/energy analysis, cache behavior, instruction profiling, and micro-kernel tuning experiments.

## Structure

```
MatrixLibrary/
├── common/
│   ├── utils.hpp          # Shared: fill_random, check_correctness, compute_gflops, scalar reference
│   └── Matrix.hpp         # Generic matrix class (addition, subtraction)
├── neon/
│   ├── GEMMKernels.hpp/.cpp   # NEON kernel + packing
│   └── test_neon.hpp/.cpp     # Packing correctness + GEMM correctness vs scalar
├── sme/
│   ├── SME-GEMMKernels.hpp/.cpp  # SME/SVE kernel + packing
│   └── test_sme.hpp/.cpp         # Packing correctness + GEMM correctness vs scalar
├── main.cpp               # Test runner + benchmark harness
├── CMakeLists.txt
└── BENCHMARKS.md          # Detailed experimental results
```

## Requirements

- Apple M4 (SME/SME2 required for the SME kernel)
- LLVM/Clang via Homebrew (`/opt/homebrew/opt/llvm`)
- libomp (`/opt/homebrew/opt/libomp`)
- CMake 3.30+

## Build

```bash
cmake -B cmake-build-release -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release
./cmake-build-release/MatrixLibrary
```

## Output

Running the binary executes unit tests for both kernels, followed by a single timed run and a 50-iteration stress test:

```
========== NEON Tests ==========
  transpose_8x4 : PASS
  Small  ( 64x60x64  ) : PASS
  ...

========== SME Tests ==========
  pack_B : PASS
  pack_A : PASS
  Small  ( 16x16x16 ) : PASS
  ...

--- Single Run (correctness + timing) ---
  Scalar : ~2 GFLOPS
  NEON   : ~120 GFLOPS  (~60x speedup)
  SME    : WIP
```

## Roadmap

- [ ] Edge-case handling for NEON (non-divisible M/K/N)
- [ ] Complete SME tail handling and validation
- [ ] Rust scheduling layer for work distribution
