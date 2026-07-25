# MatrixLibrary

Hi, I'm Ulaş, a computer engineering student.

This project started after a computer architecture course. I'd just learned about
caches, SIMD, and instruction ordering, and I wanted to actually use all of it
instead of only answering exam questions about it. The other half was plain
curiosity: how far could I push a matrix multiply on my own laptop, and how close
could I get to the libraries that ship with the machine?

Further than I expected, as it turned out.

At first I just wanted to use SIMD. Then I remembered the caching lessons from the
same course and started reading about packing and tiling. I didn't know SME even
existed; I was simply lucky to find that my laptop had it, and from there it became
the natural next step. So even though the code may look coherent and intentional
now, most of the mechanics were learned along the way, and that turned out to be
the most valuable part of the whole thing.

This repo, together with its benchmarks and write-ups, is basically a record of
that journey. It's a study rather than a library you'd drop into a project: the
kernels are here to be read and measured, the documentation is dated so you can
see how the conclusions changed over time, and the experiments that failed are
written up next to the ones that worked.

Hope you find something interesting or useful in here.

## Overview

High-performance GEMM (General Matrix Multiply) for Apple Silicon, implemented in C++20.

Explores and benchmarks three levels of compute on ARM:

| Kernel | ISA | Strategy | GFLOPS (single-thread, peak) |
|--------|-----|----------|------------------------|
| Scalar | — | Naive triple-loop reference | ~2.6 |
| NEON | ARMv8.4 | 8×12 micro-kernel, 4× K-unroll, in-register transpose | ~123 |
| NEON + OpenMP | ARMv8.4 | Above + multi-thread, dynamic scheduling | ~539 |
| SME 4×1 | ARMv8.7 + SME/SME2 | 4×SVL outer-product micro-kernel, K_tile=2048 | ~1231 (2048³) — most energy-efficient |
| SME 2×2 | ARMv8.7 + SME/SME2 | 2×2 SVL outer-product, interleaved B packing | ~1310 (1024³) |
| SME 1×4 | ARMv8.7 + SME/SME2 | B-inner packing, x1 interleaved B loads in FMOPA shadow | ~1236 (1024³) |
| SME 1×4-sym | ARMv8.7 + SME/SME2 | B-inner packing, x4-grouped B loads (true mirror of 4×1) | **~1308 (4096³) — overall peak** |
| SME 1×4ZAIO | ARMv8.7 + SME/SME2 | 1×4-sym with ZA-resident K-accumulation (`__arm_inout("za")`) | ~1272 (2048³) |
| SME 4×1 ZAPack | ARMv8.7 + SME/SME2 | 4×1 with ZA-based pack_A transpose, M_tile=128 | ~1346 (1024×1024×1020) |

All optimized kernels use cache-blocking and pack A/B into contiguous, kernel-friendly layouts. **Every kernel handles arbitrary M/N/K**: full output tiles write directly to C (hot path), partial edge tiles go through a scratch-buffer + scatter-back fallback with zero cost on aligned sizes.

The SME kernels use SVE predicated loads for K-tail handling and ZA tile accumulators for outer-product computation. Six SME micro-kernel variants are implemented; the winner depends on size — no single geometry dominates (see the comparison below). See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for full experimental results: power/energy comparison vs Accelerate / PyTorch / OpenBLAS / NumPy, cache behavior, Instruments PMU profiling, and micro-kernel tuning experiments.

## SME kernel comparison (interleaved, single-thread, 2026-07-25)

| Size | 4×1 | 2×2 | 1×4 | 1×4-sym | 1×4ZAIO | 4×1-ZAPack | Winner |
|---|---:|---:|---:|---:|---:|---:|---|
| 256³ | 715.7 | **766.4** | 729.7 | 731.4 | 609.0 | 762.4 | 2×2 |
| 512³ | 1057.1 | **1111.4** | 1050.2 | 1064.8 | 917.8 | 1110.7 | 2×2 |
| 1024³ | 1142.6 | 1186.2 | 1235.7 | **1243.6** | 1119.8 | 1190.4 | 1×4-sym |
| 2048³ | 1202.7 | 1067.2 | 999.2 | 1128.1 | **1259.0** | 1120.0 | 1×4ZAIO |
| 4096³ | 1173.7 | 1059.1 | 1215.4 | **1308.5** | 1086.7 | 1087.4 | 1×4-sym |

## Single-Thread Comparison vs Industry Libraries

Benchmarked on Apple M4, single-threaded float32 GEMM, threads pinned to 1. Numbers from the 2026-07-25 measurement (`bench/run_bench.sh`). Full table in docs/BENCHMARKS.md §0.2.

| Library | Backend | Peak GFLOPS (4096³ unless noted) | Notes |
|---------|---------|-------------|-------|
| Accelerate (`cblas_sgemm`) | AMX | ~1594 (peak ~1837 at 1024-band) | Apple's matrix coprocessor |
| PyTorch 2.10 | AMX | ~1509 (2048³) | Same backend; Python dispatch overhead |
| **Our SME 1×4-sym** | SME/ZA | **~1308** | Overall single-thread peak — ~82% of AMX at 4096³ |
| **Our SME 4×1** | SME/ZA | **~1200 (4096³), ~1231 (2048³)** | ~75% of AMX; **most energy-efficient variant** at 0.0065 J/GFLOP |
| **Our SME 2×2** | SME/ZA | **~1288 (1024³)** | Wins the ≤512³ band; balanced 32×32 output tile |
| Our NEON kernel | NEON | ~123 | At theoretical NEON ceiling; correct at all shapes |
| NumPy 2.1 (macOS) | vecLib NEON | ~112 | Does *not* take the AMX path for `np.matmul(float32)` |
| OpenBLAS 0.3.32 | NEON | ~113–1590 | Strong at L3 sizes, collapses to ~110 at 2048³+ on non-aligned shapes |

Correctness: MaxDiff vs Accelerate/OpenBLAS ≤ 0.0004 across all 19 benchmark shapes (pure fp32 rounding). Against float64 ground truth at 4096³, our kernels are *more accurate* than Accelerate (maxErr: NEON 0.000115, SME 4×1 0.000209, Accelerate 0.000348 — docs/BENCHMARKS.md §0.3).

### Energy efficiency (2048³ × 100 iters, `profile_power.sh`, 2026-04-26)

| Library | Energy (J) | J / GFLOP | GFLOPS / W |
|---------|-----------:|----------:|-----------:|
| Accelerate (AMX) | 9.44 | **0.0055** | **182** |
| **Our SME 4×1** | 11.18 | **0.0065** | **154** |
| Our SME 2×2 | 11.47 | 0.0067 | 150 |
| Our SME 1×4 | 12.24 | 0.0071 | 140 |
| PyTorch (AMX) | 13.13 | 0.0076 | 131 |
| OpenBLAS | 16.73 | 0.0097 | 103 |
| NumPy | 110.66 | 0.0644 | 16 |

**Key findings:**

- **SME closes the gap with AMX to ~1.2×.** Our peak (1×4-sym, ~1308) reaches ~82% of Accelerate's ~1594 at 4096³. Up from the ~7% ratio with NEON alone.
- **SME is ~10× faster than NEON** on the same single core — the ZA accumulator is a fundamentally different compute tier.
- **Our SME 4×1 beats PyTorch on energy efficiency** (0.0065 vs 0.0076 J/GFLOP) despite running slower in wall-clock terms — PyTorch's dispatcher overhead inflates total energy at 2048³.
- **NumPy is ~10× less efficient than our SME** in J/GFLOP. Single-threaded `np.matmul(float32)` on macOS takes the vecLib NEON path, not AMX; PyTorch's own dispatcher reaches AMX correctly.
- **Size-dependent SME ranking:** 2×2 owns ≤512³, 1×4-sym the 1024³ band and 4096³, 1×4ZAIO wins 2048³. 4×1 is never first but never far behind — it remains the safe default and the most energy-efficient variant.

### A note on the 4×1 vs 1×4 IPC story

The 1×4 (B-inner) kernel is the mathematical transpose of 4×1 (A-inner) — same FMOPA count, same tile area, symmetric load count per k-step. Naively we expected the same throughput, but the 2026-04 investigation found B-inner consistently slower and traced it to FMOPA same-tile dependency chains:

- **x4-load baseline:** one `svld1_f32_x4` per k-step on the B side reached ~1148 GFLOPS but stalled on FMOPA same-tile dependency chains — IPC collapsed to 0.62 (vs 4×1's 1.07) because the instruction stream was too lean to fill FMOPA's 6–8 cycle execution shadow.
- **x1 interleaved variant:** four independent `svld1_f32` loads dropped into the FMOPA shadow. IPC jumped to 1.40 but the ~2.5× instruction-count inflation ate the gain.

**2026-07-25 update:** the story has a new chapter — the same x4-grouped 1×4-sym kernel now *wins* at 4096³ (1308.5 vs 4×1's 1173.7, interleaved measurement), and the ZA-resident-accumulation variant (1×4ZAIO) wins at 2048³. Driver/tiling evolution since the April measurements rehabilitated the B-inner geometry; the IPC analysis in docs/BENCHMARKS.md §10 documents why it was hard, and re-profiling with PMU counters to explain the turnaround is an open item. Full data in docs/BENCHMARKS.md §0.1 and §10.

### Other findings

- Accelerate and PyTorch use AMX — a dedicated coprocessor. The remaining gap to AMX is hardware, not algorithmic.
- All 2026-04 correctness bugs are fixed (2026-07-25): SME small-size heap overruns (scratch-buffer fallback everywhere), the NEON exact-2× bug at `N > Nc_cache` (cache-tile width violated a multiple-of-12 invariant), and the ZAPack wrong-result bug (packing layout collision). Details in TODO.md and docs/BENCHMARKS.md §0.
- The "accumulator precision" issue turned out to be the NEON 2× bug corrupting the comparison baseline — fp64-truth measurement shows our kernels beat Accelerate's accuracy (§0.3).

## Structure

```
MatrixLibrary/
├── common/
│   ├── utils.hpp                  # fill_random, check_correctness, compute_gflops, scalar reference
│   └── Matrix.hpp                 # Generic matrix class (addition, subtraction)
├── neon/
│   ├── neon-8x12.hpp/.cpp         # NEON 8×12 kernel + packing + OpenMP driver
│   └── test_neon.hpp/.cpp         # Packing + GEMM correctness vs scalar
├── sme/
│   ├── sme-4x1.hpp/.cpp               # SME 4×1 kernel + packing — energy-efficiency default
│   ├── sme-2x2.hpp/.cpp               # SME 2×2 kernel + interleaved B packing
│   ├── sme-1x4.hpp/.cpp               # SME 1×4 B-inner kernel (x1 interleaved loads)
│   ├── sme-1x4-sym.hpp/.cpp           # SME 1×4 x4-grouped B loads — overall peak at 4096³
│   ├── sme-1x4-sym-zainout.hpp/.cpp   # 1×4-sym with split zero/compute/store sharing ZA via __arm_inout
│   ├── sme-4x1-zapack.hpp/.cpp        # SME 4×1 with ZA-based pack_A transpose, M_tile=128
│   └── test_sme.hpp/.cpp              # Single dispatch surface: SMETest::Kernel enum + run / run_comparison / run_timing_breakdown / profile
├── bench/
│   ├── bench_compare.cpp          # C++ comparison: NEON / SME / Accelerate / OpenBLAS
│   ├── bench_python.py            # Python comparison: NumPy / PyTorch
│   ├── bench_profile.cpp          # Single-library driver (accel|oblas) for profiling scripts
│   ├── bench_profile.py           # Single-library driver (numpy|pytorch)
│   └── run_bench.sh               # Build + run both comparison benchmarks
├── experiments/
│   └── matmul.cpp                 # Standalone N=2048 head-to-head: our SME kernel vs Accelerate
├── scripts/
│   ├── run.sh                     # Build + run main binary
│   ├── run_matmul.sh              # Build + run the matmul experiment (vs Accelerate)
│   ├── profile.sh                 # Instruments PMU profiling (L1 misses, INST_ALL); accepts --binary / --args
│   └── profile_power.sh           # powermetrics-based energy profiling (J / GFLOPS-W / thermal pressure)
├── docs/
│   ├── BENCHMARKS.md              # Detailed experimental results (power, cache, instructions)
│   └── COMPARISON.md              # Older detailed comparison report
├── traces/                        # Output of the profiling scripts (PMU + power summaries)
├── main.cpp                       # Test runner + benchmark harness
├── TODO.md                        # Roadmap + fixed-bug archive
├── CMakeLists.txt
└── GemmTemplate.tracetemplate     # Xcode Instruments template (PMU counter list)
```

## Requirements

- Apple M4 (SME/SME2 required for the SME kernels)
- LLVM/Clang — `brew install llvm` (Apple's bundled Clang lacks SME/SME2 intrinsics)
- libomp — `brew install libomp`
- CMake 3.30+
- OpenBLAS — `brew install openblas`, for `bench_compare` only (loaded via `dlopen`, optional at runtime)
- Python with `numpy` and `torch` — for `bench_python.py` only

The build discovers the toolchain rather than hardcoding paths. Override any of it:

```bash
cmake -B build -DCMAKE_CXX_COMPILER=/path/to/clang++   # different compiler
HOMEBREW_PREFIX=/custom/prefix cmake -B build          # non-standard Homebrew
OPENBLAS_DYLIB=/path/to/libopenblas.dylib ./bench/run_bench.sh
PYTHON=/opt/anaconda3/bin/python ./bench/run_bench.sh  # interpreter with numpy+torch
```

## Build & Run

```bash
# Main binary (NEON tests + all six SME suites + comparison + timing breakdown)
./scripts/run.sh

# Single-thread competitor comparison (NEON / SME / Accelerate / OpenBLAS / NumPy / PyTorch)
./bench/run_bench.sh

# Instruments PMU profiling — saves traces/<name>.txt
./scripts/profile.sh 4x1                                                # edit main.cpp to leave only the target kernel active
./scripts/profile.sh --binary cmake-build-release/bench_profile \
                     --args "accel 2048 2048 2048 100" accel            # competitor library

# Energy / thermal profiling — saves traces/<name>_power.txt (sudo for powermetrics)
sudo ./scripts/profile_power.sh --no-build 4x1                          # SME kernel
sudo ./scripts/profile_power.sh --binary /opt/anaconda3/bin/python \
                                --args "bench/bench_profile.py pytorch 2048 2048 2048 100" pytorch
```

## Output

Running `./scripts/run.sh` executes NEON unit tests, NEON speed sweep, all six SME kernel suites (pack correctness → GEMM correctness → benchmark), the cross-kernel comparison, and the 4×1 timing breakdown.

## Roadmap

- [x] Edge-case handling for NEON (non-divisible M/K/N)
- [x] OpenMP parallelism with correct thread structure and dynamic scheduling
- [x] Single-thread competitor benchmark (Accelerate, OpenBLAS, NumPy, PyTorch)
- [x] Fix SME bugs (BUG-1 through BUG-6, BUG-2x2-1/2 + butterfly permutation + software pipelining)
- [x] Optimize SME kernel (K_tile=2048, 2×2 kernel variant, interleaved B packing)
- [x] Experimental 1×4 B-inner kernel + PMU profiling (IPC/L1 miss analysis)
- [x] Cross-library energy profiling vs Accelerate / PyTorch / OpenBLAS / NumPy
- [x] SME-ZA transpose packing (ZAPack) + ZA-resident K-accumulation (1×4ZAIO)
- [x] Scratch-buffer edge-tile fallback in all kernels (arbitrary M/N/K safe)
- [x] Fix NEON exact-2× bug at `N > Nc_cache` shapes
- [x] Fix ZAPack packing-layout bug
- [x] Verify fp32 accumulation error vs fp64 ground truth (beats Accelerate)
- [ ] Dynamic tiling
- [ ] Re-profile 1×4-sym vs 4×1 with PMU counters (explain the B-inner turnaround)
- [ ] SME multi-threading
- [ ] Rust scheduling layer for heterogeneous work distribution (P-core vs E-core)

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

```
Copyright 2026 Ulaş Sertan KEMEÇ

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
