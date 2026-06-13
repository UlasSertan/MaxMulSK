# MatrixLibrary

High-performance GEMM (General Matrix Multiply) for Apple Silicon, implemented in C++20.

Explores and benchmarks three levels of compute on ARM:

| Kernel | ISA | Strategy | GFLOPS (single-thread, peak) |
|--------|-----|----------|------------------------|
| Scalar | — | Naive triple-loop reference | ~2 |
| NEON | ARMv8.4 | 8×12 micro-kernel, 4× K-unroll, in-register transpose | ~123 |
| NEON + OpenMP | ARMv8.4 | Above + multi-thread, dynamic scheduling | ~539 |
| SME 4×1 | ARMv8.7 + SME/SME2 | 4×SVL outer-product micro-kernel, K_tile=2048 | ~1191 (4096³) |
| SME 2×2 | ARMv8.7 + SME/SME2 | 2×2 SVL outer-product, interleaved B packing | ~1192 (1024³), ~1057 (2048³) |
| SME 1×4 (experimental) | ARMv8.7 + SME/SME2 | B-inner packing, 1×4 SVL tile, x1 interleaved B loads in FMOPA shadow | ~1190 (4096³), ~1025 (2048³) |
| SME 4×1 ZAPack | ARMv8.7 + SME/SME2 | 4×1 layout with ZA-based pack_A transpose | **disabled — has known wrong-result / heap-corrupt bug** (TODO BUG-ZAPACK-WRONG) |

All optimized kernels use cache-blocking and pack A/B into contiguous, kernel-friendly layouts. The 2×2 kernel handles arbitrary M/N/K via a scratch-buffer fallback for partial output tiles. **Caveat (TODO §1):** the 4×1 / 1×4 / ZAPack micro-kernels write a fixed 4·SVL × SVL (or SVL × 4·SVL) output tile and *do not* yet have a scratch-buffer path — sizes with M < 64 (or N < 64 for 1×4) silently write past `C` and corrupt the heap. Production usage targets large M/N where this never triggers; the test harness either skips small sizes or pads `C` to absorb the overrun.

The SME kernels use SVE predicated loads for K-tail handling and ZA tile accumulators for outer-product computation. Three SME micro-kernel layouts are implemented (4×1, 2×2, 1×4); peak winners depend on size — 4×1 and 1×4 tie at 4096³, 2×2 wins at 1024³–2048³. See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for full experimental results: power/energy comparison vs Accelerate / PyTorch / OpenBLAS / NumPy, cache behavior, Instruments PMU profiling, and micro-kernel tuning experiments.

## Single-Thread Comparison vs Industry Libraries

Benchmarked on Apple M4, single-threaded float32 GEMM, threads pinned to 1. Numbers below are from the 2026-04-26 fresh measurement (`bench/run_bench.sh`). Full table in docs/BENCHMARKS.md §8.4.

| Library | Backend | Peak GFLOPS (4096³ unless noted) | Notes |
|---------|---------|-------------|-------|
| Accelerate (`cblas_sgemm`) | AMX | ~1523 | Apple's matrix coprocessor |
| PyTorch 2.10 | AMX | ~1488 (2048³) | Same backend; Python dispatch overhead |
| **Our SME 4×1 kernel** | SME/ZA | **~1183** | ~78% of AMX, ~10× our NEON. **Most energy-efficient SME variant** at 0.0065 J/GFLOP. |
| **Our SME 1×4 kernel (experimental)** | SME/ZA | **~1190 (4096³), ~1025 (2048³)** | x1-interleaved B loads in FMOPA shadow. Ties 4×1 at 4096³ but lags at smaller sizes. |
| **Our SME 2×2 kernel** | SME/ZA | **~1192 (1024³), ~1057 (2048³)** | Wins the L3-fitting band; balanced 32×32 output tile. |
| Our NEON kernel | NEON | ~123 | At theoretical NEON ceiling. **Bug:** exact-2× wrong output at certain `N > Nc_cache` shapes (TODO BUG-NEON-2X). |
| NumPy 2.1 (macOS) | vecLib NEON | ~113 | ~10% below our NEON kernel. Does *not* take the AMX path for `np.matmul(float32)`. |
| OpenBLAS 0.3.32 | NEON | ~113–1300 | Strong at L3 sizes (~1300 at 1024³), collapses to ~110 at 2048³+ on non-aligned shapes. |

### Energy efficiency (2048³ × 100 iters, `profile_power.sh`)

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

- **SME closes the gap with AMX to ~1.3×.** SME 4×1 peaks at ~1183 GFLOPS vs Accelerate's ~1523 (78% ratio at 4096³). Up from the ~7% ratio with NEON alone.
- **SME is ~10× faster than NEON** on the same single core — the ZA accumulator is a fundamentally different compute tier.
- **Our SME 4×1 beats PyTorch on energy efficiency** (0.0065 vs 0.0076 J/GFLOP) despite running slower in wall-clock terms — PyTorch's dispatcher overhead inflates total energy at 2048³.
- **NumPy is ~10× less efficient than our SME** in J/GFLOP. Single-threaded `np.matmul(float32)` on macOS takes the vecLib NEON path, not AMX; PyTorch's own dispatcher reaches AMX correctly.
- **Size-dependent SME ranking:** 4×1 and 1×4 tie at 4096³ (~1190 each), 2×2 wins at 1024³ (1192) and 2048³ (1057). Use 4×1 as the production default — it's the only one consistently strong *and* most energy-efficient.

### A note on the 4×1 vs 1×4 IPC story

The 1×4 (B-inner) kernel is the mathematical transpose of 4×1 (A-inner) — same FMOPA count, same tile area, symmetric load count per k-step. Naively we expected the same throughput. We didn't get it. The investigation closed both escape hatches:

- **x4-load baseline (retired):** symmetric to 4×1 — one `svld1_f32_x4` per k-step on the B side. Reached ~1148 GFLOPS but stalled on FMOPA same-tile dependency chains. IPC collapsed to 0.62 (vs 4×1's 1.07 in the same-era measurement) — the instruction stream was too lean to fill FMOPA's 6–8 cycle execution shadow.
- **x1 interleaved rescue (current):** replace the grouped x4 B-load with four independent `svld1_f32` loads issued inline between successive svmopa instructions, dropping into the FMOPA shadow. This *worked* microarchitecturally — IPC jumped to 1.40, higher than 4×1 — but at the cost of a 2.5× instruction-count inflation (~4.1 B → ~10.3 B over 100 iters in the original measurement). Throughput went *down*, not up — to ~1018 GFLOPS at 2048³.

**The takeaway:** A-inner (4×1) is the only geometry on the M4 SME pipeline that produces a scheduling-friendly load/FMOPA mix *without* an instruction tax. The 4 A-loads per k-step act as latency-hiding filler "for free". Full data and per-iteration counters in docs/BENCHMARKS.md §10.1.

**What's changed in the 2026-04-26 measurement:** the same x1-interleaved 1×4 kernel now hits ~1190 GFLOPS at 4096³ — essentially tied with 4×1 — and ~1025 at 2048³. The kernel evolved (instruction count is now 1.82 B at 100 iters per the fresh PMU profile, *not* 10.3 B); the IPC-collapse / instruction-tax narrative above is the cleanest explanation we have for *why* B-inner is hard, but the absolute numbers from §10 are now stale and replaced by §10.0. Either way, 4×1 stays the default — it's never worse than the alternatives and is the most energy-efficient.

### Other findings & known issues

- Accelerate and PyTorch use AMX — a dedicated coprocessor. The remaining gap to AMX is hardware, not algorithmic.
- **4×1 / 1×4 / 4×1ZAPack writeback overruns C for M < 64** (or N < 64 for 1×4) — silent heap corruption. Test harness skips/pads small sizes; real fix is scratch-buffer fallback (TODO §1).
- **NEON exact-2× bug** at non-aligned shapes (TODO BUG-NEON-2X) — large MaxDiff at 2048³+ shown in §8.4.
- **Accumulator precision degrades at ≥2048³** (MaxDiff up to ~100 vs Accelerate). Pairwise summation planned (TODO §7).
- **SME 4×1 ZAPack kernel disabled** pending bug fix (TODO BUG-ZAPACK-WRONG).

## Structure

```
MatrixLibrary/
├── common/
│   ├── utils.hpp                            # fill_random, check_correctness, compute_gflops, scalar reference
│   └── Matrix.hpp                           # Generic matrix class (addition, subtraction)
├── neon/
│   ├── GEMMKernels.hpp/.cpp                 # NEON kernel + packing — complete (with BUG-NEON-2X open)
│   └── test_neon.hpp/.cpp                   # Packing + GEMM correctness vs scalar
├── sme/
│   ├── SME-GEMMKernels4x1.hpp/.cpp                # SME 4×1 kernel + packing — production default
│   ├── SME-GEMMKernels2x2.hpp/.cpp                # SME 2×2 kernel + interleaved B packing
│   ├── SME-GEMMKernels1x4.hpp/.cpp                # SME 1×4 B-inner kernel (experimental, x1 interleaved loads)
│   ├── SME-GEMMKernels1x4-sym.hpp/.cpp            # SME 1×4 x4-grouped B loads (true mirror of 4×1)
│   ├── SME-GEMMKernels1x4-symZAInOut.hpp/.cpp     # SME 1×4-sym with split zero/compute/store sharing ZA via __arm_inout
│   ├── SME-GEMMKernels4x1ZAPack.hpp/.cpp          # SME 4×1 with ZA-based pack_A transpose (disabled — has bug)
│   └── test_sme.hpp/.cpp                          # Single dispatch surface: SMETest::Kernel enum + run / run_comparison / run_timing_breakdown / profile
├── bench/
│   ├── bench_compare.cpp                    # C++ comparison: NEON / SME / Accelerate / OpenBLAS
│   ├── bench_python.py                      # Python comparison: NumPy / PyTorch
│   ├── bench_profile.cpp                    # Single-library driver (accel|oblas) for profiling scripts
│   ├── bench_profile.py                     # Single-library driver (numpy|pytorch)
│   └── run_bench.sh                         # Build + run both comparison benchmarks
├── experiments/
│   └── matmul.cpp                           # Standalone N=2048 head-to-head: our SME kernel vs Accelerate
├── scripts/
│   ├── run.sh                               # Build + run main binary
│   ├── run_matmul.sh                        # Build + run the matmul experiment (vs Accelerate)
│   ├── profile.sh                           # Instruments PMU profiling (L1 misses, INST_ALL); accepts --binary / --args
│   └── profile_power.sh                     # powermetrics-based energy profiling (J / GFLOPS-W / thermal pressure)
├── docs/
│   ├── BENCHMARKS.md                        # Detailed experimental results (power, cache, instructions)
│   └── COMPARISON.md                        # Older detailed comparison report
├── traces/                                  # Output of the profiling scripts (PMU + power summaries)
├── main.cpp                                 # Test runner + benchmark harness
├── TODO.md                                  # Active bug list + roadmap
├── CMakeLists.txt
└── GemmTemplate.tracetemplate               # Xcode Instruments template (PMU counter list)
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
# Main binary (NEON tests + SME suites + comparison + timing breakdown)
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

Running `./scripts/run.sh` executes NEON unit tests, NEON speed sweep, all three working SME kernel suites (pack correctness → GEMM correctness → benchmark), the cross-kernel comparison, and the 4×1 timing breakdown.

## Roadmap

- [x] Edge-case handling for NEON (non-divisible M/K/N) *(but BUG-NEON-2X still open at certain N > Nc_cache shapes)*
- [x] OpenMP parallelism with correct thread structure and dynamic scheduling
- [x] Single-thread competitor benchmark (Accelerate, OpenBLAS, NumPy, PyTorch)
- [x] Fix SME bugs (BUG-1 through BUG-6, BUG-2x2-1, BUG-2x2-2 + butterfly permutation + software pipelining)
- [x] Optimize SME kernel (K_tile=2048, 2×2 kernel variant, interleaved B packing)
- [x] Benchmark SME vs NEON vs Accelerate (single-thread: ~1190 GFLOPS peak)
- [x] Experimental 1×4 B-inner kernel + PMU profiling (IPC/L1 miss analysis)
- [x] Investigate 1×4 IPC collapse — x1-interleaved loads attempted; explored both x4 and x1 instruction-budget extremes
- [x] Cross-library energy profiling vs Accelerate / PyTorch / OpenBLAS / NumPy
- [ ] **Fix 4×1 / 1×4 / ZAPack small-M heap-overflow** (scratch-buffer fallback) — TODO §1
- [ ] **Fix NEON exact-2× bug** at non-aligned shapes — TODO BUG-NEON-2X
- [ ] **Fix 4×1 ZAPack wrong-result bug** — TODO BUG-ZAPACK-WRONG
- [ ] Fix accumulator precision at large sizes (pairwise summation)
- [ ] SME multi-threading
- [ ] Rust scheduling layer for heterogeneous work distribution (P-core vs E-core)
