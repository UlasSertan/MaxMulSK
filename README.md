# MaxMulSK

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

| Kernel | ISA | Strategy | GFLOPS (single-thread, peak, end-to-end) |
|--------|-----|----------|------------------------|
| Scalar | — | Naive triple-loop reference | ~2.6 |
| NEON | ARMv8.4 | 8×12 micro-kernel, 4× K-unroll, in-register transpose | ~123 |
| NEON + OpenMP | ARMv8.4 | Above + multi-thread, dynamic scheduling | ~539 |
| SME 4×1 | ARMv8.7 + SME/SME2 | 4×SVL outer-product micro-kernel, K_tile=2048 | ~1248 (2048³) |
| SME 2×2 | ARMv8.7 + SME/SME2 | 2×2 SVL outer-product, interleaved B packing | ~1310 (1024³) |
| SME 1×4 | ARMv8.7 + SME/SME2 | B-inner packing, x1 interleaved B loads in FMOPA shadow | ~1236 (1024³) |
| SME 1×4-sym | ARMv8.7 + SME/SME2 | B-inner packing, x4-grouped B loads (true mirror of 4×1) | ~1326 (4096³) — our best at 4096³ |
| SME 1×4ZAIO | ARMv8.7 + SME/SME2 | 1×4-sym with ZA-resident K-accumulation (`__arm_inout("za")`) | ~1272 (2048³) |
| SME 4×1 ZAPack | ARMv8.7 + SME/SME2 | 4×1 with ZA-based pack_A transpose, M_tile=128 | ~1346 (1024×1024×1020) |
| SME 1×4-Acc | ARMv8.7 + SME/SME2 | K innermost, ZA-resident across all of K, overwriting row-major store, ZA-transpose pack_A | **~1773 (1024³) — overall peak** |
| SME 1×4-Acc-Kc | ARMv8.7 + SME/SME2 | 1×4-Acc with the K tile split into `Kc` sub-chunks | ~1739 (1024³) |

All optimized kernels use cache-blocking and pack A/B into contiguous, kernel-friendly layouts. **Every kernel handles arbitrary M/N/K**: full output tiles write directly to C (hot path), partial edge tiles go through a scratch-buffer + scatter-back fallback with zero cost on aligned sizes.

The SME kernels use SVE predicated loads for K-tail handling and ZA tile accumulators for outer-product computation. Eight SME micro-kernel variants are implemented (six of them wired into the main test binary; the two `1×4-Acc` variants are exercised through `bench/benchmark_maxmul_vs_kleidiai`). The winner depends on size — no single geometry dominates (see the comparison below).

**Two notes on reading any number here.** First, *end-to-end* figures include packing; *prepacked* figures do not, and the two are not comparable — the tables say which. Second, this machine offers no thread pinning and between-run variance reaches ~12% on the worst rows, so current figures are the median of three full runs and **differences below ~5% are not meaningful**. See [bench/results/](bench/results/) and docs/BENCHMARKS.md §0.14. See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for full experimental results: power/energy comparison vs Accelerate / PyTorch / OpenBLAS / NumPy, cache behavior, Instruments PMU profiling, and micro-kernel tuning experiments.

## SME kernel comparison (interleaved, single-thread, 2026-07-25)

> Historical: these are the six kernels that existed before the 1×4-Acc line.
> Current figures, as the median of three runs, are in docs/BENCHMARKS.md §0.14.

| Size | 4×1 | 2×2 | 1×4 | 1×4-sym | 1×4ZAIO | 4×1-ZAPack | Winner |
|---|---:|---:|---:|---:|---:|---:|---|
| 256³ | 715.7 | **766.4** | 729.7 | 731.4 | 609.0 | 762.4 | 2×2 |
| 512³ | 1057.1 | **1111.4** | 1050.2 | 1064.8 | 917.8 | 1110.7 | 2×2 |
| 1024³ | 1142.6 | 1186.2 | 1235.7 | **1243.6** | 1119.8 | 1190.4 | 1×4-sym |
| 2048³ | 1202.7 | 1067.2 | 999.2 | 1128.1 | **1259.0** | 1120.0 | 1×4ZAIO |
| 4096³ | 1173.7 | 1059.1 | 1215.4 | **1308.5** | 1086.7 | 1087.4 | 1×4-sym |

## Single-Thread Comparison vs Industry Libraries

Benchmarked on Apple M4, single-threaded float32 GEMM. The rows below are the 2026-07-25 measurement (`bench/run_bench.sh`, docs/BENCHMARKS.md §0.2) except where marked; the current head-to-head against **Arm KleidiAI** and **llama.cpp/ggml** — including a micro-kernel-only, a prepacked and an end-to-end mode — is docs/BENCHMARKS.md §0.6–§0.14, with raw data in [bench/results/](bench/results/).

| Library | Backend | Peak GFLOPS (4096³ unless noted) | Notes |
|---------|---------|-------------|-------|
| Accelerate (`cblas_sgemm`) | AMX | ~1594 (peak ~1837 at 1024-band) | Apple's matrix coprocessor |
| PyTorch 2.10 | AMX | ~1509 (2048³) | Same backend; Python dispatch overhead |
| **Our SME 1×4-Acc** | SME/ZA | **~1773 (1024³)** | Current peak; ahead of Accelerate and KleidiAI at 1024³, level with KleidiAI panel-blocked (§0.14) |
| **Our SME 1×4-sym** | SME/ZA | **~1308 (4096³)** | Previous peak; still our best at 4096³ end-to-end |
| **Our SME 4×1** | SME/ZA | **~1237 (4096³), ~1248 (2048³)** | ~75% of AMX; our most energy-efficient kernel *only* at 128×32768×512 now (§0.15) |
| **Our SME 2×2** | SME/ZA | **~1308 (1024³)** | Best of the pre-Acc kernels at ≤512³; balanced 32×32 output tile |
| Our NEON kernel | NEON | ~123 | At theoretical NEON ceiling; correct at all shapes |
| NumPy 2.1 (macOS) | vecLib NEON | ~112 | Does *not* take the AMX path for `np.matmul(float32)` |
| OpenBLAS (develop, SME) | SME/SVE | **~1240–1622** (squares), ~460–650 (large-K) | Reaches its SME kernels only with `DYNAMIC_ARCH`; no collapse once it does (§0.16) |
| OpenBLAS 0.3.34 (Homebrew) | SME direct path only | ~113–1519 | Fast to 1024³, then falls off: 628 at 2048³, 113 at 4096³ — a build/threshold artefact, not the library's ceiling |

Correctness: MaxDiff vs Accelerate/OpenBLAS ≤ 0.0004 across all 19 benchmark shapes (pure fp32 rounding). Against float64 ground truth at 4096³, our kernels are *more accurate* than Accelerate (maxErr: NEON 0.000115, SME 4×1 0.000209, Accelerate 0.000348 — docs/BENCHMARKS.md §0.3).

### Energy efficiency (2048³ × 100 iters, `profile_power.sh`, 2026-04-26)

> **Superseded by docs/BENCHMARKS.md §0.15** (2026-09-06), which measures seven
> implementations — including the 1×4-Acc kernels and KleidiAI — as the median of
> three runs. The table below is kept for the progression; it covers only three
> of our kernels and predates the whole 1×4-Acc line.

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

- **At 1024³ we now pass Accelerate end-to-end.** 1×4-Acc reaches ~1773 GFLOP/s against Accelerate's ~1683, KleidiAI's ~1620 and KleidiAI panel-blocked ~1737 (§0.14). Accelerate still leads at 256³–512³ and at 4096³.
- **Prepacked, we are level with KleidiAI at 2048³ and lead on the large-K LLM shapes.** 1810 vs 1799 is 0.6% — a tie under this repo's own ±5% noise rule, not a lead. The large-K lead is real: 1275 vs 923 at 128×32768×512, though that figure is 1×4-sym's; 1×4-Acc manages only 965 there (§0.14).
- **Our SME micro-kernel's arithmetic matches KleidiAI's exactly.** Fitting per-invocation time against Kc gives the same slope for both (~2008 GFLOP/s of pure compute); the whole difference was fixed per-call cost, which four rounds of work cut from 356 ns to 52 ns (§0.9-A).
- **SME is ~14× faster than NEON** on the same single core (1773 vs ~123) — the ZA accumulator is a fundamentally different compute tier.
- **OpenBLAS is a much closer competitor than a stock install suggests (§0.16).** Built so its SME GEMM kernels are actually reachable, it holds 1240–1622 GFLOP/s across every square and never collapses; we lead by 1.03× at 512³ rising to 1.29× at 4096³. Against the Homebrew build the gap looks like 14× at 4096³, but that number measures a mis-built OpenBLAS rather than our kernel, and it is not the figure to quote. Our real margin is on the large-K LLM shapes: **1.35×–2.18×**.
- **Our SME 4×1 beats PyTorch on energy efficiency** (0.0065 vs 0.0076 J/GFLOP, 2026-04-26) despite running slower in wall-clock terms — PyTorch's dispatcher overhead inflates total energy at 2048³. PyTorch has not been re-measured since; our own kernels have (§0.15).
- **NumPy is ~10× less efficient than our SME** in J/GFLOP. Single-threaded `np.matmul(float32)` on macOS takes the vecLib NEON path, not AMX; PyTorch's own dispatcher reaches AMX correctly.
- **Size-dependent SME ranking (2026-09-06, end-to-end):** the 1×4-Acc pair owns everything up to 2048³ — 1398 at 256³, 1660 at 512³, 1773 at 1024³, 1481 at 2048³ — while 1×4-sym still owns 4096³ (1326) and the pre-Acc kernels hold the K=32768 shapes (4×1-ZAPack 901 vs Acc's 736 at 128×32768×512), where Acc's full-K packing hurts most. The older ranking (2×2 ≤512³, 1×4ZAIO at 2048³) described the six kernels that existed before the Acc line.
- **Energy tracks throughput, it does not trade against it (§0.15).** Going from 1×4-sym to 1×4-Acc bought +37% GFLOP/s *and* −13.4% J/GFLOP at 1024³, and +25.5% / −5.8% at 2048³; at 4096³ and 128×32768×512 it lost on both axes. Against the outside libraries we are level with KleidiAI at 1024³ (0.00600 vs 0.00591), and 21% behind at 4096³.

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
MaxMulSK/
├── common/
│   ├── utils.hpp                  # fill_random, check_correctness, compute_gflops, scalar reference
│   └── Matrix.hpp                 # Generic matrix class (addition, subtraction)
├── neon/
│   ├── neon-8x12.hpp/.cpp         # NEON 8×12 kernel + packing + OpenMP driver
│   └── test_neon.hpp/.cpp         # Packing + GEMM correctness vs scalar
├── sme/
│   ├── sme-4x1.hpp/.cpp               # SME 4×1 kernel + packing — most efficient at large K
│   ├── sme-2x2.hpp/.cpp               # SME 2×2 kernel + interleaved B packing
│   ├── sme-1x4.hpp/.cpp               # SME 1×4 B-inner kernel (x1 interleaved loads)
│   ├── sme-1x4-sym.hpp/.cpp           # SME 1×4 x4-grouped B loads — our best at 4096³
│   ├── sme-1x4-sym-zainout.hpp/.cpp   # 1×4-sym with split zero/compute/store sharing ZA via __arm_inout
│   ├── sme-4x1-zapack.hpp/.cpp        # SME 4×1 with ZA-based pack_A transpose, M_tile=128
│   ├── sme-1x4-acc.hpp/.cpp           # 1×4-Acc: K innermost, ZA-resident over all K, overwriting store, ZA pack_A
│   ├── sme-1x4-acc-kc.hpp/.cpp        # 1×4-Acc with the K tile split into Kc sub-chunks
│   └── test_sme.hpp/.cpp              # Single dispatch surface: SMETest::Kernel enum + run / run_comparison / run_timing_breakdown / profile
├── bench/
│   ├── bench_compare.cpp          # C++ comparison: NEON / SME / Accelerate / OpenBLAS
│   ├── bench_python.py            # Python comparison: NumPy / PyTorch
│   ├── bench_profile.cpp          # Single-library driver (accel|oblas) for profiling scripts
│   ├── bench_profile.py           # Single-library driver (numpy|pytorch)
│   ├── run_bench.sh               # Build + run both comparison benchmarks
│   ├── benchmark_maxmul_vs_kleidiai.cpp  # Layered head-to-head: hot micro-kernel / prepacked / end-to-end / panel-blocked / ggml
│   ├── maxmulsk_sme_adapter.hpp/.cpp     # Uniform dispatch over our SME kernels + prepacked and hot-tile harnesses
│   ├── kleidiai_gemm.hpp/.cpp            # Arm KleidiAI fp32 SME2 wrapper (full-pack and panel-blocked callers)
│   ├── ggml_gemm_baseline.hpp/.cpp       # llama.cpp/ggml CPU baseline, native and KleidiAI dispatch paths
│   ├── energy_bench.cpp                  # One implementation, one shape, held busy for a fixed wall time
│   ├── energy_bench.sh                   # Wraps the above in powermetrics -> W, J, J/GFLOP
│   ├── energy_parse.py                   # powermetrics capture -> average power
│   ├── grand_benchmark.sh                # Runs every benchmark into one dated directory
│   ├── grand_summary.py                  # Aggregates a run directory into SUMMARY.md
│   ├── average_runs.py                   # Median across repeated runs + per-row spread
│   └── results/                          # Dated result archive; nothing here is deleted (see results/README.md)
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
- OpenBLAS — `brew install openblas`, for `bench_compare` and `bench_vs_openblas` (loaded via `dlopen`, optional at runtime). The Homebrew build cannot reach its own SME GEMM kernels; to compare against OpenBLAS at its best, build `develop` with `DYNAMIC_ARCH=1` and point `OPENBLAS_DYLIB` at it (docs/BENCHMARKS.md §0.16)
- Python with `numpy` and `torch` — for `bench_python.py` only
- **Arm KleidiAI** — fetched and built automatically at a pinned tag (`v1.30.0`). Turn it off with `-DMAXMULSK_WITH_KLEIDIAI=OFF`, or point at an existing checkout with `-DFETCHCONTENT_SOURCE_DIR_KLEIDIAI=...`
- **llama.cpp / ggml** *(optional)* — only for the ggml CPU baseline rows. Build it CPU-only, then point CMake at it

The build discovers the toolchain rather than hardcoding paths. Override any of it:

```bash
cmake -B build -DCMAKE_CXX_COMPILER=/path/to/clang++   # different compiler
HOMEBREW_PREFIX=/custom/prefix cmake -B build          # non-standard Homebrew
OPENBLAS_DYLIB=/path/to/libopenblas.dylib ./bench/run_bench.sh
PYTHON=/opt/anaconda3/bin/python ./bench/run_bench.sh  # interpreter with numpy+torch
```

For the llama.cpp/ggml baseline, build ggml with every external accelerator off
so the rows measure ggml's own CPU path, then point CMake at it:

```bash
cmake -S llama.cpp -B llama-build -DCMAKE_BUILD_TYPE=Release \
      -DGGML_METAL=OFF -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF \
      -DGGML_CPU_KLEIDIAI=ON -DGGML_OPENMP=OFF
cmake --build llama-build --target ggml-cpu ggml-base ggml

cmake -B cmake-build-release -S . \
      -DLLAMA_CPP_DIR=llama.cpp -DLLAMA_CPP_BUILD_DIR=llama-build \
      -DLLAMA_CPP_KLEIDIAI_LIB=llama-build/_deps/kleidiai-build/libkleidiai.a
```

`GGML_CPU_KLEIDIAI=ON` adds a second ggml row that dispatches to KleidiAI's fp32
SME2 kernel; without it only ggml's native path is measured. Note that ggml pins
its *own* KleidiAI version (v1.24.0), independent of the one this repo fetches.

## Build & Run

```bash
# Main binary (NEON tests + the six SME suites it wires up + comparison + timing breakdown)
./scripts/run.sh

# Everything, into bench/results/<date>/ — this is the one to run
./bench/grand_benchmark.sh
./bench/grand_benchmark.sh --power-only   # energy only, reusing an existing directory

# Layered head-to-head on its own (hot micro-kernel / prepacked / end-to-end / panel / ggml)
cmake --build cmake-build-release --target benchmark_maxmul_vs_kleidiai
./cmake-build-release/benchmark_maxmul_vs_kleidiai out.csv

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

Running `./scripts/run.sh` executes NEON unit tests, NEON speed sweep, the six SME kernel suites wired into that binary (pack correctness → GEMM correctness → benchmark), the cross-kernel comparison, and the 4×1 timing breakdown.

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
- [x] Head-to-head vs Arm KleidiAI and llama.cpp/ggml, separating micro-kernel from macro-flow
- [x] ZA-lifetime kernel line (1×4-Acc): K innermost, overwriting row-major store, ZA-transpose packing
- [ ] Dynamic tiling
- [ ] Re-profile 1×4-sym vs 4×1 with PMU counters (explain the B-inner turnaround)
- [ ] SME multi-threading
- [ ] Remove 1×4-Acc's full-K packing requirement (its remaining loss at 4096³ and K ≥ 16384)

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
