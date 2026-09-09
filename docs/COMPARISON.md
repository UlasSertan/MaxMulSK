# Single-Thread GEMM Benchmark — Comparison Report

> **Historical snapshot (2026-04-15).** Kept for the progression; nothing here is
> current. Today's numbers — eight SME kernels, plus head-to-head against Arm
> KleidiAI and llama.cpp/ggml across micro-kernel, prepacked and end-to-end
> modes — are in [BENCHMARKS.md §0.14](BENCHMARKS.md), with raw data under
> [bench/results/](../bench/results/).

**Date:** 2026-04-15 (updated; original NEON-only comparison: 2026-04-10)
**Hardware:** Apple M4 (MacBook Air), arm64
**OS:** macOS 14 (Sequoia)
**Compiler:** Clang (LLVM, Homebrew), `-O3 -mcpu=apple-m4`
**Kernels:** NEON 8×12 (Mc=64, Kc=256, Nc=1020), SME 4×1 (M=64, K=2048, N=1024), SME 2×2 (M=256, K=2048, N=512)

---

## Setup

All benchmarks are single-threaded:

| Library | Thread control |
|---|---|
| Our NEON / SME 4×1 / SME 2×2 | `omp_set_num_threads(1)` |
| Accelerate | `VECLIB_MAXIMUM_THREADS=1` (env, set before process launch) |
| OpenBLAS 0.3.32 | `OPENBLAS_NUM_THREADS=1` (env, set before process launch) |
| NumPy 2.1.3 | `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1` |
| PyTorch 2.10.0 | `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)` |

All matrices are float32, row-major, `C = A(M×K) × B(K×N)`.
Each size is warmed up once, then averaged over N iterations (N scales down with size: 10000 at 8^3, 3 at 2048^3).
Correctness is verified against Accelerate as ground truth; MaxDiff is the maximum absolute element-wise difference.

---

## Raw Results — C++ Benchmark (2026-04-15, with SME kernels)

```
Size (MxKxN)          Tag            NEON GF   SME 4x1   SME 2x2  Accel GF  OBlas GF  MaxDiff
----------------------------------------------------------------------------------------------
8x8x8                 tiny               6.3       0.1       0.1      19.1       1.2  0.00000
16x16x16              tiny               0.8       0.7       0.7      56.3      58.4  0.00000
32x32x32              small              6.5       4.3       6.0     444.4     380.8  0.00000
64x64x64              small             28.8      32.5      40.3     937.8     947.6  0.00000
128x128x128           L2                85.8     187.1     175.4    1483.6    1379.7  0.00000
256x256x256           L2               110.6     502.3     452.9    1722.9    1646.4  0.00000
512x512x512           L3               121.3     868.4     830.3    1813.5    1643.9  0.00000
1024x1024x1024        L3               122.6    1071.2    1083.2    1680.5    1372.1  0.00000
2048x2048x2048        mem-bound        123.9    1163.4    1001.0    1636.3     582.3 62.34431
4096x4096x4096        mem-bound        124.2    1213.0    1032.8    1532.0     113.9100.59300
4095x4095x4095        mem-bound        121.7    1175.3     972.4    1477.6     112.0 87.96959
1024x1024x1020        N=85x12 aligned  121.5    1045.7    1025.5    1813.9     110.3  0.00000
2048x512x64           tall-skinny      100.8     590.1     565.2    1340.4     105.6  0.00000
512x2048x64           wide-flat        100.1     615.9     480.1    1261.2     104.6  0.00011
65x65x65              all tails +1       9.5       6.2       5.9     463.3     527.9  0.00000
513x513x509           all tails mixed  114.2     663.2     589.3    1591.0     105.2  0.00000
1025x1025x1021        all tails large  117.8    1052.1    1062.4    1752.8     108.7  0.00007
128x128x1100          N > Nc_cache     100.8     294.9     327.9    1566.0     104.7 12.93185
512x512x2048          N >> Nc_cache    117.4    1002.9     999.4    1647.3     112.4 28.58930
```

Note: Large MaxDiff values at ≥2048 sizes are due to floating-point accumulator precision, not correctness bugs.
Pairwise summation (tree reduction instead of sequential accumulation) is planned to fix this. *(Not a real defect — see the note in Key Takeaways below.)*

---

## Raw Results — Python Benchmark

```
Size (MxKxN)          Tag                      NP ms   NP GFLOPS       PT ms   PT GFLOPS   MaxDiff
--------------------------------------------------------------------------------------------------
8x8x8                 tiny                      0.00         2.1        0.00         2.1   0.00000
16x16x16              tiny                      0.00        12.7        0.00         6.9   0.00000
32x32x32              small                     0.00        53.5        0.00        57.2   0.00000
64x64x64              small                     0.01        40.1        0.00       328.9   0.00000
128x128x128           L2                        0.05        76.8        0.00       946.4   0.00000
256x256x256           L2                        0.38        87.8        0.02      1420.1   0.00000
512x512x512           L3                        2.49       107.9        0.15      1742.3   0.00000
1024x1024x1024        L3                       20.00       107.4        1.43      1503.2   0.00061
2048x2048x2048        mem-bound               152.43       112.7       11.04      1556.1   0.00171
1024x1020x1024        N=85x12 aligned          19.02       112.5        1.40      1530.2   0.00058
2048x512x64           tall-skinny               1.32       101.7        0.11      1198.4   0.00000
512x2048x64           wide-flat                 1.32       101.9        0.11      1182.0   0.00153
65x65x65              all tails +1              0.01        78.3        0.00       222.5   0.00000
513x509x513           all tails mixed           2.47       108.3        0.19      1379.6   0.00000
1025x1021x1025        all tails large          19.48       110.1        1.48      1450.1   0.00061
128x1100x128          N > Nc_cache              0.34       106.6        0.04       993.0   0.00058
512x2048x512          N >> Nc_cache             9.75       110.2        0.60      1777.7   0.00159
```

---

## Analysis

### Tier 1 — Accelerate and PyTorch (~1500–1813 GFLOPS)

> **Wrong, corrected 2026-09-09 (BENCHMARKS.md §0.17).** This whole tier was
> named after AMX on the strength of nothing but speed. Sampling the program
> counter inside `cblas_sgemm` found Accelerate running on **SME**, the same
> unit our kernels use: 0 AMX-dominated samples out of 583 at 4096³. PyTorch
> was never measured either way. The AMX reasoning below is left as written
> because it drove real decisions for months; it should not be believed.

Apple Accelerate's `cblas_sgemm` and PyTorch both land in the 1500–1813 GFLOPS range at large sizes. This is only possible because both are using **AMX (Apple Matrix eXtensions)**, a dedicated matrix multiply coprocessor built into every M-series chip. AMX is not part of the public ARM ISA; Apple uses it internally through Accelerate and does not document the instruction encoding.

Accelerate peaks at **~1813 GFLOPS** on 512³. PyTorch follows closely and independently confirms the AMX hypothesis.

### Tier 2 — Our SME Kernels (~1000–1213 GFLOPS)

The SME kernels represent a dramatic step up from NEON, reaching **67% of Accelerate's AMX performance** on a single thread.

**SME 4×1** peaks at **~1213 GFLOPS** (4096³) — approximately **10× faster than NEON** on the same core. The 4×1 layout processes a 64×16 output tile per micro-kernel call, giving each B vector 4× reuse across svmopa instructions. With K_tile=2048, the ZA accumulator holds partial results across a large K-dimension before writeback, reducing pack overhead.

**SME 2×2** peaks at **~1083 GFLOPS** (1024³). The 2×2 layout processes a 32×32 output tile using interleaved B packing (two SVL vectors stored contiguously per k-step). It uses fewer total loads per k-step (4 vs 5) but each B vector is only reused twice. This makes 2×2 slightly faster at medium sizes (1024³) where the balanced tile shape may better fit the cache hierarchy, but 4×1 wins at larger sizes where B reuse dominates.

| Size | SME 4×1 | SME 2×2 | vs Accelerate (4×1) |
|---|---|---|---|
| 256³ | 502 | 453 | 29% |
| 512³ | 868 | 830 | 48% |
| 1024³ | 1071 | 1083 | 64% |
| 2048³ | 1163 | 1001 | 71% |
| 4096³ | 1213 | 1033 | 79% |

The remaining ~1.5× gap to Accelerate is hardware: AMX is a dedicated coprocessor with higher throughput than SME's outer-product path. This is not an algorithmic limitation.

> **Exactly backwards, corrected 2026-09-09.** Same hardware (§0.17), and the
> gap *was* algorithmic: it was per-call fixed cost. §0.9-A took one
> micro-kernel invocation from 357 ns to 52 ns and closed it — 1773 vs
> Accelerate's 1786 GFLOP/s at 1024³ (§0.14).

### Tier 3 — Our NEON Kernel (~99–124 GFLOPS)

Our NEON kernel hits **122–124 GFLOPS** at large square matrices, within 2% of the M4's theoretical single-core NEON ceiling (~125 GFLOPS). Performance is highly consistent across all sizes — tail cases cost less than 3 GFLOPS.

The NEON kernel is **~10% faster than NumPy** at all large sizes (NumPy routes through vecLib BLAS, not AMX) and **beats OpenBLAS on all non-aligned sizes**.

### Tier 3 (contested) — NumPy and OpenBLAS

NumPy peaks at ~112 GFLOPS — firmly NEON-tier. OpenBLAS continues to show erratic behavior: ~1646 on aligned squares (suspected AMX or multi-thread leak), collapsing to ~105–113 on non-aligned sizes where our kernels outperform it consistently.

### Accumulator Precision Issue

At sizes ≥ 2048, MaxDiff values between our SME kernels and Accelerate grow large (up to ~100). This is not a correctness bug — it is floating-point accumulation order sensitivity. With K=2048+, the ZA accumulator sums thousands of products sequentially, and the order of additions differs from Accelerate's implementation. Pairwise summation (tree reduction: `((A+B) + (C+D))` instead of `A+B+C+D`) is planned to reduce this. *(Superseded: the large MaxDiff was BUG-NEON-2X in the comparison baseline, not accumulation order. Closed as not-a-bug on 2026-07-25.)*

---

## Summary Table

| Library | Backend | Peak GFLOPS | Edge-case GFLOPS | Consistency |
|---|---|---|---|---|
| Accelerate | SME (§0.17) | ~1813 | ~463–1340 | High |
| PyTorch | vecLib, unverified | ~1815 | ~222–1450 | High |
| **Our SME 4×1** | SME/ZA | **~1213** | **~502–1175** | **High** |
| **Our SME 2×2** | SME/ZA | **~1083** | **~453–1062** | **High** |
| OpenBLAS 0.3.32 | NEON (+ suspected AMX on aligned) | ~1647 (aligned) / ~105 (non-aligned) | **Low** | Very low |
| Our NEON kernel | NEON | ~124 | ~100–122 | Very high |
| NumPy | vecLib BLAS (NEON) | ~112 | ~78–112 | High |
| Scalar reference | scalar | ~2 | ~2 | Perfect |

---

## Key Takeaways

1. **SME closes 67% of the gap to AMX.** Our SME 4×1 kernel reaches ~1213 GFLOPS vs Accelerate's ~1813. The remaining ~1.5× gap is hardware — AMX is a dedicated coprocessor with higher raw throughput than SME's outer-product instructions. *(Wrong on both counts — see §0.17 and the note in Tier 1. Same hardware, and the gap was fixed cost, since closed.)*

2. **SME is ~10× faster than NEON on the same core.** This confirms that the ZA accumulator is a fundamentally different compute tier, not just "wider SIMD." The 4×1 kernel at 4096³ processes more FLOPS per second than NEON + OpenMP at 10 threads.

3. **4×1 beats 2×2 at scale due to B reuse.** Each B vector in the 4×1 layout feeds 4 svmopa instructions vs 2 in 2×2. At large K_tile, this reuse advantage dominates despite 2×2 having fewer total loads per k-step.

4. **NEON kernel still beats NumPy by ~10%** and beats OpenBLAS on all non-aligned sizes. These conclusions from the original comparison remain unchanged.

5. **Accumulator precision is the next correctness frontier.** At ≥2048 sizes, sequential float32 accumulation across thousands of K-steps produces unacceptable drift. Pairwise summation is the planned fix. *(Wrong, resolved 2026-07-25: the drift was BUG-NEON-2X corrupting the comparison baseline. Measured against fp64 truth our kernels are more accurate than Accelerate; pairwise summation was never needed — BENCHMARKS.md §0.3.)*

6. **Next steps:** SME multi-threading (expected to scale similarly to NEON's ~4.4× on 10 threads), accumulator precision fix, then a tile-scheduling layer. *(Both later resolved differently: the precision item was closed as not-a-bug — see TODO.md — and the scheduling layer was dropped from the roadmap.)*

---

## Reproducibility

```bash
# From repo root
./bench/run_bench.sh
```

Requires:
- macOS with Accelerate (built-in)
- OpenBLAS: `brew install openblas`
- Python env: Anaconda base with `numpy` and `torch` (`/opt/anaconda3/bin/python`)
- CLion CMake or CMake on PATH
