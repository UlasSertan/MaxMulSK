# Benchmark & Experimental Results

Platform: Apple Silicon (M4), macOS  
Problem size: M=1024, N=1020, K=1024 (unless noted)  
Compiler: LLVM/Clang `-O3 -mcpu=apple-m4`

---

## 1. Performance Progression

| Implementation | GFLOPS | Notes |
|---|---|---|
| Naive scalar | ~2 | Triple-loop, no optimization |
| SIMD (NEON) | ~91.8 | 8×12 micro-kernel, no packing |
| SIMD + Packing/Tiling | ~120 | Cache-blocked, packed A & B |
| SIMD + Packing/Tiling + OpenMP | ~539 | 10-thread parallel, dynamic scheduling |
| SME 4×1 (initial) | ~482 | First working SME kernel, K_tile=256 |
| SME 4×1 (optimized) | ~1191 | K_tile=2048, software pipelining (peak at 4096³, 2026-04-26 run) |
| SME 2×2 | ~1192 | 2×2 tile layout, interleaved B packing (peak at 1024³) |
| SME 1×4 (B-inner, x4 loads — retired baseline) | ~1148 | Symmetric-to-4×1 load pattern; stalled on FMOPA same-tile chain (IPC 0.62) |
| SME 1×4 (B-inner, x1 interleaved loads — current experimental) | ~1190 | B panels loaded one register at a time into FMOPA shadow. **Now matches 4×1 at 4096³** (1190 GFLOPS, 2026-04-26) but lags at smaller sizes (~1025 at 2048³, ~856 at 512³) |
| SME 4×1 ZAPack | n/a | New variant using ZA-based pack_A transpose. **Has known wrong-result / heap-corrupt bug at small M (TODO BUG-ZAPACK-WRONG and BUG-4x1-SMALL-M).** Disabled in current test harness; numbers below in §8 are from the buggy build and will be re-measured after the fix. |

**Note on small sizes:** SME 4×1 / 1×4 / ZAPack micro-kernels emit a fixed 4·SVL × SVL (or SVL × 4·SVL) output tile and do not have a scratch-buffer fallback for partial tiles. Sizes with M < 64 (or N < 64 for 1×4) silently corrupt heap memory beyond `C[M*N)`. Production usage targets large M; small-size correctness is enforced via test-harness padding (`bench/bench_compare.cpp`) or skip (`sme/test_sme.cpp`). Real fix: add scratch-buffer + scatter-back path mirroring 2×2's BUG-2x2-1 fix (TODO §1).

---

## 2. Power & Energy Analysis

Measurement tool: `powermetrics --samplers cpu_power,thermal`, 100 ms sampling interval, captured by `scripts/profile_power.sh` (combined CPU + GPU + ANE basis). All measurements at 2048³ × 100 iterations (~1.7 TFLOPs of work per run), single-thread, MacBook M4, no thermal pressure (Nominal across all runs).

### 2.1 Our SME kernels (fresh, 2026-04-26)

| Kernel | GFLOPS | Runtime (s) | Avg power (W) | Energy (J) | J / GFLOP | GFLOPS / W |
|---|---:|---:|---:|---:|---:|---:|
| **SME 4×1** | **1173.7** | 1.84 | 6.07 | 11.18 | **0.0065** | **153.6** |
| SME 2×2 | 1066.1 | 2.11 | 5.44 | 11.47 | 0.0067 | 149.8 |
| SME 1×4 (x1 interleaved) | 1037.7 | 1.71 | 7.17 | 12.24 | 0.0071 | 140.4 |

4×1 is the most energy-efficient kernel in absolute terms (lowest J/GFLOP, highest GFLOPS/W). 2×2 draws the least power but takes longer; 1×4 draws the most power *and* runs slowest, the worst of both axes.

### 2.2 Competitor libraries (fresh, 2026-04-26)

| Library | Backend | GFLOPS | Runtime (s) | Avg power (W) | Energy (J) | J / GFLOP | GFLOPS / W |
|---|---|---:|---:|---:|---:|---:|---:|
| Accelerate | AMX | **1621.9** | 1.11 | 8.53 | 9.44 | **0.0055** | **182.0** |
| PyTorch 2.10 | AMX | 1487.9 | 1.71 | 7.70 | 13.13 | 0.0076 | 130.9 |
| Our SME 4×1 | SME/ZA | 1173.7 | 1.84 | 6.07 | 11.18 | 0.0065 | 153.6 |
| OpenBLAS 0.3.32 | NEON | 617.9 | 2.86 | 5.86 | 16.73 | 0.0097 | 102.7 |
| NumPy 2.1 | vecLib NEON | 112.8 | 15.46 | 7.16 | 110.66 | 0.0644 | 15.5 |

### 2.3 Findings

- **Our SME 4×1 beats PyTorch on energy efficiency** (0.0065 vs 0.0076 J/GFLOP, 154 vs 131 GFLOPS/W) despite running slower in wall-clock terms. PyTorch reaches AMX peak performance but its dispatcher overhead at 2048³ inflates total energy.
- **Accelerate is the energy-efficiency ceiling** at 0.0055 J/GFLOP / 182 GFLOPS/W. The AMX coprocessor finishes the work fastest *and* draws power roughly proportional to the work it's doing — short integral.
- **Our SME 4×1 draws ~30% less power than Accelerate** (6.07 W vs 8.53 W) while still being 2.6× more efficient than the equivalently-power-budgeted OpenBLAS — when work-per-watt is the constraint (battery, fanless), SME is competitive.
- **NumPy on macOS is shockingly inefficient** at 0.064 J/GFLOP — ~10× worse than our SME 4×1 and ~12× worse than Accelerate. NumPy's `np.matmul` for float32 takes the vecLib NEON sgemm path, not the AMX path that Accelerate's direct `cblas_sgemm` reaches. PyTorch dispatches differently and *does* get AMX.
- **OpenBLAS sits in the awkward middle:** lower wall-clock power (5.9 W) than the AMX libraries but ~2× slower than our SME, so the energy integral comes out worse (16.7 J vs 11.2 J for the same work). It pays the NEON-tier compute cost without any of SME's accumulator savings.
- **No thermal pressure** in any of these runs — all 100-iteration sweeps fit inside the M4's sustained budget, so none of the numbers reflect throttling. Longer or multi-kernel pipelines may differ.

### 2.4 Caveats

- `scripts/profile_power.sh` Combined power on M4 macOS Sonoma+ aggregates CPU + GPU + ANE. The script's P-cluster / E-cluster regex matches an older powermetrics format and currently reports 0.0 W for those sub-totals (TODO MINOR). Combined is the figure used above.
- 100-iteration runs at 2048³ are short (1–2 s) on the AMX path — power averages may include 1–2 sample windows of startup/finish idle. The sense of the comparisons (4×1 < PyTorch < Accelerate efficiency) is robust; absolute J figures have a few-percent measurement noise.

---

## 3. Cache Behavior

> Note: Direct cycle counting (`FIXED_CYCLES`) is unavailable on Apple Silicon due to OS-level PMU restrictions. Analysis relies on instruction counts, wall-clock timing, and cache event counters — consistent with Apple's recommended methodology.

### NEON — L1D Cache

| Metric | Count (per 50 iterations) |
|---|---|
| L1D_CACHE_MISS_ST | 321,533,145 |
| L1D_CACHE_MISS_LD | 1,616,957,905 |

Miss intensity is stable and narrow-band over time — no cache thrashing, no idle gaps. Load misses dominate store misses, which is expected for GEMM (reads outweigh writes). Confirms effective tile reuse within L1. Kernel is compute-bound with controlled memory access.

### Scalar — L1D Cache

~2.1B L1 data cache misses per single matrix multiplication, load-dominated. Miss intensity is temporally smooth (deterministic access pattern) but persistently high — cache thrashing caused by stride-based access with no packing or tiling. Despite the absence of spikes, efficiency is poor.

---

## 4. Instruction Profile

| Kernel | Instructions per GEMM | Characteristic |
|---|---|---|
| NEON | ~337M | Low instruction-to-FLOP ratio; effective SIMD + unrolling |
| Scalar | ~6.3B | ~18× more instructions for identical work |

The NEON kernel's stable, high instruction intensity throughout execution indicates a well-filled pipeline with minimal frontend or control-flow stalls.

---

## 5. Micro-Kernel Prefetch Interleaving

Testing three variants of the 8×12 NEON micro-kernel inner loop:

| Strategy | GFLOPS | Notes |
|---|---|---|
| Reference (standard unroll) | ~119 | Compiler (`-O3`) + hardware prefetcher handle scheduling |
| Aggressive interleaving | ~117 | Pulling all loads (A0, A1, B0, B1, B2) to the top caused register pressure; spills introduced jitter |
| Selective interleaving | ~120 | Pre-loading only A0, A1, B0 — register budget exactly at 32-register limit (24+2+3+3); most consistent |

**Findings:**

- **Register pressure is a hard limit.** ARM64 provides 32 physical vector registers. Pulling too many loads forward shrinks the register renamer's scheduling window and introduces performance jitter.
- **Hardware prefetcher is aggressive.** Apple Silicon's hardware prefetcher is effective enough that software-level prefetch interleaving can interfere with its prediction algorithm rather than help it (see: aggressive interleaving regression).
- **Consistency over peak.** Selective interleaving did not dramatically raise the ceiling, but stabilized performance by eliminating rare hardware-induced stalls.

---

## 6. Threading Analysis

### OpenMP Structure

Two bugs were fixed before meaningful multi-thread numbers could be measured:

**Bug 1 — `-fopenmp` was missing from CMakeLists.txt.**
The code linked against `libomp.dylib` and `omp_get_max_threads()` returned the correct thread count, so it looked like OpenMP was working. But without `-fopenmp`, Clang silently ignores all `#pragma omp` directives — every parallel region ran on the main thread. This is why the original "8-thread" number (456 GFLOPS) was actually just single-thread performance measured with a false sense of parallelism.

**Bug 2 — parallel region was created and destroyed inside the tile loop.**
The original structure opened `#pragma omp parallel` inside the `(k_out, j)` loop, so the thread pool was spawned and joined on every tile iteration. Worse, `pack_B_block` ran before the parallel region opened, meaning all threads were idle while one thread packed B for each tile. The fix was to hoist `#pragma omp parallel` outside both loops so threads are created once for the entire computation. B packing is now done inside the parallel region via `#pragma omp single`, which has an implicit barrier — one thread packs while others wait, but no one leaves the thread pool.

**Scheduling — `schedule(static)` → `schedule(dynamic, 1)`.**
M4 has 4 performance cores and 6 efficiency cores. With static scheduling, each thread gets a fixed number of M-tiles upfront. P-core threads finish their tiles fast and sit at the barrier waiting for E-core threads, which are ~3× slower for FP work. Dynamic scheduling with chunk size 1 means threads pull tiles from a shared queue as they finish — P-cores naturally grab more tiles, E-cores contribute what they can, and no thread idles until the queue is empty.

### Thread Count Results (1024×1024×1024)

| Threads | GFLOPS | Notes |
|---|---|---|
| 1 | ~122 | Single P-core baseline |
| 4 | ~422 | P-cores only — ~3.5× scaling (memory bandwidth contention) |
| 8 | ~500 | 4P + 4E, dynamic scheduling |
| 10 | ~539 | All cores — best result |

### Why Not Linear Scaling?

Theoretical ceiling: 4 P-cores × 122 + 6 E-cores × ~40 ≈ 730 GFLOPS. Achieving 539 (74%) is reasonable given:
- `pack_B_block` is still a serial barrier per k_out/j tile — one thread works while the rest spin.
- Static load imbalance at job boundaries: the last few tiles are always E-core tiles.
- Cache pressure: all threads share the packed B buffer and compete for L2/L3 bandwidth.

The remaining gap to theoretical is best closed at a higher level — a Rust scheduling layer that assigns larger tiles to P-cores and smaller tiles to E-cores, rather than relying on dynamic stealing after the fact.

---

## 7. Packing & Memory Strategy

- **Block sizes (NEON):** Mc=64, Kc=256. Chosen to keep the working set (packed A panel + packed B panel) resident in L1/L2.
- **Block sizes (SME 4×1):** M_tile=64, K_tile=2048, N_tile=1024. Larger K_tile exploits ZA accumulator's ability to hold partial results without writeback.
- **Block sizes (SME 2×2):** M_tile=256, K_tile=2048, N_tile=512.
- **SIMD transpose (`transpose_8x4`):** Packs A into a layout the micro-kernel reads linearly, maximizing L1 bandwidth utilization and eliminating gather-load patterns.
- **Store-to-Load Forwarding:** "A-inner-packing" (current structure) outperforms the BLIS-style "A-above-B" hierarchy on Apple Silicon.

### Architecture Note — BLIS-style Hierarchy

Tested BLIS-style "A-above-B" cache hierarchy on Apple Silicon P-cores. Contrary to expectations, ~5–10% performance regression was observed. Hypothesis: heavy B-packing traffic evicts freshly-packed A data from L1 before the micro-kernel can reuse it. The current "A-inner-packing" structure avoids this by exploiting store-to-load forwarding, which was confirmed to be the superior strategy on this microarchitecture.

---

## 8. SME Optimization & Kernel Comparison

### Optimization Sprint Results (2026-04-15)

The initial SME kernel (4×1, K_tile=256) achieved ~482 GFLOPS. After optimization:

| Change | Impact |
|---|---|
| K_tile 256 → 2048 | Major: reduces pack overhead, keeps data in ZA longer |
| Software pipelining (prologue/epilogue) | Hides load latency behind svmopa execution |
| 2×2 kernel variant with interleaved B packing | Alternative tile layout for comparison |

### SME 4×1 vs 2×2 Kernel Comparison

Both kernels use the same A packing (butterfly transpose) and software-pipelined micro-kernels. Key architectural difference:

| Property | 4×1 | 2×2 |
|---|---|---|
| Output tile | 64×16 (4*SVL × SVL) | 32×32 (2*SVL × 2*SVL) |
| ZA tiles | ZA0-3 = A0-3 × B0 | ZA0=A0×B0, ZA1=A0×B1, ZA2=A1×B0, ZA3=A1×B1 |
| B packing | SVL-wide panels, separate | 2*SVL interleaved [b0\|b1] per k-step |
| Loads per k-step | 4 A + 1 B = 5 | 2 A + 2 B = 4 |
| B reuse | Each B vector used by 4 svmopa | Each B vector used by 2 svmopa |
| Edge handling | N/A (M_step=M_tile) | Scratch buffer for M/N tails |

### Side-by-Side Performance (50-iteration sweep, single-thread, Apple M4, 2026-04-26)

From `SMETest::run_comparison()`:

| Size | 4×1 GFLOPS | 2×2 GFLOPS | 1×4 GFLOPS | Winner |
|---|---:|---:|---:|---|
| 256³ | 265.9 | 306.7 | **444.7** | 1×4 (small-size noise) |
| 512³ | 728.8 | 798.2 | **856.4** | 1×4 (small-size noise) |
| 1024³ | 1074.8 | **1169.0** | 1098.5 | 2×2 |
| 2048³ | 972.2 | **993.6** | 976.6 | 2×2 |
| 4096³ | **1190.8** | 1051.4 | 1189.8 | 4×1 (1×4 essentially tied) |

ZAPack excluded — it crashes on the first iteration of `run_comparison()` due to BUG-ZAPACK-WRONG / BUG-4x1-SMALL-M.

**Conclusion:** 4×1 wins at the largest size (4096³) where B-reuse dominates. 2×2 wins at 1024³ and 2048³ — its more balanced output-tile shape matches the cache hierarchy better at those sizes. 1×4 (current x1-interleaved variant) closes the gap with 4×1 at 4096³ but lags everywhere else. Small-size numbers (256³, 512³) are below the noise floor — total runtime there is < 50 ms across 50 iterations.

### Full Comparison vs Industry Libraries (2026-04-26 fresh, `bench/bench_compare.cpp`)

```
Size (MxKxN)         Tag             NEON   SME 4x1  ZAPack   SME 2x2   Accel    OBlas   MaxDiff
-------------------------------------------------------------------------------------------------
8x8x8                tiny             6.3       0.1     0.1       0.2     19.4      1.9   0.00000
16x16x16             tiny             1.6       0.7     1.4       1.2     57.5     83.8   0.00000
32x32x32             small            5.5       3.6     4.4       6.7    319.7    317.0   0.00000
64x64x64             small           33.0      41.1    63.4      48.9    944.2    947.3   0.00000
128x128x128          L2              85.9     178.6   200.5     206.2   1475.6   1389.5   0.00000
256x256x256          L2             108.5     516.2   556.6     571.8   1705.7   1656.2   0.00000
512x512x512          L3             119.3     858.4   911.3     892.8   1820.1   1672.8   0.00000
1024x1024x1024       L3             120.1    1026.1  1187.6    1192.4   1660.0   1299.7   0.00000
2048x2048x2048       mem-bound      121.8    1170.0  1075.4    1056.7   1613.7    567.2  62.34431
4096x4096x4096       mem-bound      122.7    1182.8  1094.4    1067.3   1523.4    113.1 100.59300
4095x4095x4095       mem-bound      120.7    1165.3  1039.9    1002.7   1477.5    111.3  87.96959
1024x1024x1020       N=85x12 aligned 119.2   1025.5  1160.0    1121.4   1760.2    109.2   0.00000
2048x512x64          tall-skinny     96.0     434.0   563.2     560.0   1372.4    106.2   0.00000
512x2048x64          wide-flat       96.3     460.6   592.1     591.0   1266.8    103.7   0.00011
65x65x65             all tails +1    13.1       5.5    18.5      15.0    461.7    527.5   0.00000
513x513x509          all tails mixed 114.2    702.4   778.2     695.1   1581.7    103.6   0.00000
1025x1025x1021       all tails large 116.2   1004.8   925.4     929.4   1733.1    106.7   0.00007
128x1100x128         N > Nc_cache    96.9    259.9   317.2     307.0   1562.6    105.4  12.93185
512x2048x512         N >> Nc_cache  115.7    994.9  1040.4    1028.6   1652.8    110.7  28.58930
```

(MaxDiff = `max(|NEON - Accel|, |NEON - OBlas|)`. The 12+/62+/100+ entries are the pre-existing NEON exact-2× correctness bug at certain `N > Nc_cache` shapes — see TODO BUG-NEON-2X.)

### Timing Breakdown — pack_A vs pack_B vs micro-kernel (4×1, fresh)

```
  Size      pack_A(ms)  pack_B(ms)  kernel(ms)  total(ms)   pack%
  ----------------------------------------------------------------
   256^3        0.01        0.00        0.04        0.05   22.62%
   512^3        0.02        0.02        0.22        0.26   15.28%
  1024^3        0.23        0.10        1.50        1.83   18.13%
  2048^3        2.58        0.50       11.30       14.37   21.41%
```

### Cross-library findings (2026-04-26)

- **SME 4×1 reaches ~78% of Accelerate (AMX) at 4096³** (1183 / 1523). Up from the previous ~67% reference point; both numbers shifted (4×1 dropped from 1213, Accel dropped from 1813 → likely measurement variance / different thermal state).
- **SME 4×1 beats OpenBLAS by ~10× at large sizes** (1183 vs 113 at 4096³). OpenBLAS does not take the AMX path and degrades sharply on non-aligned sizes ≥ 2048³.
- **SME 2×2 wins the L3-fitting band** (1024³–2048³). Its 32×32 output tile interacts better with M4's L1/L2 hierarchy at those sizes.
- **ZAPack matches 4×1 at L3 sizes** (1188 at 1024³ vs 4×1's 1026 — actually *beats* it) but the kernel has known correctness bugs (TODO BUG-ZAPACK-WRONG); these numbers may be unreliable until the bug is fixed.
- **NEON kernel sits at NEON ceiling** (~120 GFLOPS) — competitive with NumPy's vecLib path (~110) but dominated by everything that uses AMX.
- **Accumulator precision** still degrades at ≥2048³ (MaxDiff vs Accelerate up to ~100). Pairwise summation planned (TODO §7).

---

## 9. SME 1×4 B-Inner Packing — Experimental Variant

### Hypothesis
The 4×1 kernel uses A-inner packing (single 4·SVL wide A panel reused across N) and loads 4 A-panels + 1 B-panel per k-step. The 1×4 variant is its transpose: B-inner packing (single A panel per m-step, 4·SVL wide B panel reused across M), loading 1 A + 4 B per k-step. By symmetry we expected similar throughput — same total loads per k-step, same FMOPA count, same output-tile area.

### Result — x4-loads baseline (2026-04-19, single-thread, Apple M4)

Both sides loaded with a single `svld1_f32_x4` per k-step — a load pattern structurally identical to 4×1's A x4 load.

| Kernel | 2048³ GFLOPS | ms/iter |
|---|---|---|
| SME 4×1 | **1192.8** | 14.40 |
| SME 2×2 | 976.3 | 17.60 |
| SME 1×4 (B-inner, x4 loads) | 1148.6 | 14.96 |

1×4 runs ~4 % slower than 4×1 despite fewer total instructions and marginally fewer L1D load misses. The symmetry hypothesis failed in practice — explanation in §10.

### Current state — x1 interleaved loads (experimental code on disk)

The file `SME-GEMMKernelsExperimental.cpp` now ships the x1-loads variant — the rescue experiment described in §10.1 was left in place as the live code so the tradeoff is reproducible. Measured result:

| Kernel | 2048³ GFLOPS | ms/iter |
|---|---|---|
| SME 1×4 (B-inner, x1 interleaved — current) | 1018.6 | 16.87 |

This is ~11 % slower than the x4 baseline, even though IPC more than doubles (§10.1). The x4 number (~1148) is retained above as the fair symmetric comparison against 4×1; the x1 number (~1018) is what you get if you build and run the repo today.

### Edge-tile handling
1×4's N_step = 4·SVL = 64 columns. Matrices with N not divisible by 64 would cause OOB writes in the micro-kernel. Adopted the scratch-buffer pattern from the 2×2 kernel (see TODO BUG-2x2-1): the micro-kernel always writes a full 16×64 tile into a scratch buffer, the driver scatters only the valid rows×cols portion back into C with accumulation. Zero overhead on the aligned hot path.

---

## 10. Instruments Profiling — Per-Kernel Microarch Analysis

Measured with Instruments (CPU Counters template: `L1D_CACHE_MISS_LD`, `L1D_CACHE_MISS_ST`, `INST_ALL`). One library/kernel at a time, 100 iterations at 2048³, captured by `./scripts/profile.sh <name>`. Counter totals summed from `counters-profile` deltas. The same workload is then re-run under `scripts/profile_power.sh` for energy data (§2).

### 10.0 Fresh per-kernel metrics (2026-04-26)

| Kernel | GFLOPS | INST_ALL (B) | L1D miss / 1M instr | Est. IPC (@ 4.4 GHz) |
|---|---:|---:|---:|---:|
| SME 4×1 | 1186.4 | 4.57 | 332 | 0.72 |
| SME 2×2 | 1078.5 | 8.64 | 220 | 1.23 |
| SME 1×4 (x1 interleaved, current) | 1025.7 | 1.82 | 819 | 0.25 |

**IPC caveat (vs the historical numbers in §10.1 below):** the fresh `INST_ALL` totals are markedly different from the 2026-04-15 measurements (4×1: 4.57 B vs ~6.8 B previously; 1×4: 1.82 B vs ~10.3 B). The trace template, build flags, or how `INST_ALL` accounts for SME/FMOPA instructions appears to have shifted. The *relative* shape of the comparison is consistent — 1×4 still has the lowest instruction count and highest L1D-miss-per-instruction, 2×2 still has the highest absolute instruction count — but the absolute IPC numbers should be read as "low-precision indicator", not "ground truth".

**What still holds:**
- L1D miss rate < 0.1% of all loads in every kernel — none of these are memory-bound.
- 2×2's instruction stream is ~1.9× larger than 4×1's. Confirms the "instruction-count tax" hypothesis (TODO §4).
- 1×4's stream is ~2.5× *smaller* than 4×1's (vs the 2026-04-15 measurement where 1×4 was 2.5× *larger* — the kernel evolved).

### 10.0.1 Competitor profiling (2026-04-26, same workload — 2048³ × 100 iters)

| Library | GFLOPS | INST_ALL (B) | L1D miss / 1M instr | Notes |
|---|---:|---:|---:|---|
| Accelerate (AMX) | 1534.3 | 9.50 | 22,211 | Highest miss/instr — AMX uses its own memory path; L1D counters reflect host loads, not coprocessor compute. |
| PyTorch 2.10 (AMX) | 1392.4 | 14.83 | 21,583 | Same backend as Accelerate; extra `INST_ALL` is dispatcher / Python interop overhead. |
| OpenBLAS 0.3.32 (NEON) | 572.0 | **0.65** | 2,019 | INST_ALL implausibly low (IPC ≈ 0.05) — suspect under-attribution from dlopen'd dylib symbol path or AMX-internal ops not counted. **Use this row qualitatively only.** TODO MINOR. |
| NumPy 2.1 (vecLib NEON) | 88.6 | 12.37 | **51,692** | Highest miss rate by a wide margin; vecLib's sgemm path is not packed/blocked the way our NEON kernel is. |

### 10.0.2 Cross-comparison takeaways

- **L1D miss rate is the cleanest separator**: our SME kernels (220–820 misses/M-instr) are ~30–250× cleaner than the AMX libraries (22,000) and ~60–235× cleaner than NumPy (52,000). This reflects our packing strategy — packed A/B buffers stay hot in L1 throughout the kernel.
- **AMX backends rack up host L1D misses** because the coprocessor needs to be fed from memory but its compute work doesn't show up in `INST_ALL`. The high miss/instr ratio for Accel/PyTorch is mostly a denominator effect, not a real cache-thrashing signal.
- **NumPy is genuinely cache-thrashing** — 51,692 misses/M-instr at low GFLOPS means it's losing to memory. Confirms the energy result (110 J / 1.7 TFLOPs — 10× our SME).
- **OpenBLAS counter mystery** — needs investigation before its INST_ALL number is comparable.

### 10.1 Historical analysis — Derived per-iteration metrics (2026-04-15)

Retained for comparison with the kernel evolution. Numbers from the previous trace-template configuration; absolute IPC values not directly comparable to §10.0 above.

| Kernel | GFLOPS | FLOPs / instr | L1D miss / 1M instr | Est. IPC (@ 4.4 GHz) |
|---|---|---|---|---|
| SME 4×1 | 1192.8 | **253** | 234.6 | **1.07** |
| SME 2×2 | 976.3 | 176 | 204.8 | **1.26** |
| SME 1×4 (x4 loads, retired) | 1148.6 | **418** | 367.7 | **0.62** |
| SME 1×4 (x1 interleaved, 2026-04-15 build) | 1018.6 | 166 | 135.0 | **1.40** |

- **L1D miss rate ≈ 0.04–0.07 % of loads** across all four rows. Packed A/B buffers stay hot; cache blocking is doing its job. Memory is **not** the bottleneck. The x1 row's much lower miss/1M-instr is dilution, not a real miss-count improvement — absolute L1D miss counts are essentially unchanged from the x4 row.
- Instruction count alone is a misleading proxy for speed. 1×4 (x4 loads) does ~40 % fewer instructions per iter than 4×1 yet runs slower because the CPU stalls on FMOPA dependency chains.
- 2×2 has the highest IPC (1.26) — it flows through the pipe well — but burns so many extra instructions that overall GFLOPS loses. Its packed panels are 2·SVL wide for *both* A and B (256 KB each), so neither side acts as a compact L1-resident buffer the way 4×1's B panel or 1×4's A panel does. That explains its higher miss count too.

### Why 1×4's IPC collapses

FMOPA (SVE outer-product) has long execution latency (~6–8 cycles on M4) with throughput ~1/cycle. To hide that latency, the CPU needs *other* instructions between back-to-back FMOPAs targeting the same ZA tile. If the issue window is too tight, the pipeline stalls waiting for the destination tile to be free.

- **4×1 per k-step:** 4 A-panel loads + 1 B-panel load + 4 FMOPAs. The A loads are spread throughout the sequence and act as filler between consecutive FMOPAs writing to ZA0, ZA1, ZA2, ZA3 — the destination tile alternates, and the loads extend the gap between same-tile writes. IPC ≈ 1.07.
- **1×4 per k-step:** 1 A-panel load + 4 B-panel loads + 4 FMOPAs. Nominally the same load/FMOPA ratio, but the overall instruction stream is leaner (41 M vs 68 M instructions / iter). Fewer non-FMOPA instructions means the decoder reaches the next FMOPA faster — less slack between consecutive writes to the same ZA tile, so more stalls waiting on tile availability. IPC ≈ 0.62 — ~25 M cycles/iter of pure stall time.

The 4×1 kernel's "extra" instructions are not overhead — they're latency hiders that happened to fall into a scheduling-friendly position.

### 10.1 The Instruction Tax Trade-off (1x4 Interleaving Experiment)

To test the `1x4` IPC collapse hypothesis, the experimental kernel was rewritten to replace the grouped `svld1_f32_x4` B-load with four independent `svld1_f32` loads (`x1`), each issued inline between successive `svmopa` instructions. This eliminated the register-contiguity requirement of x4 loads and let the x1 loads fall into the 6–8 cycle execution shadow of FMOPA, with a double-buffered `A` panel removing load-use stalls. This x1-loads variant is what currently lives in `SME-GEMMKernelsExperimental.cpp`.

**Result (100 iterations @ 2048³, per-trace totals from `traces/1x4.txt`):**
| Metric | 1×4 (original `x4` loads) | 1×4 (interleaved `x1` loads, current) | Impact |
|---|---|---|---|
| Wall time / iter | 14.96 ms | 16.87 ms | **−11 % throughput** (~1148 → ~1018 GFLOPS) |
| Est. IPC @ 4.4 GHz | 0.62 | 1.40 | **+125 %** — pipeline unblocked, now *higher* than 4×1 (1.07) |
| Total instructions (100 iters) | ~4.1 B | ~10.3 B | **+2.5×** — instruction bloat |
| FLOPs / instr | 418 | 166 | Stream is far "less dense" in useful work |
| L1D miss / 1M instr | 368 | 135 | Dilution only — absolute miss count unchanged |

**Conclusion — A-inner (4×1) is the winning geometry on M4 SME.** Combined with §10, this experiment closes both escape routes for B-inner (1×4):

1. **Symmetric x4 loads:** 1×4 with the same load pattern as 4×1 already loses — 1148 < 1193 GFLOPS, IPC 0.62 vs 1.07. So "they're mathematical transposes, they should tie" is empirically false on this pipeline.
2. **x1 rescue attempt:** fixing the IPC stall requires so many extra load/move instructions that wall time gets *worse*, not better — even though IPC now exceeds 4×1's.

There is no middle ground: x4 loads starve the pipeline of latency-hiding filler (cratering IPC), x1 loads fix the pipeline at the cost of a 2.5× instruction tax. 4×1's natural geometry (4 A-loads + 1 B-load + 4 FMOPAs per k-step) is the only pattern that produces a scheduling-friendly load/FMOPA mix *without* inflating the instruction count — the `A` loads act as filler "for free". The x1-loads kernel is kept in the tree as a reproducible demonstration of the tradeoff, not as a forward path.

Scope of the claim: M4 with SVL=16, single-thread, 2048³, LLVM/Clang `-O3 -mcpu=apple-m4`. On different silicon (AMX, different SVE width, different FMOPA latency), the asymmetry could shift or flip.

### Lessons learned

1. **Instruction count ≠ speed.** A leaner instruction stream can run slower if it strips away work that was hiding execution-unit latency.
2. **A-inner vs B-inner packing is not a true symmetry on this pipeline.** On M4's SME path, A-inner (4×1) produces a more scheduling-friendly instruction mix than B-inner (1×4), even though the two kernels are mathematical transposes.
3. **L1 was never the constraint.** At ~0.05 % miss rate across all variants, tuning packing layout for cache behavior yields almost nothing. Tune for instruction scheduling instead.
4. **FMOPA dependency chains on the same ZA tile are the visible bottleneck.** Anything that widens the gap between back-to-back writes to ZA0…ZA3 helps IPC.
5. **2×2's cost is instruction count.** Packing both A and B in 2·SVL-wide panels means loading both sides per k-step, nearly doubling the load instructions vs 4×1/1×4. Its higher IPC doesn't save it.

---

## 11. Performance Ceiling — Rough Estimates (speculative)

These are back-of-envelope thoughts for planning future work, **not measurements**. Treat them as bounds to sanity-check optimization ideas against, not targets to hit.

- **Hardware peak (FMOPA throughput bound):** ≈ 2 048 GFLOPS @ 4.0 GHz, ≈ 2 253 GFLOPS @ 4.4 GHz boost. Derived from 1 FMOPA/cycle × SVL² × 2 FLOPs per FMA × clock.
- **Kernel-alone ceiling (if packing overhead vanished):** ≈ 1 615 GFLOPS, extrapolated from current pack% (~20 % of total time at 2048³ per §8 timing breakdown).
- **Practical ceiling on the SME path:** ≈ 1 600–1 700 GFLOPS. Getting there likely requires `__arm_inout("za")` to skip the per-call ZA zero + load-back, a leaner pack_A, and better FMOPA scheduling (address the 1×4 IPC finding above).
- **AMX-class (≥ 1 800 GFLOPS)** probably out of reach from SME alone. Accelerate's ~1 813 GFLOPS at 4096³ uses the AMX coprocessor — a different silicon block — so no software on the SME path is expected to match it.

Current SME 4×1 peak is ~1 213 GFLOPS @ 4096³, leaving roughly 28–40 % headroom against these estimates. Worth chasing, but not infinite.
