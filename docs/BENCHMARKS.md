# Benchmark & Experimental Results

Platform: Apple Silicon (M4), macOS  
Problem size: M=1024, N=1020, K=1024 (unless noted)  
Compiler: LLVM/Clang `-O3 -mcpu=apple-m4`

## How to read this document

This file is a running lab notebook, not a snapshot. Measurements were taken on
different dates against different builds of the kernels, and some later results
overturned earlier conclusions. Rather than deleting superseded work — the wrong
turns are often the most informative part — every section is dated and labelled:

| Label | Meaning |
|---|---|
| **CURRENT** | Reflects the code as it stands. Trust these numbers. |
| **STANDS** | Older measurement, but nothing since has invalidated it. |
| **SUPERSEDED** | Numbers replaced by a later run. Kept for the progression; do not quote. |
| **PARTLY OVERTURNED** | The data is real, but a later result contradicts the conclusion drawn from it. Read the linked section too. |

If you want only the current state of the project, read **§0** and stop.
Everything after it is history, in the order it happened.

**Timeline of measurement dates** (from git history):

| Date | What was measured |
|---|---|
| 2026-04-09 | NEON kernel complete — cache, instruction, prefetch, threading analysis (§3–§6) |
| 2026-04-10 | First single-thread competitor comparison |
| 2026-04-15 | SME optimization sprint — 4×1 + 2×2, first PMU profiling (§8, §10.1) |
| 2026-04-19 | SME 1×4 B-inner experiment (§9) |
| 2026-04-26 | ZAPack kernel, energy/thermal tooling, full re-measurement (§2, §8, §10.0) |
| 2026-06-10 | 1×4-sym and 1×4ZAIO kernels added |
| 2026-07-25 | Correctness bug-fix session, full re-measurement (§0) — **current** |

---

## 0. 2026-07-25 Refresh — Bug-Fix Session

**Measured:** 2026-07-25 · **CURRENT**

All three outstanding correctness bugs were fixed this session with **zero hot-path cost** (aligned sizes take the identical instruction stream as before):

- **BUG-4x1-SMALL-M** — scratch-buffer edge-tile fallback added to 4×1 and ZAPack drivers (1×4 family already had it; harness skip-list was stale). All six kernels now pass the full correctness list (16³, 32³, 20×35×41, 67³, …).
- **BUG-NEON-2X** — `Nc_cache` 1024 → 1020 (85×12). The old value violated the multiple-of-12 invariant, making the last 12-wide panel of each cache block double-accumulate 8 columns into the next block. MaxDiff at the failing shapes: was 12–100, now ≤ 0.0004 (pure fp32 rounding).
- **BUG-ZAPACK-WRONG** — pack_A layout collision: panel index `p = m/SVL` wasn't wrapped per 4-panel group, so with `M_tile=128` panels 4–7 overwrote k+1's packed data. Fixed with `(m/SVL) % 4` + per-group base offset. ZAPack re-enabled everywhere.
- **Accumulator precision** — closed as **not a bug**. Measured against fp64 ground truth (`cblas_dgemm`) at 4096³: Accelerate maxErr 0.000348, our SME 4×1 0.000209, our NEON 0.000115 — our kernels are *more accurate* than AMX. The historical "MaxDiff ~100" was BUG-NEON-2X corrupting the comparison baseline.

### 0.1 Side-by-side kernel comparison (interleaved, single-thread, 2026-07-25)

From `SMETest::run_comparison()` — all six kernels, interleaved timing:

| Size | iters | 4×1 | 2×2 | 1×4 | 1×4-sym | 1×4ZAIO | 4×1-ZAPack | Winner |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 256³ | 400 | 715.7 | **766.4** | 729.7 | 731.4 | 609.0 | 762.4 | 2×2 |
| 512³ | 200 | 1057.1 | **1111.4** | 1050.2 | 1064.8 | 917.8 | 1110.7 | 2×2 |
| 1024³ | 100 | 1142.6 | 1186.2 | 1235.7 | **1243.6** | 1119.8 | 1190.4 | 1×4-sym |
| 2048³ | 50 | 1202.7 | 1067.2 | 999.2 | 1128.1 | **1259.0** | 1120.0 | 1×4ZAIO |
| 4096³ | 20 | 1173.7 | 1059.1 | 1215.4 | **1308.5** | 1086.7 | 1087.4 | 1×4-sym |

Notable shifts vs the 2026-04-26 measurement (§8):

- **1×4-sym takes the crown at 4096³ (1308.5)** — the "true mirror of 4×1" B-inner kernel now clearly beats 4×1 (1173.7) at the largest size, weakening the §10 "A-inner is the only winning geometry" conclusion. The x4-grouped B-load variant appears to have been rehabilitated by the kernel/driver evolution since §10's measurements.
- **1×4ZAIO wins 2048³ (1259.0)** — ZA-resident accumulation across K-chunks (`__arm_inout("za")`, K_inner_tile=40) pays off exactly where the store-back traffic hurt most.
- **ZAPack, now correct, is real competition** (1190 @ 1024³ interleaved; 1310–1355 in isolated suite runs) — the ZA-based pack_A transpose is at minimum equal to the butterfly, and its `M_tile=128` tiling wins the L3 band.
- **No single kernel wins everywhere.** 2×2 owns ≤512³, ZAPack/1×4-sym the 1024³ band, ZAIO 2048³, 1×4-sym 4096³. 4×1 is never first but never worse than ~10% off — it remains the safe default alongside its energy-efficiency lead (§2).

Run-to-run thermal variance on these kernels is a few percent; treat single-digit-percent gaps as ties.

### 0.2 Industry-library comparison (fresh, `bench/run_bench.sh`, 2026-07-25)

```
Size (MxKxN)          Tag                  NEON GF   SME 4x1    4x1 ZP   SME 2x2  Accel GF  OBlas GF  MaxDiff
--------------------------------------------------------------------------------------------------------------
8x8x8                 tiny                     5.8       0.3       0.4       0.6      18.4       6.4  0.00000
16x16x16              tiny                     8.4       4.1       4.0       4.0      51.5      52.4  0.00000
32x32x32              small                   41.0      13.5      13.6      61.8     340.3     404.4  0.00000
64x64x64              small                   84.2     193.8     206.1     203.5    1051.6     994.6  0.00000
128x128x128           L2                     106.9     414.4     447.9     441.8    1549.7    1359.9  0.00000
256x256x256           L2                     115.4     712.9     762.8     765.9    1767.3    1364.1  0.00000
512x512x512           L3                     121.5    1056.4    1110.4    1113.7    1697.9    1590.3  0.00000
1024x1024x1024        L3                     122.7    1187.4    1280.8    1287.8    1679.6    1463.6  0.00000
2048x2048x2048        mem-bound              122.9    1231.0    1096.4    1105.5    1647.4     633.2  0.00018
4096x4096x4096        mem-bound              122.9    1199.7    1059.4    1087.7    1593.8     113.6  0.00039
4095x4095x4095        mem-bound              121.5    1167.4     982.1    1019.2    1591.6     112.3  0.00041
1024x1024x1020        N=85x12 aligned        123.4    1265.4    1345.9    1275.9    1837.3     109.4  0.00000
2048x512x64           tall-skinny            101.0     667.4     879.7     883.4    1263.7     105.2  0.00000
512x2048x64           wide-flat              102.0     734.9     973.6     971.1    1057.9     103.0  0.00011
65x65x65              all tails +1            76.1      80.4      87.7      98.4     542.5     522.6  0.00000
513x513x509           all tails mixed        118.6     857.2     903.6     823.0    1672.0     104.2  0.00000
1025x1025x1021        all tails large        119.0    1145.1     933.6     891.9    1812.6     107.9  0.00007
128x128x1100          N > Nc_cache           112.7     467.5     473.5     462.5    1575.2     104.2  0.00000
512x512x2048          N >> Nc_cache          119.5    1065.5    1095.6    1075.6    1664.3     111.6  0.00000
```

- **MaxDiff ≤ 0.0004 everywhere** — the correctness column is finally flat. Compare §8's historical table where the same shapes showed 12–100.
- **SME 4×1: 1231 @ 2048³, ~1200 @ 4096³** — ~75% of Accelerate at 4096³ (Accelerate itself measured hotter this run: 1594 @ 4096³, 1837 peak at the aligned 1024 case).
- **NEON at 121–123 across every large shape** — including the previously-broken `N > Nc_cache` shapes, at full speed. `65³` jumped from 13 to 76 GFLOPS (small sizes previously took the scalar edge path much harder; the wider correctness surface is also friendlier now).
- **Small sizes (≤32³) remain Accelerate's domain** — our SME kernels pay pack + streaming-mode entry per call; that's a known non-goal.
- Python side (same session): NumPy 107–112 (vecLib NEON path), PyTorch ~1500 (AMX) — both unchanged from §2.2 within noise.

### 0.3 fp32 accumulation error vs fp64 ground truth (2026-07-25)

| | 2048³ maxErr | 2048³ rmsErr | 4096³ maxErr | 4096³ rmsErr |
|---|---:|---:|---:|---:|
| Accelerate (AMX) | 0.000159 | 0.000012 | 0.000348 | 0.000024 |
| **Our SME 4×1** | 0.000159 | 0.000012 | **0.000209** | 0.000017 |
| **Our NEON** | **0.000093** | 0.000009 | **0.000115** | 0.000012 |

(max|C| ≈ 78 at 2048³, ≈ 112 at 4096³; inputs U(−1,1).) Pairwise summation is unnecessary — dropped from the roadmap.

### 0.4 4×1 timing breakdown (fresh)

```
  Size      pack_A(ms)  pack_B(ms)  kernel(ms)  total(ms)   pack%
  ----------------------------------------------------------------
   256^3        0.01        0.00        0.04        0.05   17.57%
   512^3        0.02        0.01        0.22        0.26   12.59%
  1024^3        0.22        0.06        1.51        1.79   15.72%
  2048^3        2.46        0.44       11.29       14.19   20.43%
```

---

## 1. Performance Progression

**Measured:** 2026-04-09 → 2026-07-25 (accumulated) · **CURRENT** — peak column reflects the latest run

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
| SME 4×1 ZAPack | ~1190–1355 | ZA-based pack_A transpose (`svld1_hor_za32` / `svst1_ver_za32`). Packing layout bug fixed 2026-07-25 (see §0); wins the L3 band with M_tile=128. |
| SME 1×4-sym | ~1308 | x4-grouped B loads (true mirror of 4×1). Current peak holder at 4096³ (§0.1). |
| SME 1×4ZAIO | ~1259 | Split zero/compute/store sharing ZA via `__arm_inout`; K_inner_tile=40. Current winner at 2048³ (§0.1). |

**Note on small sizes (historical, fixed 2026-07-25):** all kernels now have a scratch-buffer + scatter-back fallback for partial output tiles — arbitrary M/K/N is safe. The old harness skip-list and `bench_compare` C-padding workarounds are removed.

---

## 2. Power & Energy Analysis

**Measured:** 2026-04-26 · **STANDS** — not re-measured on 2026-07-25. The bug fixes were edge-path-only, so the aligned-size energy figures here should still hold, but they predate the 1×4-sym / ZAPack results in §0.1 and cover only three kernels.

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

**Measured:** 2026-04-09 (NEON era) · **STANDS**

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

**Measured:** 2026-04-09 (NEON era) · **STANDS**

| Kernel | Instructions per GEMM | Characteristic |
|---|---|---|
| NEON | ~337M | Low instruction-to-FLOP ratio; effective SIMD + unrolling |
| Scalar | ~6.3B | ~18× more instructions for identical work |

The NEON kernel's stable, high instruction intensity throughout execution indicates a well-filled pipeline with minimal frontend or control-flow stalls.

---

## 5. Micro-Kernel Prefetch Interleaving

**Measured:** 2026-04-09 (NEON era) · **STANDS**

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

**Measured:** 2026-04-09 · **STANDS** — NEON multi-thread only; SME remains single-threaded (roadmap item)

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

**Measured:** 2026-04-09 → 2026-04-15 · **STANDS** (tile constants below corrected 2026-07-25 to match the code)

Tile constants as of 2026-07-25 (verified against the sources — the earlier
revision of this section quoted stale values for NEON `Kc` and SME 2×2):

| Kernel | Tiling |
|---|---|
| NEON | `Mc=64`, `Kc=1024`, `Nc_cache=1020`. Keeps the working set (packed A + packed B panel) resident in L1/L2. `Nc_cache` **must** stay a multiple of 12 — see BUG-NEON-2X in TODO.md. |
| SME 4×1 | `M_tile=64`, `K_tile=2048`, `N_tile=1024`. Large K_tile exploits ZA's ability to hold partial results without writeback. |
| SME 2×2 | `M_tile=64`, `K_tile=1024`, `N_tile=512`. |
| SME 1×4 / 1×4-sym | `M_tile=1024`, `K_tile=2048`, `N_tile=64` — transposed shape, since B is the inner-packed operand. |
| SME 1×4ZAIO | `M_tile=1024`, `N_tile=128`, `K_inner_tile=40`. K is packed in full; the inner tile is the ZA-resident accumulation chunk (sweep in the source header). |
| SME 4×1 ZAPack | `M_tile=128`, `K_tile=1024`, `N_tile=512`. The wider M_tile is what exposed the packing bug fixed on 2026-07-25. |

- **SIMD transpose (`transpose_8x4`):** Packs A into a layout the micro-kernel reads linearly, maximizing L1 bandwidth utilization and eliminating gather-load patterns.
- **Store-to-Load Forwarding:** "A-inner-packing" (current structure) outperforms the BLIS-style "A-above-B" hierarchy on Apple Silicon.

### Architecture Note — BLIS-style Hierarchy

Tested BLIS-style "A-above-B" cache hierarchy on Apple Silicon P-cores. Contrary to expectations, ~5–10% performance regression was observed. Hypothesis: heavy B-packing traffic evicts freshly-packed A data from L1 before the micro-kernel can reuse it. The current "A-inner-packing" structure avoids this by exploiting store-to-load forwarding, which was confirmed to be the superior strategy on this microarchitecture.

---

## 8. SME Optimization & Kernel Comparison

**Measured:** 2026-04-15 (sprint) + 2026-04-26 (tables) · **SUPERSEDED by §0.1 / §0.2** — these tables predate the 1×4-sym and 1×4ZAIO kernels, and ran against builds carrying the three correctness bugs fixed on 2026-07-25 (the large MaxDiff values below are that NEON bug). Kept for the optimization progression; quote §0 instead.

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

(MaxDiff = `max(|NEON - Accel|, |NEON - OBlas|)`. The 12+/62+/100+ entries were the NEON exact-2× correctness bug at `N > Nc_cache` shapes — **fixed 2026-07-25**, see §0.2 for the clean table.)

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
- **ZAPack matches 4×1 at L3 sizes** (1188 at 1024³ vs 4×1's 1026 — actually *beats* it). *(2026-07-25: bug fixed; the win is confirmed real — see §0.1.)*
- **NEON kernel sits at NEON ceiling** (~120 GFLOPS) — competitive with NumPy's vecLib path (~110) but dominated by everything that uses AMX.
- **Accumulator precision** *(2026-07-25: closed as not-a-bug — the ~100 MaxDiff was BUG-NEON-2X; fp64-truth measurement in §0.3 shows our kernels are more accurate than Accelerate.)*

---

## 9. SME 1×4 B-Inner Packing — Experimental Variant

**Measured:** 2026-04-19 · **PARTLY OVERTURNED** — the measurements are sound, but the conclusion that B-inner loses to A-inner no longer holds: 1×4-sym now leads at 4096³ (§0.1). Read alongside §10's caveat.

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

The file `sme/sme-1x4.cpp` (named `SME-GEMMKernelsExperimental.cpp` at the time of this measurement) ships the x1-loads variant — the rescue experiment described in §10.2 was left in place as the live code so the tradeoff is reproducible. Measured result:

| Kernel | 2048³ GFLOPS | ms/iter |
|---|---|---|
| SME 1×4 (B-inner, x1 interleaved — current) | 1018.6 | 16.87 |

This is ~11 % slower than the x4 baseline, even though IPC more than doubles (§10.2). The x4 number (~1148) is retained above as the fair symmetric comparison against 4×1; the x1 number (~1018) is what you get if you build and run the repo today.

### Edge-tile handling
1×4's N_step = 4·SVL = 64 columns. Matrices with N not divisible by 64 would cause OOB writes in the micro-kernel. Adopted the scratch-buffer pattern from the 2×2 kernel (see TODO BUG-2x2-1): the micro-kernel always writes a full 16×64 tile into a scratch buffer, the driver scatters only the valid rows×cols portion back into C with accumulation. Zero overhead on the aligned hot path.

---

## 10. Instruments Profiling — Per-Kernel Microarch Analysis

**Measured:** 2026-04-15 (§10.1) + 2026-04-26 (§10.0) · **PARTLY OVERTURNED** — the counter data and the FMOPA-latency mechanism still explain why B-inner *was* hard. The headline conclusion ("A-inner is the only winning geometry") is contradicted by §0.1, where the B-inner 1×4-sym kernel wins at 4096³. Re-profiling to explain the reversal is an open roadmap item.

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

### 10.2 The Instruction Tax Trade-off (1x4 Interleaving Experiment)

To test the `1x4` IPC collapse hypothesis, the experimental kernel was rewritten to replace the grouped `svld1_f32_x4` B-load with four independent `svld1_f32` loads (`x1`), each issued inline between successive `svmopa` instructions. This eliminated the register-contiguity requirement of x4 loads and let the x1 loads fall into the 6–8 cycle execution shadow of FMOPA, with a double-buffered `A` panel removing load-use stalls. This x1-loads variant is what lives in `sme/sme-1x4.cpp` today (renamed from `SME-GEMMKernelsExperimental.cpp` in the 2026-06-13 reorganization).

**Result (100 iterations @ 2048³, per-trace totals from `traces/1x4.txt`):**
| Metric | 1×4 (original `x4` loads) | 1×4 (interleaved `x1` loads, current) | Impact |
|---|---|---|---|
| Wall time / iter | 14.96 ms | 16.87 ms | **−11 % throughput** (~1148 → ~1018 GFLOPS) |
| Est. IPC @ 4.4 GHz | 0.62 | 1.40 | **+125 %** — pipeline unblocked, now *higher* than 4×1 (1.07) |
| Total instructions (100 iters) | ~4.1 B | ~10.3 B | **+2.5×** — instruction bloat |
| FLOPs / instr | 418 | 166 | Stream is far "less dense" in useful work |
| L1D miss / 1M instr | 368 | 135 | Dilution only — absolute miss count unchanged |

**Conclusion — A-inner (4×1) is the winning geometry on M4 SME.** *(2026-07-25 update: this conclusion is now partially overturned — the 1×4-sym kernel (x4-grouped B loads) measures 1308 GFLOPS at 4096³, beating 4×1's 1174 in the same interleaved run (§0.1). The IPC-collapse analysis below remains valid for the kernels as they existed on 2026-04-15; the B-inner geometry has since been rehabilitated by driver/tiling evolution. Worth re-profiling with PMU counters to understand what changed.)* Combined with §10, this experiment closes both escape routes for B-inner (1×4):

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

**Written:** 2026-04-26, headroom updated 2026-07-25 · **STANDS** — estimates, never measurements

These are back-of-envelope thoughts for planning future work, **not measurements**. Treat them as bounds to sanity-check optimization ideas against, not targets to hit.

- **Hardware peak (FMOPA throughput bound):** ≈ 2 048 GFLOPS @ 4.0 GHz, ≈ 2 253 GFLOPS @ 4.4 GHz boost. Derived from 1 FMOPA/cycle × SVL² × 2 FLOPs per FMA × clock.
- **Kernel-alone ceiling (if packing overhead vanished):** ≈ 1 615 GFLOPS, extrapolated from current pack% (~20 % of total time at 2048³ per §8 timing breakdown).
- **Practical ceiling on the SME path:** ≈ 1 600–1 700 GFLOPS. Getting there likely requires `__arm_inout("za")` to skip the per-call ZA zero + load-back, a leaner pack_A, and better FMOPA scheduling (address the 1×4 IPC finding above).
- **AMX-class (≥ 1 800 GFLOPS)** probably out of reach from SME alone. Accelerate's ~1 813 GFLOPS at 4096³ uses the AMX coprocessor — a different silicon block — so no software on the SME path is expected to match it.

Current single-thread peak is ~1 308 GFLOPS @ 4096³ (1×4-sym, 2026-07-25 — see §0.1), leaving roughly 20–30 % headroom against these estimates. Worth chasing, but not infinite.
