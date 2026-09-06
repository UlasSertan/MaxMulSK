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
| 2026-09-04 | Arm KleidiAI added to `bench_compare` (§0.5) |
| 2026-09-04 | Dedicated fair head-to-head vs KleidiAI, incl. large-K LLM shapes (§0.6) |
| 2026-09-04 | Layer-separating experiments: hot micro-kernel, controlled macro-flow, ggml CPU (§0.7) |
| 2026-09-04 | llama.cpp KleidiAI SME2 path added as a second ggml baseline (§0.8) |
| 2026-09-05 | 1x4-Acc / 1x4-Acc-Kc: ZA-lifetime experiment (§0.9) |
| 2026-09-05 | 1x4-Acc: read-modify-write dropped from the ZA→C store (§0.10) |
| 2026-09-05 | 1x4-Acc: ZA→C store made row-major, one 4-vector store per row (§0.11) |
| 2026-09-05 | 1x4-Acc: the three steps read end to end as one arc (§0.9-A) |
| 2026-09-05 | 1x4-Acc: pack_A moved from an SVE butterfly to a ZA transpose (§0.13) |
| 2026-09-06 | Current results, median of three full runs; measurement-noise floor established (§0.14) |

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
- **No single kernel wins everywhere.** Among the six kernels measured here, 2×2 owns ≤512³, ZAPack/1×4-sym the 1024³ band, ZAIO 2048³, 1×4-sym 4096³. 4×1 is never first but never worse than ~10% off, which made it the safe default. *(Ranking superseded: the 1×4-Acc kernels added in 2026-09 lead every band up to 2048³ — §0.14 — and 4×1's energy lead now holds only at 128×32768×512 — §0.15.)*

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

### 0.5 Arm KleidiAI comparison (2026-09-04)

**Measured:** 2026-09-04 · **CURRENT**

Arm's [KleidiAI](https://github.com/ARM-software/kleidiai) (pinned at `v1.30.0`)
was added to `bench_compare` as a fourth reference point. Three fp32 kernels:

| Column | KleidiAI micro-kernel | Shape |
|---|---|---|
| `2VL` | `kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa` | SME2 FMOPA, 2VL×2VL tile — the structural analogue of our SME 2×2 |
| `8VS` | `kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa` | SME2 FMOPA, 8VS×8VS tile with 16VS×4VS / 8VS×4VS edge variants |
| `6x8` | `kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla` | NEON FMLA, 6×8 tile — the analogue of our NEON 8×12 |

Two things have to be got right before the numbers mean anything, and both were
initially got wrong:

**1. Packing.** KleidiAI ships micro-kernels, not a BLAS: they consume operands
already packed into a kernel-specific layout. Our kernels, Accelerate and
OpenBLAS all pack *inside* the timed call, so `+pk` (pack + matmul) is the
column comparable to everything else, and `mm` is the micro-kernel in isolation —
what an inference runtime sees once weights are packed once up front.

**2. Cache blocking — this one is the whole story at large sizes.** KleidiAI
ships *no* blocking above the micro-kernel by design; it expects the calling
runtime (ExecuTorch, XNNPACK, …) to supply it, which is what the
`kai_get_rhs_packed_offset` / `kai_get_dst_offset` accessors are for. Called
monolithically on the full problem, the kernel re-reads all of packed B for every
`m_step`=32 row panel, and throughput collapses as soon as B outgrows L2. Driving
the *identical* 2VL kernel through an outer block loop:

| Blocking (2048³) | GFLOPS | | Blocking (4096³) | GFLOPS |
|---|---:|---|---|---:|
| monolithic call | 992 | | monolithic call | 887 |
| split M only (256 / 512 / 1024) | 997–1049 | | split M only | 890–906 |
| **split N only (N_blk = 512)** | **1780** | | **split N only (N_blk = 512)** | **1754** |

Splitting M changes nothing — A streams through once either way. Splitting N is
the entire effect, and the threshold matches the cache exactly: P-core L2 is
16 MB (`hw.perflevel0.l2cachesize`), the re-read panel is `N_blk × K × 4` bytes,
and at 4096³ an `N_blk` of 512 (8 MB panel) sustains 1754 GFLOPS while 1024
(16 MB panel) collapses back to 997. `bench/kleidiai_gemm.cpp` therefore drives
all three kernels through an N-block loop sized to a 4 MB panel budget.

Results with packing split out and N-blocking in place:

```
Size (MxKxN)          Tag                  2VL +pk    2VL mm   8VS +pk    8VS mm   6x8 +pk    6x8 mm  MaxDiff
-------------------------------------------------------------------------------------------------------------
8x8x8                 tiny                    10.0      10.2      12.8      16.5      52.3      73.4  0.00000
16x16x16              tiny                    67.4     144.8      90.7     157.0      89.1     107.8  0.00000
32x32x32              small                  439.6     822.0     426.5    1080.1      97.4     100.4  0.00000
64x64x64              small                  794.6    1216.4     955.7    1823.5     100.0     101.9  0.00000
128x128x128           L2                    1144.6    1521.7    1330.5    1915.8      93.8      95.1  0.00000
256x256x256           L2                    1463.2    1731.1    1606.4    1966.6      91.7      92.8  0.00000
512x512x512           L3                    1698.2    1860.1    1768.7    1980.2      90.5      91.2  0.00000
1024x1024x1024        L3                    1653.8    1910.4    1572.8    1892.0      89.1      89.7  0.00000
2048x2048x2048        mem-bound             1607.2    1792.6    1508.3    1776.2      89.2      89.5  0.00000
4096x4096x4096        mem-bound             1606.2    1699.4    1550.8    1683.6      88.8      89.0  0.00000
4095x4095x4095        mem-bound             1592.1    1683.3    1547.1    1663.5      88.2      88.6  0.00000
1024x1024x1020        N=85x12 aligned       1710.7    1919.1    1664.1    1948.9      89.1      89.6  0.00000
2048x512x64           tall-skinny           1225.1    1857.3    1245.3    1981.1      90.6      90.5  0.00000
512x2048x64           wide-flat             1163.5    1975.5    1209.2    2009.2      87.9      88.4  0.00000
65x65x65              all tails +1           427.4     609.3     572.6     921.6      83.8      86.6  0.00000
513x513x509           all tails mixed       1584.9    1739.7    1626.5    1843.8      89.0      89.6  0.00000
1025x1025x1021        all tails large       1590.3    1851.6    1562.7    1854.1      88.8      89.4  0.00002
128x128x1100          N > Nc_cache          1308.3    1496.7    1438.9    1707.6      90.4      93.0  0.00000
512x512x2048          N >> Nc_cache         1700.7    1845.5    1674.1    1816.6      89.6      90.3  0.00000
```

Same run, full library comparison (`Kai SME2` = 2VL, pack-inclusive):

```
Size (MxKxN)          Tag                  NEON GF   SME 4x1    4x1 ZP   SME 2x2  Kai SME2  Accel GF  OBlas GF  vs Accel  vs OBlas MaxDiff
-------------------------------------------------------------------------------------------------------------------------------------------
8x8x8                 tiny                     6.1       0.3       0.5       0.6      10.0      19.1       7.3     0.030     0.077  0.00000
16x16x16              tiny                     8.3       4.2       4.2       4.2      67.4      51.2      53.2     0.083     0.080  0.00000
32x32x32              small                   40.6      13.4      13.5      63.3     439.6     343.8     436.8     0.184     0.145  0.00000
64x64x64              small                   83.5     190.1     202.0     203.4     794.6    1011.1     940.7     0.201     0.216  0.00000
128x128x128           L2                     106.0     409.5     441.9     444.8    1144.6    1569.4    1365.9     0.283     0.326  0.00000
256x256x256           L2                     113.0     712.9     758.5     760.7    1463.2    1768.7    1574.3     0.430     0.483  0.00000
512x512x512           L3                     120.7    1048.6    1108.1    1113.2    1698.2    1757.1    1668.3     0.634     0.667  0.00000
1024x1024x1024        L3                     120.2    1299.4    1349.6    1270.6    1653.8    1632.3    1476.9     0.778     0.860  0.00000
2048x2048x2048        mem-bound              121.6    1198.6    1091.5    1092.7    1607.2    1539.7     616.4     0.710     1.773  0.00018
4096x4096x4096        mem-bound              122.7    1237.0    1080.0    1088.9    1606.2    1549.9     113.7     0.703     9.580  0.00039
4095x4095x4095        mem-bound              121.4    1176.2    1014.1    1014.4    1592.1    1541.9     112.1     0.658     9.048  0.00041
1024x1024x1020        N=85x12 aligned        122.6    1316.9    1325.5    1244.6    1710.7    1852.3     110.3     0.672    11.286  0.00000
2048x512x64           tall-skinny            102.3     658.7     881.5     878.2    1225.1    1275.3     106.5     0.689     8.249  0.00000
512x2048x64           wide-flat              103.0     758.0     993.8     996.3    1163.5    1075.5     103.5     0.926     9.623  0.00011
65x65x65              all tails +1            76.6      77.7      87.4      92.6     427.4     549.1     528.0     0.169     0.175  0.00000
513x513x509           all tails mixed        119.4     854.7     901.5     819.0    1584.9    1683.7     105.1     0.486     7.792  0.00000
1025x1025x1021        all tails large        119.8    1202.8     986.9     934.4    1590.3    1812.3     102.5     0.516     9.119  0.00007
128x128x1100          N > Nc_cache           112.1     462.6     463.7     459.3    1308.3    1597.8     104.5     0.287     4.395  0.00000
512x512x2048          N >> Nc_cache          121.3    1014.8    1106.6    1065.5    1700.7    1662.4     112.2     0.641     9.500  0.00000
```

- **KleidiAI's SME2 micro-kernels are faster than ours essentially everywhere.**
  At 1024³, 2VL does 1654 `+pk` / 1910 `mm` against our best SME kernel's ~1350.
  Peak observed is 8VS at ~2009 GFLOPS (`mm`, wide-flat), the fastest fp32 GEMM
  measured on this machine — above Accelerate's ~1850.
- **The large-size lead we appeared to have was an artifact of how we called
  them.** Before N-blocking, 2VL `mm` read 1030 @ 2048³ and 898 @ 4096³ against
  our 1273 / 1233. After, it reads 1793 and 1699. Our `Nc_cache` blocking is a
  real advantage over a *bare micro-kernel call*, but not over KleidiAI as it is
  meant to be used. The earlier reading of these rows was wrong.
- **Our NEON kernel does beat theirs** — 8×12 at 120–123 GFLOPS against 6×8's
  88–95. The wider tile plus 4× K-unroll and in-register transpose is worth
  roughly 30%, and this one is not a harness artifact: the NEON numbers are flat
  across every size, blocked or not.
- **Small sizes stay Accelerate's** (≤ 64³), as with our own kernels.
- **MaxDiff vs Accelerate is 0.00000 on 18 of 19 shapes** and 0.00002 on the
  remaining one, including every tail case (65³, 513×513×509, 1025×1025×1021).
  The KleidiAI kernels handle arbitrary M/N/K with no padding on our side.

### 0.6 MaxMulSK vs KleidiAI — dedicated head-to-head (2026-09-04)

**Measured:** 2026-09-04 · **SUPERSEDED on the MaxMulSK side by §0.9-A and
§0.14.** The methodology and the KleidiAI figures stand, but every MaxMulSK
number here predates the 1×4-Acc kernels. In particular "KleidiAI wins in both
modes, at every size" below is no longer true: at 1024³ we are now ahead
end-to-end and level prepacked (§0.14).

`bench/benchmark_maxmul_vs_kleidiai.cpp` is a separate target from
`bench_compare`, built for one question: how do our SME kernels compare against
Arm's fp32 SME2 micro-kernels when conditions are actually held equal — both
end-to-end and with packing excluded.

```
cmake --build cmake-build-release --target benchmark_maxmul_vs_kleidiai
./cmake-build-release/benchmark_maxmul_vs_kleidiai results.csv
```

**Configuration.** Apple M4 (4 P-cores, 16 MB P-core L2, SVL 512 bits), macOS
Darwin 25.5.0, Homebrew Clang 22.1.8. Benchmark TUs at `-O3 -Wall -Wextra
-mcpu=apple-m4 -fopenmp`; the MaxMulSK kernels and the adapter add
`-march=armv8.7-a+sme+sme2`; KleidiAI `v1.30.0` built from source with its own
flags (`-march=armv8.2-a+sve+sve2`, SME via `.inst`). Single-thread throughout
(`omp_set_num_threads(1)`, `VECLIB_MAXIMUM_THREADS=1`). Timer is
`std::chrono::steady_clock`. Inputs U(-1,1), `mt19937` seed 42, identical for
every implementation. Three warmups per measurement, none reported; sample count
adaptive (>= 7) targeting ~0.35 s of timed work; headline metric is GFLOP/s from
the **median**, FLOPs = `2*M*N*K` computed in `double`.

#### What is compared, and what could not be made identical

| | MaxMulSK | KleidiAI |
|---|---|---|
| Entry point | `SMEKernels*::run_multiplication`, 6 variants | `..._f32p2vlx1_f32p2vlx1biasf32_sme2_mopa` (`2VL`), `..._f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa` (`8VS`) |
| Packing | BLIS-style, interleaved with compute inside the blocking loops; small reused `M_tile x K_tile` / `K_tile x N_tile` buffers | separate `kai_lhs_pack_*` / `kai_rhs_pack_*` passes over the full operands, before any compute |
| Pre-packed entry point | none shipped | yes — that is the normal usage |
| GEMM semantics | `C += A*B` | `C = A*B` |
| Cache blocking above the micro-kernel | yes, built in | none — the caller supplies it |

Five methodology choices follow from that table. Each of them moves numbers, so
they are stated rather than buried:

1. **The `C` memset is timed for MaxMulSK.** Its micro-kernels accumulate, so
   `C = A*B` requires zeroing first, and that is a real caller cost. Measured
   separately, it is small: 0.56 % of end-to-end at 4096³ and under 0.15 % on
   every large-K shape; worst case is 5.2 % at 256³.

2. **Both sides allocate their packed buffers inside the end-to-end region.**
   `run_multiplication` calls `aligned_alloc` on every invocation and there is no
   way to hoist that without editing the kernel, so
   `KleidiAI::Gemm::run_end_to_end` was written to allocate the same way — a bare
   `aligned_alloc`, deliberately *not* a `std::vector`, which would pre-fault
   every page and quietly move that cost out of the timed region.

3. **KleidiAI's outer N-block width is autotuned per shape.** KleidiAI ships no
   blocking above the micro-kernel, so the block width is the *caller's*
   parameter; pinning it to one heuristic would report my choice as KleidiAI's
   performance — a single fixed heuristic cost it ~10 % at K=32768. MaxMulSK is
   already reported as the best of its six variants, so the symmetric treatment
   is to report KleidiAI at its best block width:

```
  KleidiAI outer N-block width chosen by autotuning (columns).
  KleidiAI has no cache blocking of its own; this is the caller's
  parameter and is tuned per shape so KleidiAI is measured at its best,
  mirroring MaxMulSK being reported as the best of its six variants.

M x K x N                 KAI 2VL      KAI 8VS
----------------------------------------------
256x256x256                   256          256
512x512x512                   512          512
1024x1024x1024               1024          512
2048x2048x2048                512         1024
4096x4096x4096                512          512
64x8192x512                   128          128
64x16384x512                   32           64
64x32768x512                   32           32
128x8192x512                   64           64
128x16384x512                  32           32
128x32768x512                 512          512
```

4. **Kernel-only is the closest fair comparison, not an exact one.** MaxMulSK has
   no pre-packed entry point. `bench/maxmulsk_sme_adapter.cpp` replays the
   shipped driver's loop nest twice — once calling only `pack_A_streaming` /
   `pack_B_streaming` into one large buffer holding every block the driver
   visits, once calling only the micro-kernel and reading those blocks back. Loop
   order, tile sizes, micro-kernel input layout, edge-tile handling and C
   accumulation order are unchanged, and no kernel source was modified.
   **The limitation:** the shipped driver reads packed panels out of a small hot
   buffer, while this reads them out of a large cold one, so kernel-only does
   **not** decompose end-to-end time. It measures the micro-kernel plus the
   traffic of streaming full-size pre-packed operands — which is exactly the
   footing KleidiAI's pre-packed path is on, so the two remain comparable to each
   other even though neither is a pure ALU number.

5. **`SME 1x4ZAIO` has no kernel-only row.** It does not block over K and splits
   its micro-kernel into zero/compute/store around ZA-resident accumulation, so
   it does not share the skeleton the adapter replays. It is present end-to-end.

Not controllable: macOS exposes no thread-to-core pinning API. `USER_INTERACTIVE`
QoS is requested, which biases the scheduler toward the P-core cluster but cannot
pin residency.

#### Square GEMM

End-to-end (packing + compute):

```
M x K x N               MaxMulSK ms  MaxMulSK GF    KleidiAI ms  KleidiAI GF      ratio     diff %  correctness
------------------------------------------------------------------------------------------------------------------------
256x256x256                   0.053        637.6          0.026       1280.3      0.498     -50.2%  PASS
512x512x512                   0.250       1073.4          0.157       1712.5      0.627     -37.3%  PASS
1024x1024x1024                1.595       1346.4          1.299       1653.2      0.814     -18.6%  PASS
2048x2048x2048               13.265       1295.1         10.717       1603.0      0.808     -19.2%  PASS
4096x4096x4096              102.644       1339.0         87.046       1578.9      0.848     -15.2%  PASS
```

Kernel-only (packing excluded):

```
M x K x N               MaxMulSK ms  MaxMulSK GF    KleidiAI ms  KleidiAI GF      ratio     diff %  correctness
------------------------------------------------------------------------------------------------------------------------
256x256x256                   0.045        751.9          0.017       1968.9      0.382     -61.8%  PASS
512x512x512                   0.232       1156.0          0.135       1981.7      0.583     -41.7%  PASS
1024x1024x1024                1.468       1462.7          1.111       1933.7      0.756     -24.4%  PASS
2048x2048x2048               11.417       1504.8          9.510       1806.5      0.833     -16.7%  PASS
4096x4096x4096               90.593       1517.1         77.881       1764.7      0.860     -14.0%  PASS
```

#### LLM-like rectangular / large-K GEMM

End-to-end:

```
M x K x N               MaxMulSK ms  MaxMulSK GF    KleidiAI ms  KleidiAI GF      ratio     diff %  correctness
------------------------------------------------------------------------------------------------------------------------
64x8192x512                   0.671        800.5          1.017        528.1      1.516     +51.6%  PASS
64x16384x512                  1.477        727.0          2.114        508.0      1.431     +43.1%  PASS
64x32768x512                  2.951        727.8          4.852        442.6      1.644     +64.4%  PASS
128x8192x512                  1.160        925.6          1.435        748.0      1.237     +23.7%  PASS
128x16384x512                 2.377        903.6          3.147        682.3      1.324     +32.4%  PASS
128x32768x512                 4.740        906.2          7.754        553.9      1.636     +63.6%  PASS
```

Kernel-only:

```
M x K x N               MaxMulSK ms  MaxMulSK GF    KleidiAI ms  KleidiAI GF      ratio     diff %  correctness
------------------------------------------------------------------------------------------------------------------------
64x8192x512                   0.458       1171.1          0.369       1453.0      0.806     -19.4%  PASS
64x16384x512                  1.016       1056.4          0.819       1311.6      0.805     -19.5%  PASS
64x32768x512                  2.036       1054.8          2.267        947.2      1.113     +11.3%  PASS
128x8192x512                  0.819       1310.8          0.668       1607.0      0.816     -18.4%  PASS
128x16384x512                 1.691       1270.2          1.572       1366.3      0.930      -7.0%  PASS
128x32768x512                 3.387       1267.9          4.642        925.3      1.370     +37.0%  PASS
```

#### Correctness

176 measured rows, **0 failures**. Reference is Accelerate `cblas_sgemm`, plus
FP64 dot products on 32 sampled output entries per shape. Worst relative L2 error
anywhere is 3.36e-6 (at K=32768, where |C| ~ 104) against a 1e-4 tolerance; worst
max-abs deviation from FP64 accumulation is 1.6e-4. On the square shapes every
implementation is bit-identical to Accelerate.

#### Where the gap actually is

```
  Ratio trend (MaxMulSK best / KleidiAI best, >1 means MaxMulSK ahead):

M x K x N                end_to_end    kernel_only
--------------------------------------------------
256x256x256                   0.498          0.382
512x512x512                   0.627          0.583
1024x1024x1024                0.814          0.756
2048x2048x2048                0.808          0.833
4096x4096x4096                0.848          0.860
64x8192x512                   1.516          0.806
64x16384x512                  1.431          0.805
64x32768x512                  1.644          1.113
128x8192x512                  1.237          0.816
128x16384x512                 1.324          0.930
128x32768x512                 1.636          1.370
```

- **Square: KleidiAI wins in both modes, at every size.** The margin is widest
  where fixed overhead dominates (0.38x at 256³ kernel-only) and narrows
  monotonically with size to 0.86x at 4096³. Because kernel-only is *also* a loss
  and by a similar factor, the square-shape deficit is in the compute kernel
  itself — not in packing, and not in blocking.
- **Large-K end-to-end: MaxMulSK wins everywhere, 1.24x-1.64x.** This reverses
  the square result, and the packing breakdown says why: on these shapes KleidiAI
  spends **38-64 % of end-to-end time packing**, against MaxMulSK's 9-33 %. Both
  materialise the same packed bytes (64 MB of packed B at K=32768), but KleidiAI
  does it as one cold pass and then re-reads it, while MaxMulSK packs
  `K_tile`-sized blocks and consumes each while it is still hot. At M=64 every B
  element is reused only 64 times, so that extra pass is a large fraction of all
  the work there is.
- **Large-K kernel-only: the crossover is at K.** With packing excluded KleidiAI
  still leads at K=8192 (0.81x) and K=16384 (0.81x / 0.93x), but MaxMulSK takes
  K=32768 (1.11x at M=64, 1.37x at M=128). Packed B is 16 / 32 / 64 MB at those
  three K values against a 16 MB L2: the crossover lands where the operand stops
  fitting, which is where explicit cache blocking begins to pay.
- **Micro-kernel geometry on the rectangular shapes.** `SME 1x4sym` wins
  MaxMulSK's kernel-only column on 5 of 6 large-K shapes — its `N_tile`=64 and
  `4*SVL` N-step suit N=512 with tiny M. End-to-end the winner shifts to
  `SME 4x1ZP`, whose smaller `K_tile`=1024 and ZA-based `pack_A` give it the
  cheapest packing on these shapes (8.6 % at 128x32768x512, against 30.7 % for
  `SME 4x1`). On the square shapes the winners are the familiar ones: `4x1ZP`
  small, `1x4sym` large.
- **Bottleneck attribution.** Square: compute scheduling inside the micro-kernel
  (kernel-only and end-to-end lose by the same factor). Large-K end-to-end:
  packing and the memory traffic it generates, decisively — the single largest
  line item for KleidiAI on every one of those shapes. Large-K kernel-only at
  K=32768: memory traffic against a 16 MB L2, i.e. cache blocking. Fixed
  invocation overhead matters only at 256³, where it costs MaxMulSK more than
  half its throughput. Edge handling is not implicated anywhere: every shape here
  is a clean multiple of the relevant steps, and the tail shapes in §0.2 already
  showed the edge paths are correct and not pathological.

### 0.7 Separating micro-kernel from macro-flow, and a llama.cpp baseline (2026-09-04)

**Measured:** 2026-09-04 · **CURRENT**

§0.6 established *that* MaxMulSK wins the large-K shapes end-to-end and loses the
square ones. It could not say *why*, because its "kernel-only" mode is really a
pre-packed **full GEMM** — it still traverses the whole output and streams
full-size operands. Three experiments separate the layers. Same target, same
harness, 280 measured rows, **0 correctness failures** (worst relative L2 3.4e-6
against a 1e-4 tolerance).

#### Experiment 1 — Hot micro-kernel ceiling

Packed panels for a **single output tile**, resident in cache; the timed region
contains only repeated micro-kernel invocations. No packing, no output traversal,
no allocation, no large-matrix zeroing. Every MaxMulSK variant produces a
1024-element tile (4SVL×1SVL, 2SVL×2SVL or 1SVL×4SVL at SVL=16) and reads
`M_step*Kc + N_step*Kc` packed floats, so all variants move identical bytes.

```
    Kc    MaxMulSK GF    KleidiAI GF    ratio  MaxMul geometry  KAI geometry    
----------------------------------------------------------------------------------
   128          536.7         1350.4    0.397  SME 4x1ZP 64x16  KAI 2VL 32x32 ti
   256          849.3         1613.4    0.526  SME 4x1 64x16 ti KAI 2VL 32x32 ti
   512         1193.7         1795.3    0.665  SME 2x2 32x32 ti KAI 2VL 32x32 ti
  1024         1497.4         1898.5    0.789  SME 2x2 32x32 ti KAI 2VL 32x32 ti
  2048         1715.6         1944.6    0.882  SME 1x4sym 16x64 KAI 2VL 32x32 ti
```

One asymmetry could not be removed: MaxMulSK's micro-kernels are `__arm_streaming`
leaves, so a batch runs with streaming mode entered once (as its driver does),
while KleidiAI's `kai_run_matmul_*` does `smstart`/`smstop` on every call. Rather
than assume that away, MaxMulSK was measured both ways — and the `[iso]`
(per-call streaming) rows land within noise of the batched rows, so the confound
is not material. Full per-variant detail:

```
    Kc implementation     geometry               reps/sample    ns/invoke        GFLOP/s    correct
--------------------------------------------------------------------------------------------------
   128 SME 4x1            64x16 tile                   1489        492.3          532.5       PASS
   128 SME 4x1 [iso]      64x16, smstart/call          1489        489.5          535.5       PASS
   128 SME 2x2            32x32 tile                   3693        488.7          536.5       PASS
   128 SME 2x2 [iso]      32x32, smstart/call          3693        488.9          536.2       PASS
   128 SME 1x4            16x64 tile                   3850        488.8          536.3       PASS
   128 SME 1x4 [iso]      16x64, smstart/call          3850        488.5          536.6       PASS
   128 SME 1x4sym         16x64 tile                   3332        488.8          536.3       PASS
   128 SME 1x4sym [iso]   16x64, smstart/call          3332        489.0          536.0       PASS
   128 SME 4x1ZP          64x16 tile                   3606        488.4          536.7       PASS
   128 SME 4x1ZP [iso]    64x16, smstart/call          3606        488.8          536.4       PASS
   128 KAI 2VL            32x32 tile                   8703        194.1         1350.4       PASS
   128 KAI 8VS            16x16 tile                  21334         72.2          908.3       PASS
   256 SME 4x1            64x16 tile                   3132        617.3          849.3       PASS
   256 SME 4x1 [iso]      64x16, smstart/call          3132        617.6          849.0       PASS
   256 SME 2x2            32x32 tile                   2960        617.4          849.2       PASS
   256 SME 2x2 [iso]      32x32, smstart/call          2960        617.7          848.8       PASS
   256 SME 1x4            16x64 tile                   2986        617.4          849.1       PASS
   256 SME 1x4 [iso]      16x64, smstart/call          2986        617.7          848.8       PASS
   256 SME 1x4sym         16x64 tile                   3113        617.3          849.3       PASS
   256 SME 1x4sym [iso]   16x64, smstart/call          3113        617.8          848.7       PASS
   256 SME 4x1ZP          64x16 tile                   2998        617.4          849.2       PASS
   256 SME 4x1ZP [iso]    64x16, smstart/call          2998        617.7          848.8       PASS
   256 KAI 2VL            32x32 tile                   5874        325.0         1613.4       PASS
   256 KAI 8VS            16x16 tile                  14629        104.8         1250.8       PASS
   512 SME 4x1            64x16 tile                   2116        878.5         1193.6       PASS
   512 SME 4x1 [iso]      64x16, smstart/call          2116        878.8         1193.2       PASS
   512 SME 2x2            32x32 tile                   2192        878.4         1193.7       PASS
   512 SME 2x2 [iso]      32x32, smstart/call          2192        878.6         1193.4       PASS
   512 SME 1x4            16x64 tile                   2187        891.9         1175.7       PASS
   512 SME 1x4 [iso]      16x64, smstart/call          2187        894.0         1172.9       PASS
   512 SME 1x4sym         16x64 tile                   2170        878.6         1193.5       PASS
   512 SME 1x4sym [iso]   16x64, smstart/call          2170        878.7         1193.3       PASS
   512 SME 4x1ZP          64x16 tile                   2178        878.4         1193.7       PASS
   512 SME 4x1ZP [iso]    64x16, smstart/call          2178        878.8         1193.2       PASS
   512 KAI 2VL            32x32 tile                   3122        584.1         1795.3       PASS
   512 KAI 8VS            16x16 tile                   9366        170.2         1540.0       PASS
  1024 SME 4x1            64x16 tile                   1399       1401.2         1496.7       PASS
  1024 SME 4x1 [iso]      64x16, smstart/call          1399       1400.8         1497.1       PASS
  1024 SME 2x2            32x32 tile                   1413       1400.5         1497.4       PASS
  1024 SME 2x2 [iso]      32x32, smstart/call          1413       1400.9         1497.0       PASS
  1024 SME 1x4            16x64 tile                   1334       1450.2         1446.1       PASS
  1024 SME 1x4 [iso]      16x64, smstart/call          1334       1452.6         1443.7       PASS
  1024 SME 1x4sym         16x64 tile                   1368       1400.7         1497.2       PASS
  1024 SME 1x4sym [iso]   16x64, smstart/call          1368       1400.9         1497.0       PASS
  1024 SME 4x1ZP          64x16 tile                   1365       1400.7         1497.2       PASS
  1024 SME 4x1ZP [iso]    64x16, smstart/call          1365       1401.5         1496.4       PASS
  1024 KAI 2VL            32x32 tile                   1764       1104.6         1898.5       PASS
  1024 KAI 8VS            16x16 tile                   5679        299.9         1748.1       PASS
  2048 SME 4x1            64x16 tile                    799       2445.0         1715.4       PASS
  2048 SME 4x1 [iso]      64x16, smstart/call           799       2445.4         1715.2       PASS
  2048 SME 2x2            32x32 tile                    799       2445.0         1715.5       PASS
  2048 SME 2x2 [iso]      32x32, smstart/call           799       2445.9         1714.9       PASS
  2048 SME 1x4            16x64 tile                    778       2568.2         1633.1       PASS
  2048 SME 1x4 [iso]      16x64, smstart/call           778       2570.3         1631.9       PASS
  2048 SME 1x4sym         16x64 tile                    818       2444.7         1715.6       PASS
  2048 SME 1x4sym [iso]   16x64, smstart/call           818       2445.0         1715.4       PASS
  2048 SME 4x1ZP          64x16 tile                    799       2444.8         1715.6       PASS
  2048 SME 4x1ZP [iso]    64x16, smstart/call           799       2446.2         1714.6       PASS
  2048 KAI 2VL            32x32 tile                    919       2156.9         1944.6       PASS
  2048 KAI 8VS            16x16 tile                   3142        588.4         1782.0       PASS
```

Two further notes. KleidiAI's `8VS` entry point reports a 16×16 step, a quarter
the tile area of `2VL`'s 32×32, which is why it trails here — the comparison is
per-invocation, and a smaller tile has less arithmetic per unit of packed data.
And all five MaxMulSK variants land within ~1% of each other at any given Kc,
which is expected: they issue the same number of FMOPAs over the same bytes.

**Reading:** KleidiAI's micro-kernel is faster at every Kc. The gap is worst at
short K (0.40× at Kc=128) and closes steadily to 0.88× at Kc=2048 — the profile
of a fixed per-invocation cost being amortised. MaxMulSK's micro-kernel pays
`svzero_za` plus a 1024-element ZA→C read-modify-write store per call regardless
of Kc; at Kc=128 there are only 128 FMOPA groups to hide that behind.

#### Experiment 2 — KleidiAI's kernel inside MaxMulSK-style panel blocking

The §0.6 large-K result compared MaxMulSK's integrated pack-and-consume flow
against KleidiAI's *default* flow, which materialises the entire packed A and B
before computing. That confounds micro-kernel quality with whole-GEMM dataflow.
`KleidiAI::Gemm::run_panel_blocked` removes the confound: **KleidiAI's own
packing routines and its own micro-kernel**, unmodified, driven by a caller that
mirrors the MaxMulSK driver's organisation —

```
for N-block: for K-panel: pack B panel
                          for M-block: pack A panel, consume immediately
```

Mc/Kc/Nc are autotuned per shape, for the same reason the N-block width is: the
blocking is the caller's choice, and a bad one would be reported as KleidiAI's
performance. One thing the caller cannot mirror: these fp32 SME2 kernels
overwrite C and expose no `beta=1`, so a blocked K forces the caller to compute
into a scratch tile and add. That add is an API-imposed caller cost, not a kernel
deficiency; it is 1.6–2.5% of the panel-blocked runs where K is blocked, and zero
where Kc >= K.

```
M x K x N                MaxMulSK     KAI orig    KAI panel     MM/orig    MM/panel
------------------------------------------------------------------------------------
256x256x256                 681.3       1276.3       1276.3       0.534       0.534
512x512x512                1083.1       1711.1       1711.1       0.633       0.633
1024x1024x1024             1337.9       1647.6       1789.9       0.812       0.747
2048x2048x2048             1306.1       1620.9       1595.1       0.806       0.819
4096x4096x4096             1327.7       1574.4       1499.1       0.843       0.886
64x8192x512                 740.7        507.5        835.3       1.460       0.887
64x16384x512                719.5        504.5        849.2       1.426       0.847
64x32768x512                727.3        439.1        855.5       1.656       0.850
128x8192x512                922.6        745.8       1070.7       1.237       0.862
128x16384x512               899.7        701.0       1060.2       1.283       0.849
128x32768x512               907.0        555.1       1068.5       1.634       0.849
```

Stage breakdown of the panel-blocked runs:

```
M x K x N            kernel       config                   packA %   packB % compute %   accum %
------------------------------------------------------------------------------------------------
256x256x256          2VL          Mc=256 Kc=2048 Nc=256      10.4%      7.2%     82.0%      0.0%
256x256x256          8VS          Mc=256 Kc=2048 Nc=512      11.5%      8.3%     79.7%      0.0%
512x512x512          2VL          Mc=512 Kc=512 Nc=512        6.5%      3.7%     89.6%      0.0%
512x512x512          8VS          Mc=256 Kc=512 Nc=512        6.9%      5.6%     87.4%      0.0%
1024x1024x1024       2VL          Mc=64 Kc=2048 Nc=1024       4.7%      2.8%     92.5%      0.0%
1024x1024x1024       8VS          Mc=64 Kc=1024 Nc=1024       4.0%      3.2%     92.8%      0.0%
2048x2048x2048       2VL          Mc=256 Kc=2048 Nc=512      14.9%      3.2%     81.9%      0.0%
2048x2048x2048       8VS          Mc=64 Kc=2048 Nc=512       21.1%      3.5%     75.4%      0.0%
4096x4096x4096       2VL          Mc=256 Kc=4096 Nc=512      16.2%      1.7%     82.1%      0.0%
4096x4096x4096       8VS          Mc=64 Kc=4096 Nc=512       24.9%      2.0%     73.1%      0.0%
64x8192x512          2VL          Mc=64 Kc=2048 Nc=512       14.9%     37.9%     45.2%      1.9%
64x8192x512          8VS          Mc=64 Kc=2048 Nc=512       11.6%     48.4%     38.2%      1.6%
64x16384x512         2VL          Mc=64 Kc=2048 Nc=512       14.9%     39.1%     44.1%      1.8%
64x16384x512         8VS          Mc=64 Kc=2048 Nc=512       10.4%     48.4%     39.6%      1.6%
64x32768x512         2VL          Mc=64 Kc=2048 Nc=512       14.4%     39.6%     44.2%      1.8%
64x32768x512         8VS          Mc=256 Kc=2048 Nc=512      10.7%     48.6%     39.0%      1.7%
128x8192x512         2VL          Mc=128 Kc=2048 Nc=512      17.4%     24.0%     56.3%      2.2%
128x8192x512         8VS          Mc=256 Kc=2048 Nc=512      12.5%     31.2%     53.9%      2.3%
128x16384x512        2VL          Mc=256 Kc=2048 Nc=512      17.7%     24.1%     55.7%      2.4%
128x16384x512        8VS          Mc=128 Kc=2048 Nc=512      13.4%     31.9%     52.4%      2.2%
128x32768x512        2VL          Mc=256 Kc=2048 Nc=512      18.2%     24.2%     55.1%      2.5%
128x32768x512        8VS          Mc=256 Kc=2048 Nc=512      13.9%     31.8%     51.9%      2.3%
```

**Reading — this is the headline result.** MaxMulSK's large-K advantage is
entirely an artifact of KleidiAI's *default flow*, not of its kernel. Given the
same macro-organisation, KleidiAI goes from 439 to 856 GFLOP/s at 64×32768×512 —
a 1.95× gain from dataflow alone — and MaxMulSK/KleidiAI flips from **1.66× in
our favour to 0.85× against us**. Across all six large-K shapes the ratio lands
in a tight 0.85–0.89 band, which is the same band as the Kc=2048 micro-kernel
ratio (0.88) from Experiment 1. The residual end-to-end gap, after controlling
for macro-flow, is the micro-kernel gap.

Panel blocking is not a free win everywhere: it is neutral at 256³/512³ (the
whole problem is one panel), helps at 1024³ (1648 → 1790), and *hurts* at 4096³
(1574 → 1499), where the default full-prepack flow already streams efficiently.

#### Experiment 3 — llama.cpp / ggml CPU FP32 baseline

llama.cpp at `1548a240`, built CPU-only: `GGML_METAL=OFF`, `GGML_ACCELERATE=OFF`,
`GGML_BLAS=OFF`, `GGML_CPU_KLEIDIAI=OFF`, `GGML_OPENMP=OFF`, `GGML_NATIVE=ON`.
Tensors, graph and buffers are built outside the timed region; the timed region
is `ggml_backend_graph_compute` on a prebuilt graph. Single-thread was **verified,
not assumed**: CPU-time/wall-time over a large shape measured **1.00**.

**Which code actually runs — this took some digging and the obvious answer is
wrong.** `ggml_compute_forward_mul_mat` tries `llamafile_sgemm` (tinyBLAS) first,
and `GGML_LLAMAFILE` defaults to ON in llama.cpp, and `sgemm.cpp` *is* compiled
and linked. But `ggml-cpu.c` opens with

```c
#if defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_MATMUL_INT8)
#undef GGML_USE_LLAMAFILE
#endif
```

and ggml builds its CPU variant here with `-mcpu=native+dotprod+i8mm+nosve+sme`,
where `+i8mm` defines `__ARM_FEATURE_MATMUL_INT8`. **tinyBLAS is therefore
disabled by upstream on this hardware.** Verified two independent ways: an
instrumented build shows the `#if` block is compiled out of that TU, and a
`GGML_LLAMAFILE=OFF` build produces identical timings (42.6 vs 42.8 GFLOP/s at
1024³). The path that executes is the generic chunked fallback —
`ggml_compute_forward_mul_mat_one_chunk` → `ggml_vec_dot_f32`, one NEON dot
product per output element, no register blocking and no packing. That is what the
29–44 GFLOP/s below is measuring, and it is upstream behaviour on Apple Silicon,
not a misconfiguration here.

Layout: ggml computes `dst[n][m] = dot(src0 row m, src1 row n)`, so src1 holds B
transposed and the result is read back transposed. Both are static layout
conversions done outside the timed region — this is exactly how llama.cpp stores
weights, so it is normal usage rather than an artificial adapter. The one-off B
transpose is measured and reported separately in the benchmark output.

```
M x K x N            implementation  median ms    mean ms     min ms     stddev     reps    GFLOP/s
----------------------------------------------------------------------------------------------------------
256x256x256          SME 4x1             0.053      0.053      0.053      0.002      151      637.6
256x256x256          SME 2x2             0.049      0.050      0.049      0.001      151      681.3
256x256x256          SME 1x4             0.052      0.052      0.048      0.001      151      644.8
256x256x256          SME 1x4sym          0.052      0.052      0.052      0.001      151      644.8
256x256x256          SME 4x1ZP           0.049      0.050      0.047      0.001      151      680.7
256x256x256          SME 1x4ZAIO         0.060      0.061      0.060      0.001      151      555.4
256x256x256          KAI 2VL             0.029      0.029      0.026      0.000      151     1162.1
256x256x256          KAI 8VS             0.026      0.026      0.022      0.000      151     1276.3
256x256x256          Accelerate          0.019      0.019      0.019      0.000      151     1781.7
256x256x256          KAIpanel 2VL        0.028      0.028      0.024      0.001      151     1187.8
256x256x256          KAIpanel 8VS        0.026      0.026      0.026      0.000      151     1276.3
256x256x256          ggml CPU            0.801      0.803      0.795      0.007      101       41.9
256x256x256          ggml CPU+setup      0.805      0.808      0.800      0.008      101       41.7
512x512x512          SME 4x1             0.261      0.262      0.261      0.003      151     1027.5
512x512x512          SME 2x2             0.248      0.250      0.248      0.003      151     1081.7
512x512x512          SME 1x4             0.263      0.264      0.263      0.002      151     1019.1
512x512x512          SME 1x4sym          0.260      0.261      0.259      0.003      151     1033.9
512x512x512          SME 4x1ZP           0.248      0.249      0.248      0.003      151     1083.1
512x512x512          SME 1x4ZAIO         0.304      0.305      0.303      0.003      151      884.0
512x512x512          KAI 2VL             0.166      0.166      0.165      0.002      151     1617.9
512x512x512          KAI 8VS             0.157      0.158      0.156      0.003      151     1711.1
512x512x512          Accelerate          0.147      0.148      0.147      0.003      151     1820.9
512x512x512          KAIpanel 2VL        0.163      0.164      0.163      0.002      151     1643.9
512x512x512          KAIpanel 8VS        0.157      0.158      0.156      0.003      151     1711.1
512x512x512          ggml CPU            6.092      6.100      6.062      0.029       58       44.1
512x512x512          ggml CPU+setup      6.095      6.102      6.069      0.026       58       44.0
1024x1024x1024       SME 4x1             1.702      1.729      1.651      0.065      151     1262.0
1024x1024x1024       SME 2x2             1.651      1.669      1.621      0.068      151     1300.5
1024x1024x1024       SME 1x4             1.722      1.748      1.689      0.061      151     1247.1
1024x1024x1024       SME 1x4sym          1.704      1.722      1.654      0.069      151     1260.0
1024x1024x1024       SME 4x1ZP           1.605      1.623      1.590      0.047      151     1337.9
1024x1024x1024       SME 1x4ZAIO         1.838      1.858      1.806      0.049      151     1168.5
1024x1024x1024       KAI 2VL             1.314      1.331      1.277      0.044      151     1634.1
1024x1024x1024       KAI 8VS             1.303      1.322      1.272      0.052      151     1647.6
1024x1024x1024       Accelerate          1.272      1.275      1.265      0.011      151     1688.5
1024x1024x1024       KAIpanel 2VL        1.212      1.223      1.199      0.031      151     1771.4
1024x1024x1024       KAIpanel 8VS        1.200      1.208      1.188      0.023      151     1789.9
1024x1024x1024       ggml CPU           50.565     50.572     50.366      0.145        7       42.5
1024x1024x1024       ggml CPU+setup     50.526     50.597     50.450      0.144        7       42.5
2048x2048x2048       SME 4x1            13.413     13.479     13.255      0.191       28     1280.8
2048x2048x2048       SME 2x2            15.985     16.070     15.649      0.295       22     1074.7
2048x2048x2048       SME 1x4            14.169     14.538     13.160      1.338       27     1212.5
2048x2048x2048       SME 1x4sym         13.678     14.192     12.984      1.306       27     1256.1
2048x2048x2048       SME 4x1ZP          15.527     15.493     15.058      0.307       23     1106.5
2048x2048x2048       SME 1x4ZAIO        13.153     13.290     12.704      0.479       27     1306.1
2048x2048x2048       KAI 2VL            10.599     10.665     10.501      0.144       33     1620.9
2048x2048x2048       KAI 8VS            10.979     11.037     10.885      0.123       32     1564.7
2048x2048x2048       Accelerate         10.234     10.258     10.086      0.119       35     1678.7
2048x2048x2048       KAIpanel 2VL       10.770     10.812     10.626      0.148       32     1595.1
2048x2048x2048       KAIpanel 8VS       11.967     11.954     11.601      0.213       30     1435.6
2048x2048x2048       ggml CPU          468.552    469.107    468.296      0.945        5       36.7
2048x2048x2048       ggml CPU+setup    469.243    469.123    468.096      0.777        5       36.6
4096x4096x4096       SME 4x1           109.332    109.473    108.915      0.506        7     1257.1
4096x4096x4096       SME 2x2           129.099    129.035    128.572      0.347        7     1064.6
4096x4096x4096       SME 1x4           105.963    107.022    105.609      2.124        7     1297.1
4096x4096x4096       SME 1x4sym        103.514    103.338    100.379      2.238        7     1327.7
4096x4096x4096       SME 4x1ZP         131.122    131.188    130.963      0.243        7     1048.2
4096x4096x4096       SME 1x4ZAIO       130.539    130.728    123.547      5.323        7     1052.9
4096x4096x4096       KAI 2VL            88.706     88.311     86.857      0.960        7     1549.4
4096x4096x4096       KAI 8VS            87.298     87.808     86.777      1.213        7     1574.4
4096x4096x4096       Accelerate         86.166     86.865     85.770      1.526        7     1595.1
4096x4096x4096       KAIpanel 2VL       91.680     91.964     89.716      1.653        7     1499.1
4096x4096x4096       KAIpanel 8VS       98.646     98.938     96.094      1.739        7     1393.3
4096x4096x4096       ggml CPU         3855.232   3854.598   3853.014      1.204        5       35.6
4096x4096x4096       ggml CPU+setup   3853.669   3853.793   3853.040      0.638        5       35.7

-- square, kernel_only --

M x K x N            implementation  median ms    mean ms     min ms     stddev     reps    GFLOP/s
----------------------------------------------------------------------------------------------------------
256x256x256          SME 4x1             0.045      0.045      0.044      0.001      151      747.0
256x256x256          SME 2x2             0.045      0.045      0.045      0.001      151      742.9
256x256x256          SME 1x4             0.045      0.045      0.044      0.001      151      746.4
256x256x256          SME 1x4sym          0.045      0.045      0.044      0.001      151      740.2
256x256x256          SME 4x1ZP           0.045      0.045      0.044      0.001      151      745.7
256x256x256          KAI 2VL             0.020      0.020      0.020      0.000      151     1681.2
256x256x256          KAI 8VS             0.017      0.017      0.017      0.001      151     1964.2
512x512x512          SME 4x1             0.232      0.233      0.232      0.002      151     1156.0
512x512x512          SME 2x2             0.232      0.233      0.232      0.001      151     1155.0
512x512x512          SME 1x4             0.236      0.238      0.235      0.004      151     1137.0
512x512x512          SME 1x4sym          0.232      0.233      0.232      0.002      151     1156.0
512x512x512          SME 4x1ZP           0.232      0.234      0.231      0.003      151     1155.8
512x512x512          KAI 2VL             0.146      0.147      0.146      0.002      151     1832.3
512x512x512          KAI 8VS             0.135      0.136      0.135      0.002      151     1981.7
1024x1024x1024       SME 4x1             1.485      1.488      1.459      0.025      151     1446.1
1024x1024x1024       SME 2x2             1.470      1.483      1.463      0.030      151     1461.2
1024x1024x1024       SME 1x4             1.531      1.540      1.516      0.029      151     1403.1
1024x1024x1024       SME 1x4sym          1.475      1.482      1.466      0.019      151     1456.3
1024x1024x1024       SME 4x1ZP           1.484      1.493      1.462      0.030      151     1447.1
1024x1024x1024       KAI 2VL             1.131      1.133      1.124      0.008      151     1898.4
1024x1024x1024       KAI 8VS             1.111      1.114      1.104      0.011      151     1933.1
2048x2048x2048       SME 4x1            11.327     11.360     11.265      0.086       31     1516.7
2048x2048x2048       SME 2x2            13.263     13.301     13.020      0.171       27     1295.3
2048x2048x2048       SME 1x4            12.410     12.482     12.287      0.258       29     1384.3
2048x2048x2048       SME 1x4sym         11.867     11.886     11.550      0.305       29     1447.7
2048x2048x2048       SME 4x1ZP          13.057     13.099     12.952      0.116       27     1315.8
2048x2048x2048       KAI 2VL             9.284      9.311      9.252      0.061       38     1850.6
2048x2048x2048       KAI 8VS             9.440      9.550      9.318      0.225       38     1819.9
4096x4096x4096       SME 4x1            91.694     91.862     91.005      0.711        7     1498.9
4096x4096x4096       SME 2x2           108.276    108.127    106.961      0.533        7     1269.3
4096x4096x4096       SME 1x4            97.948     98.129     95.269      2.080        7     1403.2
4096x4096x4096       SME 1x4sym         93.532     93.401     91.232      1.641        7     1469.4
4096x4096x4096       SME 4x1ZP         106.747    106.760    106.621      0.098        7     1287.5
4096x4096x4096       KAI 2VL            80.897     80.765     80.283      0.364        7     1698.9
4096x4096x4096       KAI 8VS            81.810     82.213     80.846      1.263        7     1680.0

-- large_k, end_to_end --

M x K x N            implementation  median ms    mean ms     min ms     stddev     reps    GFLOP/s
----------------------------------------------------------------------------------------------------------
64x8192x512          SME 4x1             0.753      0.769      0.722      0.039      151      712.5
64x8192x512          SME 2x2             0.739      0.752      0.711      0.036      151      726.0
64x8192x512          SME 1x4             0.728      0.739      0.699      0.035      151      737.4
64x8192x512          SME 1x4sym          0.725      0.732      0.690      0.028      151      740.7
64x8192x512          SME 4x1ZP           0.741      0.749      0.722      0.021      151      724.4
64x8192x512          SME 1x4ZAIO         0.808      0.831      0.771      0.056      151      664.2
64x8192x512          KAI 2VL             1.058      1.087      1.014      0.066      151      507.5
64x8192x512          KAI 8VS             1.116      1.149      1.071      0.072      151      481.1
64x8192x512          Accelerate          0.582      0.592      0.560      0.028      151      922.3
64x8192x512          KAIpanel 2VL        0.643      0.649      0.623      0.020      151      835.3
64x8192x512          KAIpanel 8VS        0.699      0.704      0.659      0.030      151      768.2
64x8192x512          ggml CPU           15.361     15.367     15.304      0.049       23       35.0
64x8192x512          ggml CPU+setup     15.359     15.403     15.289      0.118       23       35.0
64x16384x512         SME 4x1             1.548      1.567      1.501      0.053      151      693.7
64x16384x512         SME 2x2             1.513      1.529      1.489      0.038      151      709.5
64x16384x512         SME 1x4             1.535      1.550      1.516      0.038      151      699.4
64x16384x512         SME 1x4sym          1.517      1.522      1.480      0.028      151      707.9
64x16384x512         SME 4x1ZP           1.492      1.499      1.474      0.023      151      719.5
64x16384x512         SME 1x4ZAIO         1.747      1.753      1.571      0.102      151      614.5
64x16384x512         KAI 2VL             2.128      2.149      2.100      0.063      151      504.5
64x16384x512         KAI 8VS             2.229      2.285      2.180      0.116      151      481.6
64x16384x512         Accelerate          1.284      1.293      1.249      0.034      151      836.0
64x16384x512         KAIpanel 2VL        1.264      1.272      1.253      0.021      151      849.2
64x16384x512         KAIpanel 8VS        1.386      1.398      1.355      0.037      151      774.8
64x16384x512         ggml CPU           34.399     34.345     34.016      0.242       11       31.2
64x16384x512         ggml CPU+setup     34.463     34.425     34.065      0.153       11       31.2
64x32768x512         SME 4x1             3.024      3.048      2.978      0.071      117      710.1
64x32768x512         SME 2x2             2.982      2.998      2.964      0.040      118      720.3
64x32768x512         SME 1x4             3.055      3.076      3.033      0.051      111      703.0
64x32768x512         SME 1x4sym          3.052      3.060      2.990      0.049      116      703.6
64x32768x512         SME 4x1ZP           2.953      2.976      2.939      0.094      119      727.3
64x32768x512         SME 1x4ZAIO         4.924      5.004      4.784      0.214       72      436.1
64x32768x512         KAI 2VL             4.890      4.944      4.842      0.145       72      439.1
64x32768x512         KAI 8VS             4.930      4.999      4.885      0.162       70      435.6
64x32768x512         Accelerate          2.497      2.514      2.479      0.049      141      860.0
64x32768x512         KAIpanel 2VL        2.510      2.526      2.496      0.038      140      855.5
64x32768x512         KAIpanel 8VS        2.768      2.789      2.731      0.053      127      775.8
64x32768x512         ggml CPU           73.227     73.257     73.029      0.194        5       29.3
64x32768x512         ggml CPU+setup     73.111     73.101     72.926      0.126        5       29.4
128x8192x512         SME 4x1             1.226      1.232      1.188      0.029      151      876.1
128x8192x512         SME 2x2             1.172      1.186      1.151      0.035      151      916.0
128x8192x512         SME 1x4             1.194      1.201      1.153      0.031      151      899.4
128x8192x512         SME 1x4sym          1.187      1.195      1.130      0.036      151      904.4
128x8192x512         SME 4x1ZP           1.164      1.173      1.139      0.028      151      922.6
128x8192x512         SME 1x4ZAIO         1.198      1.217      1.144      0.060      151      896.4
128x8192x512         KAI 2VL             1.440      1.456      1.408      0.048      151      745.8
128x8192x512         KAI 8VS             1.501      1.505      1.459      0.035      151      715.5
128x8192x512         Accelerate          0.894      0.901      0.882      0.020      151     1201.5
128x8192x512         KAIpanel 2VL        1.003      1.011      0.991      0.023      151     1070.7
128x8192x512         KAIpanel 8VS        1.022      1.036      1.002      0.035      151     1051.1
128x8192x512         ggml CPU           30.481     30.533     30.420      0.117       12       35.2
128x8192x512         ggml CPU+setup     30.466     30.490     30.386      0.078       12       35.2
128x16384x512        SME 4x1             2.473      2.494      2.446      0.053      139      868.3
128x16384x512        SME 2x2             2.407      2.421      2.386      0.040      145      892.3
128x16384x512        SME 1x4             2.481      2.494      2.439      0.040      142      865.4
128x16384x512        SME 1x4sym          2.462      2.474      2.417      0.044      142      872.3
128x16384x512        SME 4x1ZP           2.387      2.414      2.367      0.064      148      899.7
128x16384x512        SME 1x4ZAIO         3.034      3.084      2.963      0.153      115      707.7
128x16384x512        KAI 2VL             3.082      3.114      3.038      0.107      115      696.9
128x16384x512        KAI 8VS             3.063      3.091      3.017      0.092      115      701.0
128x16384x512        Accelerate          1.894      1.911      1.874      0.046      151     1134.1
128x16384x512        KAIpanel 2VL        2.026      2.041      2.001      0.039      151     1060.2
128x16384x512        KAIpanel 8VS        2.112      2.127      2.063      0.052      151     1017.0
128x16384x512        ggml CPU           68.754     68.645     67.907      0.379        6       31.2
128x16384x512        ggml CPU+setup     68.301     68.367     68.115      0.240        6       31.4
128x32768x512        SME 4x1             4.911      4.942      4.872      0.072       72      874.6
128x32768x512        SME 2x2             4.801      4.819      4.759      0.053       73      894.6
128x32768x512        SME 1x4             4.932      4.957      4.891      0.064       71      870.9
128x32768x512        SME 1x4sym          4.926      4.950      4.803      0.207       73      871.8
128x32768x512        SME 4x1ZP           4.735      4.760      4.707      0.055       72      907.0
128x32768x512        SME 1x4ZAIO         7.722      7.788      7.577      0.187       47      556.2
128x32768x512        KAI 2VL             7.737      7.847      7.670      0.210       43      555.1
128x32768x512        KAI 8VS             7.883      8.002      7.807      0.251       45      544.8
128x32768x512        Accelerate          3.765      3.786      3.732      0.069       94     1140.8
128x32768x512        KAIpanel 2VL        4.020      4.092      3.985      0.221       87     1068.5
128x32768x512        KAIpanel 8VS        4.199      4.230      4.169      0.079       84     1022.8
128x32768x512        ggml CPU          146.981    147.150    146.775      0.553        5       29.2
128x32768x512        ggml CPU+setup    146.839    146.905    146.764      0.167        5       29.2

-- large_k, kernel_only --

M x K x N            implementation  median ms    mean ms     min ms     stddev     reps    GFLOP/s
----------------------------------------------------------------------------------------------------------
64x8192x512          SME 4x1             0.516      0.531      0.494      0.036      151     1040.1
64x8192x512          SME 2x2             0.564      0.572      0.537      0.028      151      951.8
64x8192x512          SME 1x4             0.523      0.532      0.499      0.027      151     1025.5
64x8192x512          SME 1x4sym          0.476      0.482      0.455      0.020      151     1127.0
64x8192x512          SME 4x1ZP           0.556      0.567      0.522      0.035      151      966.3
64x8192x512          KAI 2VL             0.401      0.409      0.386      0.021      151     1337.7
64x8192x512          KAI 8VS             0.414      0.425      0.391      0.030      151     1295.9
64x16384x512         SME 4x1             1.080      1.101      1.030      0.057      151      994.6
64x16384x512         SME 2x2             1.215      1.268      1.178      0.173      151      883.9
64x16384x512         SME 1x4             1.133      1.151      1.104      0.048      151      948.0
64x16384x512         SME 1x4sym          1.033      1.042      1.013      0.027      151     1039.9
64x16384x512         SME 4x1ZP           1.115      1.137      1.075      0.058      151      963.1
64x16384x512         KAI 2VL             0.867      0.881      0.826      0.055      151     1238.9
64x16384x512         KAI 8VS             0.895      0.920      0.861      0.056      151     1199.3
64x32768x512         SME 4x1             2.066      2.089      2.055      0.064      151     1039.7
64x32768x512         SME 2x2             2.389      2.405      2.378      0.043      147      898.8
64x32768x512         SME 1x4             2.219      2.233      2.208      0.043      151      967.8
64x32768x512         SME 1x4sym          2.039      2.056      2.027      0.041      151     1052.9
64x32768x512         SME 4x1ZP           2.160      2.178      2.148      0.052      151      994.4
64x32768x512         KAI 2VL             2.313      2.338      2.294      0.078      151      928.4
64x32768x512         KAI 8VS             2.322      2.373      2.292      0.123      151      924.9
128x8192x512         SME 4x1             0.837      0.844      0.814      0.026      151     1282.3
128x8192x512         SME 2x2             0.952      0.954      0.931      0.015      151     1128.3
128x8192x512         SME 1x4             0.883      0.886      0.866      0.022      151     1216.0
128x8192x512         SME 1x4sym          0.820      0.821      0.806      0.015      151     1309.7
128x8192x512         SME 4x1ZP           1.038      1.037      0.939      0.027      151     1034.8
128x8192x512         KAI 2VL             0.664      0.667      0.658      0.010      151     1615.9
128x8192x512         KAI 8VS             0.679      0.684      0.674      0.015      151     1580.2
128x16384x512        SME 4x1             1.702      1.709      1.693      0.025      151     1261.8
128x16384x512        SME 2x2             1.956      1.964      1.940      0.027      151     1098.1
128x16384x512        SME 1x4             1.876      1.886      1.843      0.042      151     1144.5
128x16384x512        SME 1x4sym          1.692      1.697      1.684      0.016      151     1269.5
128x16384x512        SME 4x1ZP           2.171      2.193      2.151      0.056      151      989.4
128x16384x512        KAI 2VL             1.671      1.688      1.655      0.053      151     1285.3
128x16384x512        KAI 8VS             1.655      1.689      1.613      0.092      151     1297.8
128x32768x512        SME 4x1             3.402      3.425      3.385      0.112       96     1262.5
128x32768x512        SME 2x2             3.917      3.929      3.885      0.044       90     1096.6
128x32768x512        SME 1x4             3.719      3.741      3.702      0.055       95     1155.0
128x32768x512        SME 1x4sym          3.386      3.399      3.377      0.036      104     1268.6
128x32768x512        SME 4x1ZP           4.334      4.343      4.291      0.038       81      991.0
128x32768x512        KAI 2VL             4.656      4.718      4.586      0.149       76      922.4
128x32768x512        KAI 8VS             4.958      5.035      4.909      0.162       71      866.3

=====================================================================
  Correctness
=====================================================================

M x K x N            mode         implementation    max abs err     rel L2 err  max abs vs FP64  status
----------------------------------------------------------------------------------------------------------------
64x128x16            hot_microkernel SME 4x1             0.000e+00      0.000e+00        0.000e+00  PASS
64x128x16            hot_microkernel SME 4x1 [iso]       0.000e+00      0.000e+00        0.000e+00  PASS
32x128x32            hot_microkernel SME 2x2             0.000e+00      0.000e+00        0.000e+00  PASS
32x128x32            hot_microkernel SME 2x2 [iso]       0.000e+00      0.000e+00        0.000e+00  PASS
16x128x64            hot_microkernel SME 1x4             0.000e+00      0.000e+00        0.000e+00  PASS
16x128x64            hot_microkernel SME 1x4 [iso]       0.000e+00      0.000e+00        0.000e+00  PASS
16x128x64            hot_microkernel SME 1x4sym          0.000e+00      0.000e+00        0.000e+00  PASS
16x128x64            hot_microkernel SME 1x4sym [iso]      0.000e+00      0.000e+00        0.000e+00  PASS
64x128x16            hot_microkernel SME 4x1ZP           0.000e+00      0.000e+00        0.000e+00  PASS
64x128x16            hot_microkernel SME 4x1ZP [iso]      0.000e+00      0.000e+00        0.000e+00  PASS
32x128x32            hot_microkernel KAI 2VL             0.000e+00      0.000e+00        0.000e+00  PASS
16x128x16            hot_microkernel KAI 8VS             3.338e-06      2.031e-07        0.000e+00  PASS
64x256x16            hot_microkernel SME 4x1             0.000e+00      0.000e+00        0.000e+00  PASS
64x256x16            hot_microkernel SME 4x1 [iso]       0.000e+00      0.000e+00        0.000e+00  PASS
32x256x32            hot_microkernel SME 2x2             0.000e+00      0.000e+00        0.000e+00  PASS
32x256x32            hot_microkernel SME 2x2 [iso]       0.000e+00      0.000e+00        0.000e+00  PASS
16x256x64            hot_microkernel SME 1x4             0.000e+00      0.000e+00        0.000e+00  PASS
16x256x64            hot_microkernel SME 1x4 [iso]       0.000e+00      0.000e+00        0.000e+00  PASS
16x256x64            hot_microkernel SME 1x4sym          0.000e+00      0.000e+00        0.000e+00  PASS
16x256x64            hot_microkernel SME 1x4sym [iso]      0.000e+00      0.000e+00        0.000e+00  PASS
64x256x16            hot_microkernel SME 4x1ZP           0.000e+00      0.000e+00        0.000e+00  PASS
64x256x16            hot_microkernel SME 4x1ZP [iso]      0.000e+00      0.000e+00        0.000e+00  PASS
32x256x32            hot_microkernel KAI 2VL             0.000e+00      0.000e+00        0.000e+00  PASS
16x256x16            hot_microkernel KAI 8VS             5.722e-06      2.753e-07        0.000e+00  PASS
64x512x16            hot_microkernel SME 4x1             0.000e+00      0.000e+00        0.000e+00  PASS
64x512x16            hot_microkernel SME 4x1 [iso]       0.000e+00      0.000e+00        0.000e+00  PASS
32x512x32            hot_microkernel SME 2x2             0.000e+00      0.000e+00        0.000e+00  PASS
32x512x32            hot_microkernel SME 2x2 [iso]       0.000e+00      0.000e+00        0.000e+00  PASS
16x512x64            hot_microkernel SME 1x4             0.000e+00      0.000e+00        0.000e+00  PASS
16x512x64            hot_microkernel SME 1x4 [iso]       0.000e+00      0.000e+00        0.000e+00  PASS
16x512x64            hot_microkernel SME 1x4sym          0.000e+00      0.000e+00        0.000e+00  PASS
16x512x64            hot_microkernel SME 1x4sym [iso]      0.000e+00      0.000e+00        0.000e+00  PASS
64x512x16            hot_microkernel SME 4x1ZP           0.000e+00      0.000e+00        0.000e+00  PASS
64x512x16            hot_microkernel SME 4x1ZP [iso]      0.000e+00      0.000e+00        0.000e+00  PASS
32x512x32            hot_microkernel KAI 2VL             0.000e+00      0.000e+00        0.000e+00  PASS
16x512x16            hot_microkernel KAI 8VS             1.431e-05      4.871e-07        0.000e+00  PASS
64x1024x16           hot_microkernel SME 4x1             0.000e+00      0.000e+00        0.000e+00  PASS
64x1024x16           hot_microkernel SME 4x1 [iso]       0.000e+00      0.000e+00        0.000e+00  PASS
32x1024x32           hot_microkernel SME 2x2             0.000e+00      0.000e+00        0.000e+00  PASS
32x1024x32           hot_microkernel SME 2x2 [iso]       0.000e+00      0.000e+00        0.000e+00  PASS
16x1024x64           hot_microkernel SME 1x4             0.000e+00      0.000e+00        0.000e+00  PASS
16x1024x64           hot_microkernel SME 1x4 [iso]       0.000e+00      0.000e+00        0.000e+00  PASS
16x1024x64           hot_microkernel SME 1x4sym          0.000e+00      0.000e+00        0.000e+00  PASS
16x1024x64           hot_microkernel SME 1x4sym [iso]      0.000e+00      0.000e+00        0.000e+00  PASS
64x1024x16           hot_microkernel SME 4x1ZP           0.000e+00      0.000e+00        0.000e+00  PASS
64x1024x16           hot_microkernel SME 4x1ZP [iso]      0.000e+00      0.000e+00        0.000e+00  PASS
32x1024x32           hot_microkernel KAI 2VL             0.000e+00      0.000e+00        0.000e+00  PASS
16x1024x16           hot_microkernel KAI 8VS             2.670e-05      6.158e-07        0.000e+00  PASS
64x2048x16           hot_microkernel SME 4x1             0.000e+00      0.000e+00        0.000e+00  PASS
64x2048x16           hot_microkernel SME 4x1 [iso]       0.000e+00      0.000e+00        0.000e+00  PASS
32x2048x32           hot_microkernel SME 2x2             0.000e+00      0.000e+00        0.000e+00  PASS
32x2048x32           hot_microkernel SME 2x2 [iso]       0.000e+00      0.000e+00        0.000e+00  PASS
16x2048x64           hot_microkernel SME 1x4             0.000e+00      0.000e+00        0.000e+00  PASS
16x2048x64           hot_microkernel SME 1x4 [iso]       0.000e+00      0.000e+00        0.000e+00  PASS
16x2048x64           hot_microkernel SME 1x4sym          0.000e+00      0.000e+00        0.000e+00  PASS
16x2048x64           hot_microkernel SME 1x4sym [iso]      0.000e+00      0.000e+00        0.000e+00  PASS
64x2048x16           hot_microkernel SME 4x1ZP           0.000e+00      0.000e+00        0.000e+00  PASS
64x2048x16           hot_microkernel SME 4x1ZP [iso]      0.000e+00      0.000e+00        0.000e+00  PASS
32x2048x32           hot_microkernel KAI 2VL             0.000e+00      0.000e+00        0.000e+00  PASS
16x2048x16           hot_microkernel KAI 8VS             4.768e-05      9.018e-07        0.000e+00  PASS
256x256x256          end_to_end   SME 4x1             0.000e+00      0.000e+00        2.248e-06  PASS
256x256x256          end_to_end   SME 2x2             0.000e+00      0.000e+00        2.248e-06  PASS
256x256x256          end_to_end   SME 1x4             0.000e+00      0.000e+00        2.248e-06  PASS
256x256x256          end_to_end   SME 1x4sym          0.000e+00      0.000e+00        2.248e-06  PASS
256x256x256          end_to_end   SME 4x1ZP           0.000e+00      0.000e+00        2.248e-06  PASS
256x256x256          end_to_end   SME 1x4ZAIO         0.000e+00      0.000e+00        2.248e-06  PASS
256x256x256          end_to_end   KAI 2VL             0.000e+00      0.000e+00        2.248e-06  PASS
256x256x256          end_to_end   KAI 8VS             0.000e+00      0.000e+00        2.248e-06  PASS
256x256x256          end_to_end   Accelerate          0.000e+00      0.000e+00        2.248e-06  PASS
256x256x256          kernel_only  SME 4x1             0.000e+00      0.000e+00        2.248e-06  PASS
256x256x256          kernel_only  SME 2x2             0.000e+00      0.000e+00        2.248e-06  PASS
256x256x256          kernel_only  SME 1x4             0.000e+00      0.000e+00        2.248e-06  PASS
256x256x256          kernel_only  SME 1x4sym          0.000e+00      0.000e+00        2.248e-06  PASS
256x256x256          kernel_only  SME 4x1ZP           0.000e+00      0.000e+00        2.248e-06  PASS
256x256x256          kernel_only  KAI 2VL             0.000e+00      0.000e+00        2.248e-06  PASS
256x256x256          kernel_only  KAI 8VS             0.000e+00      0.000e+00        2.248e-06  PASS
256x256x256          end_to_end   KAIpanel 2VL        0.000e+00      0.000e+00        2.248e-06  PASS
256x256x256          end_to_end   KAIpanel 8VS        0.000e+00      0.000e+00        2.248e-06  PASS
256x256x256          end_to_end   ggml CPU            1.526e-05      3.030e-07        1.291e-06  PASS
256x256x256          end_to_end   ggml CPU+setup      1.526e-05      3.030e-07        1.291e-06  PASS
512x512x512          end_to_end   SME 4x1             0.000e+00      0.000e+00        6.646e-06  PASS
512x512x512          end_to_end   SME 2x2             0.000e+00      0.000e+00        6.646e-06  PASS
512x512x512          end_to_end   SME 1x4             0.000e+00      0.000e+00        6.646e-06  PASS
512x512x512          end_to_end   SME 1x4sym          0.000e+00      0.000e+00        6.646e-06  PASS
512x512x512          end_to_end   SME 4x1ZP           0.000e+00      0.000e+00        6.646e-06  PASS
512x512x512          end_to_end   SME 1x4ZAIO         0.000e+00      0.000e+00        6.646e-06  PASS
512x512x512          end_to_end   KAI 2VL             0.000e+00      0.000e+00        6.646e-06  PASS
512x512x512          end_to_end   KAI 8VS             0.000e+00      0.000e+00        6.646e-06  PASS
512x512x512          end_to_end   Accelerate          0.000e+00      0.000e+00        6.646e-06  PASS
512x512x512          kernel_only  SME 4x1             0.000e+00      0.000e+00        6.646e-06  PASS
512x512x512          kernel_only  SME 2x2             0.000e+00      0.000e+00        6.646e-06  PASS
512x512x512          kernel_only  SME 1x4             0.000e+00      0.000e+00        6.646e-06  PASS
512x512x512          kernel_only  SME 1x4sym          0.000e+00      0.000e+00        6.646e-06  PASS
512x512x512          kernel_only  SME 4x1ZP           0.000e+00      0.000e+00        6.646e-06  PASS
512x512x512          kernel_only  KAI 2VL             0.000e+00      0.000e+00        6.646e-06  PASS
512x512x512          kernel_only  KAI 8VS             0.000e+00      0.000e+00        6.646e-06  PASS
512x512x512          end_to_end   KAIpanel 2VL        0.000e+00      0.000e+00        6.646e-06  PASS
512x512x512          end_to_end   KAIpanel 8VS        0.000e+00      0.000e+00        6.646e-06  PASS
512x512x512          end_to_end   ggml CPU            3.052e-05      4.231e-07        1.436e-06  PASS
512x512x512          end_to_end   ggml CPU+setup      3.052e-05      4.231e-07        1.436e-06  PASS
1024x1024x1024       end_to_end   SME 4x1             0.000e+00      0.000e+00        1.274e-05  PASS
1024x1024x1024       end_to_end   SME 2x2             0.000e+00      0.000e+00        1.274e-05  PASS
1024x1024x1024       end_to_end   SME 1x4             0.000e+00      0.000e+00        1.274e-05  PASS
1024x1024x1024       end_to_end   SME 1x4sym          0.000e+00      0.000e+00        1.274e-05  PASS
1024x1024x1024       end_to_end   SME 4x1ZP           0.000e+00      0.000e+00        1.274e-05  PASS
1024x1024x1024       end_to_end   SME 1x4ZAIO         0.000e+00      0.000e+00        1.274e-05  PASS
1024x1024x1024       end_to_end   KAI 2VL             0.000e+00      0.000e+00        1.274e-05  PASS
1024x1024x1024       end_to_end   KAI 8VS             0.000e+00      0.000e+00        1.274e-05  PASS
1024x1024x1024       end_to_end   Accelerate          0.000e+00      0.000e+00        1.274e-05  PASS
1024x1024x1024       kernel_only  SME 4x1             0.000e+00      0.000e+00        1.274e-05  PASS
1024x1024x1024       kernel_only  SME 2x2             0.000e+00      0.000e+00        1.274e-05  PASS
1024x1024x1024       kernel_only  SME 1x4             0.000e+00      0.000e+00        1.274e-05  PASS
1024x1024x1024       kernel_only  SME 1x4sym          0.000e+00      0.000e+00        1.274e-05  PASS
1024x1024x1024       kernel_only  SME 4x1ZP           0.000e+00      0.000e+00        1.274e-05  PASS
1024x1024x1024       kernel_only  KAI 2VL             0.000e+00      0.000e+00        1.274e-05  PASS
1024x1024x1024       kernel_only  KAI 8VS             0.000e+00      0.000e+00        1.274e-05  PASS
1024x1024x1024       end_to_end   KAIpanel 2VL        0.000e+00      0.000e+00        1.274e-05  PASS
1024x1024x1024       end_to_end   KAIpanel 8VS        0.000e+00      0.000e+00        1.274e-05  PASS
1024x1024x1024       end_to_end   ggml CPU            7.248e-05      5.953e-07        3.050e-06  PASS
1024x1024x1024       end_to_end   ggml CPU+setup      7.248e-05      5.953e-07        3.050e-06  PASS
2048x2048x2048       end_to_end   SME 4x1             0.000e+00      0.000e+00        6.557e-05  PASS
2048x2048x2048       end_to_end   SME 2x2             1.526e-04      7.692e-07        1.675e-05  PASS
2048x2048x2048       end_to_end   SME 1x4             0.000e+00      0.000e+00        6.557e-05  PASS
2048x2048x2048       end_to_end   SME 1x4sym          0.000e+00      0.000e+00        6.557e-05  PASS
2048x2048x2048       end_to_end   SME 4x1ZP           1.526e-04      7.692e-07        1.675e-05  PASS
2048x2048x2048       end_to_end   SME 1x4ZAIO         0.000e+00      0.000e+00        6.557e-05  PASS
2048x2048x2048       end_to_end   KAI 2VL             0.000e+00      0.000e+00        6.557e-05  PASS
2048x2048x2048       end_to_end   KAI 8VS             0.000e+00      0.000e+00        6.557e-05  PASS
2048x2048x2048       end_to_end   Accelerate          0.000e+00      0.000e+00        6.557e-05  PASS
2048x2048x2048       kernel_only  SME 4x1             0.000e+00      0.000e+00        6.557e-05  PASS
2048x2048x2048       kernel_only  SME 2x2             1.526e-04      7.692e-07        1.675e-05  PASS
2048x2048x2048       kernel_only  SME 1x4             0.000e+00      0.000e+00        6.557e-05  PASS
2048x2048x2048       kernel_only  SME 1x4sym          0.000e+00      0.000e+00        6.557e-05  PASS
2048x2048x2048       kernel_only  SME 4x1ZP           1.526e-04      7.692e-07        1.675e-05  PASS
2048x2048x2048       kernel_only  KAI 2VL             0.000e+00      0.000e+00        6.557e-05  PASS
2048x2048x2048       kernel_only  KAI 8VS             0.000e+00      0.000e+00        6.557e-05  PASS
2048x2048x2048       end_to_end   KAIpanel 2VL        0.000e+00      0.000e+00        6.557e-05  PASS
2048x2048x2048       end_to_end   KAIpanel 8VS        0.000e+00      0.000e+00        6.557e-05  PASS
2048x2048x2048       end_to_end   ggml CPU            1.450e-04      8.400e-07        9.470e-06  PASS
2048x2048x2048       end_to_end   ggml CPU+setup      1.450e-04      8.400e-07        9.470e-06  PASS
4096x4096x4096       end_to_end   SME 4x1             3.510e-04      1.091e-06        4.956e-05  PASS
4096x4096x4096       end_to_end   SME 2x2             3.510e-04      1.195e-06        2.551e-05  PASS
4096x4096x4096       end_to_end   SME 1x4             3.510e-04      1.091e-06        4.956e-05  PASS
4096x4096x4096       end_to_end   SME 1x4sym          3.510e-04      1.091e-06        4.956e-05  PASS
4096x4096x4096       end_to_end   SME 4x1ZP           3.510e-04      1.195e-06        2.551e-05  PASS
4096x4096x4096       end_to_end   SME 1x4ZAIO         0.000e+00      0.000e+00        5.875e-05  PASS
4096x4096x4096       end_to_end   KAI 2VL             0.000e+00      0.000e+00        5.875e-05  PASS
4096x4096x4096       end_to_end   KAI 8VS             0.000e+00      0.000e+00        5.875e-05  PASS
4096x4096x4096       end_to_end   Accelerate          0.000e+00      0.000e+00        5.875e-05  PASS
4096x4096x4096       kernel_only  SME 4x1             3.510e-04      1.091e-06        4.956e-05  PASS
4096x4096x4096       kernel_only  SME 2x2             3.510e-04      1.195e-06        2.551e-05  PASS
4096x4096x4096       kernel_only  SME 1x4             3.510e-04      1.091e-06        4.956e-05  PASS
4096x4096x4096       kernel_only  SME 1x4sym          3.510e-04      1.091e-06        4.956e-05  PASS
4096x4096x4096       kernel_only  SME 4x1ZP           3.510e-04      1.195e-06        2.551e-05  PASS
4096x4096x4096       kernel_only  KAI 2VL             0.000e+00      0.000e+00        5.875e-05  PASS
4096x4096x4096       kernel_only  KAI 8VS             0.000e+00      0.000e+00        5.875e-05  PASS
4096x4096x4096       end_to_end   KAIpanel 2VL        0.000e+00      0.000e+00        5.875e-05  PASS
4096x4096x4096       end_to_end   KAIpanel 8VS        0.000e+00      0.000e+00        5.875e-05  PASS
4096x4096x4096       end_to_end   ggml CPU            3.586e-04      1.190e-06        2.047e-05  PASS
4096x4096x4096       end_to_end   ggml CPU+setup      3.586e-04      1.190e-06        2.047e-05  PASS
64x8192x512          end_to_end   SME 4x1             4.349e-04      1.682e-06        4.106e-05  PASS
64x8192x512          end_to_end   SME 2x2             4.578e-04      1.686e-06        4.641e-05  PASS
64x8192x512          end_to_end   SME 1x4             4.349e-04      1.682e-06        4.106e-05  PASS
64x8192x512          end_to_end   SME 1x4sym          4.349e-04      1.682e-06        4.106e-05  PASS
64x8192x512          end_to_end   SME 4x1ZP           4.578e-04      1.686e-06        4.641e-05  PASS
64x8192x512          end_to_end   SME 1x4ZAIO         0.000e+00      0.000e+00        1.943e-04  PASS
64x8192x512          end_to_end   KAI 2VL             0.000e+00      0.000e+00        1.943e-04  PASS
64x8192x512          end_to_end   KAI 8VS             0.000e+00      0.000e+00        1.943e-04  PASS
64x8192x512          end_to_end   Accelerate          0.000e+00      0.000e+00        1.943e-04  PASS
64x8192x512          kernel_only  SME 4x1             4.349e-04      1.682e-06        4.106e-05  PASS
64x8192x512          kernel_only  SME 2x2             4.578e-04      1.686e-06        4.641e-05  PASS
64x8192x512          kernel_only  SME 1x4             4.349e-04      1.682e-06        4.106e-05  PASS
64x8192x512          kernel_only  SME 1x4sym          4.349e-04      1.682e-06        4.106e-05  PASS
64x8192x512          kernel_only  SME 4x1ZP           4.578e-04      1.686e-06        4.641e-05  PASS
64x8192x512          kernel_only  KAI 2VL             0.000e+00      0.000e+00        1.943e-04  PASS
64x8192x512          kernel_only  KAI 8VS             0.000e+00      0.000e+00        1.943e-04  PASS
64x8192x512          end_to_end   KAIpanel 2VL        4.349e-04      1.682e-06        4.106e-05  PASS
64x8192x512          end_to_end   KAIpanel 8VS        4.349e-04      1.682e-06        4.106e-05  PASS
64x8192x512          end_to_end   ggml CPU            4.425e-04      1.679e-06        3.018e-05  PASS
64x8192x512          end_to_end   ggml CPU+setup      4.425e-04      1.679e-06        3.018e-05  PASS
64x16384x512         end_to_end   SME 4x1             1.038e-03      2.372e-06        6.551e-05  PASS
64x16384x512         end_to_end   SME 2x2             1.038e-03      2.347e-06        1.054e-04  PASS
64x16384x512         end_to_end   SME 1x4             1.038e-03      2.372e-06        6.551e-05  PASS
64x16384x512         end_to_end   SME 1x4sym          1.038e-03      2.372e-06        6.551e-05  PASS
64x16384x512         end_to_end   SME 4x1ZP           1.038e-03      2.347e-06        1.054e-04  PASS
64x16384x512         end_to_end   SME 1x4ZAIO         0.000e+00      0.000e+00        1.445e-04  PASS
64x16384x512         end_to_end   KAI 2VL             0.000e+00      0.000e+00        1.445e-04  PASS
64x16384x512         end_to_end   KAI 8VS             0.000e+00      0.000e+00        1.445e-04  PASS
64x16384x512         end_to_end   Accelerate          0.000e+00      0.000e+00        1.445e-04  PASS
64x16384x512         kernel_only  SME 4x1             1.038e-03      2.372e-06        6.551e-05  PASS
64x16384x512         kernel_only  SME 2x2             1.038e-03      2.347e-06        1.054e-04  PASS
64x16384x512         kernel_only  SME 1x4             1.038e-03      2.372e-06        6.551e-05  PASS
64x16384x512         kernel_only  SME 1x4sym          1.038e-03      2.372e-06        6.551e-05  PASS
64x16384x512         kernel_only  SME 4x1ZP           1.038e-03      2.347e-06        1.054e-04  PASS
64x16384x512         kernel_only  KAI 2VL             0.000e+00      0.000e+00        1.445e-04  PASS
64x16384x512         kernel_only  KAI 8VS             0.000e+00      0.000e+00        1.445e-04  PASS
64x16384x512         end_to_end   KAIpanel 2VL        1.038e-03      2.372e-06        6.551e-05  PASS
64x16384x512         end_to_end   KAIpanel 8VS        1.038e-03      2.372e-06        6.551e-05  PASS
64x16384x512         end_to_end   ggml CPU            9.766e-04      2.364e-06        6.724e-05  PASS
64x16384x512         end_to_end   ggml CPU+setup      9.766e-04      2.364e-06        6.724e-05  PASS
64x32768x512         end_to_end   SME 4x1             1.831e-03      3.363e-06        1.614e-04  PASS
64x32768x512         end_to_end   SME 2x2             1.816e-03      3.323e-06        6.619e-05  PASS
64x32768x512         end_to_end   SME 1x4             1.831e-03      3.363e-06        1.614e-04  PASS
64x32768x512         end_to_end   SME 1x4sym          1.831e-03      3.363e-06        1.614e-04  PASS
64x32768x512         end_to_end   SME 4x1ZP           1.816e-03      3.323e-06        6.619e-05  PASS
64x32768x512         end_to_end   SME 1x4ZAIO         0.000e+00      0.000e+00        6.383e-04  PASS
64x32768x512         end_to_end   KAI 2VL             0.000e+00      0.000e+00        6.383e-04  PASS
64x32768x512         end_to_end   KAI 8VS             0.000e+00      0.000e+00        6.383e-04  PASS
64x32768x512         end_to_end   Accelerate          0.000e+00      0.000e+00        6.383e-04  PASS
64x32768x512         kernel_only  SME 4x1             1.831e-03      3.363e-06        1.614e-04  PASS
64x32768x512         kernel_only  SME 2x2             1.816e-03      3.323e-06        6.619e-05  PASS
64x32768x512         kernel_only  SME 1x4             1.831e-03      3.363e-06        1.614e-04  PASS
64x32768x512         kernel_only  SME 1x4sym          1.831e-03      3.363e-06        1.614e-04  PASS
64x32768x512         kernel_only  SME 4x1ZP           1.816e-03      3.323e-06        6.619e-05  PASS
64x32768x512         kernel_only  KAI 2VL             0.000e+00      0.000e+00        6.383e-04  PASS
64x32768x512         kernel_only  KAI 8VS             0.000e+00      0.000e+00        6.383e-04  PASS
64x32768x512         end_to_end   KAIpanel 2VL        1.831e-03      3.363e-06        1.614e-04  PASS
64x32768x512         end_to_end   KAIpanel 8VS        1.831e-03      3.363e-06        1.614e-04  PASS
64x32768x512         end_to_end   ggml CPU            1.755e-03      3.385e-06        1.218e-04  PASS
64x32768x512         end_to_end   ggml CPU+setup      1.755e-03      3.385e-06        1.218e-04  PASS
128x8192x512         end_to_end   SME 4x1             4.349e-04      1.686e-06        5.253e-05  PASS
128x8192x512         end_to_end   SME 2x2             4.578e-04      1.691e-06        4.641e-05  PASS
128x8192x512         end_to_end   SME 1x4             4.349e-04      1.686e-06        5.253e-05  PASS
128x8192x512         end_to_end   SME 1x4sym          4.349e-04      1.686e-06        5.253e-05  PASS
128x8192x512         end_to_end   SME 4x1ZP           4.578e-04      1.691e-06        4.641e-05  PASS
128x8192x512         end_to_end   SME 1x4ZAIO         0.000e+00      0.000e+00        1.943e-04  PASS
128x8192x512         end_to_end   KAI 2VL             0.000e+00      0.000e+00        1.943e-04  PASS
128x8192x512         end_to_end   KAI 8VS             0.000e+00      0.000e+00        1.943e-04  PASS
128x8192x512         end_to_end   Accelerate          0.000e+00      0.000e+00        1.943e-04  PASS
128x8192x512         kernel_only  SME 4x1             4.349e-04      1.686e-06        5.253e-05  PASS
128x8192x512         kernel_only  SME 2x2             4.578e-04      1.691e-06        4.641e-05  PASS
128x8192x512         kernel_only  SME 1x4             4.349e-04      1.686e-06        5.253e-05  PASS
128x8192x512         kernel_only  SME 1x4sym          4.349e-04      1.686e-06        5.253e-05  PASS
128x8192x512         kernel_only  SME 4x1ZP           4.578e-04      1.691e-06        4.641e-05  PASS
128x8192x512         kernel_only  KAI 2VL             0.000e+00      0.000e+00        1.943e-04  PASS
128x8192x512         kernel_only  KAI 8VS             0.000e+00      0.000e+00        1.943e-04  PASS
128x8192x512         end_to_end   KAIpanel 2VL        4.349e-04      1.686e-06        5.253e-05  PASS
128x8192x512         end_to_end   KAIpanel 8VS        4.349e-04      1.686e-06        5.253e-05  PASS
128x8192x512         end_to_end   ggml CPU            4.425e-04      1.680e-06        3.059e-05  PASS
128x8192x512         end_to_end   ggml CPU+setup      4.425e-04      1.680e-06        3.059e-05  PASS
128x16384x512        end_to_end   SME 4x1             1.038e-03      2.380e-06        7.223e-05  PASS
128x16384x512        end_to_end   SME 2x2             1.038e-03      2.353e-06        4.074e-05  PASS
128x16384x512        end_to_end   SME 1x4             1.038e-03      2.380e-06        7.223e-05  PASS
128x16384x512        end_to_end   SME 1x4sym          1.038e-03      2.380e-06        7.223e-05  PASS
128x16384x512        end_to_end   SME 4x1ZP           1.038e-03      2.353e-06        4.074e-05  PASS
128x16384x512        end_to_end   SME 1x4ZAIO         0.000e+00      0.000e+00        2.991e-04  PASS
128x16384x512        end_to_end   KAI 2VL             0.000e+00      0.000e+00        2.991e-04  PASS
128x16384x512        end_to_end   KAI 8VS             0.000e+00      0.000e+00        2.991e-04  PASS
128x16384x512        end_to_end   Accelerate          0.000e+00      0.000e+00        2.991e-04  PASS
128x16384x512        kernel_only  SME 4x1             1.038e-03      2.380e-06        7.223e-05  PASS
128x16384x512        kernel_only  SME 2x2             1.038e-03      2.353e-06        4.074e-05  PASS
128x16384x512        kernel_only  SME 1x4             1.038e-03      2.380e-06        7.223e-05  PASS
128x16384x512        kernel_only  SME 1x4sym          1.038e-03      2.380e-06        7.223e-05  PASS
128x16384x512        kernel_only  SME 4x1ZP           1.038e-03      2.353e-06        4.074e-05  PASS
128x16384x512        kernel_only  KAI 2VL             0.000e+00      0.000e+00        2.991e-04  PASS
128x16384x512        kernel_only  KAI 8VS             0.000e+00      0.000e+00        2.991e-04  PASS
128x16384x512        end_to_end   KAIpanel 2VL        1.038e-03      2.380e-06        7.223e-05  PASS
128x16384x512        end_to_end   KAIpanel 8VS        1.038e-03      2.380e-06        7.223e-05  PASS
128x16384x512        end_to_end   ggml CPU            9.766e-04      2.372e-06        7.791e-05  PASS
128x16384x512        end_to_end   ggml CPU+setup      9.766e-04      2.372e-06        7.791e-05  PASS
128x32768x512        end_to_end   SME 4x1             1.831e-03      3.344e-06        1.539e-04  PASS
128x32768x512        end_to_end   SME 2x2             1.816e-03      3.307e-06        8.695e-05  PASS
128x32768x512        end_to_end   SME 1x4             1.831e-03      3.344e-06        1.539e-04  PASS
128x32768x512        end_to_end   SME 1x4sym          1.831e-03      3.344e-06        1.539e-04  PASS
128x32768x512        end_to_end   SME 4x1ZP           1.816e-03      3.307e-06        8.695e-05  PASS
128x32768x512        end_to_end   SME 1x4ZAIO         0.000e+00      0.000e+00        8.057e-04  PASS
128x32768x512        end_to_end   KAI 2VL             0.000e+00      0.000e+00        8.057e-04  PASS
128x32768x512        end_to_end   KAI 8VS             0.000e+00      0.000e+00        8.057e-04  PASS
128x32768x512        end_to_end   Accelerate          0.000e+00      0.000e+00        8.057e-04  PASS
128x32768x512        kernel_only  SME 4x1             1.831e-03      3.344e-06        1.539e-04  PASS
128x32768x512        kernel_only  SME 2x2             1.816e-03      3.307e-06        8.695e-05  PASS
128x32768x512        kernel_only  SME 1x4             1.831e-03      3.344e-06        1.539e-04  PASS
128x32768x512        kernel_only  SME 1x4sym          1.831e-03      3.344e-06        1.539e-04  PASS
128x32768x512        kernel_only  SME 4x1ZP           1.816e-03      3.307e-06        8.695e-05  PASS
128x32768x512        kernel_only  KAI 2VL             0.000e+00      0.000e+00        8.057e-04  PASS
128x32768x512        kernel_only  KAI 8VS             0.000e+00      0.000e+00        8.057e-04  PASS
128x32768x512        end_to_end   KAIpanel 2VL        1.831e-03      3.344e-06        1.539e-04  PASS
128x32768x512        end_to_end   KAIpanel 8VS        1.831e-03      3.344e-06        1.539e-04  PASS
128x32768x512        end_to_end   ggml CPU            1.755e-03      3.370e-06        1.154e-04  PASS
128x32768x512        end_to_end   ggml CPU+setup      1.755e-03      3.370e-06        1.154e-04  PASS

=====================================================================
  Table A - Hot Microkernel (inner-kernel quality)
=====================================================================

  Packed panels for ONE output tile, resident; timed region is the
  micro-kernel only. MaxMulSK enters streaming mode once per batch
  (as its driver does); KleidiAI's entry point does smstart/smstop per
  call. The [iso] rows re-measure MaxMulSK with per-call streaming so
  that difference is visible rather than assumed away.

    Kc    MaxMulSK GF    KleidiAI GF    ratio  MaxMul geometry  KAI geometry    
----------------------------------------------------------------------------------
   128          536.7         1350.4    0.397  SME 4x1ZP 64x16  KAI 2VL 32x32 ti
   256          849.3         1613.4    0.526  SME 4x1 64x16 ti KAI 2VL 32x32 ti
   512         1193.7         1795.3    0.665  SME 2x2 32x32 ti KAI 2VL 32x32 ti
  1024         1497.4         1898.5    0.789  SME 2x2 32x32 ti KAI 2VL 32x32 ti
  2048         1715.6         1944.6    0.882  SME 1x4sym 16x64 KAI 2VL 32x32 ti

  Per-variant detail

    Kc implementation     geometry               reps/sample    ns/invoke        GFLOP/s    correct
--------------------------------------------------------------------------------------------------
   128 SME 4x1            64x16 tile                   1489        492.3          532.5       PASS
   128 SME 4x1 [iso]      64x16, smstart/call          1489        489.5          535.5       PASS
   128 SME 2x2            32x32 tile                   3693        488.7          536.5       PASS
   128 SME 2x2 [iso]      32x32, smstart/call          3693        488.9          536.2       PASS
   128 SME 1x4            16x64 tile                   3850        488.8          536.3       PASS
   128 SME 1x4 [iso]      16x64, smstart/call          3850        488.5          536.6       PASS
   128 SME 1x4sym         16x64 tile                   3332        488.8          536.3       PASS
   128 SME 1x4sym [iso]   16x64, smstart/call          3332        489.0          536.0       PASS
   128 SME 4x1ZP          64x16 tile                   3606        488.4          536.7       PASS
   128 SME 4x1ZP [iso]    64x16, smstart/call          3606        488.8          536.4       PASS
   128 KAI 2VL            32x32 tile                   8703        194.1         1350.4       PASS
   128 KAI 8VS            16x16 tile                  21334         72.2          908.3       PASS
   256 SME 4x1            64x16 tile                   3132        617.3          849.3       PASS
   256 SME 4x1 [iso]      64x16, smstart/call          3132        617.6          849.0       PASS
   256 SME 2x2            32x32 tile                   2960        617.4          849.2       PASS
   256 SME 2x2 [iso]      32x32, smstart/call          2960        617.7          848.8       PASS
   256 SME 1x4            16x64 tile                   2986        617.4          849.1       PASS
   256 SME 1x4 [iso]      16x64, smstart/call          2986        617.7          848.8       PASS
   256 SME 1x4sym         16x64 tile                   3113        617.3          849.3       PASS
   256 SME 1x4sym [iso]   16x64, smstart/call          3113        617.8          848.7       PASS
   256 SME 4x1ZP          64x16 tile                   2998        617.4          849.2       PASS
   256 SME 4x1ZP [iso]    64x16, smstart/call          2998        617.7          848.8       PASS
   256 KAI 2VL            32x32 tile                   5874        325.0         1613.4       PASS
   256 KAI 8VS            16x16 tile                  14629        104.8         1250.8       PASS
   512 SME 4x1            64x16 tile                   2116        878.5         1193.6       PASS
   512 SME 4x1 [iso]      64x16, smstart/call          2116        878.8         1193.2       PASS
   512 SME 2x2            32x32 tile                   2192        878.4         1193.7       PASS
   512 SME 2x2 [iso]      32x32, smstart/call          2192        878.6         1193.4       PASS
   512 SME 1x4            16x64 tile                   2187        891.9         1175.7       PASS
   512 SME 1x4 [iso]      16x64, smstart/call          2187        894.0         1172.9       PASS
   512 SME 1x4sym         16x64 tile                   2170        878.6         1193.5       PASS
   512 SME 1x4sym [iso]   16x64, smstart/call          2170        878.7         1193.3       PASS
   512 SME 4x1ZP          64x16 tile                   2178        878.4         1193.7       PASS
   512 SME 4x1ZP [iso]    64x16, smstart/call          2178        878.8         1193.2       PASS
   512 KAI 2VL            32x32 tile                   3122        584.1         1795.3       PASS
   512 KAI 8VS            16x16 tile                   9366        170.2         1540.0       PASS
  1024 SME 4x1            64x16 tile                   1399       1401.2         1496.7       PASS
  1024 SME 4x1 [iso]      64x16, smstart/call          1399       1400.8         1497.1       PASS
  1024 SME 2x2            32x32 tile                   1413       1400.5         1497.4       PASS
  1024 SME 2x2 [iso]      32x32, smstart/call          1413       1400.9         1497.0       PASS
  1024 SME 1x4            16x64 tile                   1334       1450.2         1446.1       PASS
  1024 SME 1x4 [iso]      16x64, smstart/call          1334       1452.6         1443.7       PASS
  1024 SME 1x4sym         16x64 tile                   1368       1400.7         1497.2       PASS
  1024 SME 1x4sym [iso]   16x64, smstart/call          1368       1400.9         1497.0       PASS
  1024 SME 4x1ZP          64x16 tile                   1365       1400.7         1497.2       PASS
  1024 SME 4x1ZP [iso]    64x16, smstart/call          1365       1401.5         1496.4       PASS
  1024 KAI 2VL            32x32 tile                   1764       1104.6         1898.5       PASS
  1024 KAI 8VS            16x16 tile                   5679        299.9         1748.1       PASS
  2048 SME 4x1            64x16 tile                    799       2445.0         1715.4       PASS
  2048 SME 4x1 [iso]      64x16, smstart/call           799       2445.4         1715.2       PASS
  2048 SME 2x2            32x32 tile                    799       2445.0         1715.5       PASS
  2048 SME 2x2 [iso]      32x32, smstart/call           799       2445.9         1714.9       PASS
  2048 SME 1x4            16x64 tile                    778       2568.2         1633.1       PASS
  2048 SME 1x4 [iso]      16x64, smstart/call           778       2570.3         1631.9       PASS
  2048 SME 1x4sym         16x64 tile                    818       2444.7         1715.6       PASS
  2048 SME 1x4sym [iso]   16x64, smstart/call           818       2445.0         1715.4       PASS
  2048 SME 4x1ZP          64x16 tile                    799       2444.8         1715.6       PASS
  2048 SME 4x1ZP [iso]    64x16, smstart/call           799       2446.2         1714.6       PASS
  2048 KAI 2VL            32x32 tile                    919       2156.9         1944.6       PASS
  2048 KAI 8VS            16x16 tile                   3142        588.4         1782.0       PASS

=====================================================================
  Table B - Controlled End-to-End Flow (dataflow / packing quality)
=====================================================================

  Same micro-kernels as Table A, now inside three different whole-GEMM
  organizations. 'KAI panel' is KleidiAI's own kernel and packing
  routines driven by a MaxMulSK-style panel-blocked caller.

M x K x N                MaxMulSK     KAI orig    KAI panel     MM/orig    MM/panel
------------------------------------------------------------------------------------
256x256x256                 681.3       1276.3       1276.3       0.534       0.534
512x512x512                1083.1       1711.1       1711.1       0.633       0.633
1024x1024x1024             1337.9       1647.6       1789.9       0.812       0.747
2048x2048x2048             1306.1       1620.9       1595.1       0.806       0.819
4096x4096x4096             1327.7       1574.4       1499.1       0.843       0.886
64x8192x512                 740.7        507.5        835.3       1.460       0.887
64x16384x512                719.5        504.5        849.2       1.426       0.847
64x32768x512                727.3        439.1        855.5       1.656       0.850
128x8192x512                922.6        745.8       1070.7       1.237       0.862
128x16384x512               899.7        701.0       1060.2       1.283       0.849
128x32768x512               907.0        555.1       1068.5       1.634       0.849

  Panel-blocked configuration chosen, and its stage breakdown.
  Percentages are of the instrumented run, whose total is slightly
  above the headline median because of the clock calls.

M x K x N            kernel       config                   packA %   packB % compute %   accum %
------------------------------------------------------------------------------------------------
256x256x256          2VL          Mc=256 Kc=2048 Nc=256      10.4%      7.2%     82.0%      0.0%
256x256x256          8VS          Mc=256 Kc=2048 Nc=512      11.5%      8.3%     79.7%      0.0%
512x512x512          2VL          Mc=512 Kc=512 Nc=512        6.5%      3.7%     89.6%      0.0%
512x512x512          8VS          Mc=256 Kc=512 Nc=512        6.9%      5.6%     87.4%      0.0%
1024x1024x1024       2VL          Mc=64 Kc=2048 Nc=1024       4.7%      2.8%     92.5%      0.0%
1024x1024x1024       8VS          Mc=64 Kc=1024 Nc=1024       4.0%      3.2%     92.8%      0.0%
2048x2048x2048       2VL          Mc=256 Kc=2048 Nc=512      14.9%      3.2%     81.9%      0.0%
2048x2048x2048       8VS          Mc=64 Kc=2048 Nc=512       21.1%      3.5%     75.4%      0.0%
4096x4096x4096       2VL          Mc=256 Kc=4096 Nc=512      16.2%      1.7%     82.1%      0.0%
4096x4096x4096       8VS          Mc=64 Kc=4096 Nc=512       24.9%      2.0%     73.1%      0.0%
64x8192x512          2VL          Mc=64 Kc=2048 Nc=512       14.9%     37.9%     45.2%      1.9%
64x8192x512          8VS          Mc=64 Kc=2048 Nc=512       11.6%     48.4%     38.2%      1.6%
64x16384x512         2VL          Mc=64 Kc=2048 Nc=512       14.9%     39.1%     44.1%      1.8%
64x16384x512         8VS          Mc=64 Kc=2048 Nc=512       10.4%     48.4%     39.6%      1.6%
64x32768x512         2VL          Mc=64 Kc=2048 Nc=512       14.4%     39.6%     44.2%      1.8%
64x32768x512         8VS          Mc=256 Kc=2048 Nc=512      10.7%     48.6%     39.0%      1.7%
128x8192x512         2VL          Mc=128 Kc=2048 Nc=512      17.4%     24.0%     56.3%      2.2%
128x8192x512         8VS          Mc=256 Kc=2048 Nc=512      12.5%     31.2%     53.9%      2.3%
128x16384x512        2VL          Mc=256 Kc=2048 Nc=512      17.7%     24.1%     55.7%      2.4%
128x16384x512        8VS          Mc=128 Kc=2048 Nc=512      13.4%     31.9%     52.4%      2.2%
128x32768x512        2VL          Mc=256 Kc=2048 Nc=512      18.2%     24.2%     55.1%      2.5%
128x32768x512        8VS          Mc=256 Kc=2048 Nc=512      13.9%     31.8%     51.9%      2.3%

=====================================================================
  Table C - Runtime-Style CPU Comparison (vs a real inference runtime)
=====================================================================

M x K x N            implementation     median ms     GFLOP/s  vs MaxMul    correct
------------------------------------------------------------------------------------
256x256x256          MaxMulSK               0.049       681.3      1.000       PASS
256x256x256          KleidiAI orig          0.026      1276.3      1.873       PASS
256x256x256          KleidiAI panel         0.026      1276.3      1.873       PASS
256x256x256          ggml CPU               0.801        41.9      0.061       PASS

512x512x512          MaxMulSK               0.248      1083.1      1.000       PASS
512x512x512          KleidiAI orig          0.157      1711.1      1.580       PASS
512x512x512          KleidiAI panel         0.157      1711.1      1.580       PASS
512x512x512          ggml CPU               6.092        44.1      0.041       PASS

1024x1024x1024       MaxMulSK               1.605      1337.9      1.000       PASS
1024x1024x1024       KleidiAI orig          1.303      1647.6      1.231       PASS
1024x1024x1024       KleidiAI panel         1.200      1789.9      1.338       PASS
1024x1024x1024       ggml CPU              50.565        42.5      0.032       PASS

2048x2048x2048       MaxMulSK              13.153      1306.1      1.000       PASS
2048x2048x2048       KleidiAI orig         10.599      1620.9      1.241       PASS
2048x2048x2048       KleidiAI panel        10.770      1595.1      1.221       PASS
2048x2048x2048       ggml CPU             468.552        36.7      0.028       PASS

4096x4096x4096       MaxMulSK             103.514      1327.7      1.000       PASS
4096x4096x4096       KleidiAI orig         87.298      1574.4      1.186       PASS
4096x4096x4096       KleidiAI panel        91.680      1499.1      1.129       PASS
4096x4096x4096       ggml CPU            3855.232        35.6      0.027       PASS

64x8192x512          MaxMulSK               0.725       740.7      1.000       PASS
64x8192x512          KleidiAI orig          1.058       507.5      0.685       PASS
64x8192x512          KleidiAI panel         0.643       835.3      1.128       PASS
64x8192x512          ggml CPU              15.361        35.0      0.047       PASS

64x16384x512         MaxMulSK               1.492       719.5      1.000       PASS
64x16384x512         KleidiAI orig          2.128       504.5      0.701       PASS
64x16384x512         KleidiAI panel         1.264       849.2      1.180       PASS
64x16384x512         ggml CPU              34.399        31.2      0.043       PASS

64x32768x512         MaxMulSK               2.953       727.3      1.000       PASS
64x32768x512         KleidiAI orig          4.890       439.1      0.604       PASS
64x32768x512         KleidiAI panel         2.510       855.5      1.176       PASS
64x32768x512         ggml CPU              73.227        29.3      0.040       PASS

128x8192x512         MaxMulSK               1.164       922.6      1.000       PASS
128x8192x512         KleidiAI orig          1.440       745.8      0.808       PASS
128x8192x512         KleidiAI panel         1.003      1070.7      1.160       PASS
128x8192x512         ggml CPU              30.481        35.2      0.038       PASS

128x16384x512        MaxMulSK               2.387       899.7      1.000       PASS
128x16384x512        KleidiAI orig          3.063       701.0      0.779       PASS
128x16384x512        KleidiAI panel         2.026      1060.2      1.178       PASS
128x16384x512        ggml CPU              68.754        31.2      0.035       PASS

128x32768x512        MaxMulSK               4.735       907.0      1.000       PASS
128x32768x512        KleidiAI orig          7.737       555.1      0.612       PASS
128x32768x512        KleidiAI panel         4.020      1068.5      1.178       PASS
128x32768x512        ggml CPU             146.981        29.2      0.032       PASS
```

(Table C continues for the large-K shapes; see the CSV.)

**Reading:** MaxMulSK is 16–34× faster than ggml's CPU FP32 path on every shape,
and KleidiAI is faster still. This is a much larger margin than the SME-vs-NEON
gap alone would predict, because the ggml path is not a blocked GEMM at all — it
is a dot-product kernel. The honest framing is that llama.cpp's *FP32* CPU matmul
is not an optimised GEMM on this hardware; its engineering investment is in the
quantised paths, which is where inference actually runs.

#### What the three experiments establish

| Layer | Question | Answer |
|---|---|---|
| Micro-kernel | Who schedules SME2 better with hot operands? | KleidiAI, at every Kc: 0.40× at Kc=128 rising to 0.88× at Kc=2048 |
| Macro-kernel / blocking | Who traverses packed operands better past L2? | A tie. Given the same panel-blocked caller, KleidiAI's kernel gets the same benefit MaxMulSK's blocking gives it |
| Packing / data movement | How much time moving operands? | MaxMulSK 9–33% of end-to-end; KleidiAI's default flow 38–64% on large-K, dropping to 24–47% panel-blocked |
| End-to-end library flow | Does the integrated flow overcome the slower kernel? | Only against KleidiAI's *default* flow. Against an equally-organised KleidiAI, no: 0.85–0.89× on large-K |
| llama.cpp relevance | Faster than the real ggml CPU flow? | Yes, 16–34× on every shape — but against a dot-product fallback, not a tuned GEMM |

**"MaxMulSK has a faster micro-kernel" is not supported.** Experiment 1 measures
the opposite, directly and at every Kc.

**"MaxMulSK has a better execution/dataflow organisation for this workload" is
supported, but weakly and conditionally.** Its pack-and-consume structure is
genuinely better than KleidiAI's *default* full-prepack flow on large-K shapes
(1.24×–1.66×). It is not better than the same structure applied to KleidiAI's
kernel — that reverses the result. The organisation is a good, standard
BLIS-style choice; it is not a differentiator, because KleidiAI's kernel benefits
from it at least as much.

Of the five candidate conclusions, **#4 is the best supported**: the apparent
advantage disappears once KleidiAI uses equivalent macro blocking. **#1 is the
right corollary** — the remaining, consistent 0.85–0.89× large-K gap and the
0.53–0.89× square gap both track the Experiment-1 micro-kernel ratio, so the
inner kernel is where the headroom is. #5 is defensible only in the weak sense
that 0.85× is competitive rather than embarrassing. #2 and #3 are not supported.

The concrete headroom, from Experiment 1: MaxMulSK's per-invocation overhead
(`svzero_za` + the 1024-element ZA→C read-modify-write) is what costs it 0.40×
at Kc=128 and still 0.12 at Kc=2048. Amortising that — deeper K per ZA residency,
or accumulating in ZA across tiles rather than round-tripping through C — is the
change the measurements point at. The `1x4ZAIO` kernel already explores exactly
this, and is the one variant the hot-tile harness cannot isolate.

### 0.8 llama.cpp's own KleidiAI SME2 path (2026-09-04)

**Measured:** 2026-09-04 · **CURRENT**

§0.7 measured llama.cpp's *native* FP32 CPU path and found it is a dot-product
fallback at 29–44 GFLOP/s. But current llama.cpp also ships a KleidiAI
integration with an FP32 SME2 kernel. This adds it as a second llama.cpp
baseline, so there are now two ggml rows:

| Row | Path |
|---|---|
| `llama-native-f32` | src0 in the default CPU buffer type → ggml's generic `ggml_vec_dot_f32` |
| `llama-kleidiai-f32` | src0 in the `CPU_KLEIDIAI` extra buffer type → ggml's KleidiAI integration → fp32 SME2 kernel |

Both go through the ordinary ggml graph/backend flow. Neither calls a KleidiAI
entry point directly — that is what the `KAI direct` rows already do.

#### Dispatch was verified, not assumed

The dispatch is gated: `kleidiai.cpp::supports_op()` requires
`op->src[0]->buffer->buft->context == this`, i.e. **src0 must be allocated in the
`CPU_KLEIDIAI` extra buffer type**, plus `ggml_n_dims(src0) == 2`,
`ctx.kernels_f32 != nullptr`, and src1/dst both F32. Simply building with
`GGML_CPU_KLEIDIAI=ON` and allocating normally gets you the native path and
nothing else. The benchmark obtains that buffer type through the same public
`ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts")`
mechanism llama.cpp uses for model weights.

Three independent checks:

1. **Static.** `kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa` is
   defined (`T`) in `libkleidiai.a` and referenced (`U`) by `libggml-cpu.a`,
   along with `kai_run_lhs_pack_f32p2vlx1_f32_sme` and
   `kai_run_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme`. Absent entirely from the
   non-KleidiAI build.
2. **Runtime, from ggml itself** (captured via `ggml_log_set`):

```
  ggml's own KleidiAI init diagnostics (captured via ggml_log_set):
    kleidiai: primary q4 kernel feature SME2
    kleidiai: primary q8 kernel feature SME2
    kleidiai: primary f32 kernel feature SME2
    kleidiai: SME enabled (runtime-detected SME thread cap=1)
```

3. **Executed-kernel identity.** A separate instrumented build `dladdr`'d the
   function pointer actually invoked in `compute_forward_f32`:

```
[KAIDBG] compute_forward_f32 entered: slots=1 gemv=0 ne00(k)=1024 ne01(kai_n)=1024 ne11(kai_m)=1024
[KAIDBG] run_kernel_ex resolves to:
    _ZL15kernel_run_fn10IXadL_Z61kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopaEEE...
[KAIDBG] m_step=32 n_step=32 mr=32 nr=32 kr=1 sr=1
```

That is the **same micro-kernel and the same geometry** as our direct `KAI 2VL`
rows. The instrumentation lives only in that debug tree; the timing build
contains no such logging. Thread count was measured, not assumed:
CPU-time/wall-time = **1.00** for both ggml rows.

Note that ggml pins **KleidiAI v1.24.0** while the direct rows use **v1.30.0**,
so the two are the same kernel from different releases.

#### Same primitive, two callers

```
M x K x N               KAI direct     KAI panel     llama+KAI  llama/best    wt pack ms
--------------------------------------------------------------------------------------------
256x256x256                 1276.3        1187.8        1104.7       0.866         0.022
512x512x512                 1707.1        1710.2        1634.7       0.956         0.108
1024x1024x1024              1660.7        1773.4        1797.2       1.013         0.354
2048x2048x2048              1589.0        1590.6         988.9       0.622         1.335
4096x4096x4096              1641.7        1465.0         892.7       0.544         5.829
64x8192x512                  536.6         858.8         416.2       0.485         0.206
64x16384x512                 503.2         850.0         397.5       0.468         0.427
64x32768x512                 434.6         855.4         341.8       0.400         0.916
128x8192x512                 739.7        1082.9         674.3       0.623         0.445
128x16384x512                682.9        1063.5         604.5       0.568         0.907
128x32768x512                551.8        1077.2         469.5       0.436         1.742
```

```
M x K x N                 native GF    KleidiAI GF    speedup
--------------------------------------------------------------
256x256x256                    42.0         1104.7      26.31x
512x512x512                    43.9         1634.7      37.22x
1024x1024x1024                 42.4         1797.2      42.35x
2048x2048x2048                 36.6          988.9      27.02x
4096x4096x4096                 35.6          892.7      25.08x
64x8192x512                    35.2          416.2      11.83x
64x16384x512                   31.6          397.5      12.60x
64x32768x512                   29.4          341.8      11.62x
128x8192x512                   35.3          674.3      19.12x
128x16384x512                  31.3          604.5      19.29x
128x32768x512                  29.3          469.5      16.05x
```

#### Where llama.cpp's overhead comes from

Reading `kleidiai.cpp::compute_forward_f32` rather than guessing:

- **No cache blocking at one thread.** Chunking is
  `kleidiai_chunk_cols(n, nth, ...)`, and for `nth_total == 1` the multiplier and
  divisor are both 1, so `chunk_cols = align_up(n, n_step) = n`. The whole GEMM
  is a **single** `run_kernel_ex(m, n, k, ...)` call. llama.cpp's n-chunking is a
  *threading* device, not a blocking device — single-threaded it does none.
  This is precisely the monolithic pattern §0.5 showed costs ~2× once the packed
  operand outgrows the 16 MB L2, and the numbers agree: llama+KAI falls to
  0.54–0.62× of direct at 2048³/4096³ and 0.40–0.62× on the large-K shapes,
  while our panel-blocked caller — same kernel — holds 850–1080 GFLOP/s.
- **The LHS is fully materialised before any compute.** `lhs_info->pack_func_ex`
  packs all `m` rows over the full `k` into `params->wdata` in one pass, then the
  kernel re-reads it cold. Same full-prepack shape as KleidiAI's default flow.
- **Layout: src0 is KleidiAI's RHS.** ggml maps src0→RHS and src1→LHS, so on the
  skinny shapes the kernel sees `kai_n = 64` and `kai_m = 512` — M and N swapped
  relative to our direct rows. That is a materially different problem for it, and
  part of why the skinny ratios are the worst.
- **Weight repack is a load-time cost, not per-call.** Putting src0 in the
  KleidiAI buffer repacks it once (0.02–5.8 ms across these shapes, reported in
  Table D). Excluded from the timed region, because llama.cpp pays it at model
  load. This is why llama+KAI actually *beats* direct end-to-end at 1024³
  (1797 vs 1661): it gets one operand pre-packed for free.
- **Graph/operator dispatch is not the problem.** The `+setup` rows (graph
  rebuild plus execute) are within noise of plain execute; ggml's operator
  dispatch overhead is negligible at these sizes.

#### Conclusion

Enabling `GGML_CPU_KLEIDIAI` is a large, real win for llama.cpp on this machine:
**11.6×–42.4×** over its own native FP32 path. Anyone benchmarking llama.cpp's
FP32 CPU matmul without it is measuring a dot-product fallback.

But llama.cpp does not extract the full value of the primitive it links. Against
the same kernel driven by a panel-blocked caller it reaches only 0.40–0.62× on
large-K and 0.54× at 4096³, and the reason is structural and visible in its
source: single-threaded, it performs no cache blocking and issues one monolithic
kernel call over a cold, fully-materialised packed LHS. The gap is **caller
orchestration, not kernel quality** — the same conclusion §0.7 reached about
KleidiAI's own default flow, now reproduced independently inside a real inference
runtime.

Ranking on this hardware, single-thread FP32: KleidiAI panel-blocked > KleidiAI
default ≳ MaxMulSK > llama.cpp+KleidiAI >> llama.cpp native.

### 0.9-A  1x4-Acc, start to finish: from ZA accumulation to the C store (2026-09-05)

**Measured:** 2026-09-05 · narrative summary; the sections it summarises (§0.9,
§0.10, §0.11, and later §0.13) follow below in the order the work happened.

> The per-shape GFLOP/s in this section are single-run figures from the dates
> given. §0.14 carries the current numbers, taken as the median of three full
> runs. Nothing here is deleted; read §0.14 for what the code does today.

§0.9, §0.10, §0.11 and §0.13 were four separate sessions chasing one number. Read on
their own each looks like an isolated tweak; together they are a single chain,
and the chain is the useful part. This section is the arc. The detail, the caveats
and the negative results stay in the individual sections.

#### Where we started

`sme/sme-1x4-sym.cpp`. Its micro-kernel owned the whole ZA lifetime:

```
micro_kernel_1x4(pA, pB, C, K_curr, ldc):
    svzero_za()                 <- zero on entry
    ...FMOPA over K_curr...
    for TILE 0..3:              <- drain on exit, tile-major
      for i 0..SVL-1:
        C[...] = C[...] + ZA_TILE[i]     <- read-modify-write, 16 floats, strided
```

The driver's loop order was `m → k → n → (jr, ir)`, so **K sat outside the tile
loop**: one C tile was visited `ceil(K/K_tile)` times and each visit paid a full
zero and a full drain.

§0.7 had already measured the consequence without knowing the cause: fitting
per-invocation time against Kc gave `t(Kc) = 356 + 1.02·Kc` ns. The slope is the
arithmetic and matches KleidiAI's to four significant figures (2008 GFLOP/s
both). The intercept — **356 ns of fixed cost against KleidiAI's 43.6** — was the
entire gap. Three steps followed, each aimed at that intercept.

#### Step 1 — move the ZA lifetime out of the micro-kernel (§0.9)

Zero and drain were lifted into the driver and the loop order became `M → N → K`
with K innermost, so partial sums accumulate **in ZA** across the whole K and
each C tile is written exactly once. A and B are packed over the full K.

*Result: no speedup.* The fixed cost stayed at ~356 ns, because the work did not
change — only where it was paid. Two things did come out of it:

- an **accuracy** improvement, unlooked for: keeping all of K in one ZA
  accumulation chain removed the per-K-tile rounding through memory, making the
  kernel bit-identical to Accelerate at K = 8192 and 32768 where 1x4-sym is not;
- a **regression** at 4096³ and K ≥ 16384, from the full-K packing this
  restructure forces. That one is still open.

It also created the precondition for everything after: because each tile is now
written once, the store no longer has to preserve what is in C.

#### Step 2 — stop reading C (§0.10)

With one store per tile there is nothing to accumulate onto, so the
read-modify-write became a plain store. Isolating it first showed the read and
add were roughly half the store's cost (490 ns vs 261 ns for overwrite-only).

*Result: ~164 ns saved per invocation, flat in Kc. Fixed cost 356 → 192 ns.* That
is +43 % at 256³ and +13 % at 1024³ prepacked.

The semantics changed with it: 1x4-Acc computes `C = A*B`, not `C += A*B`, and
the caller must not pre-zero C. The benchmark was changed to charge the memset
only to the kernels that still need one.

#### Step 3 — turn the drain sideways (§0.11)

The remaining 192 ns was all store. The four ZA tiles stack *horizontally* in C,
so slice `i` of ZA0..ZA3 is one contiguous 4·SVL row of the output tile — but the
loop was walking it tile-major, producing 64 narrow strided stores. Walking it
row-major instead gives 16 contiguous 4-vector stores.

*Result: ~140 ns saved per invocation, flat in Kc. Fixed cost 192 → 52 ns.*

#### Cumulative

Fixed cost per micro-kernel invocation, the thing the first three steps targeted
(the fourth, §0.13, targets packing instead):

**356 ns → 192 ns → 52 ns, a 6.8× reduction**, against KleidiAI's 43.6 ns.
1x4-sym still sits at 356 ns; nothing was changed there.

Hot micro-kernel, ns per invocation:

| Kc | 1x4-sym (start) | step 1 | step 2 | **step 3** |
|---:|---:|---:|---:|---:|
| 128 | 489 ns | 486 ns | 322 ns | **183 ns** |
| 256 | 619 ns | 618 ns | 455 ns | **313 ns** |
| 512 | 879 ns | 878 ns | 715 ns | **574 ns** |
| 1024 | 1401 ns | 1401 ns | 1237 ns | **1096 ns** |
| 2048 | 2445 ns | 2445 ns | 2283 ns | **2142 ns** |

Prepacked full GEMM, GFLOP/s:

| Shape | 1x4-sym (start) | step 1 §0.9 | step 2 §0.10 | **step 3 §0.11** | vs start |
|---|---:|---:|---:|---:|---:|
| 256x256x256 | 746 | 746 | 1162 | **1660** | 2.22x |
| 512x512x512 | 1156 | 1156 | 1473 | **1834** | 1.59x |
| 1024x1024x1024 | 1467 | 1456 | 1692 | **1911** | 1.30x |
| 2048x2048x2048 | 1497 | 1482 | 1715 | **1827** | 1.22x |
| 4096x4096x4096 | 1478 | 929 | 925 | **1012** | 0.68x |
| 64x8192x512 | 1164 | 1290 | 1376 | **1381** | 1.19x |
| 64x32768x512 | 1056 | 829 | 854 | **856** | 0.81x |
| 128x8192x512 | 1295 | 1510 | 1545 | **1564** | 1.21x |
| 128x32768x512 | 1271 | 954 | 965 | **972** | 0.76x |

#### What this bought, and what it did not

The fixed cost is paid once per micro-kernel call, so the payoff scales with how
much of the runtime those calls' overhead represented. Small and mid shapes gain
heavily — 256³ prepacked more than doubles, 2.22× over the 1x4-sym starting
point — and by 2048³ the kernel **passes KleidiAI** (1827 vs 1823). In the hot
path the two are now level, 0.955× at Kc=128 and 0.996× at Kc=2048, where the
starting point was 0.36×.

4096³ and K ≥ 16384 are the exception and they went the other way (0.68×–0.81×
of the starting point). Those rows are not limited by the store at all; they are
limited by the memory traffic of the full-K packing that step 1 introduced. Step 4
(§0.13) reduced the cost of that packing without changing its footprint, which
helped end-to-end but left the prepacked picture unchanged. Capturing the ZA-lifetime win without that
packing cost is the open problem this arc leaves behind.

Every step was validated the same way: 376 benchmark rows plus a 15-shape
standalone check, with C prefilled with a sentinel rather than zero so a missed
overwrite cannot pass silently. 0 correctness failures at every step.

---

### 0.9 1x4-Acc / 1x4-Acc-Kc — extending the ZA lifetime (2026-09-05)

**Measured:** 2026-09-05 · **PARTLY SUPERSEDED by §0.10** — the analysis stands,
but every 1x4-Acc / 1x4-Acc-Kc number below is from the read-modify-write
version of the store and is superseded. The 1x4-sym and KleidiAI numbers stand.

§0.7 attributed MaxMulSK's micro-kernel deficit to fixed per-invocation cost:
every `micro_kernel_*` call zeroes ZA on entry and drains a 1024-element ZA→C
read-modify-write on exit, regardless of Kc. These two experimental kernels test
that directly by moving both out of the micro-kernel and giving the driver
ownership of the ZA lifetime.

#### The kernels

Both are copies of `sme/sme-1x4-sym.cpp`. The FMOPA schedule and the packing
layouts are untouched.

| | `sme/sme-1x4-acc.cpp` (`SMEKernels1x4Acc`) | `sme/sme-1x4-acc-kc.cpp` (`SMEKernels1x4AccKc`) |
|---|---|---|
| Micro-kernel | no `svzero_za()`, no ZA→C store; `C`/`wide_of_C` params dropped; `__arm_out("za")` → `__arm_inout("za")` | identical |
| Driver loop order | `M → N → K`, K innermost | identical |
| Packing | A packed first over the **full K**, then B over the full K | identical |
| ZA lifetime | zeroed once per C tile, accumulated across all of K, stored once after the K loop | identical |
| Micro-kernel calls per C tile | one per K tile | one per `Kc` sub-chunk of a K tile |

Parameters, both kernels (SVL = 16 words at SVL = 512 bits):

```
M_tile = 1024    M_step = 1*SVL = 16
K_tile = 2048    N_step = 4*SVL = 64
N_tile = 64      Kc     = 2048   (1x4-Acc-Kc only)
```

**`Kc = 2048` makes 1x4-Acc-Kc degenerate into 1x4-Acc.** `Kc` equals `K_tile`,
so the inner sub-chunk loop always runs exactly one iteration. The two kernels
are then the same computation issued the same way, and the measurements below
confirm it (every Acc-vs-AccKc ratio is within noise). The variant is kept
because `Kc` is a one-line constant, but at this value it is a control, not an
independent data point. It must stay in sync with `AccChunk<true>` in
`bench/maxmulsk_sme_adapter.cpp`, which replays the same loop.

Buffer consequence of full-K packing: `packed_A` is now `M_tile × K` floats
rather than `M_tile × K_tile`. At K = 32768 that is a 128 MB allocation (only
`mc × K` is ever touched — 16 MB at M = 128).

#### How this was measured

Same target and harness as §0.6–§0.8: `benchmark_maxmul_vs_kleidiai`, Apple M4,
single thread, Homebrew Clang 22.1.8, benchmark TUs at
`-O3 -Wall -Wextra -mcpu=apple-m4 -fopenmp`, the SME sources at
`-march=armv8.7-a+sme+sme2`. `std::chrono::steady_clock`; 3 warmups per
measurement, none reported; adaptive sample count (≥ 7) targeting ~0.35 s;
GFLOP/s from the **median**, FLOPs = `2*M*N*K` in `double`. Inputs U(−1,1),
`mt19937` seed 42, identical for every implementation. 376 measured rows,
**0 correctness failures**.

Both kernels were additionally checked standalone against Accelerate on 13
shapes (square, large-K, all-tails, tiny-odd): **bit-identical** on every one.
Note that they are bit-identical where 1x4-sym is *not*: at 64×8192×512 the
original differs from Accelerate by 4.2e-04 and at 128×32768×512 by 1.6e-03,
because it round-trips C through memory at every K-tile boundary. Keeping the
whole K in one ZA accumulation chain removes those intermediate roundings.

Three benchmark modes, all three wired for the new kernels:

- **hot micro-kernel** — packed panels for one output tile, resident; timed
  region is micro-kernel calls only. The Acc kernels get an extra `[amort]` row:
  zero once, `reps` accumulating calls, store once. The plain row keeps
  zero+compute+store per invocation so it stays comparable with every other
  kernel.
- **prepacked full GEMM** — all packing hoisted out; the driver's loop nest
  replayed with packing removed (`acc_plan` / `acc_prepack` / `acc_compute` in
  `bench/maxmulsk_sme_adapter.cpp`, since the Acc kernels do not fit the generic
  templated path).
- **end-to-end** — the shipped `run_multiplication`, C zeroing included.

#### Hot micro-kernel (GFLOP/s)

| Kc | 1x4sym | 1x4Acc | **1x4Acc [amort]** | 1x4AccKc | 1x4AccKc [amort] | KleidiAI 2VL |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 538.9 | 539.0 | **2008.1** | 537.6 | 2003.1 | 1506.4 |
| 256 | 848.5 | 848.9 | **2007.1** | 849.8 | 2007.5 | 1717.8 |
| 512 | 1193.8 | 1193.8 | **2009.9** | 1195.1 | 2010.1 | 1854.0 |
| 1024 | 1497.4 | 1497.4 | **2008.7** | 1497.4 | 2007.9 | 1928.7 |
| 2048 | 1715.7 | 1715.5 | **2007.8** | 1717.9 | 2006.8 | 1967.1 |

This is the result the experiment was for. The FMOPA loop runs at **~2008
GFLOP/s and is flat in Kc**. The entire Kc dependence in §0.7's Table A — 0.40×
at Kc=128 rising to 0.88× at Kc=2048 — was `svzero_za` plus the ZA→C store, not
arithmetic. In nanoseconds per invocation the fixed cost is **~356 ns and
constant**: 486.3 → 130.5 at Kc=128, 2445.0 → 2089.0 at Kc=2048. That is 73% of
the invocation at Kc=128 and 15% at Kc=2048.

**Which column is comparable to KleidiAI: the plain one, not `[amort]`.**
KleidiAI's kernel is invoked here on a single `m_step × n_step` tile, so it too
initialises ZA and writes the tile out on every call — structurally the same work
as the plain Acc row. Against that, on equal terms, we are **0.36× at Kc=128 and
0.87× at Kc=2048**. `[amort]` has no KleidiAI counterpart: its zero and store are
inside the hand-written `.S`, and hoisting them would mean editing that assembly,
which was not done. So `[amort]` is a diagnostic — it localises our deficit to
the fixed per-call cost rather than to the arithmetic — and **not** a claim that
we beat KleidiAI. It is also an upper bound the full GEMM cannot reach: there the
amortisation factor is the number of K chunks per C tile, `K / K_tile`, which is
1 at K = 2048 and 16 at K = 32768.

#### Separating fixed cost from pure compute, for both sides

`[amort]` gives our pure-compute rate directly but has no KleidiAI counterpart.
Both can be obtained on equal terms without touching anyone's code, by fitting
per-invocation time against Kc: `t(Kc) = a + b·Kc`, where `a` is the fixed
per-call cost (ZA zero + ZA→C writeback + call) and `b` is the marginal cost of
one k-step. Every kernel's tile is 1024 outputs except KAI 8VS, so `2·1024/b`
converts the slope to a pure-compute GFLOP/s.

Fitted over the five Kc points (128 … 2048):

| kernel | fixed `a` (ns) | slope `b` (ns/k-step) | pure compute | max residual |
|---|---:|---:|---:|---:|
| SME 1x4sym | 356.3 | 1.0197 | 2008.4 | 0.50 ns |
| SME 1x4Acc | 356.1 | 1.0200 | 2007.9 | 0.37 ns |
| SME 1x4AccKc | 356.9 | 1.0181 | 2011.6 | 1.16 ns |
| SME 4x1 | 358.8 | 1.0179 | 2012.0 | 4.37 ns |
| SME 2x2 | 357.2 | 1.0190 | 2009.8 | 1.61 ns |
| **KleidiAI 2VL** | **43.6** | **1.0197** | **2008.4** | 0.55 ns |

The fit is validated by the `[amort]` measurement, which is an independent route
to the same quantity: measured 1.0200 ns/k-step, fitted 1.0200.

**The two micro-kernels have the same pure compute rate — 2008.4 GFLOP/s each,
equal to four significant figures.** The inner FMOPA loops are equally good; the
entire per-invocation gap is fixed cost, 356 ns for ours against 43.6 ns for
KleidiAI, a factor of 8.2.

A likely contributor, though this has not been isolated: our ZA→C writeback is a
read-modify-write (`existing + result`) because the MaxMulSK kernels accumulate
into C, whereas KleidiAI's fp32 SME2 kernel overwrites. That is ~1024 extra loads
and adds per tile, on top of `svzero_za`.

KAI 8VS is excluded from the table: its tile is 16×16 rather than 32×32, so it is
not doing the same work per invocation, and its fit residual is 27.9 ns.

A useful control: the plain Acc row matches 1x4-sym to within 0.1 ns at every Kc
(486.3 vs 486.4, 2445.0 vs 2444.7, …), confirming that moving zero/store out of
the micro-kernel is itself free — only *when* they are paid changed.

#### Prepacked full GEMM (GFLOP/s)

| Shape (MxKxN) | 1x4sym | 1x4Acc | 1x4AccKc | Acc/sym |
|---|---:|---:|---:|---:|
| 256x256x256 | 746.4 | 746.4 | 746.3 | 1.000 |
| 512x512x512 | 1156.0 | 1156.0 | 1156.0 | 1.000 |
| 1024x1024x1024 | 1455.3 | 1456.4 | 1452.4 | 1.001 |
| 2048x2048x2048 | 1495.6 | 1481.9 | 1479.4 | 0.991 |
| 4096x4096x4096 | 1395.9 | 928.5 | 881.4 | **0.665** |
| 64x8192x512 | 1130.4 | 1290.0 | 1290.4 | **1.141** |
| 64x16384x512 | 1055.0 | 1090.0 | 1086.5 | **1.033** |
| 64x32768x512 | 1047.9 | 829.3 | 835.1 | **0.791** |
| 128x8192x512 | 1293.5 | 1510.3 | 1502.7 | **1.168** |
| 128x16384x512 | 1228.2 | 1065.8 | 1090.9 | **0.868** |
| 128x32768x512 | 1268.3 | 954.5 | 938.9 | **0.753** |

#### End-to-end (GFLOP/s)

| Shape (MxKxN) | 1x4sym | 1x4Acc | 1x4AccKc | Acc/sym |
|---|---:|---:|---:|---:|
| 256x256x256 | 646.3 | 644.8 | 644.8 | 0.998 |
| 512x512x512 | 1034.9 | 1034.4 | 1034.3 | 1.000 |
| 1024x1024x1024 | 1254.8 | 1229.8 | 1231.3 | 0.980 |
| 2048x2048x2048 | 1186.9 | 1134.9 | 1130.5 | **0.956** |
| 4096x4096x4096 | 1368.4 | 1068.6 | 1059.9 | **0.781** |
| 64x8192x512 | 755.4 | 646.2 | 703.4 | **0.855** |
| 64x16384x512 | 698.4 | 718.7 | 726.5 | 1.029 |
| 64x32768x512 | 698.6 | 567.4 | 524.3 | **0.812** |
| 128x8192x512 | 890.0 | 760.5 | 779.2 | **0.854** |
| 128x16384x512 | 875.9 | 871.7 | 899.3 | 0.995 |
| 128x32768x512 | 861.9 | 688.8 | 688.4 | **0.799** |

#### Reading

The micro-kernel hypothesis is confirmed and the ceiling is real, but **it does
not carry into the full GEMM**. K ≤ 2048 is unchanged, as it must be — there is
a single K tile there, so ZA already lived the whole K and nothing moved. Above
that the picture splits:

- **K = 8192 wins**: +14% (64×8192×512) and +17% (128×8192×512) prepacked.
- **K ≥ 16384 and 4096³ lose**, badly: −21% at 64×32768×512, −25% at
  128×32768×512, −34% at 4096³ prepacked.

The amortisation the hot test promises is only available if the K loop is inside
the tile loop, and that forces packing over the full K. That in turn enlarges the
per-tile working set: each C tile now streams an `M_step × K` A panel and an
`N_step × K` B panel, and the A panel is re-read for every N block. With
`K_tile` blocking those panels were small and stayed resident across tiles.
So the change trades a fixed per-call cost for a memory-traffic cost that grows
with K, and past K ≈ 8192 the trade stops paying. **This mechanism is inferred
from the shape of the results and the loop structure; it has not been isolated
with counters.**

Net: the ZA-lifetime idea is validated as a *micro-kernel* result (~2008 GFLOP/s,
above KleidiAI) and refuted as a *whole-GEMM* strategy in this form. Capturing it
would need the amortisation without the full-K packing that currently pays for
it.

### 0.10 Dropping the read-modify-write from the ZA→C store (2026-09-05)

**Measured:** 2026-09-05 · **PARTLY SUPERSEDED by §0.11** — the reasoning stands
and the read-modify-write removal is still in; §0.11 then restructured the same
store and moved the Acc numbers again.

§0.9 localised the micro-kernel deficit to a fixed ~356 ns per invocation and
showed it was entirely the ZA→C writeback, not `svzero_za` and not the FMOPA
loop. Isolating that store further showed roughly half of it was the
read-modify-write: 490 ns for `load C, add, store` against 261 ns for a plain
store. This removes the read and the add.

#### Why it is sound here and not in 1x4-sym

The writeback read C back because MaxMulSK's kernels accumulate. That is
mandatory when the same C tile is stored more than once — which is exactly what
1x4-sym does, since its K loop sits *outside* the tile loop, so a tile is written
`ceil(K/K_tile)` times (16 times at K = 32768).

In 1x4-Acc the K loop is *inside*, partial sums accumulate in ZA, and each C tile
is stored exactly once with the finished sum. There is nothing in C to preserve,
so the store can overwrite.

The consequence is a semantic change, and it is deliberate: **1x4-Acc and
1x4-Acc-Kc compute `C = A*B`, not `C += A*B`.** The caller must not pre-zero C.
Every other MaxMulSK SME kernel still accumulates, so the repository now holds
two contracts; both headers state which.

What changed, in full:

| file | change |
|---|---|
| `sme/sme-1x4-acc.cpp`, `sme/sme-1x4-acc-kc.cpp` | `store_za`: dropped `svld1_f32` + `svadd_f32_x`, store `result` directly |
| same, edge-tile path | dropped the `C_scratch` pre-zeroing (now redundant); scatter assigns instead of accumulating |
| `bench/maxmulsk_sme_adapter.cpp` | `acc_store_za` mirrored; `acc_compute` edge path mirrored; new `overwrites_C()` |
| `bench/benchmark_maxmul_vs_kleidiai.cpp` | the C memset is charged only to kernels that need it |

**A trap worth recording.** Rewriting the edge-tile scatter from `+=` to a plain
copy made the compiler recognise it as a memcpy and emit `__arm_sc_memcpy`, the
streaming-mode variant that has no implementation — a link failure. This is the
same class as TODO.md's BUG-5 (`std::fill` → `__arm_sc_memset`), now via memcpy.
Both scatters are predicated SVE loops rather than scalar loops for this reason.

**Correctness.** C is prefilled with a 12345.0 sentinel rather than zero before
every validation run, so a forgotten accumulate cannot hide: it would leave the
sentinel in the result. 15 shapes standalone (square, large-K, all-tails,
tiny-odd) and 376 rows in the benchmark, all bit-identical to Accelerate, 0
failures.

#### Hot micro-kernel, per invocation

| Kc | 1x4sym (RMW) | 1x4Acc (RMW, §0.9) | **1x4Acc (overwrite)** | saving | KleidiAI 2VL |
|---:|---:|---:|---:|---:|---:|
| 128 | 488.6 ns | 486.3 ns | **322.3 ns** | 166.3 ns | 173.8 ns |
| 256 | 619.4 ns | 617.6 ns | **455.0 ns** | 164.4 ns | 305.4 ns |
| 512 | 878.5 ns | 878.4 ns | **714.6 ns** | 163.9 ns | 565.7 ns |
| 1024 | 1400.6 ns | 1400.6 ns | **1236.8 ns** | 163.9 ns | 1086.7 ns |
| 2048 | 2444.8 ns | 2445.0 ns | **2282.6 ns** | 162.1 ns | 2132.8 ns |

A flat **~164 ns saved at every Kc**, exactly as the isolated store measurement
predicted (~166 ns). The fixed cost falls from 356 ns to ~192 ns. Against
KleidiAI on equal terms the ratio moves from 0.36× to **0.54×** at Kc=128 and
from 0.87× to **0.93×** at Kc=2048. KleidiAI's remaining 43.6 ns advantage is
still unexplained by anything measured here.

#### Prepacked full GEMM

| Shape (MxKxN) | 1x4sym | 1x4Acc §0.9 | **1x4Acc now** | Acc/sym | KleidiAI |
|---|---:|---:|---:|---:|---:|
| 256x256x256 | 747.0 | 746.4 | **1162.1** | **1.556** | 1731.8 |
| 512x512x512 | 1155.0 | 1156.0 | **1472.6** | **1.275** | 1864.1 |
| 1024x1024x1024 | 1470.1 | 1456.4 | **1692.2** | **1.151** | 1926.6 |
| 2048x2048x2048 | 1506.1 | 1481.9 | **1715.1** | **1.139** | 1832.5 |
| 4096x4096x4096 | 1375.3 | 928.5 | **924.5** | **0.672** | 1707.1 |
| 64x8192x512 | 1167.6 | 1290.0 | **1375.6** | **1.178** | 1435.6 |
| 64x16384x512 | 1058.2 | 1090.0 | **1099.5** | **1.039** | 1288.6 |
| 64x32768x512 | 1053.6 | 829.3 | **853.9** | **0.810** | 968.6 |
| 128x8192x512 | 1295.4 | 1510.3 | **1544.9** | **1.193** | 1601.6 |
| 128x16384x512 | 1271.6 | 1065.8 | **1046.6** | **0.823** | 1292.9 |
| 128x32768x512 | 1267.7 | 954.5 | **965.3** | **0.761** | 921.2 |

#### End-to-end

| Shape (MxKxN) | 1x4sym | 1x4Acc §0.9 | **1x4Acc now** | Acc/sym | KleidiAI |
|---|---:|---:|---:|---:|---:|
| 256x256x256 | 644.2 | 644.8 | **938.6** | **1.457** | 1189.5 |
| 512x512x512 | 1033.3 | 1034.4 | **1283.1** | **1.242** | 1644.3 |
| 1024x1024x1024 | 1248.6 | 1229.8 | **1526.6** | **1.223** | 1623.0 |
| 2048x2048x2048 | 1218.0 | 1134.9 | **1346.0** | **1.105** | 1615.4 |
| 4096x4096x4096 | 1334.1 | 1068.6 | **1222.5** | **0.916** | 1630.2 |
| 64x8192x512 | 797.4 | 646.2 | **682.8** | **0.856** | 533.3 |
| 64x16384x512 | 700.9 | 718.7 | **737.4** | **1.052** | 503.9 |
| 64x32768x512 | 707.5 | 567.4 | **658.8** | **0.931** | 443.1 |
| 128x8192x512 | 904.3 | 760.5 | **756.2** | **0.836** | 748.9 |
| 128x16384x512 | 866.3 | 871.7 | **944.9** | **1.091** | 660.5 |
| 128x32768x512 | 865.5 | 688.8 | **692.2** | **0.800** | 518.2 |

#### Reading

Where the full-K packing does not hurt, 1x4-Acc now **beats 1x4-sym outright**:
+56% at 256³, +28% at 512³, +15% at 1024³ and 2048³ prepacked, and +18–19% on the
K=8192 shapes. End-to-end the same, +46% at 256³ down to +11% at 2048³. The small
shapes gain most, which is expected — that is where the per-invocation fixed cost
was the largest share.

The losses that remain are the ones §0.9 already attributed to full-K packing and
they are unchanged by this: 4096³ (0.672 prepacked) and K ≥ 16384. Removing the
read-modify-write does nothing for a working-set problem, and the numbers agree —
those rows moved by −0.4% to +3%.

Net: two independent costs were stacked in §0.9's result. One of them, the
read-modify-write, is now gone and worth 15–56% depending on shape. The other,
the memory traffic that full-K packing forces, is untouched and is what still
keeps 1x4-Acc behind at 4096³ and at large K.

### 0.11 Row-major ZA→C store: one 4-vector store per output row (2026-09-05)

**Measured:** 2026-09-05 · **CURRENT** · supersedes the Acc numbers of §0.10

§0.10 left the ZA→C writeback as the whole of 1x4-Acc's fixed per-invocation
cost, ~192 ns. This restructures how that store walks the output tile.

#### The geometry

The four ZA tiles stack horizontally, so the 16×64 output tile looks like:

```
          cols 0-15   16-31      32-47      48-63
row 0    [ ZA0[0] ] [ ZA1[0] ] [ ZA2[0] ] [ ZA3[0] ]   <- 64 CONTIGUOUS floats
row 1    [ ZA0[1] ] [ ZA1[1] ] [ ZA2[1] ] [ ZA3[1] ]
  ...
row 15   [ ZA0[15]] [ ZA1[15]] [ ZA2[15]] [ ZA3[15]]
```

The old loop was tile-major — all 16 rows of ZA0, then ZA1, and so on — so
consecutive stores were `wide_of_C` apart: 64 separate narrow stores.

Walking it row-major instead, the four tiles' slice `i` form **one contiguous
4·SVL row**, which is exactly what `svst1_f32_x4` wants:

```c
for (size_t i = 0; i < SVL; i++) {
    svfloat32x4_t row = svcreate4_f32(
        svread_hor_za32_f32_m(inactive, pg, 0, i), ... 1 ... 2 ... 3 ...);
    svst1_f32_x4(pn, C + i * wide_of_C, row);
}
```

Clang emits exactly the intended form — 4 `MOVA` plus one 4-register `ST1W`:

```asm
mov  z0.s, p0/m, za0h.s[w12, 0]
mov  z1.s, p0/m, za1h.s[w12, 0]
mov  z2.s, p0/m, za2h.s[w12, 0]
mov  z3.s, p0/m, za3h.s[w12, 0]
st1w { z0.s - z3.s }, pn8, [x0]
```

80 instructions per tile instead of 128, and 16 wide stores instead of 64 narrow
strided ones. Same 4 KiB written.

#### Two things that do NOT work, both checked rather than assumed

**Grouping four rows of one tile.** `svread_hor_za32_f32_vg4(TILE, i)` returns
slices `i..i+3` of one tile in a register quad, which looks like the natural
pairing for a 4-vector store. It is not: those four rows sit `wide_of_C` apart in
C, and SME2 has no strided multi-vector store. Only the transposed grouping —
four tiles, same slice — lands on contiguous memory.

**Collapsing the four ZA reads into one.** `svread_za32_f32_vg1x4(s)` returns a
register quad directly, which would remove the four `MOVA`s. Its slice mapping
was measured by filling ZA with a tile/row-identifying pattern and reading it
back:

```
s= 0 -> t0/r0  t0/r4  t0/r8  t0/r12
s= 1 -> t1/r0  t1/r4  t1/r8  t1/r12
s= 4 -> t0/r1  t0/r5  t0/r9  t0/r13
```

So `vg1x4(s)` gives four rows of tile `s mod 4`, strided by 4 — again one tile,
not four. There is no multi-vector ZA read producing
`{ZA0[i], ZA1[i], ZA2[i], ZA3[i]}`, so the four `MOVA`s stay. An earlier guess
that the mapping was `4i+n` was wrong and produced incorrect results; it was
caught by comparing against the tile-major store before any of this was believed.

#### Results

Hot micro-kernel, per invocation — a flat **~140 ns saved at every Kc**; the
fixed cost falls from ~192 ns to ~52 ns, against KleidiAI's 43.6 ns:

| Kc | tile-major ns | **row-major ns** | saved | GFLOP/s | KleidiAI |
|---:|---:|---:|---:|---:|---:|
| 128 | 322.3 | **182.7** | 139.6 | 1434.5 | 1502.0 |
| 256 | 455.0 | **312.9** | 142.1 | 1675.7 | 1720.5 |
| 512 | 714.6 | **574.4** | 140.3 | 1825.6 | 1856.4 |
| 1024 | 1236.8 | **1096.4** | 140.3 | 1912.7 | 1926.5 |
| 2048 | 2282.6 | **2142.5** | 140.2 | 1957.7 | 1965.1 |

The kernels are now level with KleidiAI in the hot path: 0.955× at Kc=128 (was
0.54×) and 0.996× at Kc=2048 (was 0.93×).

Prepacked full GEMM:

| Shape | tile-major | **row-major** | change | 1x4sym | KleidiAI |
|---|---:|---:|---:|---:|---:|
| 256x256x256 | 1162.1 | **1660.5** | +42.9% | 746.4 | 1731.8 |
| 512x512x512 | 1472.6 | **1834.4** | +24.6% | 1156.0 | 1864.1 |
| 1024x1024x1024 | 1692.2 | **1911.3** | +12.9% | 1467.1 | 1923.2 |
| 2048x2048x2048 | 1715.1 | **1827.1** | +6.5% | 1497.1 | 1822.7 |
| 4096x4096x4096 | 924.5 | **1011.8** | +9.4% | 1478.1 | 1744.4 |
| 64x8192x512 | 1375.6 | **1381.5** | +0.4% | 1163.7 | 1363.8 |
| 64x32768x512 | 853.9 | **856.4** | +0.3% | 1055.8 | 939.8 |
| 128x8192x512 | 1544.9 | **1564.3** | +1.3% | 1294.8 | 1623.2 |
| 128x32768x512 | 965.3 | **972.0** | +0.7% | 1270.9 | 923.3 |

End-to-end:

| Shape | tile-major | **row-major** | change | 1x4sym | KleidiAI |
|---|---:|---:|---:|---:|---:|
| 256x256x256 | 938.6 | **1246.6** | +32.8% | 644.8 | 1189.5 |
| 512x512x512 | 1283.1 | **1542.4** | +20.2% | 1033.4 | 1643.5 |
| 1024x1024x1024 | 1526.6 | **1688.5** | +10.6% | 1241.7 | 1635.1 |
| 2048x2048x2048 | 1346.0 | **1419.8** | +5.5% | 1294.0 | 1594.6 |
| 4096x4096x4096 | 1222.5 | **1154.8** | -5.5% | 1343.8 | 1647.8 |
| 64x8192x512 | 682.8 | **693.9** | +1.6% | 794.9 | 524.1 |
| 64x32768x512 | 658.8 | **627.8** | -4.7% | 700.4 | 441.0 |
| 128x8192x512 | 756.2 | **745.5** | -1.4% | 899.2 | 752.8 |
| 128x32768x512 | 692.2 | **706.7** | +2.1% | 842.1 | 555.3 |

1x4-Acc now leads every MaxMulSK kernel on the square shapes by a wide margin
(+122% over 1x4-sym at 256³ prepacked, +30% at 1024³) and **passes KleidiAI at
2048³ prepacked** (1827 vs 1823). The large-K and 4096³ rows barely move, as
expected: those are limited by the full-K packing working set (§0.9), not by the
store. 376 measured rows, 0 correctness failures, verified with a sentinel-filled
C so a missed overwrite cannot hide.

#### Attribution

The optimisation site was identified by the repository author, who asked whether
the store loop could be widened to 4-vector reads and writes. That first
formulation — four rows of one tile — was analysed together and rejected: the
rows are not contiguous in C and SME2 has no strided multi-vector store. Claude
(Anthropic) proposed the transposed formulation used here, grouping the four
tiles' same slice into one contiguous row, and implemented it along with the
`vg1x4` mapping measurement that ruled out collapsing the reads.

### 0.13 pack_A: SVE butterfly → ZA transpose (2026-09-05)

**Measured:** 2026-09-05 · superseded in absolute terms by §0.14, mechanism stands

The repository already had a ZA-based `pack_A` in `sme-4x1-zapack.cpp`: write 16
rows of A into ZA tile 0 as horizontal slices with `svld1_hor_za32`, read them
back as vertical slices with `svst1_ver_za32`, and the transpose is free. 1x4-Acc
inherited 1x4-sym's SVE butterfly instead — 16 loads, a zip/uzp network, 16
stores. This swaps in the ZA form, keeping the output layout byte-identical.

It is sound here for the same reason the ZA-lifetime work is: the driver packs A
and B for a block *before* it zeroes ZA and starts accumulating, and never packs
while an accumulation is in flight. `pack_A_streaming` is therefore declared
`__arm_out("za")`. One subtlety: predicated-off horizontal loads **merge**, so a
short m-tail would otherwise read whatever the previous block left in ZA; the
tail path zeroes ZA once to prevent that.

#### End-to-end, with the control kernels shown alongside

The two runs compared here are separate sessions, so unchanged kernels drift too.
That drift is the yardstick: a change is only real if it clears it.

| Shape | butterfly | **ZA pack** | change | control drift (1x4sym / 4x1) |
|---|---:|---:|---:|---|
| 256x256x256 | 1245 | **1396** | +12.1% | +1.8% / +0.1% |
| 512x512x512 | 1522 | **1660** | +9.0% | +0.0% / +0.0% |
| 1024x1024x1024 | 1672 | **1773** | +6.0% | -1.4% / +1.8% |
| 2048x2048x2048 | 1388 | **1422** | +2.4% | +5.0% / +2.8% |
| 4096x4096x4096 | 1100 | **1186** | +7.8% | +7.9% / +6.5% |
| 64x8192x512 | 685 | **748** | +9.2% | +4.9% / +1.4% |
| 64x16384x512 | 781 | **823** | +5.4% | +1.6% / -0.5% |
| 64x32768x512 | 643 | **671** | +4.3% | +1.9% / -1.3% |
| 128x8192x512 | 683 | **915** | +34.0% | +4.8% / +2.3% |
| 128x16384x512 | 884 | **999** | +13.0% | +2.4% / -0.8% |
| 128x32768x512 | 662 | **736** | +11.1% | +6.0% / +4.0% |

Net of control drift the gain is roughly **+6 to +12%** on the shapes where
packing is a large share of the work. The 2048³ and 4096³ rows move no more than
their controls and cannot be claimed.

#### Prepacked — the corroboration

| Shape | butterfly | ZA pack | change |
|---|---:|---:|---:|
| 256x256x256 | 1657 | 1685 | +1.7% |
| 512x512x512 | 1806 | 1834 | +1.6% |
| 1024x1024x1024 | 1908 | 1907 | -0.0% |
| 2048x2048x2048 | 1795 | 1810 | +0.9% |
| 4096x4096x4096 | 987 | 1048 | +6.2% |
| 64x8192x512 | 1400 | 1360 | -2.8% |
| 64x16384x512 | 1107 | 1119 | +1.1% |
| 64x32768x512 | 845 | 850 | +0.6% |
| 128x8192x512 | 1579 | 1554 | -1.6% |
| 128x16384x512 | 1077 | 1080 | +0.3% |
| 128x32768x512 | 963 | 965 | +0.2% |

Essentially zero, which is exactly right: in prepacked mode the packing happens
outside the timed region, so a faster pack_A *should* be invisible. The
end-to-end gain and the prepacked null result together are stronger evidence
than either alone.

#### A correction worth recording

The first version of this measurement was wrong and was published in this
session before being caught. The build that was supposed to contain the ZA pack
had actually **failed** — the adapter's `acc_pack_A` forwarder still lacked
`__arm_out("za")` and would not compile — but the benchmark ran anyway, against
the stale binary, and the differences that appeared were pure run-to-run noise
attributed to a change that was not in the binary.

The cause was a shell pipeline: `cmake --build … | grep -E "error|Built target"`
returns *grep's* exit status, so the failure was invisible and the script carried
on. `grand_benchmark.sh` checks the build's exit status and aborts; ad-hoc runs
now do the same, and every archived result directory carries its `build.log` as
the receipt.

---

### 0.14 Current results — median of three full runs (2026-09-06)

**Measured:** 2026-09-06 · **CURRENT** · supersedes the absolute figures in
§0.9–§0.13 for the Acc kernels

Everything above is a single run per data point. That turned out not to be good
enough.

#### The measurement-noise floor

Re-running the identical binary produced end-to-end figures differing by up to
14% on short shapes, while the *within-run* sample spread was only 1–5.6%. The
noise therefore sits at a longer timescale than sample-to-sample jitter: it is
constant within a measurement batch and changes between batches.

It is **not** thermal. Thermal drift would be directional — everything slower
later in a run — and would show up as drift across the shape sequence. Instead,
in the same run one kernel gained 14% while two others lost 5%, and a fourth did
not move at all. The remaining candidates are core placement (macOS offers QoS
biasing but no thread pinning), physical page placement of the freshly-allocated
packing buffers, and background load. Which of those dominates has **not** been
established.

The practical consequence: this section reports the **median of three full
runs**, and `maxmul_vs_kleidiai_avg.csv` carries a `spread_pct` per row. Across
those three runs the median row moves ~1%, the 90th percentile 4–8%, the worst
12.6%. **Differences below roughly 5% are not real.**

#### End-to-end (unpacked A, B in; C out; packing timed)

| Shape | 1x4Acc | best MaxMulSK | (which) | KAI full-pack | KAI panel | Accelerate |
|---|---:|---:|---:|---:|---:|---:|
| 256x256x256 | 1396 | 1398 | 1x4AccKc | 1276 | 1274 | **1786** |
| 512x512x512 | 1660 | 1660 | 1x4AccKc | 1710 | 1709 | **1829** |
| 1024x1024x1024 | **1773** | **1773** | 1x4Acc | 1620 | 1737 | 1683 |
| 2048x2048x2048 | 1422 | 1481 | 1x4AccKc | 1603 | 1573 | **1642** |
| 4096x4096x4096 | 1186 | 1326 | 1x4sym | **1636** | 1474 | 1541 |
| 64x8192x512 | 748 | 806 | 1x4sym | 525 | 855 | **966** |
| 64x16384x512 | 823 | 823 | 1x4Acc | 505 | 850 | **856** |
| 64x32768x512 | 671 | 724 | 4x1ZP | 439 | 850 | **855** |
| 128x8192x512 | 915 | 928 | 2x2 | 749 | 1075 | **1197** |
| 128x16384x512 | 999 | 1006 | 1x4AccKc | 684 | 1067 | **1138** |
| 128x32768x512 | 736 | 901 | 4x1ZP | 553 | 1066 | **1137** |

#### Prepacked full GEMM (packing hoisted out)

| Shape | 1x4Acc | best MaxMulSK | KleidiAI 2VL | KleidiAI 8VS |
|---|---:|---:|---:|---:|
| 256x256x256 | 1685 | 1685 | 1732 | **1969** |
| 512x512x512 | 1834 | 1834 | 1864 | **1982** |
| 1024x1024x1024 | 1907 | 1908 | 1924 | **1931** |
| 2048x2048x2048 | **1810** | **1810** | 1799 | 1748 |
| 4096x4096x4096 | 1048 | 1499 | **1749** | 1740 |
| 64x8192x512 | 1360 | 1360 | 1369 | **1415** |
| 64x16384x512 | 1119 | 1119 | **1311** | 1255 |
| 64x32768x512 | 850 | **1060** | 939 | 941 |
| 128x8192x512 | 1554 | 1558 | **1591** | 1579 |
| 128x16384x512 | 1080 | 1279 | 1309 | **1320** |
| 128x32768x512 | 965 | **1275** | 923 | 868 |

#### Reading

- **1024³ end-to-end is the standout: 1773 GFLOP/s, ahead of Accelerate (1683)
  and KleidiAI's own full-pack flow (1620), and level with its panel-blocked
  flow (1737 — a 2.1 % gap, inside this document's own noise floor).** That
  is the first shape where the whole chain — ZA-resident K accumulation,
  overwriting store, row-major drain, ZA-transpose packing — lands together.
- **Prepacked at 2048³ we are level with KleidiAI** — 1810 vs 1799 is 0.6 %,
  inside the noise floor, so a tie and not a lead. **On the large-K shapes we do
  lead**: 1275 vs 923 at 128×32768×512 (1.38×). That large-K figure is
  **1×4-sym's**, not 1×4-Acc's; 1×4-Acc reaches only 965 there, so the two
  clauses are about different kernels.
- **Accelerate still wins most shapes end-to-end**, decisively at 256³–512³ and
  at 4096³. Its advantage is largest exactly where our per-call fixed costs and
  our packing footprint hurt most.
- **The full-K packing regression is unchanged.** 1x4-Acc still falls to 1048
  prepacked at 4096³ against 4x1's 1499. Neither the store work nor the ZA pack
  addresses it; it remains the open problem from §0.9.

### 0.15 Energy per implementation (2026-09-06, median of three runs)

**Measured:** 2026-09-06 · **CURRENT** — supersedes §2, which covered three
kernels in 2026-04 and predates the whole 1×4-Acc line.

`bench/energy_bench.sh` gives each (implementation, shape) pair its own
`powermetrics` session wrapped tightly around its workload, so every sample
belongs to that workload and no timestamp alignment is needed. Energy is average
power × the workload's **own measured wall time**, not the sampler's capture
length. Each point is held busy for 3 s; the whole sweep was run three times and
the tables below are the median, with `spread_pct` over GFLOP/s carried as the
reliability estimate. Raw data: `bench/results/2026-09-06/energy.csv` and
`energy_run{1,2,3}/`.

#### J/GFLOP — lower is better

| Shape | 1×4-Acc | 1×4-Acc-Kc | KAI full-pack | KAI panel | Accelerate | 4×1 | 1×4-sym |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1024³ | 0.00606 | **0.00600** | 0.00605 | 0.00591 | 0.00630 | 0.00699 | 0.00700 |
| 2048³ | 0.00640 | 0.00639 | **0.00604** | 0.00617 | 0.00619 | 0.00682 | 0.00680 |
| 4096³ | 0.00687 | 0.00689 | 0.00604 | **0.00566** | 0.00629 | 0.00686 | 0.00672 |
| 128×32768×512 | 0.00897 | 0.00902 | 0.01070 | **0.00769** | 0.00790 | 0.00826 | 0.00827 |

#### What the four improvements bought, on the energy axis

1×4-sym is 1×4-Acc's starting point and both were measured in the *same* run, on
the same machine state, so this is a clean before/after for the chain in §0.9-A:

| Shape | GFLOP/s sym → Acc | Δ | J/GFLOP sym → Acc | Δ | worst spread |
|---|---|---:|---|---:|---:|
| 1024³ | 1254 → 1720 | +37.2 % | 0.00700 → 0.00606 | **−13.4 %** | 3.4 % |
| 2048³ | 1241 → 1557 | +25.5 % | 0.00680 → 0.00640 | **−5.8 %** | 11.0 % |
| 4096³ | 1350 → 1230 | −8.9 % | 0.00672 → 0.00687 | +2.2 % | 3.9 % |
| 128×32768×512 | 870 → 724 | −16.8 % | 0.00827 → 0.00897 | +8.4 % | 2.3 % |

The energy axis tells the same story as the throughput axis, and that agreement
is the useful part: at ≤2048³ the chain won on both, at 4096³ and large K it lost
on both. There is no shape where it bought speed at the cost of energy or the
reverse. That is what you would expect if the work is the same work done with
fewer instructions and less traffic, rather than something traded against power —
the 2048³ energy gain (−5.8 %) is smaller than its throughput gain (+25.5 %)
because the kernel also draws more power while it runs (8.31 W → 9.83 W), so the
win shows up as a shorter run rather than a cheaper one.

#### Against the outside libraries

- **1024³ — level.** 0.00600–0.00606 against KleidiAI's 0.00591–0.00605, and
  ahead of Accelerate's 0.00630. The 2.5 % margin over KleidiAI panel-blocked
  sits under a 3.4–5.6 % spread, so it is a tie, not a win.
- **2048³ — close, slightly behind.** 0.00640 against 0.00604–0.00619. These
  rows carry the worst spread in the table (8.5–11.0 %), so the gap is marginal.
- **4096³ and large-K — clearly behind.** 0.00687 against KleidiAI
  panel-blocked's 0.00566 is 21 % at a 3–4 % spread, so it is real. This is the
  full-K packing problem (§0.9-A; TODO.md "OPEN: 1×4-Acc's full-K packing")
  showing up on the energy axis rather than the throughput one.
- Among our own kernels, 4×1 keeps its efficiency lead **only** at
  128×32768×512 (0.00826). At 1024³ and 2048³ the Acc kernels beat it by 14 %
  and 6 %.

#### Two caveats

- **The P-cluster and E-cluster columns read 0.** `powermetrics --samplers
  cpu_power` on this machine does not emit the per-cluster power lines the
  parser expects — only the combined CPU figure. Tracked in TODO.md under
  "MINOR: profile_power.sh P/E-cluster regex". J/GFLOP is derived from the
  combined figure, which is the meaningful one, so the tables above are
  unaffected.
- **Do not read `bench/results/2026-09-05/energy.csv` as a before/after against
  this run.** That single run is kept for the record but is
  background-contaminated: the two kernels that did **not** change between the
  two dates, 4×1 and 1×4-sym, moved on their own by 4 % and 15 % in J/GFLOP at
  4096³ and by 31 % and 38 % at 128×32768×512, and every implementation's
  average power at that shape fell by 20–28 %. The control group did not hold,
  so no code-attributable conclusion can be drawn across the two dates. The
  before/after table above avoids this entirely by staying inside one run.

---

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

**Measured:** 2026-04-26 · **SUPERSEDED by §0.15** — kept for the progression.
These figures cover only three kernels and predate both the 1×4-sym / ZAPack
results in §0.1 and the whole 1×4-Acc line. §0.15 re-measures seven
implementations, including KleidiAI and Accelerate, as the median of three runs.

Measurement tool: `powermetrics --samplers cpu_power,thermal`, 100 ms sampling interval, captured by `scripts/profile_power.sh` (combined CPU + GPU + ANE basis). All measurements at 2048³ × 100 iterations (~1.7 TFLOPs of work per run), single-thread, MacBook M4, no thermal pressure (Nominal across all runs).

### 2.1 Our SME kernels (fresh, 2026-04-26)

| Kernel | GFLOPS | Runtime (s) | Avg power (W) | Energy (J) | J / GFLOP | GFLOPS / W |
|---|---:|---:|---:|---:|---:|---:|
| **SME 4×1** | **1173.7** | 1.84 | 6.07 | 11.18 | **0.0065** | **153.6** |
| SME 2×2 | 1066.1 | 2.11 | 5.44 | 11.47 | 0.0067 | 149.8 |
| SME 1×4 (x1 interleaved) | 1037.7 | 1.71 | 7.17 | 12.24 | 0.0071 | 140.4 |

4×1 is the most energy-efficient kernel in absolute terms (lowest J/GFLOP, highest GFLOPS/W). 2×2 draws the least power but takes longer; 1×4 draws the most power *and* runs slowest, the worst of both axes.

> **Superseded (2026-09-06).** Of the kernels measured here 4×1 is still the
> best, but the kernel set has since grown. §0.15 measures 1×4-Acc at
> 0.00606 J/GFLOP against 4×1's 0.00699 at 1024³ — 4×1 now leads only at
> 128×32768×512.

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

The remaining gap to theoretical is best closed at a higher level — a scheduling layer that assigns larger tiles to P-cores and smaller tiles to E-cores, rather than relying on dynamic stealing after the fact.

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
