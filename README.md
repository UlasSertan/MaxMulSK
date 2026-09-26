# MaxMulSK

A single-threaded FP32 GEMM library for Apple M4, written against Arm SME/SME2.

This is not just a micro-kernel. MaxMulSK handles the whole execution path. It
packs A and B into kernel-friendly layouts, blocks the work over M, K and N for
cache, runs the ZA outer-product compute, deals with partial tiles at the edges,
and writes ZA back to C. You pass three pointers and three sizes.

![C++20](https://img.shields.io/badge/C%2B%2B-20-blue) ![Arm SME2](https://img.shields.io/badge/ISA-Arm%20SME%2FSME2-lightgrey) ![License](https://img.shields.io/badge/license-Apache--2.0-green)

## Results

- **~2.0 TFLOP/s** in the raw SME micro-kernel. That is 2008 to 2010 GFLOP/s of
  pure compute, isolated by fitting `t(Kc) = a + b·Kc` and reading the slope.
- **v5, the current development kernel, against every FP32 GEMM tested on Apple
  M4** (single thread, 35 shapes: squares, LLM-shaped, DeepSeek-V3 and LLaMA;
  geometric mean of paired ratios, 27 September 2026):
  - **1.76× tuned OpenBLAS.** Ahead on 34 of 35 shapes, tied on 256³.
  - **1.12× Apple Accelerate.** Ahead by more than 5% on 21 shapes, behind on
    256³ (0.80×).
  - **1.01× MpGEMM.** Ahead by more than 5% on 13 shapes (large M, long-K LLM
    shapes, wide DeepSeek-V3 shapes), behind on 5: 256³ (0.86×), two
    DeepSeek-V3 shapes with N = 2112 (0.91× and 0.94×) and the two long-K LLaMA
    shapes (0.92× and 0.93×).
  - **1.11× the published v4c path**, and never behind it by more than 2.2%.
- **End to end means end to end.** Allocation, packing, blocking and writeback
  are inside every timed call, for every library.

> **v5 is in development and its code is not in this repository yet.** The
> numbers above are measured, but the published kernels are v3
> (`sme/v3/`) and v4c (`sme/v4/`). v5 is bit-identical to v4 in its output; what
> changed is how it packs and how it writes C. It will be documented here when
> its code is published.

<img src="docs/img/grand_square.svg" width="100%" alt="Square GEMM: MaxMulSK v5 vs MpGEMM, Accelerate and OpenBLAS">

<img src="docs/img/grand_llm.svg" width="100%" alt="LLM-shaped GEMM: MaxMulSK v5 vs MpGEMM, Accelerate and OpenBLAS">

<img src="docs/img/grand_deepseek_m64.svg" width="100%" alt="DeepSeek-V3 shapes, M = 64: MaxMulSK v5 vs MpGEMM, Accelerate and OpenBLAS">

<img src="docs/img/grand_deepseek_m128.svg" width="100%" alt="DeepSeek-V3 shapes, M = 128: MaxMulSK v5 vs MpGEMM, Accelerate and OpenBLAS">

<img src="docs/img/grand_deepseek_m4096.svg" width="100%" alt="DeepSeek-V3 shapes, M = 4096: MaxMulSK v5 vs MpGEMM, Accelerate and OpenBLAS">

<img src="docs/img/grand_llama.svg" width="100%" alt="LLaMA shapes: MaxMulSK v5 vs MpGEMM, Accelerate and OpenBLAS">

<sub>Apple M4, macOS 26.6.2, AC power, single thread, FP32, row-major C = A·B.
One session on 27 September 2026, two runs in separate processes; each chart
point is the mean of the two runs' medians. All six participants (v5, v4c, v3,
MpGEMM, OpenBLAS, Accelerate) ran in the same rounds with the call order rotated,
and every ratio quoted above is paired within a round. Before timing, every
shape was checked: v5 bit-identical to v4 with the same plan, and every library
within tolerance of an FP64 reference. Accelerate was pinned to one thread with
`BLASSetThreading`; OpenBLAS is the build that reaches its own SME kernels
(`OPENBLAS_DIRECT_LIMIT=1792`, one thread). Absolute GFLOP/s in this session
came out 5 to 10% below earlier sessions for every library alike, most likely
thermal, so read the ratios rather than the absolute numbers. The y axes start
near the data, not at zero, as stated on each chart.</sub>

<details>
<summary><b>Earlier sessions</b>: 9 September (v3 vs Accelerate and OpenBLAS) and 13 September (v4c vs Accelerate)</summary>

<br>

In the 9 September session the published v3 path reached **1.40 to 1.77
TFLOP/s** end to end on squares from 256³ to 4096³, was faster than tuned
OpenBLAS on every workload tested (1.01× to 1.31× on squares, 1.36× to 2.23× on
LLM shapes), and sat in the same band as Accelerate: ahead at 1024³ and 4096³ by
1.05× and 1.02×, behind by 5 to 17% elsewhere.

<img src="docs/img/headline_square.svg" width="100%" alt="Square GEMM: MaxMulSK vs Apple Accelerate vs tuned OpenBLAS">

<img src="docs/img/headline_llm.svg" width="100%" alt="LLM-shaped GEMM: MaxMulSK vs Apple Accelerate vs tuned OpenBLAS">

<sub>Apple M4, single thread, FP32. Median of three runs with cooldowns between
them; the spread between runs was 0.1 to 2.1%. Both charts crop the y axis to
the data, at 1200 for squares and 400 for the LLM shapes, because starting at
zero squeezes everything into an unreadable band. They use points rather than
bars on purpose. A point shows its value by position, so cropping the axis is
fine. A bar shows its value by length, so cropping it would exaggerate every
difference. Every marker is labelled with its number. Raw data is in
[bench/results/2026-09-09/](bench/results/2026-09-09/).</sub>

<img src="docs/img/v4c_square.svg" width="100%" alt="Square GEMM: MaxMulSK v4c path vs Apple Accelerate, 13 September 2026">

<img src="docs/img/v4c_llm.svg" width="100%" alt="LLM-shaped GEMM: MaxMulSK v4c path vs Apple Accelerate, 13 September 2026">

<img src="docs/img/v4c_deepseek_m64.svg" width="100%" alt="DeepSeek-V3 GEMM shapes, M=64: MaxMulSK v4c path vs Apple Accelerate">

<img src="docs/img/v4c_deepseek_m128.svg" width="100%" alt="DeepSeek-V3 GEMM shapes, M=128: MaxMulSK v4c path vs Apple Accelerate">

<sub>The newer v4c path (`sme/v4/`) against Apple Accelerate on 13 September
2026: one session, same binary, call order rotated, median of 5 runs, AC power.
OpenBLAS was not part of that session. The DeepSeek-V3 shapes are the MpGEMM
paper's table; the 6 LLaMA shapes from the same run (N = 256) are not charted
here: v4c is behind Accelerate on them by 3 to 5%. Raw data:
[bench/results/2026-09-13/v4c3.csv](bench/results/2026-09-13/v4c3.csv).</sub>

</details>

---

> Hey, it's Ulaş. If you're wondering why this project exists in the first place
> when Accelerate and OpenBLAS already do, the answer is pretty simple: I like
> building things myself, I like using my own code, and I genuinely enjoy
> optimization work. So this project is a direct result of those reasons.
>
> It turned out much better than I expected. If you're looking for an
> open source FP32 GEMM library for Apple M4 and just want to call a fast matrix
> multiplication routine, I honestly believe MaxMulSK is now one of the best
> options out there.

---

## Benchmarking and comparison

| | Shape (M×N×K) | v5 | MpGEMM | Accelerate | OpenBLAS | v5 vs MpGEMM | vs Accelerate | vs OpenBLAS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| square | 256×256×256 | 1244 | 1448 | **1601** | 1270 | 0.86× | 0.80× | 0.98× |
|  | 512×512×512 | 1580 | **1648** | 1631 | 1503 | 0.96× | 0.97× | **1.05×** |
|  | 1024×1024×1024 | 1618 | **1680** | 1530 | 1362 | 0.97× | **1.07×** | **1.18×** |
|  | 2048×2048×2048 | 1494 | 1404 | **1533** | 1189 | **1.06×** | 0.97× | **1.26×** |
|  | 4096×4096×4096 | **1518** | 1382 | 1464 | 1158 | **1.10×** | 1.04× | **1.31×** |
| llm | 64×512×8192 | 995 | **1003** | 914 | 680 | 0.99× | **1.09×** | **1.46×** |
|  | 64×512×16384 | **906** | 857 | 808 | 450 | **1.06×** | **1.12×** | **2.00×** |
|  | 64×512×32768 | **891** | 856 | 801 | 458 | 1.04× | **1.11×** | **1.94×** |
|  | 128×512×8192 | 1152 | **1162** | 1094 | 609 | 0.99× | **1.05×** | **1.89×** |
|  | 128×512×16384 | 1123 | **1126** | 1042 | 462 | 1.00× | **1.08×** | **2.42×** |
|  | 128×512×32768 | **1135** | 1131 | 1062 | 481 | 1.00× | **1.07×** | **2.36×** |
| deepseek | 64×2112×7168 | 925 | **1022** | 962 | 483 | 0.91× | 0.96× | **1.92×** |
|  | 64×24576×1536 | **896** | 838 | 588 | 396 | **1.07×** | **1.53×** | **2.26×** |
|  | 64×32768×512 | **949** | 849 | 593 | 365 | **1.12×** | **1.60×** | **2.60×** |
|  | 64×7168×16384 | **921** | 866 | 624 | 381 | **1.06×** | **1.48×** | **2.42×** |
|  | 64×4096×7168 | 895 | **933** | 588 | 412 | 0.96× | **1.52×** | **2.16×** |
|  | 64×7168×2048 | **900** | 885 | 643 | 411 | 1.02× | **1.40×** | **2.18×** |
|  | 128×2112×7168 | 1200 | **1273** | 1229 | 497 | 0.94× | 0.98× | **2.42×** |
|  | 128×24576×1536 | **1176** | 1100 | 876 | 406 | **1.07×** | **1.35×** | **2.89×** |
|  | 128×32768×512 | **1232** | 1120 | 830 | 370 | **1.10×** | **1.48×** | **3.33×** |
|  | 128×7168×16384 | **1198** | 1078 | 815 | 593 | **1.11×** | **1.47×** | **2.01×** |
|  | 128×4096×7168 | 1196 | **1208** | 877 | 417 | 0.99× | **1.37×** | **2.87×** |
|  | 128×7168×2048 | **1183** | 1147 | 941 | 415 | 1.03× | **1.26×** | **2.85×** |
|  | 4096×2112×7168 | **1495** | 1426 | 1402 | 1258 | 1.05× | **1.07×** | **1.19×** |
|  | 4096×24576×1536 | 1474 | 1352 | **1477** | 1228 | **1.09×** | 1.00× | **1.20×** |
|  | 4096×32768×512 | **1483** | 1418 | 1408 | 1330 | 1.04× | **1.05×** | **1.12×** |
|  | 4096×7168×16384 | **1463** | 1286 | 1347 | 1156 | **1.13×** | **1.08×** | **1.26×** |
|  | 4096×4096×7168 | **1424** | 1325 | 1307 | 1114 | **1.07×** | **1.09×** | **1.28×** |
|  | 4096×7168×2048 | 1366 | 1297 | **1422** | 1189 | **1.05×** | 0.96× | **1.15×** |
| llama | 4096×256×4096 | 1180 | **1253** | 1216 | 961 | 0.95× | 0.97× | **1.23×** |
|  | 11008×256×4096 | 1251 | **1292** | 1278 | 506 | 0.96× | 0.98× | **2.47×** |
|  | 4096×256×11008 | 1164 | **1265** | 1224 | 736 | 0.92× | 0.95× | **1.59×** |
|  | 5120×256×5120 | 1225 | **1265** | 1209 | 726 | 0.98× | 1.02× | **1.69×** |
|  | 13824×256×5120 | 1233 | **1271** | 1210 | 735 | 0.98× | 1.02× | **1.68×** |
|  | 5120×256×13824 | 1170 | **1264** | 1222 | 756 | 0.93× | 0.97× | **1.55×** |

GFLOP/s, single thread, mean of two runs, 27 September 2026 (the session behind
the charts above). Ratios are v5's speed over the other library, paired within
rounds and averaged geometrically over the two runs. Bold marks the fastest
library on a row and ratios of 1.05× or more. Measured with the project's
separate benchmark harness.

<details>
<summary>9 September table (v3, Accelerate, OpenBLAS)</summary>

<br>

| | Shape | MaxMulSK | Accelerate | OpenBLAS | vs Accelerate | vs OpenBLAS |
|---|---|---:|---:|---:|---:|---:|
| square | 256³ | 1395.7 | **1681.2** | 1326.7 | 0.83× | 1.05× |
| | 512³ | 1657.4 | **1776.2** | 1641.4 | 0.93× | 1.01× |
| | 1024³ | **1767.8** | 1687.7 | 1546.4 | **1.05×** | 1.14× |
| | 2048³ | 1613.2 | **1667.5** | 1311.2 | 0.97× | 1.23× |
| | 4096³ | **1638.2** | 1598.3 | 1246.3 | **1.02×** | 1.31× |
| llm | 64×8192×512 | 919.9 | **964.3** | 678.5 | 0.95× | 1.36× |
| | 64×16384×512 | 813.5 | **857.0** | 475.2 | 0.95× | 1.71× |
| | 64×32768×512 | 810.9 | **857.5** | 480.3 | 0.95× | 1.69× |
| | 128×8192×512 | 1111.2 | **1185.9** | 625.4 | 0.94× | 1.78× |
| | 128×16384×512 | 1047.0 | **1132.2** | 468.4 | 0.92× | 2.23× |
| | 128×32768×512 | 1052.9 | **1126.7** | 488.1 | 0.93× | 2.16× |

GFLOP/s, single thread. Generated by `bench/bench_headline.cpp`.

</details>

### What is compared, and how

**Apple Accelerate**, **OpenBLAS** and **MpGEMM** are the end-to-end baselines.
All three are complete GEMM entry points that you call the same way you call
MaxMulSK, so comparing them side by side is fair. MpGEMM is the SME GEMM from
the paper whose DeepSeek-V3 and LLaMA shape tables this repo reuses; it is
called through its `row_sgemm` entry point.

**Arm KleidiAI** is deliberately left out of the table. KleidiAI ships
micro-kernels and packing routines, not a blocked end-to-end GEMM. To benchmark
it end to end you have to give it a blocking strategy. If we give it ours, the
result says more about our blocking than about KleidiAI. It belongs in the
micro-kernel and prepacked comparisons instead, where the units match
([§0.6 to §0.9](docs/BENCHMARKS.md)). This repo does contain hybrid runs, such
as KleidiAI's kernel driven by MaxMulSK's panel blocking, but those are
ablations. They exist to separate cause from effect and are never quoted as
headline numbers.

**NumPy and PyTorch** are reference points only. Both route through a vendor
BLAS, so neither is an independent implementation.

<details>
<summary><b>The wider landscape</b>: every FP32 GEMM path on this machine, three orders of magnitude</summary>

<br>

<img src="docs/img/landscape.svg" width="100%" alt="Every FP32 GEMM path on this machine, across sizes, log scale">

There are three tiers here. The top band holds the implementations that use a
matrix extension. The bottom band holds the ones that do not: our own NEON
kernel at its ceiling, NumPy's vecLib path, and llama.cpp's native FP32 CPU
path.

The two OpenBLAS lines are the **same library, built two different ways**. The
dashed line is what `brew install openblas` gives you today. It falls off a
cliff above 1024³. The solid line is the same source, built so that it can reach
its own SME kernels. Both lines are shown because they answer different
questions: the dashed one is what you get from a default install, and the solid
one is what the library can do when built for this hardware. The cause
is a size threshold that was calibrated against a kernel which is no longer the
alternative ([§0.16](docs/BENCHMARKS.md)).

Unlike the two charts above, this one is assembled from **runs on different
dates**, and NumPy's sweep stops at 2048³. Each series is dated on the chart.
Read it for tiers, not for close calls. Anything within about 2× belongs to the
headline benchmark, which measures its rows together in one process.

</details>

Ground rules for every number here:

- same hardware, same process, single thread, FP32 throughout
- warmup before timing, cooldown between runs, and nothing allocated or
  initialised inside a timed region
- three runs, median reported, with the per-row spread next to it
- **the order of the implementations rotates between timing blocks**, so none of
  them keeps the cache warm for the next one
- correctness checked on every run against a reference. C is prefilled with a
  sentinel value, so a tile that never gets written cannot pass silently
- micro-kernel, packing and full end-to-end costs are measured separately,
  because they answer different questions

This machine has no thread pinning, and the variance between runs can reach 12%
on the worst shapes. Differences under about 5% are not treated as real.

Full methodology, every dated result including superseded ones, and the
experiments that failed: **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)**.

## Usage

```cpp
#include "sme/v3/sme-1x4-acc-kcout.hpp"

// C = A * B, row-major FP32. A is M x K, B is K x N, C is M x N.
// Packing, cache blocking, edge tiles and the ZA->C writeback all happen inside.
// C is OVERWRITTEN, so there is no need to zero it first.
SMEKernels1x4AccKcOut::run_multiplication(A, B, C, M, K, N);
```

Any M, N and K work. Partial output tiles go through a scratch-buffer fallback,
which costs nothing on aligned sizes. Blocking parameters are picked per shape
from a small table in [`sme/support/gemm_tuning.hpp`](sme/support/gemm_tuning.hpp).
That table only holds entries that beat the default across three separate runs.

## Kernel evolution

There are six generations. Each one targets whatever was actually limiting the
one before it.

| Generation | When | What changed | Bottleneck it attacked | Result |
|---|---|---|---|---|
| Prototype | 2026-01 | Matrix type, naive triple loop, first OpenMP pass | none yet, just correctness | ~2.6 GFLOP/s |
| **NEON 8×12** | 2026-04 | 8×12 micro-kernel, packing, cache blocking, in-register transpose | scalar code leaving SIMD unused | ~123 GFLOP/s, ~98% of the NEON ceiling |
| **v1, SME** <br><sub>4×1, 2×2, 1×4, 1×4-sym</sub> | 2026-04 → 07 | ZA outer-product accumulators, BLIS-style M/K/N blocking, four tile geometries | SIMD registers being the wrong tool for a matrix product | ~1200–1330 GFLOP/s |
| **v2, Acc** <br><sub>1×4-Acc, 1×4-Acc-Kc</sub> | 2026-09-06 | ZA lifetime lifted out of the micro-kernel: K innermost, ZA held across all of K, read-modify-write dropped from the store, store turned row-major, pack_A moved to a ZA transpose | per-call fixed cost, **357 ns → 52 ns** | 1773 at 1024³, but full-K packing cost it 4096³ and large K |
| **v3, Acc-KcOut** <br><sub>all three geometries</sub> | 2026-09-09 | An outer Kc panel above the tile nest, so packed panels scale with Kc instead of K | v2's full-K packing footprint | **1.40–1.77 TFLOP/s** across the range |
| **v4, Nc-blocked** <br><sub>v4c path</sub> | 2026-09-13 | An Nc block above the Mc loop bounds the packed working set whatever N is; a shape rule (v4c) picks between two fixed plans | packed B growing with N | 1.00× v3 over the 35 shapes: 1.11× on LLaMA, 1.05× on LLM shapes, 0.96× on squares |

The fixed-cost reduction in v2 and v3 is worth a closer look, because it scales
with the tile geometry in a way that has a clear explanation:

| Geometry | Output tile | Fixed cost | ZA→C stores |
|---|---|---|---|
| 1×4 | 16×64 | 357 → **52 ns** | 64 → 16 |
| 2×2 | 32×32 | 358 → **98 ns** | 64 → 32 |
| 4×1 | 64×16 | 358 → **191 ns** | 64 → 64 |

All three start from the same place and do identical arithmetic, at 2002 to 2010
GFLOP/s of pure compute. What separates them is how wide the ZA tiles let the
writeback be. The 1×4 tile is 4·SVL across, so one row drains in a single
`svst1_f32_x4`. The 4×1 tile is one vector wide, so there is nothing to widen.

### Ideas that did not work, and one belief that turned out to be false

These are kept in `sme/experimental/`. They are useful because each one measured
something, even though none of them made the code faster.

- **Pack once and reuse** (`acc-fast`, `acc-fast-kcout`). The idea was to pack A
  and B for the whole matrix instead of per tile. It gained nothing, and it lost
  on large-K shapes. The bigger buffer is colder than the small one it replaced.
- **Skip pack_B entirely** (`bdirect`). It turns out pack_B is a pure copy, so
  the micro-kernel can read B in place with a stride instead. Results came out
  bit-identical, and performance dropped 66% at 4096³. The copy itself is cheap;
  what it buys is a contiguous layout for the inner loop, and that turns out to
  be worth far more than the copy costs.
- **1×4ZAIO and 4×1-ZAPack.** v1-era experiments, superseded by v2 and v3.
- **"The gap to Accelerate is hardware."** This was believed for months. It
  rested on Accelerate using AMX, a separate coprocessor. Sampling the program
  counter inside `cblas_sgemm` found 0 AMX-dominated samples out of 583 at
  4096³, which points the other way. A major part of the gap turned out to be
  per-call fixed cost. While the hardware explanation stood, there was no reason
  to look for that cost, so it went unexamined
  ([§0.17](docs/BENCHMARKS.md)).

## Project layout

<details>
<summary>Full source tree</summary>

```
MaxMulSK/
├── common/
│   ├── utils.hpp                  # fill_random, check_correctness, compute_gflops, scalar reference
│   └── Matrix.hpp                 # Generic matrix class (addition, subtraction)
├── neon/
│   ├── neon-8x12.hpp/.cpp         # NEON 8×12 kernel + packing + OpenMP driver
│   └── test_neon.hpp/.cpp         # Packing + GEMM correctness vs scalar
├── sme/                               # grouped by generation; v3 is current
│   ├── v1/                            # ZA lifetime lives inside the micro-kernel
│   │   ├── sme-4x1.hpp/.cpp           # 4×SVL × 1×SVL tile, A-inner
│   │   ├── sme-2x2.hpp/.cpp           # 2×SVL × 2×SVL tile, interleaved B packing
│   │   ├── sme-1x4.hpp/.cpp           # 1×SVL × 4×SVL tile, x1 interleaved B loads
│   │   └── sme-1x4-sym.hpp/.cpp       # same tile, x4-grouped B loads (true mirror of 4×1)
│   ├── v2/                            # ZA lifetime lifted into the driver: K innermost,
│   │   ├── sme-1x4-acc.hpp/.cpp       #   ZA held across all of K, overwriting row-major
│   │   └── sme-1x4-acc-kc.hpp/.cpp    #   store, ZA-transpose pack_A. Costs full-K packing.
│   ├── v3/                            # v2 + an OUTER Kc loop, which fixes that packing cost
│   │   ├── sme-1x4-acc-kcout.hpp/.cpp # 1×4 geometry, the fastest of the three
│   │   ├── sme-2x2-acc-kcout.hpp/.cpp # 2×2 geometry
│   │   └── sme-4x1-acc-kcout.hpp/.cpp # 4×1 geometry
│   ├── experimental/                  # measured, kept, NOT on the fast path
│   │   ├── v1/sme-1x4-sym-zainout     #   ZA-resident K accumulation, K_inner_tile=40
│   │   ├── v1/sme-4x1-zapack          #   4×1 with ZA-based pack_A, M_tile=128
│   │   ├── v2/sme-1x4-acc-fast        #   pack A and B once for the whole matrix
│   │   ├── v3/sme-1x4-acc-fast-kcout  #   same, per K panel
│   │   └── v3/sme-1x4-acc-kcout-bdirect #  no pack_B at all, B read in place
│   └── support/
│       ├── gemm_tuning.hpp/.cpp       # shape-dependent blocking table
│       └── test_sme.hpp/.cpp          # dispatch surface: run / run_comparison / profile
├── bench/
│   ├── bench_headline.cpp         # MaxMulSK vs Accelerate vs OpenBLAS, one run, the table above
│   ├── plot_headline.py           # renders the two SVGs from that run's CSVs
│   ├── bench_compare.cpp          # C++ comparison: NEON / SME / Accelerate / OpenBLAS
│   ├── bench_python.py            # Python comparison: NumPy / PyTorch
│   ├── bench_profile.cpp          # Single-library driver (accel|oblas) for profiling scripts
│   ├── bench_profile.py           # Single-library driver (numpy|pytorch)
│   ├── run_bench.sh               # Build + run both comparison benchmarks
│   ├── benchmark_maxmul_vs_kleidiai.cpp  # Layered: hot micro-kernel / prepacked / end-to-end / panel / ggml
│   ├── maxmulsk_sme_adapter.hpp/.cpp     # Uniform dispatch over our SME kernels + prepacked and hot-tile harnesses
│   ├── kleidiai_gemm.hpp/.cpp            # Arm KleidiAI fp32 SME2 wrapper (full-pack and panel-blocked callers)
│   ├── ggml_gemm_baseline.hpp/.cpp       # llama.cpp/ggml CPU baseline, native and KleidiAI dispatch paths
│   ├── energy_bench.cpp/.sh/_parse.py    # W, J and J/GFLOP per implementation per shape
│   ├── grand_benchmark.sh                # Runs every benchmark into one dated directory
│   ├── grand_summary.py                  # Aggregates a run directory into SUMMARY.md
│   ├── average_runs.py                   # Median across repeated runs + per-row spread
│   └── results/                          # Dated result archive; nothing here is deleted
├── experiments/
│   └── matmul.cpp                 # Standalone N=2048 head-to-head: our SME kernel vs Accelerate
├── scripts/
│   ├── run.sh                     # Build + run main binary
│   ├── run_matmul.sh              # Build + run the matmul experiment (vs Accelerate)
│   ├── profile.sh                 # Instruments PMU profiling (L1 misses, INST_ALL)
│   └── profile_power.sh           # powermetrics-based energy profiling
├── docs/
│   ├── BENCHMARKS.md              # Every measurement, dated; superseded results marked, not removed
│   └── COMPARISON.md              # Historical 2026-04 snapshot, kept for the progression
├── main.cpp                       # Test runner + benchmark harness
├── TODO.md                        # Roadmap + fixed-bug archive
└── GemmTemplate.tracetemplate     # Xcode Instruments template (PMU counter list)
```

</details>

## Requirements

- Apple M4. The SME kernels need SME/SME2
- LLVM/Clang, via `brew install llvm`. Apple's bundled Clang lacks the SME/SME2 intrinsics
- libomp, via `brew install libomp`
- CMake 3.30+
- OpenBLAS, via `brew install openblas`, for `bench_compare` and `bench_headline`. It is loaded with `dlopen`, so it is optional at runtime. The Homebrew build cannot reach its own SME GEMM kernels. To compare against OpenBLAS at its best, build `develop` with `DYNAMIC_ARCH=1` and point `OPENBLAS_DYLIB` at it ([§0.16](docs/BENCHMARKS.md))
- Python with `numpy` and `torch`, needed only for `bench_python.py`
- **Arm KleidiAI**, fetched and built automatically at a pinned tag (`v1.30.0`). Turn it off with `-DMAXMULSK_WITH_KLEIDIAI=OFF`, or point at an existing checkout with `-DFETCHCONTENT_SOURCE_DIR_KLEIDIAI=...`
- **llama.cpp / ggml** *(optional)*, only needed for the ggml CPU baseline rows. Build it CPU-only, then point CMake at it

The build finds the toolchain instead of hardcoding paths. You can override any of it:

```bash
cmake -B build -DCMAKE_CXX_COMPILER=/path/to/clang++   # different compiler
HOMEBREW_PREFIX=/custom/prefix cmake -B build          # non-standard Homebrew
OPENBLAS_DYLIB=/path/to/libopenblas.dylib ./bench/run_bench.sh
PYTHON=/opt/anaconda3/bin/python ./bench/run_bench.sh  # interpreter with numpy+torch
```

For the llama.cpp/ggml baseline, build ggml with every external accelerator
turned off, so the rows measure ggml's own CPU path. Then point CMake at it:

```bash
cmake -S llama.cpp -B llama-build -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=OFF \
      -DGGML_METAL=OFF -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF \
      -DGGML_CPU_KLEIDIAI=ON -DGGML_OPENMP=OFF
cmake --build llama-build --target ggml-cpu ggml-base ggml

cmake -B cmake-build-release -S . \
      -DLLAMA_CPP_DIR=llama.cpp -DLLAMA_CPP_BUILD_DIR=llama-build \
      -DLLAMA_CPP_KLEIDIAI_LIB=llama-build/_deps/kleidiai-build/libkleidiai.a
```

`GGML_CPU_KLEIDIAI=ON` adds a second ggml row that dispatches to KleidiAI's fp32
SME2 kernel. Without it, only ggml's native path is measured. Note that ggml
pins its own KleidiAI version, v1.24.0, independent of the one this repo
fetches.

## Build and run

```bash
# The headline comparison — MaxMulSK vs Accelerate vs OpenBLAS
cmake --build cmake-build-release --target bench_headline
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    ./cmake-build-release/bench_headline out.csv
python3 bench/plot_headline.py docs/img out.csv        # regenerate the charts

# Main binary (NEON tests + the SME suites it wires up + comparison + timing breakdown)
./scripts/run.sh

# Everything, into bench/results/<date>/
./bench/grand_benchmark.sh
./bench/grand_benchmark.sh --power-only   # energy only, reusing an existing directory

# Layered head-to-head (hot micro-kernel / prepacked / end-to-end / panel / ggml)
cmake --build cmake-build-release --target benchmark_maxmul_vs_kleidiai
./cmake-build-release/benchmark_maxmul_vs_kleidiai out.csv

# Single-thread competitor comparison (NEON / SME / Accelerate / OpenBLAS / NumPy / PyTorch)
./bench/run_bench.sh

# Instruments PMU profiling — saves traces/<name>.txt
./scripts/profile.sh 4x1
./scripts/profile.sh --binary cmake-build-release/bench_profile \
                     --args "accel 2048 2048 2048 100" accel

# Energy / thermal profiling — saves traces/<name>_power.txt (sudo for powermetrics)
sudo ./scripts/profile_power.sh --no-build 4x1
```

`./scripts/run.sh` runs the NEON unit tests, the NEON speed sweep, the SME kernel
suites wired into that binary (pack correctness, then GEMM correctness, then
benchmark), the cross-kernel comparison, and the 4×1 timing breakdown.

## Future work

- [ ] Dynamic tiling
- [ ] SME multi-threading
- [ ] Re-profile 1×4-sym vs 4×1 with PMU counters to explain the B-inner turnaround
- [ ] Apply the shape-dependent blocking table to the 2×2 and 4×1 v3 kernels (it was tuned on 1×4 only)
- [ ] Report the stale OpenBLAS direct-path threshold upstream

Completed work is in [TODO.md](TODO.md), including every bug that was fixed
along the way and its root cause.

## More

- [Benchmarks, methodology and the full result history](docs/BENCHMARKS.md)
- [Historical 2026-04 comparison snapshot](docs/COMPARISON.md)
- [Roadmap and fixed-bug archive](TODO.md)
- [Raw result archive](bench/results/)

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
