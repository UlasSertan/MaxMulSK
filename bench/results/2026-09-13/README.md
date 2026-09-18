# 2026-09-13 — experiment F2, and F2 + `__arm_preserves("za")`

One shape: **M=11008, K=4096, N=256**, FP32, row-major, single thread, Apple M4.
This is LLaMA ID 20 in Table III of the MpGEMM paper, and it sits in the regime
where MaxMulSK was weakest — narrow N.

Hardware, compiler, flags and the exact source state are in `environment.txt` and
`source/`.

## What was built

| Name | What it is |
|---|---|
| `R0` | `sme/v3/sme-1x4-acc-kcout.cpp`, the shipped kernel with its table blocking, Mc1024 / Kc2048 / Nc64 |
| `F2` | `sme/v4/sme-1x4-kcout-mc16-apack4za-bonce.cpp`, new |
| `F2-ZaPres` | `F2` with one attribute added; see below |

F2 changes three things against R0, and nothing else:

- **Mc 1024 → 16.** An A panel is now a single 16-row micro-tile, consumed by its
  four N panels immediately after it is built.
- **A is packed inline in the driver, across all four ZA tiles.** R0 calls
  `pack_A_streaming`, which uses ZA tile 0 alone and 16 k-steps per round. F2
  covers 64 k-steps per round: x4 loads feed horizontal ZA writes, and x4 stores
  drain vertical ZA reads, two independent four-Z groups on each side.
- **B is packed once per (Kc, N panel)** on the first M round and reused by every
  later M round, instead of being repacked per M block. 88 panel packs become 8.

`pack_B_streaming`, `micro_kernel_1x4`, `store_za` and `store_za_add` are carried
over verbatim. The compute load/MOPA schedule and the C writeback are not part of
this experiment.

## Method

Both entrants allocate their packed buffers inside their own call, so the
allocation scope is identical and neither is charged a cost the other avoids.
Reps are calibrated on the slowest entrant so all of them get the same count.
Order is a cyclic rotation — run `r` starts at entrant `r % n` — so over any `n`
consecutive runs each entrant occupies each slot exactly once and none of them
systematically runs first into a cold cache. Each run reports the median of its
per-call timings; every run is in the CSV so the spread is visible. `caffeinate
-dimsu` for the whole session.

Timings are taken in integer nanoseconds and converted afterwards. That is not
cosmetic: MpGEMM's hand-written assembly clobbers the callee-saved SIMD registers
d8–d15 without restoring them, which silently corrupted `double` timing
accumulators held across the call. The harness wraps that call and names those
registers as clobbered. See the 2026-09-10 session notes.

## Results

**`f2_vs_r0.csv`** — R0 vs F2, 9 runs, 15 reps each.

| | GFLOP/s | ms | min–max | spread |
|---|---:|---:|---|---:|
| R0 | 1023.6 | 22.55 | 1005.7 – 1036.7 | 3.03% |
| **F2** | **1353.7** | **17.05** | 1349.6 – 1355.8 | 0.46% |

**F2 / R0 = 1.322×**, i.e. +32.2%. The two distributions do not overlap: R0's
best run is 1036.7, F2's worst is 1349.6.

**`f2_5way.csv`** — all five entrants, same binary, same session, 10 runs, 12 reps
each.

| | GFLOP/s | spread | vs R0 |
|---|---:|---:|---:|
| R0 | 945.7 | 6.29% | — |
| F2 | 1361.0 | 1.75% | 1.439× |
| F2-ZaPres | 1356.5 | 1.42% | 1.434× |
| Accelerate | 1392.7 | 2.30% | 1.473× |
| MpGEMM | 1407.2 | 3.27% | 1.488× |

At this shape F2 lands within 3.3% of MpGEMM and 2.3% of Accelerate — below the
repo's ~5% threshold, so the honest reading is that the three are tied and R0 was
the outlier.

**R0's own number moves a lot between binaries**: 1023.6 in the two-way build,
888.3 in the three-way, 945.7 in the five-way. F2's moves by 1.3% across the same
three. Do not carry an R0 figure from one CSV into a comparison made in another.

## Does `__arm_preserves("za")` buy anything?

`pack_B_streaming` is declared `__arm_streaming` with no ZA annotation. Calling a
ZA-agnostic function from one that holds live ZA state forces the caller to arm a
TPIDR2 lazy save. `F2-ZaPres` is byte-for-byte F2 except that this function is
declared `__arm_preserves("za")` — truthfully: its generated code is 104
instructions with zero ZA or streaming-state opcodes and no calls.

It was given its own copy of the function rather than editing a shared one, so
neither R0 nor the already-measured F2 was disturbed.

The annotation does exactly what it should, in `run_f2_streaming`:

| | F2 | F2-ZaPres |
|---|---:|---:|
| `smstart za` | 2 | 1 |
| `msr TPIDR2_EL0` | 3 | 1 |
| `mrs TPIDR2_EL0` | 2 | 1 |
| `bl ___arm_tpidr2_restore` | 1 | **0** |
| static instructions | 396 | 381 |
| x4 loads / x4 stores / Z→ZA / ZA→Z | 16 / 16 / 64 / 64 | unchanged |
| Z register spills | 0 | 0 |

**End to end it changes nothing: F2-ZaPres / F2 = 0.9967× in the five-way run and
0.9961× in the three-way**, against per-entrant spreads of 1.4–2.0%. That is a
null result, and it was the expected one — see the execution-count analysis below.

## Assembly: where the ZA transitions actually are

An earlier note in this session claimed this lazy-save sequence ran once per
(m, n), i.e. 5504 times per call. **That was wrong.** The block structure of
`run_f2_streaming` is:

```
LBB4_10  (n-loop header)   cbz  x10, LBB4_12      ; x10 = (m != 0)
                           cbnz x25, LBB4_17      ; x25 = (n != 0)
                           b    LBB4_15
LBB4_12  (only when m==0)  msr  TPIDR2_EL0, x8    ; arm the lazy save
                           bl   pack_B_streaming
                           smstart za
                           mrs  x8, TPIDR2_EL0
                           cbnz x8, LBB4_14       ; TPIDR2 != 0 -> skip restore
                           bl   ___arm_tpidr2_restore
LBB4_14                    msr  TPIDR2_EL0, xzr
LBB4_17                    zero {za}              ; svzero_za, per micro-tile
```

The whole sequence lives inside `LBB4_12`, which is reached only on the `m == 0`
path. Per GEMM call:

| | executions |
|---|---:|
| `smstart za` at function entry (`__arm_new("za")`; the streaming-mode entry is the separate `smstart sm`) | 1 |
| `smstart za` on the B-pack path, `LBB4_12` | 8 (2 Kc × 4 panels) |
| restore condition `mrs TPIDR2` + `cbnz` | 8 |
| `bl ___arm_tpidr2_restore` | **0** |
| the two compare-and-branch instructions in `LBB4_10` | 5504 |

`___arm_tpidr2_restore` is called only when `TPIDR2_EL0` reads back as zero,
which means the lazy save was actually taken and ZA was spilled. Since
`pack_B_streaming` never touches ZA, it is never taken, and the branch always
skips the call. What the 5504 iterations really carry is two compare-and-branch
instructions in the loop header, not a ZA transition. Removing 8 occurrences of a
five-instruction sequence next to 45,088,768 `fmopa` was never going to be
measurable, and it was not.

## Correctness

The A packing layout was checked elementwise against its scalar definition,
`packed_A[k*16 + r] == A[(m+r)*K + kk + k]`, over all 32768 elements of five
panels: `(kk=0, m=0)`, `(kk=0, m=M−16)`, `(kk=2048, m=0)`, `(kk=2048, m=M−16)`,
`(kk=0, m=10992)`. Zero wrong, zero untouched, for both F2 and F2-ZaPres. The
probe expands the driver's own `F2_PACK_A_KP_CHUNK` macro, so what is verified is
the code that runs; the hot path stays inline rather than being refactored into a
callable helper.

Full GEMM:

- Against R0: **bit-identical**, relative Frobenius 0.000e+00. Expected — the
  compute path and accumulation order are unchanged and `packed_A` holds the same
  bytes, only the route that fills it differs.
- Against an fp64 reference on 48 rows (first 24 and last 24): 8.089e-07.
- C pre-filled with 7.5 is overwritten, not accumulated into.
- A second call with different B is correct, so no stale packed B survives a call
  or a Kc slice.
- `256³` is refused (`run_multiplication_f2` returns false) and C is left
  untouched. There is no tail path and no dispatch in this experiment.

In `f2_5way.csv` all five entrants agree: R0, F2 and F2-ZaPres produce identical
checksums, Accelerate and MpGEMM agree with each other and differ from ours by
6.63e-06 relative — fp32 accumulation order, not a defect.

## Historical context, not a comparison

The 2026-09-10 session measured this same shape as `llama-id20` in
`bench/results/2026-09-10/vs_mpgemm.csv`: MaxMulSK-1x4 1087, Accelerate 1397,
MpGEMM 1458 GFLOP/s. Those numbers come from a **different binary, a different
harness and a different session**, and the MaxMulSK entrant there is R0, whose
figure has since been observed to move by more than 10% between builds. They are
recorded here as background for why this shape was picked, and must not be put in
a table next to the numbers above as if they were measured together. The
same-session comparison is `f2_5way.csv`.


---

# Full-table sweep — generalised v4 vs v3, Accelerate and MpGEMM

`v4_sweep.csv`, `v4_sweep_summary.txt`. Same binary, same session, 35 shapes,
5 runs each, cyclic Latin square over both shapes and runs, reps calibrated on
the slowest entrant, median of per-call timings, `caffeinate -dimsu`.

## What was measured

| Entrant | Entry point | Blocking |
|---|---|---|
| v3 | `SMEKernels1x4AccKcOut::run_multiplication` | `MaxMulSK::tuning::select(Sme1x4KcOut, M, K, N)`; the `v3_Mtile/Ntile/Kc` columns record what it actually chose |
| v4 | `SMEKernels1x4KcOutMc16Apack4ZaBonceGen::run_multiplication` | Mc=16, Nc=64, Kc=min(2048, K) |
| Accelerate | `cblas_sgemm` | `BLASSetThreading(BLAS_THREADING_SINGLE_THREADED)` |
| MpGEMM | `row_sgemm`, wrapped for the d8–d15 ABI bug | its own `ROW_MC/KC/NC` = 512/1024/256 |

Every entrant allocates its own packing buffers inside its own timed call, so
allocation scope is identical. All four are single-threaded.

## Generalisation, and what it refuses

F2 was written for one shape. The generalised version derives everything from
the shape: `Kc = min(2048, K)`, the per-slice length `kc = min(Kc, K − kk)` is
passed to A packing, B packing and the micro-kernel from one variable, the panel
count is `N/64`, and `packed_B` is `panels × Kc × 64` floats.

There is still **no tail path**. `classify()` requires `M % 16 == 0`,
`N % 64 == 0` and `K % 64 == 0`, and returns which one failed. A refused shape
leaves C untouched and is not silently handed to another kernel — that choice
belongs to the caller, and a fallback number is not this kernel's performance.

**All 35 shapes in the table happen to satisfy all three conditions**, so the
sweep contains no refused rows and no fallback rows. The refusal path is
therefore present and tested but not exercised here.

## Result

| shape | M×K×N | packB MiB | v3 | v4 | Accel | MpGEMM | v4/v3 | v4/MpGEMM | v4/Accel |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `square` | 256×256×256 | 0 | 1410 | 1167 | 1786 | 1488 | **0.83×** | 0.78× | 0.65× |
| `square` | 512×512×512 | 1 | 1664 | 1602 | 1778 | 1808 | **0.96×** | 0.89× | 0.90× |
| `llama-id19` | 4096×4096×256 | 2 | 978 | 1312 | 1334 | 1347 | **1.34×** | 0.97× | 0.98× |
| `llama-id20` | 11008×4096×256 | 2 | 988 | 1321 | 1350 | 1351 | **1.34×** | 0.98× | 0.98× |
| `llama-id21` | 4096×11008×256 | 2 | 1117 | 1281 | 1352 | 1384 | **1.15×** | 0.93× | 0.95× |
| `llama-id22` | 5120×5120×256 | 2 | 1000 | 1294 | 1314 | 1344 | **1.29×** | 0.96× | 0.98× |
| `llama-id23` | 13824×5120×256 | 2 | 997 | 1277 | 1304 | 1313 | **1.28×** | 0.97× | 0.98× |
| `llama-id24` | 5120×13824×256 | 2 | 1058 | 1274 | 1318 | 1334 | **1.20×** | 0.96× | 0.97× |
| `llm` | 64×8192×512 | 4 | 898 | 905 | 961 | 1011 | **1.01×** | 0.90× | 0.94× |
| `llm` | 64×16384×512 | 4 | 794 | 863 | 857 | 910 | **1.09×** | 0.95× | 1.01× |
| `llm` | 64×32768×512 | 4 | 798 | 864 | 858 | 913 | **1.08×** | 0.95× | 1.01× |
| `llm` | 128×8192×512 | 4 | 1066 | 1156 | 1190 | 1238 | **1.08×** | 0.93× | 0.97× |
| `llm` | 128×16384×512 | 4 | 1011 | 1135 | 1137 | 1193 | **1.12×** | 0.95× | 1.00× |
| `llm` | 128×32768×512 | 4 | 1018 | 1137 | 1133 | 1194 | **1.12×** | 0.95× | 1.00× |
| `square` | 1024×1024×1024 | 4 | 1753 | 1765 | 1686 | 1892 | **1.01×** | 0.93× | 1.05× |
| `square` | 2048×2048×2048 | 16 | 1583 | 827 | 1662 | 1545 | **0.52×** | 0.54× | 0.50× |
| `ds-id1` | 64×7168×2112 | 16 | 948 | 478 | 1004 | 1069 | **0.50×** | 0.45× | 0.48× |
| `ds-id7` | 128×7168×2112 | 16 | 1168 | 571 | 1288 | 1346 | **0.49×** | 0.42× | 0.44× |
| `ds-id13` | 4096×7168×2112 | 16 | 1539 | 759 | 1527 | 1481 | **0.49×** | 0.51× | 0.50× |
| `square` | 4096×4096×4096 | 32 | 1630 | 560 | 1554 | 1489 | **0.34×** | 0.38× | 0.36× |
| `ds-id5` | 64×7168×4096 | 32 | 745 | 400 | 596 | 951 | **0.54×** | 0.42× | 0.67× |
| `ds-id11` | 128×7168×4096 | 32 | 1165 | 478 | 888 | 1230 | **0.41×** | 0.39× | 0.54× |
| `ds-id17` | 4096×7168×4096 | 32 | 1426 | 572 | 1437 | 1390 | **0.40×** | 0.41× | 0.40× |
| `ds-id4` | 64×16384×7168 | 56 | 791 | 390 | 632 | 872 | **0.49×** | 0.45× | 0.62× |
| `ds-id6` | 64×2048×7168 | 56 | 812 | 395 | 660 | 900 | **0.49×** | 0.44× | 0.60× |
| `ds-id10` | 128×16384×7168 | 56 | 1152 | 463 | 821 | 1119 | **0.40×** | 0.41× | 0.56× |
| `ds-id12` | 128×2048×7168 | 56 | 1143 | 468 | 974 | 1168 | **0.41×** | 0.40× | 0.48× |
| `ds-id16` | 4096×16384×7168 | 56 | 1499 | 562 | 1434 | 1362 | **0.37×** | 0.41× | 0.39× |
| `ds-id18` | 4096×2048×7168 | 56 | 1504 | 563 | 1531 | 1405 | **0.37×** | 0.40× | 0.37× |
| `ds-id3` | 64×512×32768 | 64 | 859 | 374 | 605 | 870 | **0.44×** | 0.43× | 0.62× |
| `ds-id9` | 128×512×32768 | 64 | 1167 | 461 | 870 | 1158 | **0.39×** | 0.40× | 0.53× |
| `ds-id15` | 4096×512×32768 | 64 | 1571 | 542 | 1525 | 1475 | **0.34×** | 0.37× | 0.35× |
| `ds-id2` | 64×1536×24576 | 144 | 890 | 393 | 602 | 851 | **0.44×** | 0.46× | 0.65× |
| `ds-id8` | 128×1536×24576 | 144 | 1182 | 466 | 890 | 1100 | **0.39×** | 0.42× | 0.52× |
| `ds-id14` | 4096×1536×24576 | 144 | 1521 | 560 | 1562 | 1403 | **0.37×** | 0.40× | 0.36× |

Sorted by packed-B size, because that turns out to be the variable that
explains the table.

## Win / tie / loss, ±5%

| | win | tie | loss | geomean |
|---|---:|---:|---:|---:|
| v4 vs v3 | 11 | 3 | 21 | 0.645× |
| v4 vs MpGEMM | 0 | 7 | 28 | 0.594× |
| v4 vs Accelerate | 0 | 11 | 24 | 0.649× |

| regime | n | v4/v3 | v4/MpGEMM | v4/Accel | w/t/l vs v3 |
|---|---:|---:|---:|---:|---|
| narrow N (≤256) | 7 | **1.191×** | 0.933× | 0.919× | 6/0/1 |
| narrow M (≤128) | 18 | 0.601× | 0.552× | 0.673× | 5/1/12 |
| all three large | 8 | 0.455× | 0.476× | 0.457× | 0/1/7 |
| mixed | 2 | 0.576× | 0.570× | 0.566× | 0/1/1 |

Most improved vs v3: `llama-id19` 1.34×, `llama-id20` 1.34×, `llama-id22` 1.29×,
`llama-id23` 1.28×, `llama-id24` 1.21×. Every one of them has N=256.

Most regressed vs v3: `square 4096³` 0.34×, `ds-id15` 0.34×, `ds-id14` 0.37×,
`ds-id18` 0.37×, `ds-id16` 0.37×. All of them have packed B ≥ 32 MiB.

## Where the limit is

The M/N regime split above is not the real variable. Sorting by `packB_MiB`
separates the table cleanly:

| packed B | shapes | v4/v3 |
|---|---:|---|
| ≤ 4 MiB | 15 | 0.83× – 1.34×, median ≈ 1.08× |
| 16 MiB | 4 | 0.49× – 0.52× |
| ≥ 32 MiB | 16 | 0.34× – 0.54× |

The cliff sits between 4 and 16 MiB. This machine's P-core L2 is **16 MiB**
(`hw.perflevel0.l2cachesize`), so a 4 MiB packed B stays resident across the many
M rounds while a 16 MiB one does not.

The mechanism is Mc. Logical packed-B read traffic for a whole GEMM is

    (M / Mc) × K × N × 4 bytes

because every one of the `M/Mc` rounds re-reads the whole `Kc × N` packed panel,
for each of the `K/Kc` slices. Dropping Mc from v3's 1024 to v4's 16 multiplies
that by `Mc_v3 / 16` — 64× on most of these shapes. At 4096³ it is 256 MiB for v3
against 16 GiB for v4.

That multiplier is paid in cache hits when packed B fits in L2, which is why the
N=256 family wins: its packed B is 2 MiB, the 64× re-read never leaves L2, and
what is left is F2's real benefit — A packed inline four ZA tiles at a time and
consumed immediately, and B packed once per Kc slice instead of per M block.
Once packed B exceeds L2 the same 64× becomes DRAM traffic and the kernel
collapses to roughly 560 GFLOP/s, almost independently of the shape.

So the strategy is not "good at narrow N" in itself. **It is good while
`Kc × N × 4` stays inside L2**, and N is just the term that usually decides that.
The llm family (N=512, packed B 4 MiB) confirms it: those are narrow-M shapes,
which the regime table lumps into the losing group, yet they win 1.01×–1.12×.

## Correctness

Checked on every shape, in the same run, against v3. Largest v4 relative
Frobenius error across the 35 shapes: **1.091e-06** (at 4096³, where fp32
accumulation over K=4096 dominates). v4 is bit-identical to v3 on most shapes —
the compute path and accumulation order are unchanged, only the route that fills
`packed_A` differs. `relerr_acc` and `relerr_mp` are in the CSV for the same
reason and are of the same order.

## Caveats

- Largest run-to-run spread in the sweep is 18.91%, at 256³, where a single call
  is ~24 µs and the harness is at its resolution limit. Treat the two smallest
  square rows as indicative only.
- `v3_Mtile/Ntile/Kc` are recorded per shape because v3's blocking comes from a
  table and is not constant across the sweep.
- Do not compare these v3 numbers against the v3 numbers in `f2_5way.csv` or in
  `bench/results/2026-09-10/`. Different binaries; v3's figure has been observed
  to move by more than 10% between builds.


---

# Nc/Mc/Kc parameter sweep — `sme-1x4-kcout-ncblock-apack4za`

`nb_sweep.csv`, `nb_sweep_summary.txt`. 100 blockings over 35 shapes, 3500
pre-scan rows plus 478 longer re-measurements. **Zero candidates produced a wrong
result**, so nothing was dropped from the rankings.

## The kernel

F2's four-ZA inline A packing and its compute body, with an Nc block added. The
previous generalised F2 packed every N panel of a Kc slice at once, so its packed
B grew with N and fell out of L2 on wide shapes. Here B is packed one `(Kc, Nc)`
block at a time, capping the packed working set at `(Nc + Mc) * Kc` floats.

Loop order `Kc → Nc → Mc → n(64) → m(16) → compute(kcl)`.

- B is packed once per `(Kc, Nc)` block, on the first Mc round, held for the rest.
- A is packed once per `(Kc, Nc, Mc)` block, on that block's first 64-column
  panel, and reused by the remaining panels of the same Nc block.
- A is re-packed for the next Nc block. That is the deliberate price: the A-pack
  count carries a factor `ceil(N/Nc)`, and `counts()` reports it.
- Mc is not packed up front — each 16-row chunk is packed and immediately
  computed, keeping preparation next to consumption.

Panel slot strides use the **allocated** Kc depth, not the current block's real
length, so a short final block leaves a gap instead of shifting later slots.
Packing and compute address slots identically. The real lengths `kcl`, `ncl`,
`mcl` reach all three consumers from one variable each.

A short *final macroblock* is supported and is not a microkernel tail: with
Mc=128 against M=64, the single block is 64 rows and that is correct. What
`classify()` rejects is a macroblock size that is not itself a multiple of the
microtile (Mc%16, Nc%64, Kc%64).

## Correctness, before the sweep

219 cases, 0 failures. A panel layout matched its scalar definition exactly in
six cases including a 64-deep `kcl` and the last Kc slice. Every one of the 35
shapes was run under five blockings chosen to force short final blocks
(`16/256/2048`, `128/1024/4096`, `32/64/256`, `64/512/1024`, `128/128/512`),
plus a repeat call with different B to rule out stale packed B, with C pre-filled
to 3.75 to confirm it is overwritten rather than accumulated into. Invalid
blockings (`Mc=24`, `Nc=100`, `Kc=100`) are refused with C untouched.

## Method

Two stages, because one short timing is not enough to eliminate a candidate.
Stage 1 measures all 100 blockings in two passes and takes the median of the
passes; stage 2 re-measures each shape's leaders, anything within 3% of its
stage-1 leader, and the fixed reference, with five runs each. Candidate order is
rotated per shape and per pass. Every candidate allocates its own packed buffers
inside its own timed call, so allocation scope is identical throughout. Same
binary, same flags, single thread, `caffeinate -dimsu`.

## A. One fixed blocking for all 35 shapes

Ratios are against the fixed internal reference **Mc16/Nc256/Kc2048**.

| Mc/Nc/Kc | geomean | worst | best | >+5% | within | <−5% |
|---|---:|---:|---:|---:|---:|---:|
| **32/1024/2048** | **1.051×** | 0.902× | 1.209× | 17 | 16 | 2 |
| 16/1024/2048 | 1.047× | 0.913× | 1.191× | 16 | 17 | 2 |
| 32/512/2048 | 1.047× | 0.900× | 1.145× | 17 | 17 | 1 |
| **16/512/2048** | 1.046× | **0.980×** | 1.122× | 15 | 20 | **0** |
| 64/512/2048 | 1.045× | 0.819× | 1.140× | 17 | 17 | 1 |
| 128/512/2048 | 1.041× | 0.819× | 1.145× | 17 | 16 | 2 |
| … | | | | | | |
| 16/64/256 (worst of 100) | 0.635× | 0.412× | 0.798× | | | |

Two different answers depending on what is being optimised:

- **32/1024/2048** has the highest geometric mean, 1.051×. Its worst case is
  0.902× on `ds-id4` (64×16384×7168), and it also loses on `ds-id1`.
- **16/512/2048** gives up 0.5% of geomean and in exchange **never regresses more
  than 2%** — zero shapes below −5%, worst case 0.980×. If one blocking has to be
  shipped without a dispatch, this is the defensible one.

Best cases for 32/1024/2048: `4096³` 1.209×, `ds-id13` 1.177×, `ds-id18` 1.168×,
`ds-id17` 1.166×.

## Trends

Geometric mean of the ratio against the reference, one axis at a time, over all
3500 pre-scan rows:

| axis | values |
|---|---|
| **Mc** | 16: 0.896× · 32: 0.902× · 64: 0.900× · 128: 0.898× |
| **Nc** | 64: 0.714× · 128: 0.868× · **256: 0.963× · 512: 0.994× · 1024: 0.992×** |
| **Kc** | 256: 0.820× · 512: 0.894× · 1024: 0.918× · **2048: 0.941×** · 4096: 0.928× |

- **Mc is flat.** Four values spanning 8× in size move the geomean by 0.6%. Mc is
  not a lever for this algorithm on this machine — which also means it can be
  chosen for whatever else it helps, such as keeping packed A small.
- **Nc matters and saturates.** Going 64 → 512 is worth 39%; 512 → 1024 is worth
  nothing. Small Nc is expensive because the A-pack count scales with
  `ceil(N/Nc)`, and at Nc=64 that is the whole cost of the kernel.
- **Kc peaks at 2048** and turns back down at 4096.

## Per regime

Best fixed blocking inside each regime, pre-scan geomean:

| regime | n | best | 2nd | 3rd |
|---|---:|---|---|---|
| narrow N (≤256) | 7 | 32/256/**4096** 1.065× | 32/512/4096 1.065× | 16/1024/4096 1.041× |
| narrow M (≤128) | 18 | 16/**512**/2048 1.047× | 32/512/2048 1.046× | 64/512/2048 1.045× |
| all three large | 8 | 128/**1024**/2048 1.157× | 32/1024/2048 1.150× | 16/1024/2048 1.149× |
| mixed | 2 | 64/1024/**512** 1.089× | 32/1024/512 1.088× | 32/1024/1024 1.086× |

The regimes genuinely disagree on two axes. Narrow N wants **Kc=4096** — with
N=256 the packed B is small whatever Nc is, so depth is free and a longer K run
per C tile pays. The all-three-large regime wants **Nc=1024** and gains the most
from tuning at all, 1.157×. Narrow M sits at Nc=512.

## B. Per-shape measured leaders

**This is a table of measured per-shape leaders, not the result of an implemented
dispatch.** No automatic selection has been added. The last column counts how
many of the 100 blockings landed within 3% of that shape's leader.

| shape | M×K×N | top 3 (Mc/Nc/Kc — GFLOP/s ±spread) | within 3% |
|---|---|---|---:|
| `ds-id1` | 64×7168×2112 | 128/1024/1024 — 878 ±1.4% \| 128/512/1024 — 864 ±1.4% \| 64/512/1024 — 860 ±1.0% | 6 |
| `ds-id10` | 128×16384×7168 | 32/512/2048 — 972 ±0.1% \| 32/1024/1024 — 965 ±0.2% \| 64/1024/1024 — 962 ±1.1% | 7 |
| `ds-id11` | 128×7168×4096 | 32/1024/2048 — 988 ±1.3% \| 64/512/2048 — 988 ±2.8% \| 128/1024/2048 — 988 ±0.0% | 8 |
| `ds-id12` | 128×2048×7168 | 32/512/4096 — 1007 ±0.9% \| 16/512/4096 — 1006 ±0.8% \| 64/512/2048 — 1005 ±1.1% | 13 |
| `ds-id13` | 4096×7168×2112 | 32/1024/2048 — 1261 ±0.7% \| 16/1024/2048 — 1260 ±3.5% \| 64/1024/2048 — 1256 ±1.1% | 8 |
| `ds-id14` | 4096×1536×24576 | 32/1024/2048 — 1366 ±0.9% \| 16/1024/4096 — 1361 ±1.4% \| 16/1024/2048 — 1361 ±0.5% | 6 |
| `ds-id15` | 4096×512×32768 | 64/1024/512 — 1522 ±2.1% \| 32/1024/1024 — 1503 ±7.6% \| 32/1024/4096 — 1480 ±6.0% | 4 |
| `ds-id16` | 4096×16384×7168 | 32/1024/2048 — 1285 ±0.3% \| 64/1024/2048 — 1282 ±0.7% \| 16/512/4096 — 1265 ±0.3% | 6 |
| `ds-id17` | 4096×7168×4096 | 64/1024/2048 — 1336 ±0.6% \| 16/1024/2048 — 1335 ±0.3% \| 128/1024/2048 — 1334 ±0.2% | 7 |
| `ds-id18` | 4096×2048×7168 | 16/1024/4096 — 1501 ±2.6% \| 32/1024/2048 — 1463 ±0.9% \| 128/1024/2048 — 1412 ±3.6% | 2 |
| `ds-id2` | 64×1536×24576 | 32/1024/4096 — 791 ±0.5% \| 128/1024/2048 — 790 ±2.7% \| 32/512/4096 — 784 ±2.0% | 13 |
| `ds-id3` | 64×512×32768 | 128/1024/2048 — 778 ±0.2% \| 128/1024/1024 — 776 ±0.4% \| 16/1024/512 — 776 ±0.6% | 20 |
| `ds-id4` | 64×16384×7168 | 16/1024/512 — 766 ±1.3% \| 128/1024/1024 — 763 ±1.1% \| 128/1024/512 — 762 ±1.4% | 8 |
| `ds-id5` | 64×7168×4096 | 64/512/2048 — 750 ±0.7% \| 64/1024/2048 — 750 ±0.6% \| 128/512/2048 — 741 ±1.3% | 7 |
| `ds-id6` | 64×2048×7168 | 32/1024/1024 — 775 ±1.5% \| 16/1024/1024 — 768 ±1.0% \| 16/512/4096 — 768 ±0.9% | 13 |
| `ds-id7` | 128×7168×2112 | 128/512/2048 — 1058 ±0.8% \| 128/1024/1024 — 1053 ±1.2% \| 64/1024/1024 — 1050 ±1.7% | 10 |
| `ds-id8` | 128×1536×24576 | 16/1024/4096 — 1003 ±0.3% \| 16/512/2048 — 1001 ±0.1% \| 16/512/4096 — 1000 ±0.2% | 16 |
| `ds-id9` | 128×512×32768 | 128/1024/4096 — 987 ±0.3% \| 128/1024/512 — 986 ±0.3% \| 64/1024/1024 — 985 ±0.2% | 31 |
| `llama-id19` | 4096×4096×256 | 64/256/4096 — 1292 ±3.0% \| 32/256/4096 — 1291 ±1.4% \| 128/1024/4096 — 1291 ±3.2% | 10 |
| `llama-id20` | 11008×4096×256 | 64/256/4096 — 1285 ±3.3% \| 64/512/4096 — 1279 ±1.2% \| 32/256/4096 — 1273 ±2.4% | 8 |
| `llama-id21` | 4096×11008×256 | 64/256/4096 — 1247 ±3.9% \| 32/1024/4096 — 1242 ±1.3% \| 32/512/4096 — 1238 ±1.4% | 12 |
| `llama-id22` | 5120×5120×256 | 64/512/4096 — 1251 ±0.4% \| 64/1024/4096 — 1249 ±0.9% \| 128/512/4096 — 1247 ±2.4% | 10 |
| `llama-id23` | 13824×5120×256 | 64/256/4096 — 1247 ±0.8% \| 128/256/4096 — 1242 ±0.8% \| 32/1024/4096 — 1240 ±5.1% | 7 |
| `llama-id24` | 5120×13824×256 | 128/1024/4096 — 1237 ±0.3% \| 128/256/4096 — 1222 ±1.5% \| 64/256/4096 — 1220 ±0.8% | 8 |
| `llm` | 64×8192×512 | 16/512/512 — 925 ±2.5% \| 16/1024/1024 — 922 ±2.6% \| 128/512/2048 — 915 ±1.1% | 17 |
| `llm` | 64×16384×512 | 128/512/2048 — 872 ±0.4% \| 64/512/2048 — 872 ±0.2% \| 128/1024/2048 — 871 ±0.3% | 14 |
| `llm` | 64×32768×512 | 16/1024/2048 — 871 ±0.3% \| 128/1024/2048 — 871 ±0.5% \| 64/512/2048 — 870 ±0.2% | 15 |
| `llm` | 128×8192×512 | 16/512/2048 — 1111 ±0.1% \| 64/512/2048 — 1111 ±0.2% \| 16/1024/2048 — 1111 ±1.2% | 13 |
| `llm` | 128×16384×512 | 16/512/2048 — 1078 ±0.1% \| 16/1024/2048 — 1077 ±0.2% \| 32/512/2048 — 1076 ±0.1% | 11 |
| `llm` | 128×32768×512 | 16/512/1024 — 1050 ±5.9% \| 16/1024/1024 — 1050 ±0.1% \| 32/512/1024 — 1049 ±0.1% | 10 |
| `square` | 256×256×256 | 16/256/512 — 1167 ±18.1% \| 16/1024/1024 — 1167 ±0.3% \| 16/1024/2048 — 1167 ±11.5% | 13 |
| `square` | 512×512×512 | 128/512/512 — 1605 ±0.0% \| 32/512/512 — 1605 ±0.0% \| 128/1024/512 — 1605 ±0.0% | 32 |
| `square` | 1024×1024×1024 | 32/1024/2048 — 1700 ±1.8% \| 32/1024/4096 — 1696 ±3.8% \| 128/1024/2048 — 1694 ±4.8% | 13 |
| `square` | 2048×2048×2048 | 128/1024/2048 — 1621 ±3.5% \| 16/1024/4096 — 1618 ±2.0% \| 64/1024/4096 — 1616 ±5.5% | 7 |
| `square` | 4096×4096×4096 | 64/1024/2048 — 1543 ±1.2% \| 128/1024/2048 — 1538 ±5.8% \| 32/1024/2048 — 1502 ±5.2% | 3 |

On many shapes the choice barely matters: `ds-id9` has 31 blockings within 3% of
its leader, `512³` has 32. The shapes where the choice is tight are `4096³`
(3 within 3%) and `ds-id18` (2).

## Measurement uncertainty, and where the space needs extending

- `256³` carries ±18% spread even at 5000 reps per run. One call is ~29 µs; its
  ranking is indicative only. `ds-id15` reaches ±17% at reps=1.
- Many 256³ candidates report identical times. That is not a harness artefact:
  with M=N=K=256, every Nc≥256, Kc≥256 and Mc≥256 clamps to a single block, so
  different parameters collapse to the same execution.
- **Thermal drift was checked and ruled out.** Six early shapes were re-measured
  with the reference blocking immediately after the 25-minute sweep, while hot:
  256³ 1167.1→1167.1, 512³ 1513.4→1513.7, 1024³ 1595.6→1641.9, 2048³
  1385.4→1358.6, 4096³ 1304.9→1343.6, `llm 64×8192×512` 893.5→883.1. Deviations
  within ±3% with **no consistent direction** — two shapes were faster when hot.
  `pmset -g therm` recorded no thermal or performance warning, and low-power mode
  was off. Raw data in `thermal_recheck_hot.csv`.
- **Two axes want extending.** Among per-shape winners, **Nc=1024 wins on 19 of
  35 shapes and Kc=4096 on 11 of 35** — both are the top of the swept range. The
  results reported here are the best of the space that was tested, and are not a
  claim of an optimum. Mc needs no extension: it is flat, and its boundary values
  win 9 and 8 times respectively, which is what a flat axis looks like.


---

## Extended axes — Nc up to 4096, Kc up to 16384

`nb_sweep_ext.csv` (96 new blockings + the reference, 35 shapes, 3885 rows, zero
incorrect candidates), merged with the base run into `nb_sweep_merged.csv`:
6895 pre-scan rows covering the **full 4 × 7 × 7 = 196 grid, each blocking
exactly once**.

Nc and Kc were extended because their per-shape winners sat on the old top edge.
Mc was not: over the base grid it was flat, and a flat axis is what boundary wins
on both ends look like. Both runs measure the same fixed reference per shape in
the same pass, which is what makes the `vs_ref` ratios comparable between them —
absolute GFLOP/s are not comparable across the two binaries.

A global warm-up (512³ × 12) and a per-shape warm-up (reference blocking × 3,
after A/B/C are filled) were added for this run. The base run had neither, which
is why its very first measured candidate carried a 54% spread.

**The two sweeps therefore ran under different warm-up policies**, and the merged
table mixes them. That is acceptable for what the merged table is used for --
narrowing 196 blockings down to a handful of candidates -- and it is not
acceptable as the basis for a final choice. The selection between candidates is
made in a separate run under common conditions; see `final5.csv`.

### The extension did not move the answer

| Mc/Nc/Kc | geomean | worst | best | >+5% | within | <−5% |
|---|---:|---:|---:|---:|---:|---:|
| 32/1024/2048 | 1.051× | 0.902× | 1.209× | 17 | 16 | 2 |
| **32/2048/1024** (new) | **1.051×** | 0.819× | 1.174× | 18 | 16 | 1 |
| 16/1024/2048 | 1.047× | 0.913× | 1.191× | 16 | 17 | 2 |
| **16/512/2048** | 1.046× | **0.980×** | 1.122× | 15 | 20 | **0** |
| 64/2048/1024 (new) | 1.046× | 0.819× | 1.187× | 18 | 15 | 2 |
| … | | | | | | |
| 16/4096/16384 (new) | 0.602× | 0.395× | 1.117× | | | |

The best new candidate ties the old best rather than beating it, and the largest
buffers are among the worst blockings in the whole grid. `16/512/2048` remains
the only candidate near the top that never regresses more than 2%.

### Both axes now turn over

Marginal geomean across the full 196 grid, one axis at a time:

| axis | values |
|---|---|
| Mc | 16: 0.858× · 32: 0.886× · 64: 0.905× · **128: 0.907×** |
| Nc | 64: 0.725× · 128: 0.880× · **256: 0.965×** · 512: 0.961× · 1024: 0.936× · 2048: 0.911× · 4096: 0.863× |
| Kc | 256: 0.852× · 512: 0.927× · **1024: 0.939×** · 2048: 0.931× · 4096: 0.905× · 8192: 0.855× · 16384: 0.816× |

Nc peaks at 256–512 and declines monotonically after; Kc peaks at 1024–2048 and
declines. In the base grid Nc looked flat-topped at 512–1024 only because the
range stopped there.

These marginals are not directly comparable to the base run's, and they do **not
isolate the effect of one axis**. Each number averages over every value of the
other two, so extending Nc and Kc changed the Mc marginal as well, purely through
which combinations now exist. A marginal that moves when a *different* axis gains
values is reporting an interaction, not that axis's own effect. Read these as a
coarse map of where the good region is, not as a per-axis response curve.

Mc's marginal is flat over the base grid and rises over the full grid, 0.858× →
0.907× from Mc=16 to Mc=128. Subject to the caveat above, that rise is not
evidence that Mc itself matters more than before: the full grid adds large Kc and
Nc values, and larger Mc may simply be less bad in combination with those. Mc
remains the weakest of the three axes, and per-shape winners stay spread out:
Mc=16 wins 14 shapes, 32 wins 3, 64 wins 10, 128 wins 8.

### No clear overall gain from extending further, within this sweep's scope

Boundary winners are shapes whose parameter **clamps to the dimension**, not
shapes that want a larger value:

| shape | winning Kc | actual K | effective Kc |
|---|---|---:|---|
| `ds-id3` 64×512×32768 | 16384 | 512 | 512 = K |
| `ds-id9` 128×512×32768 | 16384 | 512 | 512 = K |
| `ds-id8` 128×1536×24576 | 16384 | 1536 | 1536 = K |
| `ds-id14` 4096×1536×24576 | 16384 | 1536 | 1536 = K |
| `2048³` | 16384 | 2048 | 2048 = K |

Both shapes picking Nc=4096 have N=2112, so their effective Nc is 2112 = N.

For these shapes, picking the top of the range means **"do not block this axis at
all"**, and the clamp already delivers that — a larger swept value would produce
the same execution. **17 of 35 shapes want an effective Kc equal to K**, i.e. a
single K slice with no K blocking.

That argument covers the shapes whose winner clamps. It does not prove the space
is exhausted in general: a shape with both a very large N and a very large K
could still prefer a value above the range swept here, and none of the 35 shapes
puts both dimensions at their maximum at once. The claim this sweep supports is
narrower: **extending Nc to 4096 and Kc to 16384 produced no clear overall gain**
— the best new candidate ties the previous best rather than beating it.

### Caveat on the large-buffer candidates

Every candidate allocates its packed buffers inside its own timed call, which is
what keeps the allocation scope identical across the grid. In the base run the
largest packed B was 16 MiB; here it reaches 256 MiB (Nc=4096 × Kc=16384), and
that allocation plus its first-touch page faults is charged per call. A
production caller would hoist it. So the measured decline at the far end of both
axes is an **upper bound** on the real cost of those blockings.

**How much of that decline is allocation has not been measured.** The split
between allocation, first-touch page faults and the genuine working-set cost is
unknown, and attributing the loss to any one of them would be a guess. Hoisting
the allocation out of the timed call is a separate experiment, and it would have
to be applied to every entrant to stay fair.

The clamping argument above does not depend on these timings, so it stands
regardless.


---

# Five-way comparison on fresh common conditions

`final5.csv`, `final5_summary.txt`. One binary, one session, all 35 shapes,
5 runs each, cyclic Latin square over shapes and runs, reps calibrated on the
slowest entrant, median of per-call timings, common warm-up plus a per-shape
warm-up on the freshly filled matrices, `caffeinate -dimsu`.

**No timing from the parameter sweep is carried into this table.** The sweep
narrowed 196 blockings to two candidates; the comparison between them is made
here, under conditions all five entrants share.

| entrant | what |
|---|---|
| `v3` | `SMEKernels1x4AccKcOut::run_multiplication`, blocking from `tuning::select` |
| `v4a` | Nc-blocked v4 at **Mc16 / Nc512 / Kc2048** — default candidate |
| `v4b` | Nc-blocked v4 at **Mc32 / Nc1024 / Kc2048** — alternative fixed profile |
| `mp` | MpGEMM `row_sgemm`, wrapped for its d8–d15 ABI bug |
| `acc` | Accelerate `cblas_sgemm`, `BLAS_THREADING_SINGLE_THREADED` |

**Allocation scope.** Every entrant allocates its packing buffers inside its own
timed call and frees them before returning. v3 and v4 are identical here by
construction — the same `aligned_alloc` / `unique_ptr` path in their own driver.
MpGEMM does its own `posix_memalign`/`free` inside `row_sgemm`. Accelerate is
opaque but is called identically every time. No entrant gets a hoisted buffer.

**The two v4 profiles are reported separately everywhere.** Picking the better of
the two per row would describe a dispatch that does not exist.

## Per shape, GFLOP/s

| shape | M×K×N | v3 | v4a | v4b | MpGEMM | Accel | v4a/v3 · v4b/v3 | v4a/mp · v4b/mp | v4a/acc · v4b/acc |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| `square` | 256×256×256 | 1401 | 1167 | 1141 | 1478 | 1790 | 0.833 / 0.815 | 0.790 / 0.772 | 0.652 / 0.637 |
| `square` | 512×512×512 | 1660 | 1601 | 1602 | 1804 | 1780 | 0.965 / 0.965 | 0.888 / 0.888 | 0.900 / 0.900 |
| `square` | 1024×1024×1024 | 1740 | 1730 | 1744 | 1889 | 1677 | 0.994 / 1.003 | 0.916 / 0.924 | 1.032 / 1.040 |
| `square` | 2048×2048×2048 | 1566 | 1581 | 1706 | 1548 | 1687 | 1.010 / 1.090 | 1.022 / 1.102 | 0.937 / 1.011 |
| `square` | 4096×4096×4096 | 1618 | 1536 | 1657 | 1434 | 1596 | 0.950 / 1.025 | 1.071 / 1.155 | 0.963 / 1.038 |
| `llm` | 64×8192×512 | 892 | 895 | 892 | 1005 | 955 | 1.003 / 0.999 | 0.891 / 0.887 | 0.938 / 0.934 |
| `llm` | 64×16384×512 | 804 | 866 | 865 | 907 | 856 | 1.077 / 1.076 | 0.955 / 0.954 | 1.012 / 1.011 |
| `llm` | 64×32768×512 | 800 | 868 | 865 | 914 | 862 | 1.084 / 1.080 | 0.949 / 0.946 | 1.006 / 1.003 |
| `llm` | 128×8192×512 | 1069 | 1159 | 1151 | 1240 | 1195 | 1.084 / 1.077 | 0.935 / 0.928 | 0.971 / 0.964 |
| `llm` | 128×16384×512 | 1040 | 1141 | 1140 | 1191 | 1131 | 1.096 / 1.096 | 0.958 / 0.957 | 1.009 / 1.008 |
| `llm` | 128×32768×512 | 1035 | 1141 | 1138 | 1196 | 1142 | 1.102 / 1.099 | 0.954 / 0.952 | 0.999 / 0.997 |
| `ds-id1` | 64×7168×2112 | 928 | 940 | 836 | 1077 | 1007 | 1.013 / 0.901 | 0.873 / 0.776 | 0.933 / 0.830 |
| `ds-id2` | 64×1536×24576 | 890 | 843 | 808 | 853 | 601 | 0.947 / 0.907 | 0.988 / 0.947 | 1.403 / 1.344 |
| `ds-id3` | 64×512×32768 | 862 | 831 | 842 | 872 | 616 | 0.964 / 0.977 | 0.952 / 0.965 | 1.349 / 1.367 |
| `ds-id4` | 64×16384×7168 | 799 | 836 | 745 | 879 | 636 | 1.047 / 0.932 | 0.951 / 0.847 | 1.314 / 1.171 |
| `ds-id5` | 64×7168×4096 | 739 | 758 | 807 | 968 | 598 | 1.026 / 1.093 | 0.783 / 0.834 | 1.266 / 1.349 |
| `ds-id6` | 64×2048×7168 | 806 | 867 | 762 | 906 | 662 | 1.075 / 0.945 | 0.957 / 0.841 | 1.310 / 1.152 |
| `ds-id7` | 128×7168×2112 | 1212 | 1201 | 1104 | 1371 | 1296 | 0.991 / 0.911 | 0.876 / 0.805 | 0.927 / 0.852 |
| `ds-id8` | 128×1536×24576 | 1178 | 1131 | 1113 | 1105 | 890 | 0.961 / 0.945 | 1.024 / 1.007 | 1.271 / 1.251 |
| `ds-id9` | 128×512×32768 | 1162 | 1102 | 1126 | 1158 | 872 | 0.948 / 0.969 | 0.951 / 0.972 | 1.264 / 1.292 |
| `ds-id10` | 128×16384×7168 | 1152 | 1092 | 1008 | 1121 | 831 | 0.948 / 0.875 | 0.974 / 0.899 | 1.315 / 1.212 |
| `ds-id11` | 128×7168×4096 | 1167 | 1101 | 1096 | 1240 | 892 | 0.944 / 0.939 | 0.888 / 0.883 | 1.235 / 1.228 |
| `ds-id12` | 128×2048×7168 | 1091 | 1151 | 1075 | 1172 | 971 | 1.055 / 0.986 | 0.982 / 0.918 | 1.186 / 1.108 |
| `ds-id13` | 4096×7168×2112 | 1558 | 1454 | 1541 | 1495 | 1557 | 0.934 / 0.989 | 0.973 / 1.031 | 0.934 / 0.990 |
| `ds-id14` | 4096×1536×24576 | 1540 | 1549 | 1637 | 1433 | 1610 | 1.006 / 1.063 | 1.081 / 1.142 | 0.963 / 1.017 |
| `ds-id15` | 4096×512×32768 | 1585 | 1513 | 1559 | 1503 | 1529 | 0.955 / 0.984 | 1.007 / 1.037 | 0.989 / 1.019 |
| `ds-id16` | 4096×16384×7168 | 1557 | 1467 | 1544 | 1398 | 1465 | 0.942 / 0.991 | 1.049 / 1.104 | 1.001 / 1.054 |
| `ds-id17` | 4096×7168×4096 | 1527 | 1475 | 1565 | 1484 | 1489 | 0.966 / 1.025 | 0.994 / 1.055 | 0.990 / 1.051 |
| `ds-id18` | 4096×2048×7168 | 1573 | 1551 | 1634 | 1461 | 1600 | 0.986 / 1.039 | 1.062 / 1.119 | 0.969 / 1.021 |
| `llama-id19` | 4096×4096×256 | 1115 | 1349 | 1355 | 1418 | 1383 | 1.210 / 1.216 | 0.951 / 0.956 | 0.976 / 0.980 |
| `llama-id20` | 11008×4096×256 | 1115 | 1351 | 1361 | 1400 | 1390 | 1.212 / 1.221 | 0.965 / 0.972 | 0.973 / 0.979 |
| `llama-id21` | 4096×11008×256 | 1199 | 1329 | 1329 | 1425 | 1375 | 1.108 / 1.108 | 0.932 / 0.932 | 0.967 / 0.967 |
| `llama-id22` | 5120×5120×256 | 1120 | 1337 | 1348 | 1393 | 1365 | 1.193 / 1.204 | 0.960 / 0.968 | 0.980 / 0.988 |
| `llama-id23` | 13824×5120×256 | 1136 | 1341 | 1354 | 1422 | 1378 | 1.180 / 1.192 | 0.943 / 0.952 | 0.973 / 0.983 |
| `llama-id24` | 5120×13824×256 | 1229 | 1327 | 1321 | 1409 | 1381 | 1.080 / 1.075 | 0.942 / 0.938 | 0.961 / 0.957 |

Spreads are in the CSV; they are mostly under 4%. The exceptions are `256³`
(13.6% on v3, one call is ~18 µs), `ds-id16` (9.1% on v3 at reps=3), `ds-id2`
(6.8%), `llama-id20` (6.4%) and `4096³` (5.1%). Rankings on those five rows are
indicative.

## Overall

Geometric mean of the ratio, shapes equally weighted, and win/tie/loss at ±5%:

| | geomean | win / tie / loss |
|---|---:|---|
| v4a vs v3 | **1.022×** | 13 / 14 / 8 |
| v4a vs MpGEMM | 0.952× | 3 / 19 / 13 |
| v4a vs Accelerate | **1.041×** | 10 / 18 / 7 |
| v4b vs v3 | **1.019×** | 14 / 12 / 9 |
| v4b vs MpGEMM | 0.949× | 6 / 12 / 17 |
| v4b vs Accelerate | **1.037×** | 12 / 18 / 5 |
| v4b vs v4a | 0.997× | 8 / 21 / 6 |

Both v4 profiles are ahead of v3 and ahead of Accelerate on the geometric mean,
and behind MpGEMM by about 5%. Against each other they are a wash overall —
0.997×, with 21 of 35 shapes inside ±5%.

## By regime — where the two profiles actually differ

| regime | n | v4a/v3 | v4a/mp | v4a/acc | v4b/v3 | v4b/mp | v4b/acc | v4b/v4a |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| narrow N (≤256) | 7 | **1.109×** | 0.924× | 0.918× | **1.109×** | 0.925× | 0.918× | 1.000× |
| narrow M (≤128) | 18 | **1.019×** | 0.934× | **1.138×** | 0.986× | 0.904× | 1.102× | 0.968× |
| all three large | 8 | 0.973× | 1.019× | 0.973× | **1.028×** | **1.077×** | **1.028×** | 1.056× |
| mixed | 2 | 0.960× | 0.946× | 0.943× | 0.974× | 0.960× | 0.958× | 1.015× |

The overall wash hides a clean split:

- **Narrow N: the two profiles are identical** (1.000×). With N=256, Nc=512 and
  Nc=1024 both clamp to 256, and Mc is the only difference left — which does
  nothing here. Both are 1.109× over v3, and both are ~8% behind MpGEMM and
  Accelerate. This is the regime the Nc-blocked kernel improved most over v3.
- **Narrow M: v4a wins**, 0.968× the other way. It is also where v4 is furthest
  ahead of Accelerate (1.138×) and furthest behind MpGEMM (0.934×).
- **All three large: v4b wins**, 1.056×. This is the only regime where a v4
  profile beats MpGEMM (1.077×), and v4a actually loses to v3 here (0.973×)
  while v4b beats it (1.028×).

## The 14 shapes where the profiles differ by more than 5%

| shape | M×K×N | v4b/v4a | spreads (v4a / v4b) |
|---|---|---:|---|
| `2048³` | 2048×2048×2048 | 1.079× | 2.2% / 2.5% |
| `4096³` | 4096×4096×4096 | 1.079× | 0.9% / 2.0% |
| `ds-id5` | 64×7168×4096 | 1.065× | 1.9% / 2.4% |
| `ds-id17` | 4096×7168×4096 | 1.062× | 2.1% / 1.0% |
| `ds-id13` | 4096×7168×2112 | 1.060× | 2.1% / 3.1% |
| `ds-id14` | 4096×1536×24576 | 1.057× | 1.8% / 2.9% |
| `ds-id18` | 4096×2048×7168 | 1.054× | 3.7% / 0.8% |
| `ds-id16` | 4096×16384×7168 | 1.053× | 1.0% / 3.5% |
| `ds-id12` | 128×2048×7168 | 0.934× | 0.3% / 2.7% |
| `ds-id10` | 128×16384×7168 | 0.922× | 1.4% / 1.2% |
| `ds-id7` | 128×7168×2112 | 0.919× | 4.2% / 1.8% |
| `ds-id4` | 64×16384×7168 | 0.891× | 1.8% / 3.2% |
| `ds-id1` | 64×7168×2112 | 0.889× | 1.2% / 2.4% |
| `ds-id6` | 64×2048×7168 | 0.879× | 0.4% / 3.2% |

Every shape v4b wins has M=4096 except `ds-id5`; every shape v4a wins has
M ≤ 128. The differences run to 12% and the spreads on those rows are 0.3–4.2%,
so they are above measurement noise — but this is **one session**. Whether the
split repeats has not been tested, and a two-profile selection should not be
built on a single run.

No automatic dispatch has been added.

## Correctness

Checked on every shape in the same run, against v3. Largest relative Frobenius
error: **1.091e-06** for either v4 profile, 3.331e-06 for MpGEMM and the same for
Accelerate. These are fp32 accumulation-order differences over long K, not
defects.


---

# v4c — a shape-driven choice between two fixed profiles

`v4c3.csv`, `v4c3_summary.txt`. Accelerate / v4c / MpGEMM, one binary, one
session, all 35 shapes, 5 runs each, cyclic Latin square, common warm-up plus a
per-shape warm-up, 0.5 s of timed work per entrant per run (reps capped at
20000, which is what the small shapes needed), `caffeinate -dimsu`.

## The rule

```cpp
Blocking choose(size_t M, size_t K, size_t N) {
    if (M >= 2048 && N > 512) return {32, 1024, 2048};   // v4b
    return {16, 512, 2048};                              // v4a
}
```

Evaluated once at entry, before anything is allocated and before streaming mode
is entered; the chosen parameters go to the existing Nc-blocked driver unchanged.
K is deliberately not a condition. No shape-ID lookup, no exact-dimension
matching, no runtime autotuning, and it never runs both profiles — every input
takes exactly one path. The rule was fixed before this run and not changed after.

v4c lives in its own translation unit, so the kernel TU measured in `final5.csv`
is byte-identical here; `source/SHA256SUMS.txt` records that it did not change.
The dispatch and the packing-buffer allocation are inside v4c's timed call.

The rule chose **v4a on 27 shapes and v4b on 8**. All 35 ran natively; there are
no fallback rows.

## Predicted from `final5.csv` before this run — not a v4c measurement

Applying the rule to the already-measured per-shape profile timings, it picks the
faster of the two on 26 of 35 shapes, and its geometric mean against a
per-shape oracle is **0.9961×**, i.e. 0.39% behind. The nine misses:

| shape | M×K×N | rule | measured faster | shortfall |
|---|---|---|---|---:|
| **`ds-id5`** | 64×7168×4096 | v4a | v4b | **−6.1%** |
| `ds-id9` | 128×512×32768 | v4a | v4b | −2.1% |
| `ds-id3` | 64×512×32768 | v4a | v4b | −1.3% |
| `llama-id23` | 13824×5120×256 | v4a | v4b | −1.0% |
| `llama-id22` | 5120×5120×256 | v4a | v4b | −0.9% |
| `1024³` | 1024×1024×1024 | v4a | v4b | −0.8% |
| `llama-id20` | 11008×4096×256 | v4a | v4b | −0.7% |
| `llama-id19` | 4096×4096×256 | v4a | v4b | −0.5% |
| `512³` | 512×512×512 | v4a | v4b | −0.04% |

`ds-id5` is the known exception: M=64 but v4b wins, so it sits outside the
"large M → v4b" pattern the rule encodes. The rule was not bent to catch it.
The six `llama` and square misses cost ≤1%: at N=256 both profiles clamp Nc to N
and differ only in Mc, which `final5` measured at exactly 1.000×.

## Measured

| shape | M×K×N | profile | Mc/Nc/Kc | Accel | v4c | MpGEMM | v4c/acc | v4c/mp | spreads a/v/m % |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `square` | 256×256×256 | **v4a** | 16/512/2048 | 1790 | 1165 | 1480 | 0.651 | 0.787 | 0.2 / 0.0 / 0.2 |
| `square` | 512×512×512 | **v4a** | 16/512/2048 | 1779 | 1602 | 1804 | 0.901 | 0.888 | 0.1 / 0.0 / 0.4 |
| `square` | 1024×1024×1024 | **v4a** | 16/512/2048 | 1677 | 1710 | 1880 | 1.020 | 0.910 | 1.0 / 1.8 / 0.7 |
| `square` | 2048×2048×2048 | **v4b** | 32/1024/2048 | 1697 | 1680 | 1550 | 0.990 | 1.083 | 3.5 / 3.0 / 2.4 |
| `square` | 4096×4096×4096 | **v4b** | 32/1024/2048 | 1603 | 1678 | 1482 | 1.046 | 1.132 | 1.8 / 1.0 / 6.4 |
| `llm` | 64×8192×512 | **v4a** | 16/512/2048 | 972 | 905 | 1050 | 0.932 | 0.863 | 0.5 / 0.2 / 2.2 |
| `llm` | 64×16384×512 | **v4a** | 16/512/2048 | 859 | 873 | 910 | 1.016 | 0.959 | 2.4 / 3.4 / 1.8 |
| `llm` | 64×32768×512 | **v4a** | 16/512/2048 | 835 | 849 | 894 | 1.017 | 0.951 | 1.1 / 0.7 / 1.0 |
| `llm` | 128×8192×512 | **v4a** | 16/512/2048 | 1164 | 1147 | 1213 | 0.985 | 0.946 | 1.2 / 2.8 / 3.8 |
| `llm` | 128×16384×512 | **v4a** | 16/512/2048 | 1131 | 1137 | 1198 | 1.006 | 0.950 | 1.6 / 0.3 / 0.1 |
| `llm` | 128×32768×512 | **v4a** | 16/512/2048 | 1137 | 1141 | 1198 | 1.004 | 0.953 | 1.5 / 0.2 / 0.1 |
| `ds-id1` | 64×7168×2112 | **v4a** | 16/512/2048 | 1017 | 921 | 1093 | 0.905 | 0.843 | 0.2 / 1.0 / 0.1 |
| `ds-id2` | 64×1536×24576 | **v4a** | 16/512/2048 | 603 | 858 | 860 | 1.423 | 0.998 | 0.1 / 0.6 / 0.2 |
| `ds-id3` | 64×512×32768 | **v4a** | 16/512/2048 | 617 | 840 | 875 | 1.363 | 0.961 | 0.3 / 0.5 / 0.6 |
| `ds-id4` | 64×16384×7168 | **v4a** | 16/512/2048 | 640 | 844 | 888 | 1.319 | 0.950 | 0.1 / 0.1 / 0.1 |
| `ds-id5` | 64×7168×4096 | **v4a** | 16/512/2048 | 604 | 755 | 978 | 1.251 | 0.772 | 0.9 / 1.0 / 0.1 |
| `ds-id6` | 64×2048×7168 | **v4a** | 16/512/2048 | 657 | 876 | 918 | 1.333 | 0.954 | 1.4 / 0.1 / 0.3 |
| `ds-id7` | 128×7168×2112 | **v4a** | 16/512/2048 | 1307 | 1202 | 1380 | 0.920 | 0.872 | 0.3 / 0.2 / 0.3 |
| `ds-id8` | 128×1536×24576 | **v4a** | 16/512/2048 | 902 | 1143 | 1123 | 1.267 | 1.018 | 0.1 / 0.1 / 0.6 |
| `ds-id9` | 128×512×32768 | **v4a** | 16/512/2048 | 871 | 1114 | 1163 | 1.278 | 0.958 | 0.3 / 0.4 / 0.6 |
| `ds-id10` | 128×16384×7168 | **v4a** | 16/512/2048 | 860 | 1107 | 1126 | 1.287 | 0.983 | 1.9 / 0.2 / 0.2 |
| `ds-id11` | 128×7168×4096 | **v4a** | 16/512/2048 | 897 | 1104 | 1259 | 1.231 | 0.877 | 0.5 / 0.4 / 0.2 |
| `ds-id12` | 128×2048×7168 | **v4a** | 16/512/2048 | 947 | 1158 | 1178 | 1.223 | 0.983 | 5.0 / 0.6 / 0.2 |
| `ds-id13` | 4096×7168×2112 | **v4b** | 32/1024/2048 | 1539 | 1546 | 1497 | 1.005 | 1.033 | 6.4 / 10.0 / 7.3 |
| `ds-id14` | 4096×1536×24576 | **v4b** | 32/1024/2048 | 1594 | 1566 | 1389 | 0.982 | 1.127 | 6.5 / 6.0 / 15.1 |
| `ds-id15` | 4096×512×32768 | **v4b** | 32/1024/2048 | 1468 | 1554 | 1481 | 1.058 | 1.049 | 18.5 / 5.1 / 8.8 |
| `ds-id16` | 4096×16384×7168 | **v4b** | 32/1024/2048 | 1480 | 1527 | 1398 | 1.032 | 1.092 | 4.5 / 2.2 / 0.6 |
| `ds-id17` | 4096×7168×4096 | **v4b** | 32/1024/2048 | 1500 | 1566 | 1490 | 1.044 | 1.050 | 3.0 / 1.1 / 2.6 |
| `ds-id18` | 4096×2048×7168 | **v4b** | 32/1024/2048 | 1610 | 1644 | 1460 | 1.021 | 1.126 | 0.8 / 3.1 / 2.5 |
| `llama-id19` | 4096×4096×256 | **v4a** | 16/512/2048 | 1376 | 1326 | 1414 | 0.964 | 0.938 | 3.5 / 3.7 / 2.9 |
| `llama-id20` | 11008×4096×256 | **v4a** | 16/512/2048 | 1383 | 1308 | 1396 | 0.945 | 0.937 | 3.7 / 4.4 / 5.3 |
| `llama-id21` | 4096×11008×256 | **v4a** | 16/512/2048 | 1377 | 1316 | 1417 | 0.956 | 0.929 | 4.3 / 3.2 / 3.1 |
| `llama-id22` | 5120×5120×256 | **v4a** | 16/512/2048 | 1365 | 1330 | 1416 | 0.974 | 0.939 | 3.6 / 2.9 / 3.7 |
| `llama-id23` | 13824×5120×256 | **v4a** | 16/512/2048 | 1374 | 1334 | 1413 | 0.971 | 0.944 | 4.6 / 3.2 / 1.4 |
| `llama-id24` | 5120×13824×256 | **v4a** | 16/512/2048 | 1369 | 1316 | 1406 | 0.961 | 0.936 | 3.7 / 1.5 / 2.2 |

| | geomean | win / tie / loss |
|---|---:|---|
| v4c vs Accelerate | **1.053×** | 11 / 18 / 6 |
| v4c vs MpGEMM | 0.959× | 6 / 13 / 16 |

| regime | n | v4c/acc | v4c/mp | w/t/l acc | w/t/l mp |
|---|---:|---:|---:|---|---|
| narrow N (≤256) | 7 | 0.910× | 0.914× | 0/5/2 | **0/0/7** |
| narrow M (≤128) | 18 | **1.141×** | 0.931× | 10/5/3 | 0/11/7 |
| all three large | 8 | 1.017× | **1.067×** | 0/8/0 | **6/1/1** |
| mixed | 2 | 0.976× | 0.965× | 1/0/1 | 0/1/1 |

Largest losses:

- **vs Accelerate**: `256³` 0.651×, `512³` 0.901×, `ds-id1` 0.905×,
  `ds-id7` 0.920×, `llm 64×8192×512` 0.932×.
- **vs MpGEMM**: `ds-id5` 0.772×, `256³` 0.787×, `ds-id1` 0.843×,
  `llm 64×8192×512` 0.863×, `ds-id7` 0.872×.

`ds-id5` is the worst row against MpGEMM and it is partly the rule's doing: the
prediction table says the rule gives up 6.1% there by choosing v4a.

## Reading it

v4c is ahead of Accelerate overall (1.053×) and behind MpGEMM (0.959×), which is
where the two fixed profiles already sat. What the rule buys is that the
all-three-large regime now uses v4b: it is the one regime where v4c **beats
MpGEMM**, 1.067× with 6 wins, 1 tie, 1 loss, and it ties Accelerate 0/8/0.

The narrow-M regime is where v4c is furthest ahead of Accelerate, 1.141×, driven
by the `ds-id2/3/4/6/8/9/10/11/12` family where Accelerate runs at 600–950
GFLOP/s against v4c's 840–1160.

Narrow N is the weak regime: 0.910× against Accelerate and **0/0/7 against
MpGEMM**. The profile choice cannot help there — both profiles clamp to the same
execution — so closing that gap needs a different change, not a different rule.

## Caveats

- Spreads are mostly under 4%, but five rows are noisy: `ds-id15` (18.5% on
  Accelerate), `ds-id14` (15.1% MpGEMM), `ds-id13` (10.0% v4c), `ds-id16` (4.5%),
  `4096³` (6.4% MpGEMM). Those are the low-rep shapes; their individual rankings
  are indicative.
- No timing from `final5.csv` is carried into this table. The two share no
  binary and no session.
- Correctness checked on every shape against v3 in the same run: largest relative
  Frobenius error **1.091e-06** for v4c, 3.331e-06 for Accelerate and MpGEMM.
- The rule rests on one prior session's regime split. A second rule candidate,
  if one is wanted, belongs in a separate experiment rather than an edit to this
  one.

## Files

| File | Contents |
|---|---|
| `f2_vs_r0.csv` | R0 vs F2, 9 runs |
| `f2_zapres_3way.csv` | R0 vs F2 vs F2-ZaPres, 9 runs |
| `f2_5way.csv` | the above plus Accelerate and MpGEMM, 10 runs |
| `v4_sweep.csv` | 35-shape sweep: v3 / generalised v4 / Accelerate / MpGEMM |
| `v4_sweep_summary.txt` | win/tie/loss, geomeans, regime breakdown, outliers |
| `nb_sweep.csv` | Nc/Mc/Kc sweep, 100 blockings x 35 shapes, both stages |
| `nb_sweep_summary.txt` | fixed-blocking ranking, per-shape leaders, axis trends |
| `nb_sweep_ext.csv` | extended axes: Nc to 4096, Kc to 16384; new blockings only |
| `nb_sweep_merged.csv` | base + extension, the full 4x7x7 grid, one row per blocking |
| `nb_sweep_merged_summary.txt` | analysis over the merged grid |
| `thermal_recheck_hot.csv` | early shapes re-measured hot, straight after the sweep |
| `final5.csv` | five-way comparison on common conditions, the decision table |
| `final5_summary.txt` | geomeans, win/tie/loss, regime split, profile differences |
| `v4c3.csv` | Accelerate / v4c / MpGEMM, the shape-driven two-profile rule |
| `v4c3_summary.txt` | geomeans, win/tie/loss, regime split, largest losses |
| `source/RERUN.txt` | exact commands to reproduce every run here |
| `environment.txt` | hardware, compiler, flags, entrant versions |
| `source/SHA256SUMS.txt` | sha256 of every source in the measured binary, the binary, and the same tracked files at `8f1ec86` |
| `source/tracked-relevant.diff` | working-tree diff of the tracked files these binaries were built from |
| `source/cmake-relevant.diff` | the build-file changes for these targets |
| `source/*.cpp`, `source/*.hpp` | full copies of the untracked sources |

The measured code is **not** the committed code: `sme/v3/sme-1x4-acc-kcout.cpp`
and `sme/support/gemm_tuning.*` carry uncommitted edits. Unrelated working-tree
changes in the 2x2 and 4x1 kernels, `bench_headline.cpp` and `.idea/` are not part
of these binaries and are excluded from `source/` on purpose.
