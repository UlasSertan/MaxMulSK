# TODO / Active Bug List

---

## SME/SVE — All Bugs Fixed

### ~~BUG-1: Dispatch wrappers call `__arm_streaming` functions from normal mode~~ — FIXED
**Fix applied:** Deleted `pack_A_dispatch`/`pack_B_dispatch` wrappers from `.cpp` and `.hpp`. `run_multiplication` now calls `pack_A_streaming`/`pack_B_streaming` directly (legal because it is itself in streaming mode).

### ~~BUG-2: `run_multiplication` is `__arm_streaming` — uncallable from normal code~~ — FIXED
**Fix applied:** Changed to `__arm_locally_streaming` (attribute goes before the function, not after params — Clang rejects it as a type qualifier). Function now manages its own SMSTART/SMSTOP.

### ~~BUG-3: M-tail missing in `pack_A_streaming`~~ — FIXED
**Fix applied:** Unified main loop and tail into a single loop using SVE predicates. Each panel computes `rows_here = min(SVL, M_curr - m)` and builds 16 per-row predicates (`pg` or `pfalse`). Predicated `svld1_f32` returns zero for invalid rows — no OOB access, no branching in the k loop. K scalar tail uses `rows_here` bound + SVE-zeroed tmp buffer (avoids `__arm_sc_memset`).

### ~~BUG-4: `pack_B_streaming` tail loop stores with full predicate~~ — FIXED (pre-existing)
pack_B test was already passing. The garbage lanes in inactive positions don't affect correctness because the micro-kernel's B panel access stays within valid column bounds. Functionally harmless on current usage patterns.

### ~~BUG-5: `tmp[16]` fixed-size array in K-tail of `pack_A_streaming`~~ — FIXED
**Fix applied:** K-tail now uses SVE-zeroed tmp buffer (`svst1_f32(pg, tmp, svdup_f32(0.0f))`) and only writes `rows_here` elements. Avoids `__arm_sc_memset` linker error that occurs when the compiler optimizes scalar zero-fill loops into streaming-incompatible memset calls.

### ~~BUG-6: C is never zeroed in `run_multiplication`~~ — FIXED
**Fix applied:** Caller (test harness) pre-zeroes C. Micro-kernel accumulates correctly.

### Additional fixes discovered during bug work:
- **Butterfly transpose permutation:** Original zip groupings `(r0,r1), (r2,r3)...` produced a non-identity row permutation within output columns. Fixed by regrouping stage 1 to distance-8 pairs `(r0,r8), (r1,r9)...` which produces correct identity row ordering.
- **`svread_hor_za32_f32_m` tile constant:** Tile index must be compile-time constant. Unrolled with `STORE_ZA_TILE` macro.
- **`__arm_sc_memset` linker error:** Scalar zero-fill loops in streaming functions get optimized to `__arm_sc_memset` which doesn't exist. All zero-fills replaced with SVE stores.
- **Micro-kernel software pipelining:** Prologue/epilogue pattern interleaves k+1 loads between svmopa instructions to hide load latency.

---

## SME/SVE — Known Non-Issues (deliberate)

- **Transpose hardcoded to 16 rows / 4 zip stages** (`pack_A_streaming` inner loop): only correct when SVL=16. M4 streaming SVL is 512-bit → `svcntsw()=16`, so this works on the target hardware. Not portable but not a bug for this project.

---

## SME 2x2 Micro-Kernel — Bugs & Fixes

### BUG-2x2-1: Heap corruption — micro-kernel writes past C buffer on edge tiles — FIXED

**Symptom:** `malloc: Incorrect checksum for freed object` crash. The 16×16×16 test appeared to PASS but silently corrupted heap metadata. The crash manifested one test later at 64×64×64 when malloc detected the damage.

**Root cause:** `run_multiplication` called `micro_kernel_2x2` unconditionally for every `(ir, jr)` block. The 2x2 kernel always writes a full `2*SVL × 2*SVL` (32×32) output tile into C. When remaining rows (`mc - ir`) or remaining columns (`nc - jr`) were less than 32, the kernel wrote past the valid region of C into unallocated heap memory.

Example at 16×16×16: `mc=16, nc=16, M_step=32, N_step=32`. The kernel wrote 32×32 = 1024 floats starting from `C[0,0]`, but C was only 16×16 = 256 floats. The STORE_ZA_TILE_2x2 macro writes at `C + ROW_OFF * wide_of_C + COL_OFF` — tile 3 (bottom-right) wrote to row offsets 16..31 and column offsets 16..31, all completely outside the buffer.

The 4x1 kernel never hit this because its N_step was 1×SVL=16 (matches the smallest test size) and M_step=64 matched M_tile=64 exactly, so the inner loops never had a partial tile.

**Fix:** Scratch buffer approach (same pattern as NEON N-tail). In `run_multiplication`:
1. Allocate a persistent `M_step × N_step` (32×32) scratch buffer once, outside all loops.
2. At each `(ir, jr)` iteration, check if `m_rem < M_step` or `n_rem < N_step`.
3. **Full tile (no tail):** Call `micro_kernel_2x2` directly with `C + offset` and `wide_of_C = N`. Zero overhead — this is the hot path.
4. **Edge tile (any tail):** Zero the scratch buffer, call `micro_kernel_2x2` with `C_scratch` and `wide_of_C = N_step` (so the kernel writes safely within the 32×32 buffer), then scatter only the valid `rows × cols` portion back into C with accumulation (`C[r][c] += scratch[r][c]`).

This keeps the micro-kernel completely unaware of tails — it always writes a full 32×32 tile. The driver handles the boundary.

### BUG-2x2-2: `__arm_sc_memset` linker error from `std::fill` in streaming function — FIXED

**Symptom:** `Undefined symbols: ___arm_sc_memset` at link time.

**Root cause:** Same issue as BUG-5 in the 4x1 kernel. The scratch buffer zeroing (`std::fill(C_scratch.get(), ..., 0.0f)`) inside `run_multiplication` (which is `__arm_locally_streaming`) gets optimized by the compiler into a call to `__arm_sc_memset` — the streaming-compatible version of memset that doesn't exist in any standard library.

Any scalar loop or standard library call that writes a repeated pattern to memory inside a streaming function is at risk of this optimization. The compiler sees "write zero to N consecutive floats" and emits a memset call, but in streaming mode it must use the streaming-compatible variant which has no implementation.

**Fix:** Replace `std::fill` with an explicit SVE store loop:
```cpp
svbool_t pg_z = svptrue_b32();
svfloat32_t zero = svdup_f32(0.0f);
float* zp = C_scratch.get();
for (size_t i = 0; i < M_step * N_step; i += SVL)
    svst1_f32(pg_z, zp + i, zero);
```
`M_step * N_step = (2*SVL)² = 4*SVL²` is always a multiple of SVL, so no predicate tail needed. The SVE store is a valid streaming-mode instruction and the compiler won't try to lower it to memset.

**General rule:** Never use `std::fill`, `std::memset`, `memset`, or any scalar zero-fill loop inside `__arm_streaming` or `__arm_locally_streaming` functions. Always zero memory with SVE vector stores.

---

## Recently Completed

### ~~Processor Trace / PMU counter instrumentation~~ — DONE
Built `profile.sh` around Instruments' `GemmTemplate.tracetemplate`. Records `L1D_CACHE_MISS_LD`, `L1D_CACHE_MISS_ST`, `INST_ALL` via `xctrace record`, extracts per-process totals with a Python XML parser over the `counters-profile` schema, derives L1D-misses/1M-instructions and (combined with stdout timing) FLOPs/instr and IPC. Saves compact `traces/<name>.txt` summaries; drops the heavy `.trace` bundle by default. See BENCHMARKS.md §10.

### ~~A-inner vs B-inner packing A/B test~~ — DONE (moved from "spare time" item, now a primary result)
Built an experimental 1×4 B-inner kernel (transpose of 4×1) and profiled all three layouts with PMU counters. 1×4 has the lowest instruction count and fewest L1D misses per instruction, but **loses on wall-clock GFLOPS** (1148 vs 4×1's 1213). Root cause per counters: IPC drops from 1.07 (4×1) to 0.62 (1×4) — the tighter instruction stream leaves too little slack between back-to-back FMOPAs targeting the same ZA tile, and pipeline stalls on same-tile dependency chains. Lesson: fewer instructions ≠ faster when FMOPA latency (6–8 cycles on M4) isn't hidden. Full write-up in BENCHMARKS.md §9–§10.

### ~~1×4 experimental kernel shape~~ — DONE
Implemented in `sme/SME-GEMMKernelsExperimental.{hpp,cpp}`. Correctness verified; benchmark wired into `test_sme.cpp::profile_1x4`. Kept in-tree as the B-inner reference point even though 4×1 remains the production kernel.

---

## Active Bugs

### BUG-4x1-SMALL-M: 4x1 micro-kernel writes past C for M < 4·SVL
**Symptom:** `malloc: Incorrect checksum for freed object` (SIGABRT, exit 134) when running GEMM correctness tests at sizes with M < 64. Surfaced by the wider correctness list (16³, 32³, 20×35×41) introduced during the 2026-04-26 cleanup. The kernel was always broken at small M; the old per-kernel test list happened to avoid SIGABRT due to heap-layout luck.

**Root cause:** `micro_kernel_4x1`'s store-back loop unconditionally writes 4 ZA tiles → 4·SVL = 64 rows × SVL = 16 cols. `run_multiplication` calls it directly (no scratch buffer). For M < 64, tiles 1..3 land past `C[M*N)` and corrupt heap. Same class as BUG-2x2-1.

**Workaround in place:** test harness skips small-M cases for K4x1 (`kernel_can_handle()` in `sme/test_sme.cpp`); `bench/bench_compare.cpp` pads `C_sme41` to `(4·SVL) * N` floats.

**Real fix needed:** scratch-buffer fallback in `run_multiplication`, mirroring the BUG-2x2-1 fix. Apply to 4x1, 1x4, and 4x1ZAPack (1x4's case is harder: writes interleave columns rather than overrunning rows, so a scratch buffer + scatter-back is mandatory there).

**Severity:** medium — production usage targets large M, but any caller passing M < 64 silently corrupts heap.

### BUG-ZAPACK-WRONG: 4x1ZAPack produces wrong results / SIGABRT
**Symptom:** Same SIGABRT as BUG-4x1-SMALL-M plus reportedly wrong outputs. Currently disabled in `main.cpp` and `run_comparison()` until the kernel logic is fixed.

**Status:** ZAPack uses ZA-based pack_A transpose (a new technique you are exploring), separate from the small-M issue above. Kernel logic itself is the work item.

### BUG-NEON-2X: NEON correctness fails (exact 2× output) at non-aligned shapes
**Symptom:** `bench_compare` MaxDiff blows up at 128×1100×128 (12.9), 2048³ (62.3), 4096³ (100.6), 4095³ (88.0), 1067³ (43.5), 512×2048×512 (28.6). For early indices the value is *exactly 2×* the reference, suggesting the kernel runs an extra pass over part of the K dimension at certain `N > Nc_cache` shapes. Pre-existing — surfaced by the cleanup, not introduced by it. Bench numbers (GFLOPS) are still valid; the result in C is wrong.

**Severity:** high for correctness — silently gives 2× wrong output. Investigate the K-loop / packed-B reuse path when N exceeds the cache-blocking parameter `Nc_cache`.

### MINOR: profile_power.sh P/E-cluster regex
On M4 macOS Sonoma+, `powermetrics --samplers cpu_power` no longer prints the `P-Cluster Power: NNN mW` / `E-Cluster Power: NNN mW` lines that the parser expects — only `Combined Power` is captured. P-cluster and E-cluster averages currently report 0.0 W. Combined is the meaningful number for J/GFLOP, so this is cosmetic, but worth fixing.

### MINOR: OpenBLAS PMU counter undercount
`profile.sh oblas` reports only ~650M instructions for a 3-second run at 2048³ × 100 iters (IPC ≈ 0.05 — implausible). Suspected: AMX-internal compute path doesn't count toward `INST_ALL`, OR the trace template loses events from dlopen'd dylib symbol attribution. Comparison against AMX/NEON paths via this counter is unreliable for OpenBLAS until investigated.


## Hypothesis (2026-04-29)
--------------

Hypothesis: Packing Cost vs Reuse Asymmetry (4×1 vs 1×4)
Observation:
In the 4×1 kernel, pack_A takes a significant portion of runtime (~22%).
pack_A is inherently more expensive than pack_B due to:
Butterfly transpose
Operating on large SVL×SVL tiles
pack_B is relatively cheap:
Mostly pointer rearrangement / streaming
Works on smaller chunks (e.g., 2×SVL per step)
Hypothesis:
Performance differences between 4×1 and 1×4(-sym) are driven by how often each operand is repacked vs reused.
In 4×1 (A-inner):
A is consumed more aggressively inside the microkernel (4 A loads per step)
This causes A panels to be “drained” faster
As a result, pack_A is invoked more frequently
Since pack_A is expensive, this creates a sustained overhead
CPU spends more time in packing relative to compute
In 1×4 (B-inner / sym variant):
B is reused more heavily across computation
A is packed less frequently and reused longer
Since pack_A is the more expensive operation, reducing its frequency improves performance
Even if B is packed more often, its lower cost makes this tradeoff favorable
Interpretation:
The dominant factor is not just the raw cost of packing, but:
How long each packed panel stays useful before needing to be repacked
If an expensive operand (A) is reused longer → better amortization → higher performance
If it is repacked frequently → overhead dominates
Additional context:
Current experiments use large K (≈2048), while M/N tile sizes are relatively small (e.g., 64×512 or 512×64)
This setup minimizes total packing iterations but still exposes differences in reuse patterns
Results are tuned for peak GFLOPS (large matrices), not small-size efficiency
What to verify:
Count how often pack_A and pack_B are called in each kernel
Measure time split between:
pack_A
pack_B
compute
Check whether A is actually repacked more frequently in 4×1 vs 1×4(-sym)
Validate whether longer reuse of A correlates with better performance
Status:
Unverified hypothesis — needs targeted experiments


---

## Next Steps (in order)

0. **SME-ZA transpose** (learned its existence from AI, implementing myself).
0. **Fix 4x1ZAPack** (BUG-ZAPACK-WRONG above) — wrong-result bug.
0. **Create dynamic tiling.**

1. **Add scratch-buffer fallback to 4x1 / 1x4 / ZAPack** (BUG-4x1-SMALL-M). Closes silent heap corruption at small M/N; lets us drop the test-harness padding workaround and re-enable small-size correctness sweeps for these kernels.
2. **Fix NEON 2× correctness bug** (BUG-NEON-2X). Investigate the K-blocking / packed-B reuse path when `N > Nc_cache`.
3. **Investigate 1×4 IPC collapse.** Counter evidence says the kernel is FMOPA-latency-bound, not bandwidth-bound. Try: rotate across all 4 ZA tiles every instruction so no two adjacent FMOPAs touch the same tile; add an extra software-pipelining stage; experiment with K-unroll. Target: push IPC from 0.62 back toward 1.0+ and surpass 4×1.
4. **Investigate 2×2's high instruction count.** 2×2 burns ~8.6B instructions at 2048³ vs 4×1's ~4.6B (~87% more). Profile the pack phase separately from the compute phase to confirm where the instructions go.
5. **Optimize SME kernel:**
   - Try `__arm_inout("za")` to accumulate across K-blocks without flushing ZA each call — enables smaller K_tile (e.g. 256) that fits L1, with store phase only once at the end.
   - Explore tiling parameters (M_tile, K_tile, N_tile sweep).
   - Improve N-tail handling.
   - Add OpenMP threading.
6. **Optimize pack_A in SME** so the micro-kernel does not need ps-strided access — enable x4 load instructions.
7. **Solve accumulator errors at large matrices** (4096³, MaxDiff up to ~100) via pairwise summation `((A + B) + (C + D))` instead of `(A + B + C + D)`.
8. **Rust scheduling layer** — heterogeneous work distribution (P-core vs E-core tile sizing).
9. *Stretch:* explore other kernel shapes (4×8, 4×12, 8×8, 8×16 for NEON; alternative SME tile patterns).
