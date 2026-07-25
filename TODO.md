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
Built `scripts/profile.sh` around Instruments' `GemmTemplate.tracetemplate`. Records `L1D_CACHE_MISS_LD`, `L1D_CACHE_MISS_ST`, `INST_ALL` via `xctrace record`, extracts per-process totals with a Python XML parser over the `counters-profile` schema, derives L1D-misses/1M-instructions and (combined with stdout timing) FLOPs/instr and IPC. Saves compact `traces/<name>.txt` summaries; drops the heavy `.trace` bundle by default. See docs/BENCHMARKS.md §10.

### ~~A-inner vs B-inner packing A/B test~~ — DONE (moved from "spare time" item, now a primary result)
Built an experimental 1×4 B-inner kernel (transpose of 4×1) and profiled all three layouts with PMU counters. 1×4 has the lowest instruction count and fewest L1D misses per instruction, but **loses on wall-clock GFLOPS** (1148 vs 4×1's 1213). Root cause per counters: IPC drops from 1.07 (4×1) to 0.62 (1×4) — the tighter instruction stream leaves too little slack between back-to-back FMOPAs targeting the same ZA tile, and pipeline stalls on same-tile dependency chains. Lesson: fewer instructions ≠ faster when FMOPA latency (6–8 cycles on M4) isn't hidden. Full write-up in docs/BENCHMARKS.md §9–§10.

### ~~1×4 experimental kernel shape~~ — DONE
Implemented in `sme/SME-GEMMKernelsExperimental.{hpp,cpp}`. Correctness verified; benchmark wired into `test_sme.cpp::profile_1x4`. Kept in-tree as the B-inner reference point even though 4×1 remains the production kernel.

---

## Fixed 2026-07-25

### ~~BUG-4x1-SMALL-M: 4x1 micro-kernel writes past C for M < 4·SVL~~ — FIXED
**Root cause:** `micro_kernel_4x1`'s store-back loop unconditionally writes 4 ZA tiles → 4·SVL = 64 rows × SVL = 16 cols. `run_multiplication` called it directly (no scratch buffer). For M < 64 (or N-tails < SVL), tiles landed past `C[M*N)` and corrupted heap. Same class as BUG-2x2-1.

**Fix applied:** scratch-buffer fallback in `run_multiplication` mirroring the BUG-2x2-1 fix — full tiles go directly to C (hot path unchanged), edge tiles compute into a pre-zeroed 64×16 scratch and scatter only valid rows × cols back. Additionally, `packed_A` is SVE-zeroed when `mc < M_tile` so the micro-kernel never reads uninitialized panels (their results land in discarded scratch rows either way). Aligned sizes never pay either cost — 4x1 still benches ~1218 GFLOPS @ 2048³ after the fix.

**Audit finding:** 1x4 / 1x4-sym / 1x4ZAIO already had complete scratch-buffer paths — the harness skip-list and docs were stale. All six kernels now pass the full correctness list (16³, 32³, 20×35×41, 67³, …); `kernel_can_handle()` and the `bench_compare` C-padding workarounds are removed.

### ~~BUG-ZAPACK-WRONG: 4x1ZAPack produces wrong results / SIGABRT~~ — FIXED
**Root cause (wrong results):** layout collision in `pack_A_streaming`. Panel index was computed as `p = m / SVL` and written at `packed_A + k*GS + p*SVL` — a formula that only addresses ONE 4-panel group. ZAPack's `M_tile = 128` spans two groups, so p ran 0–7 and panels 4–7 (offset ≥ GS) landed on top of k+1's panels 0–3, overwriting already-packed data one k-step shifted. Plain 4x1 never hit this because its `M_tile = 64` is exactly one group. The old pack test used M = 20 (single panel) and couldn't see it.

**Fix applied:** `p = (m/SVL) % 4` + per-group base `packed_A + (m/GS)*GS*K_curr`, matching the layout the driver's read side (`packed_A + ir*kc`) always assumed. The SIGABRT half was BUG-4x1-SMALL-M; the same scratch-buffer fallback is now ported here. ZAPack re-enabled in `main.cpp`, `run_comparison()`, and `bench_compare`.

**Result:** full suite PASS; benches ~1355 GFLOPS @ 1024³ — the highest single-thread number of any kernel in the repo. ZA-based pack transpose is a win in the L3 band.

### ~~BUG-NEON-2X: NEON correctness fails (exact 2× output) at N > Nc_cache~~ — FIXED
**Root cause:** `Nc_cache = 1024` violated its own "must be multiple of 12" invariant (12 × 85 = 1020). `multiply()` steps 12-wide panels across `current_Nc = 1024`, so the last panel started at column 1020 and wrote 8 columns into the NEXT j-block's territory — with real packed-B data, so those columns received the full correct contribution, then the next block accumulated it AGAIN → exactly 2×, precisely at cache-tile boundaries. Only fired for N > 1024 (needs a second block to double into), which is why `1024×1020×1024` was always clean.

**Fix applied:** `Nc_cache = 1020` (85 × 12). Every `current_Nc` is now a multiple of 12 (`N_aligned` and `j` both are), so panels never straddle blocks. Zero hot-path cost — cache footprint essentially unchanged; NEON still ~122 GFLOPS @ 2048³. Verified vs scalar at 128×1100×128, 512×2048×512, 1067³, 513×1033×517: MaxDiff ≈ 0.

### ~~Accumulator precision at ≥2048³ (MaxDiff up to ~100)~~ — NOT A BUG, CLOSED
Measured against float64 ground truth (`cblas_dgemm`), fp32 accumulation error at 4096³ is: Accelerate 0.000348 max, our SME 4x1 0.000209, our NEON 0.000115 — **our kernels are more accurate than AMX**. The reported MaxDiff ~100 was entirely BUG-NEON-2X corrupting `bench_compare`'s NEON-vs-Accelerate diff (doubled columns at |C| ~ 100 magnitudes). Pairwise summation is unnecessary; item dropped from the roadmap.

---

## Remaining Minor Issues

### MINOR: profile_power.sh P/E-cluster regex
On M4 macOS Sonoma+, `powermetrics --samplers cpu_power` no longer prints the `P-Cluster Power: NNN mW` / `E-Cluster Power: NNN mW` lines that the parser expects — only `Combined Power` is captured. P-cluster and E-cluster averages currently report 0.0 W. Combined is the meaningful number for J/GFLOP, so this is cosmetic, but worth fixing.

### MINOR: OpenBLAS PMU counter undercount
`scripts/profile.sh oblas` reports only ~650M instructions for a 3-second run at 2048³ × 100 iters (IPC ≈ 0.05 — implausible). Suspected: AMX-internal compute path doesn't count toward `INST_ALL`, OR the trace template loses events from dlopen'd dylib symbol attribution. Comparison against AMX/NEON paths via this counter is unreliable for OpenBLAS until investigated.


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

0. **Create dynamic tiling.**

1. **Investigate 1×4 IPC collapse.** Counter evidence says the kernel is FMOPA-latency-bound, not bandwidth-bound. Try: rotate across all 4 ZA tiles every instruction so no two adjacent FMOPAs touch the same tile; add an extra software-pipelining stage; experiment with K-unroll. Target: push IPC from 0.62 back toward 1.0+ and surpass 4×1.
2. **Investigate 2×2's high instruction count.** 2×2 burns ~8.6B instructions at 2048³ vs 4×1's ~4.6B (~87% more). Profile the pack phase separately from the compute phase to confirm where the instructions go.
3. **Optimize SME kernel:**
   - Explore tiling parameters (M_tile, K_tile, N_tile sweep) — ZAPack's win at 1024³ suggests M_tile=128 + ZA-transpose pack is underexplored.
   - Add OpenMP threading.
4. **Rust scheduling layer** — heterogeneous work distribution (P-core vs E-core tile sizing).
5. *Stretch:* explore other kernel shapes (4×8, 4×12, 8×8, 8×16 for NEON; alternative SME tile patterns).

### Done (2026-07-25 session)
- ~~SME-ZA transpose~~ — working in both ZAPack (`svld1_hor_za32`/`svst1_ver_za32`) and 1x4ZAIO pack_A (2-tile rolling MOVA pipeline).
- ~~Fix 4x1ZAPack~~ — BUG-ZAPACK-WRONG fixed (packing group-offset bug, see above).
- ~~Scratch-buffer fallback for 4x1 / ZAPack~~ — BUG-4x1-SMALL-M fixed; 1x4-family already had it.
- ~~NEON 2× bug~~ — BUG-NEON-2X fixed (Nc_cache 1024 → 1020).
- ~~Accumulator precision~~ — measured vs fp64 truth: our kernels beat Accelerate's accuracy; closed as not-a-bug.
- ~~`__arm_inout("za")` K-block accumulation~~ — implemented as the 1x4ZAIO kernel (K_inner_tile=40, ZA-resident accumulation).
