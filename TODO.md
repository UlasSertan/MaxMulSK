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

## Next Steps (in order)

1. Update README.md and BENCHMARK.md where we claim 98% of theoretical limit and make it 96%, and check if our tests are performed on 4 GHz or 4.4 GHz (4.4GHz would make percentage 88%)
2. Optimize SME kernel — explore tiling, N-tail handling, threading
3. Benchmark SME vs NEON vs Accelerate (single-thread and multi-thread)
4. Solve accumulator errors in big matricies like 4096^3 via ((A + B) + (C + D) instead of A + B + C + D)
4. Rust scheduling layer — heterogeneous work distribution (P-core vs E-core tile sizing)
5. During spare time, do A/B testing for L1 cache misses in A-inner packing and B-inner packing in NEON
6. IF AND ONLY IF EVERYTHING IS DONE explore some different kernel sizes
