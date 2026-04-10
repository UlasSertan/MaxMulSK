# TODO / Active Bug List

---

## SME/SVE — Active Bugs (not yet building)

### BUG-1: Dispatch wrappers call `__arm_streaming` functions from normal mode
**File:** `sme/SME-GEMMKernels.cpp` — `pack_A_dispatch`, `pack_B_dispatch`  
**Problem:** Plain functions directly calling `__arm_streaming` functions — UB, Clang rejects it.  
**Fix:** Eliminate the wrappers entirely. `run_multiplication` is already `__arm_streaming`, so call `pack_A_streaming`/`pack_B_streaming` directly from it. No mode transition needed.

### BUG-2: `run_multiplication` is `__arm_streaming` — uncallable from normal code
**File:** `sme/SME-GEMMKernels.cpp` — `run_multiplication`  
**Problem:** The top-level driver requires the caller to already be in streaming mode. Cannot be called from `main` or any normal function.  
**Fix:** Change to `__arm_locally_streaming` so it manages SMSTART/SMSTOP itself.

### BUG-3: M-tail missing in `pack_A_streaming`
**File:** `sme/SME-GEMMKernels.cpp` — `pack_A_streaming` outer loop  
**Problem:** Loop iterates `m += SVL` and unconditionally loads SVL rows. If `M_curr % SVL != 0`, the last panel reads out-of-bounds rows.  
**Fix:** Predicated loads for the tail panel, or a scalar fallback for the partial M panel.

### BUG-4: `pack_B_streaming` tail loop stores with full predicate
**File:** `sme/SME-GEMMKernels.cpp` — `pack_B_streaming` tail loop  
**Problem:** Load uses `svwhilelt_b32_u64(n, N_curr)` (partial), but store uses `svptrue_b32()` (full). Inactive lanes contain garbage and get written into `packed_B`.  
**Fix:** Use `ld_mask` for the store as well.

### BUG-5: `tmp[16]` fixed-size array in K-tail of `pack_A_streaming`
**File:** `sme/SME-GEMMKernels.cpp` — `pack_A_streaming` K-tail  
**Problem:** `tmp` is fixed at 16 elements but the loop writes `SVL` of them. On M4 SVL=16 so it barely doesn't overflow, but it's fragile and non-portable.  
**Fix:** Replace with scalar stores directly into `panel_base + k * SVL + row`, no temp buffer needed.

### BUG-6: C is never zeroed in `run_multiplication`
**File:** `sme/SME-GEMMKernels.cpp` — `run_multiplication`  
**Problem:** Micro-kernel reads existing C and accumulates into it (`svadd`), but `run_multiplication` never zero-initializes C. Caller is silently required to pre-zero.  
**Fix:** Zero C at the start of `run_multiplication`, same as the NEON `package` does.

---

## SME/SVE — Known Non-Issues (deliberate)

- **Transpose hardcoded to 16 rows / 4 zip stages** (`pack_A_streaming` inner loop): only correct when SVL=16. M4 streaming SVL is 512-bit → `svcntsw()=16`, so this works on the target hardware. Not portable but not a bug for this project.

---

## Next Steps (in order)

1. Fix BUG-1 and BUG-2 to get SME compiling
2. Fix BUG-3, BUG-4, BUG-5, BUG-6 for correctness
3. Re-enable SME in `CMakeLists.txt` and `main.cpp`
4. Run full correctness suite against scalar reference at all tail combinations
5. Benchmark SME vs NEON
6. Rust scheduling layer — heterogeneous work distribution (P-core vs E-core tile sizing)
