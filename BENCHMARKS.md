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
| SME/SVE | ~1200 | 1 thread, 2x2 and 4x1 kernels |

---

## 2. Power & Energy Analysis

Measurement tool: `powermetrics --samplers cpu_power`, 100 ms sampling interval.

### Experiment 1 — NEON (50 iterations)

| Metric | Value |
|---|---|
| Wall time | ~900 ms |
| Power samples (mW) | 135, 236, 84, 90, 53, 80, 117, 136, 235, 99 |
| Average power | ~126.5 mW |
| Estimated energy | ~0.11 J |

The NEON kernel produces short, dense compute bursts. Work completes fast enough that the CPU does not sustain high power draw for long. The low average power reflects the combination of fast execution and idle/tail periods falling within the 100 ms sampling window. Profile is compute-dense and energy-efficient.

### Experiment 2 — Scalar (1 iteration)

| Metric | Value |
|---|---|
| Wall time | ~790 ms |
| Power samples (mW) | 413, 133, 92, 90, 62, 72, 81, 126, 181, 108 |
| Average power | ~136 mW |
| Estimated energy | ~0.11 J |

A single scalar run produces one short burst. Only one sampling window captures the high-power phase; the remaining samples reflect cache-miss stalls and post-completion idle. The 100 ms resolution is too coarse to faithfully represent such a short compute phase.

### Experiment 3 — Scalar (3 iterations)

| Metric | Value |
|---|---|
| Wall time | ~2.37 s |
| Power samples (mW) | 230, 315, 361, 996, 208, 775, 383, 183, 91, 117 |
| Average power | ~366 mW |
| Estimated energy | ~0.87 J |

With 3 iterations the scalar kernel stays runnable long enough to fill sampling windows. The absence of tiling makes it memory-bound: execution units idle while waiting on memory, yet the frontend and memory subsystem continue drawing power. The 996 mW and 775 mW spikes correspond to brief moments of cache locality where compute-bound behavior temporarily emerges.

### Energy Comparison

| Kernel | Wall time | Avg power | Total energy | Relative energy |
|---|---|---|---|---|
| NEON (50×) | ~0.9 s | ~0.13 W | ~0.11 J | 1× |
| Scalar (1×) | ~0.79 s | ~0.14 W | ~0.11 J | ~1× |
| Scalar (3×) | ~2.37 s | ~0.37 W | ~0.87 J | ~8× |

**Key takeaway:** average wattage alone is not a useful efficiency metric. Total energy consumed (J) is. The scalar kernel consumes ~8× more energy to do the same work due to its memory-bound, inefficient execution pattern.

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

- **Block sizes:** Mc=64, Kc=256. Chosen to keep the working set (packed A panel + packed B panel) resident in L1/L2.
- **SIMD transpose (`transpose_8x4`):** Packs A into a layout the micro-kernel reads linearly, maximizing L1 bandwidth utilization and eliminating gather-load patterns.
- **Store-to-Load Forwarding:** "A-inner-packing" (current structure) outperforms the BLIS-style "A-above-B" hierarchy on Apple Silicon.

### Architecture Note — BLIS-style Hierarchy

Tested BLIS-style "A-above-B" cache hierarchy on Apple Silicon P-cores. Contrary to expectations, ~5–10% performance regression was observed. Hypothesis: heavy B-packing traffic evicts freshly-packed A data from L1 before the micro-kernel can reuse it. The current "A-inner-packing" structure avoids this by exploiting store-to-load forwarding, which was confirmed to be the superior strategy on this microarchitecture.
