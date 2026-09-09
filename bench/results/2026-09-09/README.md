# 2026-09-09 — MaxMulSK vs OpenBLAS

`bench/bench_vs_openblas.cpp`, single thread, Apple M4. Four OpenBLAS
configurations were measured, because the first one turned out not to be a fair
representation of what OpenBLAS can do on this machine.

| file | OpenBLAS build |
|---|---|
| `vs_openblas_0334_brew.*`        | 0.3.34 from Homebrew, DYNAMIC_ARCH, detects `vortexm4` |
| `vs_openblas_develop_sme.*`      | develop @2154be86, static `TARGET=ARMV9SME` |
| `vs_openblas_develop_dynarch.*`  | develop @2154be86, `DYNAMIC_ARCH=1` |
| `vs_openblas_develop_tuned.*`    | as above + `OPENBLAS_DIRECT_LIMIT=1792` (local patch) |

## Why four

Two separate things were holding OpenBLAS back, and neither is about MaxMulSK.

**1. The blocked SME GEMM is unreachable without DYNAMIC_ARCH.**
`sme_sgemm_kernel.c` landed upstream on 2026-08-11 and is absent from the
v0.3.34 release. `interface/gemm.c` reaches it only under
`#if defined(USE_SGEMM_KERNEL_DIRECT) || defined(DYNAMIC_ARCH)`, gated on
`gotoblas_corename()` being `armv9sme` or (under clang) `vortexm4`. A static
`TARGET=ARMV9SME` build compiles the kernel but never calls it: `KERNEL.ARMV9SME`
is one line, `include KERNEL.ARMV8SVE`, so SGEMMKERNEL stays SVE. Measured: the
static SME build performs the same as stock and dies with SIGILL at 4096^3.
`OPENBLAS_CORETYPE=armv9sme` on the DYNAMIC_ARCH build also SIGILLs at load, so
the working path on this machine is the auto-detected `vortexm4`.

**2. The direct-path threshold is calibrated against a kernel that is no longer
the alternative.** `kernel/arm64/sgemm_direct_performant.c` returns "use the
direct path" for `M*N*K < 3100^3`, and its own comment says the number is where
the direct path "crosses the graph of the NEON SGEMM". With the SME GEMM kernel
now on the other side of that branch, the crossover moved. Measured, forcing
each path at every size:

| n | SGEMM_DIRECT | SME_SGEMM_KERNEL |
|---:|---:|---:|
| 1280 | **1615.8** | 1307.3 |
| 1536 | **1504.2** | 1367.5 |
| 1792 | 1017.3 | **1404.5** |
| 2048 | 637.3 | **1302.5** |
| 4096 | 407.9 | **1242.7** |

The direct path peaks at 1280 and decays; the SME kernel is flat at ~1240-1400.
The real crossover is near 1792, not 3100, so every square between roughly 1800
and 3100 takes the wrong branch. That is the whole of the 2048^3 collapse:
632 -> 1299 GFLOP/s once the threshold is corrected.

`OPENBLAS_DIRECT_LIMIT` is a LOCAL patch to the clone in `~/src/OpenBLAS`, not
an upstream feature. It exists to find the crossover; the finding is worth
reporting upstream, the env var is not.

## Caveats

- Single run each. Not yet repeated three times, which this repo's own rules
  require before a number is quoted as settled.
- The `ARMV9SME` code path is broken on this machine in two independent ways
  (SIGILL at load when forced, SIGILL at 4096^3 when built statically, plus an
  illegal instruction in the `ismin` test). Only the `vortexm4` path was usable.
- OpenBLAS builds used `USE_THREAD=0 USE_OPENMP=0 NO_LAPACK=1 NO_FORTRAN=1`
  and a compiler wrapper forcing `-isysroot`, because Homebrew clang here
  ignores `SDKROOT` and defaults to an SDK path that does not exist.
