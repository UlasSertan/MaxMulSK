// matmul.cpp — head-to-head FP32 SGEMM (2048x2048) on Apple M4:
//   our SME kernel (SMEKernels1x4SymZAInOut) vs Apple Accelerate.
//
// Both run on the *same* A, B buffers, with the *same* warmup/best-of-7
// timing protocol, single-threaded (VECLIB_MAXIMUM_THREADS=1 +
// omp_set_num_threads(1)) so the comparison is apples-to-apples.
//
// Target hardware (verified on this machine via sysctl):
//   Apple M4 (Mac16,12, MacBook Air), ARMv9.2-A.
//     4 Performance cores (L1d 128 KiB, L2 16 MiB shared), 6 Efficiency cores.
//     128-byte cache line, 16 KiB page, 16 GiB unified memory.
//   FEAT_SME, FEAT_SME2 = 1.  Streaming Vector Length (SVL) = 512 bits
//   (16 FP32 lanes).  ZA tile array = 4096 B = 4 × (16x16) FP32 tiles.
//   Measured FP32 SME throughput: ~2.0 TFLOPS / P-core, ~2.34 TFLOPS multi-core
//   (Zakharko, "M4 SME exploration", 2024; Wagner, Breuer, Bader,
//   "Hello SME!", arXiv:2409.18779, 2024).
//
// Algorithm: Goto-style 5-loop blocked SGEMM with packing.
//   References:
//     [1] K. Goto and R. van de Geijn, "Anatomy of High-Performance Matrix
//         Multiplication," ACM TOMS 34(3):12, 2008.
//     [2] F. Van Zee and R. van de Geijn, "BLIS: A Framework for Rapidly
//         Instantiating BLAS Functionality," ACM TOMS 41(3):14, 2015.
//     [3] Wagner, Breuer, Bader, "Hello SME! Generating Fast Matrix
//         Multiplication Kernels Using the Scalable Matrix Extension,"
//         arXiv:2409.18779, 2024.
//
// Apple's Accelerate `cblas_sgemm` ships hand-tuned AMX/SME micro-kernels on
// Apple Silicon (developer.apple.com/documentation/accelerate/blas).  At
// M=N=K=2048 the vendor kernel beats published user-space SME kernels — see
// [3] Fig. 8, where the authors' generator matches Accelerate only at
// M=N=K ≤ 512.  So Accelerate is the max-GFLOPS path here.
//
// Build:  via CMake target `matmul` (see top-level CMakeLists.txt)
// Run  :  ./matmul              # benchmark SME kernel + Accelerate
//         ./matmul --verify     # also run naive reference and check both

// _DARWIN_C_SOURCE keeps CLOCK_MONOTONIC_RAW visible; ACCELERATE_NEW_LAPACK
// selects Apple's current cblas headers (the legacy ones are deprecated as
// of macOS 13.3).
#define _DARWIN_C_SOURCE
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <ctime>
#include <cerrno>
#include <omp.h>

#include "../sme/sme-1x4-sym-zainout.hpp"

#define N         2048
#define MAT_ALIGN  128           // M4 cache-line size (sysctl hw.cachelinesize)
#define NRUNS        7           // benchmark repetitions; report min time

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static float *aligned_matrix(size_t elems) {
    void *p = NULL;
    if (posix_memalign(&p, MAT_ALIGN, elems * sizeof(float)) != 0) {
        fprintf(stderr, "posix_memalign failed: %s\n", strerror(errno));
        exit(1);
    }
    return (float *)p;
}

static void fill_random(float *M, size_t elems, uint32_t seed) {
    // Deterministic xorshift32 → reproducible inputs across runs.
    // Values in [-1, 1) keep K=2048 dot products well inside FP32 range.
    uint32_t s = seed ? seed : 0x9E3779B9u;
    for (size_t i = 0; i < elems; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        M[i] = ((float)(int32_t)s) * (1.0f / 2147483648.0f);
    }
}

// Naive ijk SGEMM, row-major, C = A·B.  Correctness reference only.
// Walks B sequentially in the inner loop so clang -O3 vectorizes to NEON.
static void sgemm_naive(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            const float a = A[i * n + k];
            const float *Brow = &B[k * n];
            float       *Crow = &C[i * n];
            for (int j = 0; j < n; j++) {
                Crow[j] += a * Brow[j];
            }
        }
    }
}

// Best-of-NRUNS GFLOPS reporter, with per-run echo to stdout. `run_one` must
// leave its output in `C` and is called NRUNS times. Returns best wall-clock
// seconds for the labeled kernel.
template <typename RunFn>
static double bench_best_of(const char* label, float* C, size_t elems,
                            double gflops_per_run, RunFn&& run_one) {
    double best = 1e18;
    for (int r = 0; r < NRUNS; r++) {
        memset(C, 0, elems * sizeof(float));
        double t0 = now_seconds();
        run_one();
        double dt = now_seconds() - t0;
        if (dt < best) best = dt;
        printf("  %-12s run %d: %8.3f ms  →  %7.1f GFLOPS\n",
               label, r + 1, dt * 1e3, gflops_per_run / dt);
    }
    return best;
}

// FP32 relative error vs the naive ijk reference. K=2048 terms in [-1,1) gives
// stddev ~26 and a worst-case forward error of ~K·eps·||a||·||b|| ≈ 1e-4
// relative. Allow some margin: different kernels use different summation
// orders.
static bool check_vs_reference(const char* label,
                               const float* C, const float* Cref,
                               size_t elems) {
    double max_abs = 0.0, max_rel = 0.0;
    for (size_t i = 0; i < elems; i++) {
        double a = (double)C[i], b = (double)Cref[i];
        double d = fabs(a - b);
        if (d > max_abs) max_abs = d;
        double m = fabs(b);
        if (m > 1.0 && d / m > max_rel) max_rel = d / m;
    }
    bool ok = max_rel < 1e-3;
    printf("  %-12s  max |Δ| = %.3e   max rel = %.3e   %s\n",
           label, max_abs, max_rel, ok ? "OK" : "MISMATCH");
    return ok;
}

int main(int argc, char **argv) {
    bool verify = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verify") == 0) verify = true;
    }

    // Apples-to-apples: Accelerate has an internal thread pool that auto-scales
    // for large GEMMs. Our SME kernel is single-threaded. Cap Accelerate (and
    // OpenMP, defensively) at 1 thread so both kernels contest the same single
    // P-core SME budget. bench_compare.cpp uses the same convention.
    setenv("VECLIB_MAXIMUM_THREADS", "1", 1);
    omp_set_num_threads(1);

    const size_t elems          = (size_t)N * (size_t)N;
    const double flops_per_run  = 2.0 * (double)N * (double)N * (double)N;
    const double gflops_per_run = flops_per_run / 1e9;

    printf("Apple M4 SGEMM head-to-head — N = %d (FP32, row-major)\n", N);
    printf("Kernels: SMEKernels1x4SymZAInOut  vs  Apple Accelerate (cblas_sgemm)\n");
    printf("Threads: 1 (VECLIB_MAXIMUM_THREADS=1, omp_set_num_threads(1))\n");
    printf("Protocol: best of %d runs, shared A/B buffers, memset(C) before each run\n", NRUNS);
    printf("Work per multiply: %.3f GFLOP (2·N³)\n", gflops_per_run);
    printf("Buffers: 3 × %zu B (~%.1f MiB), %d-byte aligned\n\n",
           elems * sizeof(float),
           3.0 * (double)elems * sizeof(float) / (1024.0 * 1024.0),
           MAT_ALIGN);

    float *A = aligned_matrix(elems);
    float *B = aligned_matrix(elems);
    float *C = aligned_matrix(elems);

    fill_random(A, elems, 0xA11CEu);
    fill_random(B, elems, 0xB0BB1Eu);

    // Warmup — first call faults in code pages and lets the SME unit
    // transition into streaming mode; exclude it from timing. Warm up both
    // paths so neither is penalized by cold caches on its first timed run.
    memset(C, 0, elems * sizeof(float));
    SMEKernels1x4SymZAInOut::run_multiplication(A, B, C, N, N, N);
    memset(C, 0, elems * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N,
                1.0f, A, N,
                      B, N,
                0.0f, C, N);

    // --- Our SME kernel (SMEKernels1x4SymZAInOut) --------------------------
    // SME signature is (M, K, N); square here so order doesn't matter.
    printf("[SME 1x4SymZAInOut]\n");
    float *C_sme = aligned_matrix(elems);
    double best_sme = bench_best_of("SME", C_sme, elems, gflops_per_run, [&]{
        SMEKernels1x4SymZAInOut::run_multiplication(A, B, C_sme, N, N, N);
    });
    const double gflops_sme = gflops_per_run / best_sme;
    printf("  SME best:  %.3f ms  →  %.1f GFLOPS\n\n",
           best_sme * 1e3, gflops_sme);

    // --- Accelerate (vendor reference, AMX/SME-backed) ----------------------
    printf("[Apple Accelerate]\n");
    float *C_acc = aligned_matrix(elems);
    double best_acc = bench_best_of("Accelerate", C_acc, elems, gflops_per_run, [&]{
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N,
                    1.0f, A, N,
                          B, N,
                    0.0f, C_acc, N);
    });
    const double gflops_acc = gflops_per_run / best_acc;
    printf("  Accelerate best: %.3f ms  →  %.1f GFLOPS"
           "  (single-P-core SME peak ≈ 2000 GFLOPS [3])\n\n",
           best_acc * 1e3, gflops_acc);

    // --- Summary ------------------------------------------------------------
    printf("================ Summary ================\n");
    printf("  SME 1x4SymZAInOut : %7.1f GFLOPS  (%.3f ms)\n", gflops_sme, best_sme * 1e3);
    printf("  Apple Accelerate  : %7.1f GFLOPS  (%.3f ms)\n", gflops_acc, best_acc * 1e3);
    printf("  ratio (SME / Acc) : %.3f\n", gflops_sme / gflops_acc);
    printf("=========================================\n");

    // --- Optional naive correctness check ----------------------------------
    int rc = 0;
    if (verify) {
        printf("\nVerifying against naive ijk reference…\n");
        float *Cref = aligned_matrix(elems);
        memset(Cref, 0, elems * sizeof(float));
        double t0 = now_seconds();
        sgemm_naive(A, B, Cref, N);
        double dt = now_seconds() - t0;
        printf("  naive: %.3f s  →  %.2f GFLOPS\n", dt, gflops_per_run / dt);

        bool ok_sme = check_vs_reference("SME",        C_sme, Cref, elems);
        bool ok_acc = check_vs_reference("Accelerate", C_acc, Cref, elems);
        free(Cref);
        if (!ok_sme || !ok_acc) rc = 2;
    }

    free(A); free(B); free(C); free(C_sme); free(C_acc);
    return rc;
}
