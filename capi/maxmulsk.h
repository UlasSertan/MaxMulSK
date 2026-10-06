#pragma once

#include <stddef.h>

// =============================================================================
// Thin C entry points for local use from other codebases. Nothing here is a
// stable public interface yet.
//
// Both compute C = A*B for row-major, tightly packed FP32 matrices:
// A is M x K, B is K x N, C is M x N. C is overwritten (alpha = 1, beta = 0);
// its previous contents are never read. Single thread.
//
// Dimensions follow the internal kernels' order: (M, K, N), not BLAS's
// (M, N, K).
//
// There is no runtime SME check: on hardware without SME (anything before M4)
// these die with SIGILL.
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MAXMULSK_OK = 0,
    MAXMULSK_ERR_UNSUPPORTED,   // the kernel rejected this shape or vector length
    MAXMULSK_ERR_ALLOC,         // a packing buffer could not be allocated
} maxmulsk_status;

// v3: SMEKernels1x4AccKcOut. Never returns MAXMULSK_ERR_ALLOC -- v3 does not
// check its allocations, so a failed allocation crashes instead.
maxmulsk_status maxmulsk_sgemm_v3(const float* A, const float* B, float* C,
                                  size_t M, size_t K, size_t N);

// v4: the v4c dispatcher (SMEKernels1x4V4C) over the Nc-blocked v4 kernel.
maxmulsk_status maxmulsk_sgemm_v4(const float* A, const float* B, float* C,
                                  size_t M, size_t K, size_t N);

#ifdef __cplusplus
}
#endif
