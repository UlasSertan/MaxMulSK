#include "maxmulsk.h"

#include "sme/v3/sme-1x4-acc-kcout.hpp"
#include "sme/v4/sme-1x4-v4c-dispatch.hpp"

#include <cstring>

namespace {

    // Empty shapes are answered here so both kernels see the same contract.
    // With K == 0 the product is the zero matrix, but v3 would leave C untouched
    // and v4 rejects the shape, so C is cleared instead of calling either.
    bool handle_empty(float* C, size_t M, size_t K, size_t N) {
        if (M == 0 || N == 0) return true;
        if (K == 0) {
            std::memset(C, 0, M * N * sizeof(float));
            return true;
        }
        return false;
    }

    maxmulsk_status to_status(SMEKernels1x4V4C::Support s) {
        using S = SMEKernels1x4V4C::Support;
        switch (s) {
            case S::Native:           return MAXMULSK_OK;
            case S::AllocationFailed: return MAXMULSK_ERR_ALLOC;
            default:                  return MAXMULSK_ERR_UNSUPPORTED;
        }
    }

} // namespace

extern "C" maxmulsk_status maxmulsk_sgemm_v3(const float* A, const float* B, float* C,
                                             size_t M, size_t K, size_t N) {
    if (handle_empty(C, M, K, N)) return MAXMULSK_OK;
    SMEKernels1x4AccKcOut::run_multiplication(A, B, C, M, K, N);
    return MAXMULSK_OK;
}

extern "C" maxmulsk_status maxmulsk_sgemm_v4(const float* A, const float* B, float* C,
                                             size_t M, size_t K, size_t N) {
    if (handle_empty(C, M, K, N)) return MAXMULSK_OK;
    return to_status(SMEKernels1x4V4C::run_multiplication(A, B, C, M, K, N));
}
