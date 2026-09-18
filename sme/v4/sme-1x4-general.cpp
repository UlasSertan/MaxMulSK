#include "sme-1x4-general.hpp"
#include "sme-1x4-v4c-dispatch.hpp"
#include "../v3/sme-1x4-acc-kcout.hpp"

namespace V4C = SMEKernels1x4V4C;
namespace NB  = SMEKernels1x4KcOutNcBlockApack4Za;

namespace SMEKernels1x4General {

    Path route(size_t M, size_t K, size_t N) {
        if (M == 0 || K == 0 || N == 0) return Path::Unsupported;
        if (M == K && K == N) return Path::V3Square;
        return NB::classify(M, K, N, V4C::choose(M, K, N)) == NB::Support::Native
                   ? Path::V4C : Path::Unsupported;
    }

    Path run_multiplication(const float* A, const float* B, float* C,
                            size_t M, size_t K, size_t N) {
        const Path p = route(M, K, N);
        switch (p) {
            case Path::V3Square:
                SMEKernels1x4AccKcOut::run_multiplication(A, B, C, M, K, N);
                return p;
            case Path::V4C:
                V4C::run_multiplication(A, B, C, M, K, N);
                return p;
            default:
                return Path::Unsupported;   // C untouched
        }
    }

} // namespace SMEKernels1x4General
