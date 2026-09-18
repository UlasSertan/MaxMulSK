#include "sme-1x4-v4c-dispatch.hpp"

namespace NB = SMEKernels1x4KcOutNcBlockApack4Za;

namespace SMEKernels1x4V4C {

    // First rule. Derived from final5.csv (2026-09-13): across the 14 shapes
    // where the two profiles differed by more than 5%, every shape v4b won had
    // M = 4096 (the one exception being ds-id5, 64x7168x4096) and every shape
    // v4a won had M <= 128. K is deliberately NOT a condition here. At N <= 512
    // the two profiles were within 5% of each other on every shape -- at N = 256
    // they are literally identical, since Nc512 and Nc1024 both clamp to N -- so
    // the rule only needs to separate the wide-N cases.
    //
    // The threshold is experimental and rests on one session. It is not tuned
    // per shape and it is not re-derived from timings at runtime.
    Blocking choose(size_t M, size_t K, size_t N) {
        (void)K;
        if (M >= 2048 && N > 512)
            return Blocking{32, 1024, 2048};   // v4b
        return Blocking{16, 512, 2048};        // v4a
    }

    // The choice is made here, once, before the kernel allocates anything and
    // before it enters streaming mode. The chosen parameters are handed to the
    // existing Nc-blocked driver unchanged: same micro-kernel, same packing
    // pipelines, same ZA attributes, same zeroing and the same C writeback.
    Support run_multiplication(const float* A, const float* B, float* C,
                               size_t M, size_t K, size_t N) {
        return NB::run_multiplication(A, B, C, M, K, N, choose(M, K, N));
    }

} // namespace SMEKernels1x4V4C
