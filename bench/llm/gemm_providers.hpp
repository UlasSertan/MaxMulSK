#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// The matrix-multiplication provider plugged into the patched ggml BLAS hook.
//
// Hook contract (ggml/src/ggml-blas/ggml-blas.cpp, F32 mul_mat path):
//     C[m x n] = A[m x k] * B^T ,  A row-major lda,
//     B stored n x k row-major ldb, C row-major ldc, alpha = 1, beta = 0.
//
// Accelerate takes that shape natively. MaxMulSK v4c and MpGEMM are both
// NoTrans x NoTrans only, so they need one of two routes:
//
//   m == 1  : C^T[n x 1] = B[n x k] * A^T[k x 1]. With m == 1, A and A^T are the
//             same bytes and so are C and C^T, so this costs NOTHING -- but it
//             makes N == 1, which v4c cannot run natively.
//   m  > 1  : B must be transposed into k x n. That copy is real work and is
//             charged to the provider inside the timed call; its cost is also
//             accumulated separately so it can be reported.
//
// Accelerate does not pay that transpose. That asymmetry is inherent to the
// runtime's calling convention, not a benchmark artefact, and it is reported.
namespace llmbench {

enum class Provider { Accelerate, V4C, V4CTail, MpGEMM };

struct ShapeStat {
    int  m = 0, n = 0, k = 0;
    long calls = 0;
    long native_calls = 0;      // ran on the provider's own kernel
    long fallback_calls = 0;    // ran on the declared common fallback
    long transposed_calls = 0;  // needed a B transpose
};

void set_provider(Provider p);
const char* provider_name(Provider p);

void counters_reset();
void counters_enable(bool on);
std::vector<ShapeStat> counters_shapes();
long   counters_total_calls();
long   counters_native_calls();
long   counters_fallback_calls();
double counters_transpose_seconds();

// Installs the hook into ggml. Call once, after llama_backend_init().
void install_hook();

} // namespace llmbench
