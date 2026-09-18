#include "gemm_providers.hpp"
#include "sme/v4/sme-1x4-v4c-dispatch.hpp"
#include "sme/v4/sme-1x4-kcout-ncblock-apack4za-tail.hpp"

#include <Accelerate/Accelerate.h>
#include <chrono>
#include <cstring>
#include <map>
#include <tuple>
#include <vector>

extern "C" {
typedef void (*maxmulsk_sgemm_hook_t)(int, int, int, const float*, int, const float*, int, float*, int);
extern maxmulsk_sgemm_hook_t maxmulsk_sgemm_hook;
void row_sgemm(int m, int n, int k, float* XA, float* XB, float* XC);
}

namespace V4C = SMEKernels1x4V4C;
namespace NB  = SMEKernels1x4KcOutNcBlockApack4Za;
namespace TL  = SMEKernels1x4NcBlockTail;

// Same selection rule as v4c, applied to the tail-capable kernel. The rule is
// unchanged; only the kernel behind it differs, which is the whole point of the
// A/B: with tail support a decode step (m == 1) no longer has to fall back.
inline TL::Blocking tail_choose(size_t M, size_t K, size_t N) {
    const auto b = V4C::choose(M, K, N);
    return TL::Blocking{b.Mc, b.Nc, b.Kc};
}

namespace llmbench {
namespace {

Provider g_provider = Provider::Accelerate;
bool     g_record   = false;
std::map<std::tuple<int,int,int>, ShapeStat> g_stats;
long   g_calls = 0, g_native = 0, g_fallback = 0;
double g_transpose_s = 0.0;

// Scratch for the B transpose. Reused across calls so the benchmark is not
// measuring malloc, but the transpose ITSELF is redone on every call -- this is
// a conversion buffer, not a packed-weight cache.
std::vector<float> g_bt;   // B^T (k x n)  or  A^T (k x m)
std::vector<float> g_ct;   // C^T (n x m), only for the swapped direction

// The hook wants C = A*B^T with everything row-major. A NoTrans x NoTrans kernel
// can get there two ways, and they cost different amounts:
//
//   direct  : transpose B (n*k elements), then C[m x n] = A[m x k] * Bt[k x n]
//   swapped : transpose A (m*k), compute Ct[n x m] = B[n x k] * At[k x m],
//             then transpose Ct into C (m*n)
//
// The weight B is fixed-size while A and C grow with the token count, so the
// direct route wins on long prefills and the swapped route wins on short ones.
// Picking per call is an adapter choice, not a cache: both transposes are redone
// on every call and are charged to the provider.
inline bool prefer_swapped(int m, int n, int k) {
    return (size_t)m*k + (size_t)m*n < (size_t)n*k;
}
template <typename Fn>
void run_transposed(int m, int n, int k, const float* A, const float* B, float* C,
                    double& acc_s, bool& transposed, Fn&& gemm) {
    // Only accumulate while recording, so an untimed prefix feed (the decode
    // cases populate the KV cache before the clock starts) cannot leak into the
    // reported transpose cost.
    const bool rec = g_record;
    auto t0 = std::chrono::steady_clock::now();
    transposed = true;
    if (prefer_swapped(m, n, k)) {
        g_bt.resize((size_t)k * m);                       // A^T, k x m
        for (int i = 0; i < m; i++)
            for (int j = 0; j < k; j++) g_bt[(size_t)j*m + i] = A[(size_t)i*k + j];
        g_ct.resize((size_t)n * m);
        if (rec) acc_s += std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
        gemm(n, m, k, B, g_bt.data(), g_ct.data());       // Ct[n x m] = B[n x k] * At[k x m]
        auto t1 = std::chrono::steady_clock::now();
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++) C[(size_t)j*n + i] = g_ct[(size_t)i*m + j];
        if (rec) acc_s += std::chrono::duration<double>(std::chrono::steady_clock::now()-t1).count();
    } else {
        g_bt.resize((size_t)k * n);                       // B^T, k x n
        for (int j = 0; j < n; j++)
            for (int i = 0; i < k; i++) g_bt[(size_t)i*n + j] = B[(size_t)j*k + i];
        if (rec) acc_s += std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
        gemm(m, n, k, A, g_bt.data(), C);
    }
}

// MpGEMM's hand-written .S kernels clobber callee-saved d8-d15 without
// restoring them (verified 2026-09-10). Naming them clobbered forces this
// wrapper to spill and reload them around the call.
__attribute__((noinline))
void mpgemm_call(int m, int n, int k, const float* A, const float* B, float* C) {
    row_sgemm(m, n, k, const_cast<float*>(A), const_cast<float*>(B), C);
    __asm__ volatile("" ::: "d8","d9","d10","d11","d12","d13","d14","d15","memory");
}

// The declared common fallback for any shape a provider cannot run itself.
// Results attributed to it are NOT that provider's kernel performance.
void fallback_sgemm(int m, int n, int k, const float* A, int lda,
                    const float* B, int ldb, float* C, int ldc) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k,
                1.0f, A, lda, B, ldb, 0.0f, C, ldc);
}

// Whichever direction the adapter picks, v4c must be able to run THAT shape.
bool v4c_ok(int m, int n, int k) {
    const int M = prefer_swapped(m, n, k) ? n : m;
    const int N = prefer_swapped(m, n, k) ? m : n;
    return NB::classify(M, k, N, V4C::choose(M, k, N)) == NB::Support::Native;
}

void hook(int m, int n, int k, const float* A, int lda,
          const float* B, int ldb, float* C, int ldc) {
    bool native = false, transposed = false;

    switch (g_provider) {
    case Provider::Accelerate:
        fallback_sgemm(m, n, k, A, lda, B, ldb, C, ldc);
        native = true;                       // this IS Accelerate's own path
        break;

    case Provider::MpGEMM:
        if (m == 1 && lda == k && ldb == k && ldc == n) {
            // C^T[n x 1] = B[n x k] * A^T[k x 1]; no transpose, no copy.
            mpgemm_call(n, 1, k, B, A, C);
            native = true;
        } else if (lda == k && ldb == k && ldc == n) {
            run_transposed(m, n, k, A, B, C, g_transpose_s, transposed,
                [](int mm, int nn, int kk, const float* aa, const float* bb, float* cc) {
                    mpgemm_call(mm, nn, kk, aa, bb, cc); });
            native = true;
        } else {
            fallback_sgemm(m, n, k, A, lda, B, ldb, C, ldc);
        }
        break;

    case Provider::V4CTail:
        if (m == 1 && lda == k && ldb == k && ldc == n) {
            // C^T[n x 1] = B[n x k] * A^T[k x 1]. No transpose, and with tail
            // support N == 1 is now a supported edge tile rather than a refusal.
            if (TL::run_multiplication(B, A, C, n, k, 1, tail_choose(n, k, 1))
                    == TL::Support::Native) native = true;
            else fallback_sgemm(m, n, k, A, lda, B, ldb, C, ldc);
        } else if (lda == k && ldb == k && ldc == n) {
            bool sw = prefer_swapped(m, n, k);
            const size_t M_ = sw ? n : m, N_ = sw ? m : n;
            if (TL::classify(M_, k, N_, tail_choose(M_, k, N_)) == TL::Support::Native) {
                run_transposed(m, n, k, A, B, C, g_transpose_s, transposed,
                    [](int mm, int nn, int kk, const float* aa, const float* bb, float* cc) {
                        TL::run_multiplication(aa, bb, cc, mm, kk, nn, tail_choose(mm, kk, nn)); });
                native = true;
            } else {
                fallback_sgemm(m, n, k, A, lda, B, ldb, C, ldc);
            }
        } else {
            fallback_sgemm(m, n, k, A, lda, B, ldb, C, ldc);
        }
        break;

    case Provider::V4C:
        if (m == 1 && lda == k && ldb == k && ldc == n) {
            // Same swap. It makes N == 1, which v4c cannot run natively, so
            // this lands on the fallback -- deliberately visible.
            if (V4C::run_multiplication(B, A, C, n, k, 1) == NB::Support::Native) native = true;
            else fallback_sgemm(m, n, k, A, lda, B, ldb, C, ldc);
        } else if (lda == k && ldb == k && ldc == n && v4c_ok(m, n, k)) {
            run_transposed(m, n, k, A, B, C, g_transpose_s, transposed,
                [](int mm, int nn, int kk, const float* aa, const float* bb, float* cc) {
                    V4C::run_multiplication(aa, bb, cc, mm, kk, nn); });
            native = true;
        } else {
            fallback_sgemm(m, n, k, A, lda, B, ldb, C, ldc);
        }
        break;
    }

    if (g_record) {
        g_calls++;
        if (native) g_native++; else g_fallback++;
        auto& s = g_stats[{m,n,k}];
        s.m = m; s.n = n; s.k = k; s.calls++;
        if (native) s.native_calls++; else s.fallback_calls++;
        if (transposed) s.transposed_calls++;
    }
}

} // namespace

void set_provider(Provider p) { g_provider = p; }
const char* provider_name(Provider p) {
    switch (p) { case Provider::Accelerate: return "accelerate";
                 case Provider::V4C: return "v4c";
                 case Provider::V4CTail: return "v4c-tail";
                 default: return "mpgemm"; }
}
void install_hook() { maxmulsk_sgemm_hook = hook; }
void counters_reset() { g_stats.clear(); g_calls = g_native = g_fallback = 0; g_transpose_s = 0; }
void counters_enable(bool on) { g_record = on; }
long counters_total_calls()    { return g_calls; }
long counters_native_calls()   { return g_native; }
long counters_fallback_calls() { return g_fallback; }
double counters_transpose_seconds() { return g_transpose_s; }
std::vector<ShapeStat> counters_shapes() {
    std::vector<ShapeStat> v; v.reserve(g_stats.size());
    for (auto& [key, s] : g_stats) v.push_back(s);
    return v;
}

} // namespace llmbench
