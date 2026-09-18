// =============================================================================
// Five-way comparison on fresh, common measurement conditions.
//
//   v3    SMEKernels1x4AccKcOut::run_multiplication, blocking from
//         MaxMulSK::tuning::select(Sme1x4KcOut, M, K, N)
//   v4a   Nc-blocked v4 at Mc16 / Nc512 / Kc2048   (default candidate)
//   v4b   Nc-blocked v4 at Mc32 / Nc1024 / Kc2048  (alternative fixed profile)
//   mp    MpGEMM row_sgemm, wrapped for its d8-d15 ABI bug
//   acc   Accelerate cblas_sgemm, single-threaded
//
// ALLOCATION SCOPE, recorded deliberately: every entrant allocates whatever
// packing buffers it needs INSIDE its own timed call and frees them before
// returning. v3 and v4 are identical in this respect by construction -- both
// take the same aligned_alloc / unique_ptr path in their own driver. MpGEMM
// does its own posix_memalign/free inside row_sgemm. Accelerate is opaque but
// is called the same way every time. No entrant gets a hoisted buffer.
//
// No timing from the earlier parameter sweep is carried into this table.
// =============================================================================
#include "sme/v3/sme-1x4-acc-kcout.hpp"
#include "sme/v4/sme-1x4-kcout-ncblock-apack4za.hpp"
#include "sme/support/gemm_tuning.hpp"
#include "bench_sweep_shapes.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <Accelerate/Accelerate.h>

extern "C" void row_sgemm(int m, int n, int k, float* XA, float* XB, float* XC);
namespace NB = SMEKernels1x4KcOutNcBlockApack4Za;
using Clock = std::chrono::steady_clock;

static const NB::Blocking kA{16, 512, 2048};
static const NB::Blocking kB{32, 1024, 2048};

__attribute__((noinline))
static void mpgemm_sgemm(size_t M, size_t K, size_t N,
                         const float* A, const float* B, float* C) {
    row_sgemm((int)M, (int)N, (int)K, const_cast<float*>(A), const_cast<float*>(B), C);
    __asm__ volatile("" ::: "d8","d9","d10","d11","d12","d13","d14","d15","memory");
}
static double med(std::vector<double> v) {
    std::sort(v.begin(), v.end()); size_t n = v.size();
    return n ? ((n%2) ? v[n/2] : 0.5*(v[n/2-1]+v[n/2])) : 0.0;
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
    const int runs = (argc > 1) ? std::atoi(argv[1]) : 5;

    struct Impl { const char* name; void (*run)(const float*, const float*, float*, size_t, size_t, size_t); };
    Impl impls[5] = {
        {"v3",  [](const float* a, const float* b, float* c, size_t M, size_t K, size_t N){
            SMEKernels1x4AccKcOut::run_multiplication(a, b, c, M, K, N); }},
        {"v4a", [](const float* a, const float* b, float* c, size_t M, size_t K, size_t N){
            NB::run_multiplication(a, b, c, M, K, N, kA); }},
        {"v4b", [](const float* a, const float* b, float* c, size_t M, size_t K, size_t N){
            NB::run_multiplication(a, b, c, M, K, N, kB); }},
        {"mp",  [](const float* a, const float* b, float* c, size_t M, size_t K, size_t N){
            mpgemm_sgemm(M, K, N, a, b, c); }},
        {"acc", [](const float* a, const float* b, float* c, size_t M, size_t K, size_t N){
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)M, (int)N, (int)K,
                        1.0f, a, (int)K, b, (int)N, 0.0f, c, (int)N); }},
    };

    std::printf("tag,M,K,N,v3_Mtile,v3_Ntile,v3_Kc,v4a_packA,v4a_packB,v4b_packA,v4b_packB,reps,"
                "v3_ms,v3_gf,v3_sp,v4a_ms,v4a_gf,v4a_sp,v4b_ms,v4b_gf,v4b_sp,"
                "mp_ms,mp_gf,mp_sp,acc_ms,acc_gf,acc_sp,"
                "v4a_v3,v4a_mp,v4a_acc,v4b_v3,v4b_mp,v4b_acc,"
                "err_v4a,err_v4b,err_mp,err_acc\n");

    std::mt19937 rng(20260913);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    // Common warm-up, once, before any shape: page faults and the frequency ramp
    // are paid here rather than by whichever entrant happens to be measured first.
    {
        const size_t w = 512;
        std::vector<float> a(w*w), b(w*w), c(w*w);
        for (auto& v : a) v = dist(rng);
        for (auto& v : b) v = dist(rng);
        for (int i = 0; i < 10; i++) for (auto& im : impls) im.run(a.data(), b.data(), c.data(), w, w, w);
    }

    for (int si = 0; si < kNShapes; si++) {
        const Shape& s = kShapes[si];
        const size_t M = s.M, K = s.K, N = s.N;
        std::vector<float> A(M*K), B(K*N), C(M*N), Cref(M*N);
        for (auto& v : A) v = dist(rng);
        for (auto& v : B) v = dist(rng);

        const auto blk = MaxMulSK::tuning::select(MaxMulSK::tuning::Kernel::Sme1x4KcOut, M, K, N);
        size_t paA, pbA, paB, pbB;
        NB::capacity(M, K, N, kA, &paA, &pbA);
        NB::capacity(M, K, N, kB, &paB, &pbB);

        // correctness, against v3, once per shape
        SMEKernels1x4AccKcOut::run_multiplication(A.data(), B.data(), Cref.data(), M, K, N);
        auto err = [&](int idx) {
            std::fill(C.begin(), C.end(), 1234.5f);
            impls[idx].run(A.data(), B.data(), C.data(), M, K, N);
            double n = 0, d = 0;
            for (size_t i = 0; i < C.size(); i++) { double e = (double)C[i]-Cref[i]; n += e*e; d += (double)Cref[i]*Cref[i]; }
            return d > 0 ? std::sqrt(n/d) : 0.0;
        };
        double e[5] = {0, err(1), err(2), err(3), err(4)};

        // per-shape warm-up on the freshly filled matrices
        for (int i = 0; i < 2; i++) for (auto& im : impls) im.run(A.data(), B.data(), C.data(), M, K, N);

        double slowest = 0;
        for (auto& im : impls) {
            auto t0 = Clock::now(); im.run(A.data(), B.data(), C.data(), M, K, N);
            slowest = std::max(slowest, (double)std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-t0).count()*1e-9);
        }
        const size_t reps = (size_t)std::clamp(0.30 / std::max(slowest, 1e-9), 3.0, 2000.0);

        std::vector<double> g[5];
        for (int r = 0; r < runs; r++)
            for (int slot = 0; slot < 5; slot++) {
                const int i = (si + r + slot) % 5;       // Latin square over shapes and runs
                std::vector<double> t; t.reserve(reps);
                for (size_t q = 0; q < reps; q++) {
                    auto a = Clock::now();
                    impls[i].run(A.data(), B.data(), C.data(), M, K, N);
                    t.push_back((double)std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-a).count()*1e-6);
                }
                g[i].push_back(med(t));
            }

        const double flops = 2.0*M*K*N;
        double ms[5], gf[5], sp[5];
        for (int i = 0; i < 5; i++) {
            ms[i] = med(g[i]); gf[i] = flops/(ms[i]*1e6);
            auto mm = std::minmax_element(g[i].begin(), g[i].end());
            sp[i] = 100.0*(*mm.second-*mm.first)/ms[i];
        }
        std::printf("%s,%zu,%zu,%zu,%zu,%zu,%zu,%.3f,%.3f,%.3f,%.3f,%zu,",
                    s.tag, M, K, N, blk.M_tile, blk.N_tile, blk.Kc,
                    paA/1048576.0, pbA/1048576.0, paB/1048576.0, pbB/1048576.0, reps);
        for (int i = 0; i < 5; i++) std::printf("%.4f,%.1f,%.2f,", ms[i], gf[i], sp[i]);
        std::printf("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,",
                    gf[1]/gf[0], gf[1]/gf[3], gf[1]/gf[4],
                    gf[2]/gf[0], gf[2]/gf[3], gf[2]/gf[4]);
        std::printf("%.3e,%.3e,%.3e,%.3e\n", e[1], e[2], e[3], e[4]);
    }
    return 0;
}
