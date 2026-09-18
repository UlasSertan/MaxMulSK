// =============================================================================
// Accelerate / 1x4-general / MpGEMM. One binary, one session, 35 shapes.
//
// 1x4-general routes on the shape alone:
//   M == K == N  ->  sme/v3/sme-1x4-acc-kcout
//   otherwise    ->  v4c (Nc-blocked, two fixed profiles chosen by M and N)
//
// The routing decision, the profile choice it delegates, and the packing-buffer
// allocation are all INSIDE the timed call, as they are for the other entrants:
// MpGEMM allocates inside row_sgemm, Accelerate is opaque but called
// identically. Nobody gets a hoisted buffer.
//
// v3 is linked for the correctness reference as well. On square shapes the
// routed kernel IS v3, so its error column is exactly zero there by
// construction; off the diagonal it is an independent reference.
// =============================================================================
#include "sme/v4/sme-1x4-general.hpp"
#include "sme/v3/sme-1x4-acc-kcout.hpp"
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
namespace GEN = SMEKernels1x4General;
using Clock = std::chrono::steady_clock;

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
static const char* route_name(GEN::Path p) {
    switch (p) {
        case GEN::Path::V3Square: return "v3-square";
        case GEN::Path::V4C:      return "v4c";
        default:                  return "unsupported";
    }
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
    const int runs = (argc > 1) ? std::atoi(argv[1]) : 5;

    struct Impl { const char* name; void (*run)(const float*, const float*, float*, size_t, size_t, size_t); };
    Impl impls[3] = {
        {"acc", [](const float* a, const float* b, float* c, size_t M, size_t K, size_t N){
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)M, (int)N, (int)K,
                        1.0f, a, (int)K, b, (int)N, 0.0f, c, (int)N); }},
        {"gen", [](const float* a, const float* b, float* c, size_t M, size_t K, size_t N){
            GEN::run_multiplication(a, b, c, M, K, N); }},
        {"mp",  [](const float* a, const float* b, float* c, size_t M, size_t K, size_t N){
            mpgemm_sgemm(M, K, N, a, b, c); }},
    };

    std::printf("tag,M,K,N,route,reps,acc_ms,acc_gf,acc_sp,gen_ms,gen_gf,gen_sp,"
                "mp_ms,mp_gf,mp_sp,gen_over_acc,gen_over_mp,err_gen,err_acc,err_mp\n");

    std::mt19937 rng(20260913);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
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

        const GEN::Path p = GEN::route(M, K, N);
        SMEKernels1x4AccKcOut::run_multiplication(A.data(), B.data(), Cref.data(), M, K, N);
        auto err = [&](int idx) {
            std::fill(C.begin(), C.end(), 1234.5f);
            impls[idx].run(A.data(), B.data(), C.data(), M, K, N);
            double n = 0, d = 0;
            for (size_t i = 0; i < C.size(); i++) { double e = (double)C[i]-Cref[i]; n += e*e; d += (double)Cref[i]*Cref[i]; }
            return d > 0 ? std::sqrt(n/d) : 0.0;
        };
        const double e_acc = err(0), e_gen = err(1), e_mp = err(2);

        for (int i = 0; i < 2; i++) for (auto& im : impls) im.run(A.data(), B.data(), C.data(), M, K, N);
        double slowest = 0;
        for (auto& im : impls) {
            auto t0 = Clock::now(); im.run(A.data(), B.data(), C.data(), M, K, N);
            slowest = std::max(slowest, (double)std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-t0).count()*1e-9);
        }
        const size_t reps = (size_t)std::clamp(0.50 / std::max(slowest, 1e-9), 3.0, 20000.0);

        std::vector<double> g[3];
        for (int r = 0; r < runs; r++)
            for (int slot = 0; slot < 3; slot++) {
                const int i = (si + r + slot) % 3;
                std::vector<double> t; t.reserve(reps);
                for (size_t q = 0; q < reps; q++) {
                    auto a = Clock::now();
                    impls[i].run(A.data(), B.data(), C.data(), M, K, N);
                    t.push_back((double)std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-a).count()*1e-6);
                }
                g[i].push_back(med(t));
            }
        const double flops = 2.0*M*K*N;
        double ms[3], gf[3], sp[3];
        for (int i = 0; i < 3; i++) {
            ms[i] = med(g[i]); gf[i] = flops/(ms[i]*1e6);
            auto mm = std::minmax_element(g[i].begin(), g[i].end());
            sp[i] = 100.0*(*mm.second-*mm.first)/ms[i];
        }
        std::printf("%s,%zu,%zu,%zu,%s,%zu,", s.tag, M, K, N, route_name(p), reps);
        for (int i = 0; i < 3; i++) std::printf("%.4f,%.1f,%.2f,", ms[i], gf[i], sp[i]);
        std::printf("%.4f,%.4f,%.3e,%.3e,%.3e\n", gf[1]/gf[0], gf[1]/gf[2], e_gen, e_acc, e_mp);
    }
    return 0;
}
