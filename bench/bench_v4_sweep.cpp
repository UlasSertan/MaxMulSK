// =============================================================================
// Full-table sweep: v3 acc-kcout / generalised v4 F2 / Accelerate / MpGEMM.
// Single binary, single session, FP32, row-major, single thread, Apple M4.
//
// Entry points, so the table can be read without guessing:
//   v3          SMEKernels1x4AccKcOut::run_multiplication
//               blocking from MaxMulSK::tuning::select(Sme1x4KcOut, M, K, N)
//   v4          SMEKernels1x4KcOutMc16Apack4ZaBonceGen::run_multiplication
//               Mc=16, Nc=64, Kc=min(2048,K); returns its own Support reason
//   Accelerate  cblas_sgemm, BLASSetThreading(BLAS_THREADING_SINGLE_THREADED)
//   MpGEMM      row_sgemm, wrapped because it clobbers callee-saved d8-d15
//
// Allocation scope is the same for all four: every entrant allocates whatever
// packing buffers it needs inside its own timed call. Timing is integer
// nanoseconds, median over reps, reps calibrated on the slowest entrant so all
// four get the same count. Call order is a cyclic Latin square over runs.
//
// v4 is never substituted silently: a shape it cannot run natively is reported
// with its refusal reason and left out of the v4 columns.
// =============================================================================
#include "sme/v3/sme-1x4-acc-kcout.hpp"
#include "sme/v4/sme-1x4-kcout-mc16-apack4za-bonce-gen.hpp"
#include "sme/support/gemm_tuning.hpp"
#include "bench_sweep_shapes.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <Accelerate/Accelerate.h>

extern "C" void row_sgemm(int m, int n, int k, float* XA, float* XB, float* XC);

namespace G = SMEKernels1x4KcOutMc16Apack4ZaBonceGen;
using Clock = std::chrono::steady_clock;

// MpGEMM's .S kernels do not restore d8-d15. Naming them clobbered forces this
// wrapper to spill and reload them, so the damage cannot reach the timing code.
__attribute__((noinline))
static void mpgemm_sgemm(size_t M, size_t K, size_t N,
                         const float* A, const float* B, float* C) {
    row_sgemm((int)M, (int)N, (int)K, const_cast<float*>(A), const_cast<float*>(B), C);
    __asm__ volatile("" ::: "d8","d9","d10","d11","d12","d13","d14","d15","memory");
}

static double med(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n/2] : 0.5 * (v[n/2-1] + v[n/2]);
}
static const char* support_name(G::Support s) {
    switch (s) {
        case G::Support::Native:                  return "native";
        case G::Support::UnsupportedMTail:        return "unsupported-M-tail";
        case G::Support::UnsupportedNTail:        return "unsupported-N-tail";
        case G::Support::UnsupportedKTail:        return "unsupported-K-tail";
        case G::Support::UnsupportedVectorLength: return "unsupported-svl";
        case G::Support::AllocationFailed:        return "alloc-failed";
        default:                                  return "unsupported";
    }
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
    const int runs   = (argc > 1) ? std::atoi(argv[1]) : 5;
    const int lo     = (argc > 2) ? std::atoi(argv[2]) : 0;
    const int hi     = (argc > 3) ? std::atoi(argv[3]) : kNShapes;

    std::printf("tag,M,K,N,v4_path,v3_Mtile,v3_Ntile,v3_Kc,packA_MiB,packB_MiB,reps,"
                "v3_ms,v3_gflops,v3_spread,v4_ms,v4_gflops,v4_spread,"
                "acc_ms,acc_gflops,acc_spread,mp_ms,mp_gflops,mp_spread,"
                "v4_over_v3,v4_over_mp,v4_over_acc,relerr_v4,relerr_acc,relerr_mp\n");

    std::mt19937 rng(20260913);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    // Global warm-up: without it the first shape absorbs page-fault and
    // frequency-ramp cost that no later shape pays, and its spread blows up.
    {
        const size_t w = 512;
        std::vector<float> a(w*w), b(w*w), c(w*w);
        for (auto& v : a) v = dist(rng);
        for (auto& v : b) v = dist(rng);
        for (int i = 0; i < 8; i++) {
            SMEKernels1x4AccKcOut::run_multiplication(a.data(), b.data(), c.data(), w, w, w);
            G::run_multiplication(a.data(), b.data(), c.data(), w, w, w);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)w, (int)w, (int)w,
                        1.0f, a.data(), (int)w, b.data(), (int)w, 0.0f, c.data(), (int)w);
            mpgemm_sgemm(w, w, w, a.data(), b.data(), c.data());
        }
    }

    for (int si = lo; si < hi; si++) {
        const Shape& s = kShapes[si];
        const size_t M = s.M, K = s.K, N = s.N;
        const G::Support sup = G::classify(M, K, N);
        size_t pa = 0, pb = 0; G::capacity(K, N, &pa, &pb);
        const auto blk = MaxMulSK::tuning::select(MaxMulSK::tuning::Kernel::Sme1x4KcOut, M, K, N);

        std::vector<float> A(M*K), B(K*N), C(M*N), Cref(M*N);
        for (auto& v : A) v = dist(rng);
        for (auto& v : B) v = dist(rng);

        // ---- correctness, once per shape, against v3 -------------------------
        SMEKernels1x4AccKcOut::run_multiplication(A.data(), B.data(), Cref.data(), M, K, N);
        auto relerr = [&](void(*run)(const float*,const float*,float*,size_t,size_t,size_t)) {
            std::fill(C.begin(), C.end(), 1234.5f);
            run(A.data(), B.data(), C.data(), M, K, N);
            double num = 0, den = 0;
            for (size_t i = 0; i < C.size(); i++) { double e = (double)C[i]-Cref[i]; num += e*e; den += (double)Cref[i]*Cref[i]; }
            return den > 0 ? std::sqrt(num/den) : 0.0;
        };
        double e_v4 = -1;
        if (sup == G::Support::Native) {
            std::fill(C.begin(), C.end(), 1234.5f);
            G::run_multiplication(A.data(), B.data(), C.data(), M, K, N);
            double num = 0, den = 0;
            for (size_t i = 0; i < C.size(); i++) { double e = (double)C[i]-Cref[i]; num += e*e; den += (double)Cref[i]*Cref[i]; }
            e_v4 = std::sqrt(num/den);
        }
        double e_acc = relerr([](const float* a, const float* b, float* c, size_t M_, size_t K_, size_t N_){
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)M_, (int)N_, (int)K_,
                        1.0f, a, (int)K_, b, (int)N_, 0.0f, c, (int)N_); });
        double e_mp  = relerr([](const float* a, const float* b, float* c, size_t M_, size_t K_, size_t N_){
            mpgemm_sgemm(M_, K_, N_, a, b, c); });

        // ---- timing ----------------------------------------------------------
        struct Impl { const char* name; void (*run)(const float*, const float*, float*, size_t, size_t, size_t); };
        Impl impls[4] = {
            {"v3", [](const float* a, const float* b, float* c, size_t M_, size_t K_, size_t N_){
                SMEKernels1x4AccKcOut::run_multiplication(a, b, c, M_, K_, N_); }},
            {"v4", [](const float* a, const float* b, float* c, size_t M_, size_t K_, size_t N_){
                G::run_multiplication(a, b, c, M_, K_, N_); }},
            {"acc", [](const float* a, const float* b, float* c, size_t M_, size_t K_, size_t N_){
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)M_, (int)N_, (int)K_,
                            1.0f, a, (int)K_, b, (int)N_, 0.0f, c, (int)N_); }},
            {"mp", [](const float* a, const float* b, float* c, size_t M_, size_t K_, size_t N_){
                mpgemm_sgemm(M_, K_, N_, a, b, c); }},
        };
        const int first = (sup == G::Support::Native) ? 0 : 0;  // v4 still called; it no-ops and is excluded below

        double slowest = 0;
        for (auto& im : impls) {
            auto t0 = Clock::now(); im.run(A.data(), B.data(), C.data(), M, K, N);
            slowest = std::max(slowest, (double)std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-t0).count()*1e-9);
        }
        const size_t reps = (size_t)std::clamp(0.30 / std::max(slowest, 1e-9), 3.0, 2000.0);

        std::vector<double> g[4];
        for (int r = 0; r < runs; r++)
            for (int slot = 0; slot < 4; slot++) {
                const int idx = (si + r + slot) % 4;      // Latin square over shapes AND runs
                std::vector<double> t; t.reserve(reps);
                for (size_t i = 0; i < reps; i++) {
                    auto a = Clock::now();
                    impls[idx].run(A.data(), B.data(), C.data(), M, K, N);
                    t.push_back((double)std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-a).count()*1e-6);
                }
                g[idx].push_back(med(t));
            }
        (void)first;

        const double flops = 2.0 * M * K * N;
        double ms[4], gf[4], sp[4];
        for (int i = 0; i < 4; i++) {
            ms[i] = med(g[i]);
            gf[i] = flops / (ms[i] * 1e6);
            auto mm = std::minmax_element(g[i].begin(), g[i].end());
            sp[i] = 100.0 * (*mm.second - *mm.first) / ms[i];
        }
        const bool nat = (sup == G::Support::Native);
        auto num = [&](double v){ static char b[32]; if (v < 0) { std::snprintf(b,sizeof b,""); } else std::snprintf(b,sizeof b,"%.4g",v); return b; };
        std::printf("%s,%zu,%zu,%zu,%s,%zu,%zu,%zu,%.2f,%.2f,%zu,"
                    "%.4f,%.1f,%.2f,", s.tag, M, K, N, support_name(sup),
                    blk.M_tile, blk.N_tile, blk.Kc, pa/1048576.0, pb/1048576.0, reps,
                    ms[0], gf[0], sp[0]);
        if (nat) std::printf("%.4f,%.1f,%.2f,", ms[1], gf[1], sp[1]); else std::printf(",,,");
        std::printf("%.4f,%.1f,%.2f,%.4f,%.1f,%.2f,", ms[2], gf[2], sp[2], ms[3], gf[3], sp[3]);
        if (nat) std::printf("%.4f,%.4f,%.4f,%s,", gf[1]/gf[0], gf[1]/gf[3], gf[1]/gf[2], num(e_v4));
        else     std::printf(",,,,");
        std::printf("%s,%s\n", num(e_acc), num(e_mp));
    }
    return 0;
}
