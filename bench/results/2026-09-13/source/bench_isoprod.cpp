// =============================================================================
// Constant-product blocking family for the Nc-blocked v4 kernel.
//
// Every entry has Mc * Nc * Kc = 2^25, i.e. the same packed volume redistributed
// between the three axes: lower Kc, proportionally higher Mc or Nc. 42 points,
// covering every power-of-two split with Mc >= 16, Nc >= 64, Kc >= 64.
//
// v4a (16/512/2048) and v4b (32/1024/2048) are included as controls. They are
// NOT on this surface -- their products are 2^24 and 2^26 -- so they anchor the
// family against what is already measured rather than belonging to it.
//
// Ratios are against v4a, measured on the same shape in the same pass.
// A candidate whose result disagrees with v3 is marked and excluded.
// =============================================================================
#include "sme/v4/sme-1x4-kcout-ncblock-apack4za.hpp"
#include "sme/v3/sme-1x4-acc-kcout.hpp"
#include "bench_sweep_shapes.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

namespace NB = SMEKernels1x4KcOutNcBlockApack4Za;
using Clock = std::chrono::steady_clock;

// Mc, Nc, Kc
static const NB::Blocking kFamily[] = {
    {  16,  512, 4096},
    {  32,  256, 4096},
    {  64,  128, 4096},
    { 128,   64, 4096},
    {  16, 1024, 2048},
    {  32,  512, 2048},
    {  64,  256, 2048},
    { 128,  128, 2048},
    { 256,   64, 2048},
    {  16, 2048, 1024},
    {  32, 1024, 1024},
    {  64,  512, 1024},
    { 128,  256, 1024},
    { 256,  128, 1024},
    { 512,   64, 1024},
    {  16, 4096,  512},
    {  32, 2048,  512},
    {  64, 1024,  512},
    { 128,  512,  512},
    { 256,  256,  512},
    { 512,  128,  512},
    {1024,   64,  512},
    {  32, 4096,  256},
    {  64, 2048,  256},
    { 128, 1024,  256},
    { 256,  512,  256},
    { 512,  256,  256},
    {1024,  128,  256},
    {2048,   64,  256},
    {  64, 4096,  128},
    { 128, 2048,  128},
    { 256, 1024,  128},
    { 512,  512,  128},
    {1024,  256,  128},
    {2048,  128,  128},
    {4096,   64,  128},
    { 128, 4096,   64},
    { 256, 2048,   64},
    { 512, 1024,   64},
    {1024,  512,   64},
    {2048,  256,   64},
    {4096,  128,   64}
};
static const NB::Blocking kA{16, 512, 2048};
static const NB::Blocking kB{32, 1024, 2048};

static double med(std::vector<double> v) {
    std::sort(v.begin(), v.end()); size_t n = v.size();
    return n ? ((n%2) ? v[n/2] : 0.5*(v[n/2-1]+v[n/2])) : 0.0;
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const double budget = (argc > 1) ? std::atof(argv[1]) : 0.10;
    const int    runs   = (argc > 2) ? std::atoi(argv[2]) : 3;

    std::vector<NB::Blocking> cands(std::begin(kFamily), std::end(kFamily));
    cands.push_back(kA);
    cands.push_back(kB);
    const int NC = (int)cands.size();
    std::fprintf(stderr, "%d blockings per shape (%d on the surface + v4a + v4b)\n", NC, NC-2);

    std::printf("tag,M,K,N,Mc,Nc,Kc,onsurface,packA_MiB,packB_MiB,reps,ms,gflops,spread,correct,vs_v4a\n");

    std::mt19937 rng(20260913);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    {   // common warm-up before any shape
        const size_t w = 512;
        std::vector<float> a(w*w), b(w*w), c(w*w);
        for (auto& v : a) v = dist(rng);
        for (auto& v : b) v = dist(rng);
        for (int i = 0; i < 12; i++) NB::run_multiplication(a.data(), b.data(), c.data(), w, w, w, kA);
    }

    for (int si = 0; si < kNShapes; si++) {
        const Shape& s = kShapes[si];
        const size_t M = s.M, K = s.K, N = s.N;
        std::vector<float> A(M*K), B(K*N), C(M*N), Cref(M*N);
        for (auto& v : A) v = dist(rng);
        for (auto& v : B) v = dist(rng);
        SMEKernels1x4AccKcOut::run_multiplication(A.data(), B.data(), Cref.data(), M, K, N);
        for (int i = 0; i < 3; i++) NB::run_multiplication(A.data(), B.data(), C.data(), M, K, N, kA);

        const size_t stride = std::max<size_t>(1, C.size() / 4096);
        auto csum = [&](const std::vector<float>& v){ double a=0; for (size_t i=0;i<v.size();i+=stride) a+=v[i]; return a; };
        const double ref = csum(Cref);

        std::vector<double> ms(NC); std::vector<size_t> rp(NC);
        std::vector<double> sp(NC); std::vector<bool> ok(NC);
        for (int j = 0; j < NC; j++) {
            const int i = (si + j) % NC;                  // rotated order
            const NB::Blocking& b = cands[i];
            NB::run_multiplication(A.data(), B.data(), C.data(), M, K, N, b);
            auto t0 = Clock::now();
            NB::run_multiplication(A.data(), B.data(), C.data(), M, K, N, b);
            double one = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-t0).count()*1e-9;
            size_t reps = (size_t)std::clamp(budget/std::max(one,1e-9), 1.0, 20000.0);
            std::vector<double> per;
            for (int r = 0; r < runs; r++) {
                std::vector<double> t; t.reserve(reps);
                for (size_t q = 0; q < reps; q++) {
                    auto a = Clock::now();
                    NB::run_multiplication(A.data(), B.data(), C.data(), M, K, N, b);
                    t.push_back((double)std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-a).count()*1e-6);
                }
                per.push_back(med(t));
            }
            auto mm = std::minmax_element(per.begin(), per.end());
            ms[i] = med(per); rp[i] = reps; sp[i] = 100.0*(*mm.second-*mm.first)/ms[i];
            ok[i] = std::fabs(csum(C)-ref) <= 1e-3*std::fabs(ref)+1e-6;
        }
        const double flops = 2.0*M*K*N;
        double ms_a = 0; for (int i=0;i<NC;i++) if (cands[i].Mc==kA.Mc&&cands[i].Nc==kA.Nc&&cands[i].Kc==kA.Kc) ms_a = ms[i];
        for (int i = 0; i < NC; i++) {
            const NB::Blocking& b = cands[i];
            size_t pa, pb; NB::capacity(M, K, N, b, &pa, &pb);
            const bool onsurf = (b.Mc*b.Nc*b.Kc) == (1u<<25);
            std::printf("%s,%zu,%zu,%zu,%zu,%zu,%zu,%s,%.3f,%.3f,%zu,%.4f,%.1f,%.2f,%s,%.4f\n",
                s.tag, M, K, N, b.Mc, b.Nc, b.Kc, onsurf?"yes":"no",
                pa/1048576.0, pb/1048576.0, rp[i], ms[i], flops/(ms[i]*1e6), sp[i],
                ok[i]?"ok":"MISMATCH", ms_a>0 ? ms_a/ms[i] : -1.0);
        }
    }
    return 0;
}
