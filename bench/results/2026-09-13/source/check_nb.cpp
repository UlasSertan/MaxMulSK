#include "sme/v4/sme-1x4-kcout-ncblock-apack4za.hpp"
#include "sme/v3/sme-1x4-acc-kcout.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include "bench_sweep_shapes.h"
namespace NB = SMEKernels1x4KcOutNcBlockApack4Za;
using B = NB::Blocking;

static int bad = 0, cases = 0;
static double relerr(const std::vector<float>& a, const std::vector<float>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); i++) { double e = (double)a[i]-b[i]; n += e*e; d += (double)b[i]*b[i]; }
    return d > 0 ? std::sqrt(n/d) : 0.0;
}
int main() {
    std::mt19937 rng(913); std::uniform_real_distribution<float> d(-1.f, 1.f);

    // ---- 1. A panel layout, against the scalar definition, incl. a short Kc --
    {
        const size_t M = 256, K = 4096;
        std::vector<float> A(M*K); for (auto& v : A) v = d(rng);
        std::vector<float> pa(16*4096);
        struct { size_t kk, m, kcl; } cs[] = {
            {0, 0, 2048}, {2048, 240, 2048}, {0, 16, 1024}, {4096-512, 0, 512},
            {0, 0, 64}, {4096-64, 240, 64},
        };
        for (auto& c : cs) {
            std::fill(pa.begin(), pa.end(), -1.f);
            bool ran = NB::probe_pack_A(A.data(), M, K, c.kk, c.m, c.kcl, pa.data());
            size_t wrong = 0;
            for (size_t k = 0; k < c.kcl; k++) for (size_t r = 0; r < 16; r++)
                if (pa[k*16+r] != A[(c.m+r)*K + c.kk + k]) wrong++;
            cases++; if (!ran || wrong) { bad++; std::printf("  A-layout kk=%zu m=%zu kcl=%zu  wrong=%zu FAIL\n", c.kk, c.m, c.kcl, wrong); }
        }
        std::printf("A panel layout: %d cases, %s\n", 6, bad ? "FAIL" : "all exact");
    }

    // ---- 2. full GEMM on every shape, with blockings that force short blocks --
    std::printf("\n%-12s %18s %-22s %10s %s\n", "tag", "MxKxN", "Mc/Nc/Kc", "relFrob", "");
    for (int i = 0; i < kNShapes; i++) {
        const Shape& s = kShapes[i];
        std::vector<float> A(s.M*s.K), Bm(s.K*s.N), B2(s.K*s.N), C(s.M*s.N), Cref(s.M*s.N);
        for (auto& v : A) v = d(rng);
        for (auto& v : Bm) v = d(rng);
        for (auto& v : B2) v = d(rng);
        SMEKernels1x4AccKcOut::run_multiplication(A.data(), Bm.data(), Cref.data(), s.M, s.K, s.N);

        // deliberately non-dividing / oversized macroblocks
        B blks[] = { {16,256,2048}, {128,1024,4096}, {32,64,256}, {64,512,1024}, {128,128,512} };
        for (auto& b : blks) {
            std::fill(C.begin(), C.end(), 3.75f);          // must be overwritten
            auto sup = NB::run_multiplication(A.data(), Bm.data(), C.data(), s.M, s.K, s.N, b);
            cases++;
            if (sup != NB::Support::Native) { bad++; std::printf("  %-12s %zux%zux%zu  Mc%zu/Nc%zu/Kc%zu  NOT NATIVE\n", s.tag,s.M,s.K,s.N,b.Mc,b.Nc,b.Kc); continue; }
            double e = relerr(C, Cref);
            if (!(e < 5e-6)) { bad++; std::printf("  %-12s %zux%zux%zu  Mc%zu/Nc%zu/Kc%zu  relFrob=%.3e FAIL\n", s.tag,s.M,s.K,s.N,b.Mc,b.Nc,b.Kc,e); }
        }
        // repeat with different B, same buffers: no stale packed B may survive
        SMEKernels1x4AccKcOut::run_multiplication(A.data(), B2.data(), Cref.data(), s.M, s.K, s.N);
        B rb{32, 512, 1024};
        NB::run_multiplication(A.data(), B2.data(), C.data(), s.M, s.K, s.N, rb);
        double e2 = relerr(C, Cref); cases++;
        if (!(e2 < 5e-6)) { bad++; std::printf("  %-12s repeat-with-new-B relFrob=%.3e FAIL\n", s.tag, e2); }
        std::printf("%-12s %6zux%zux%zu  6 blockings + repeat   %s\n", s.tag, s.M, s.K, s.N, "ok");
    }
    // ---- 3. bad blocking is refused, C untouched ------------------------------
    {
        std::vector<float> A(256*256,1.f), Bm(256*256,1.f), C(256*256,9.f);
        B bad1{24,256,2048}, bad2{16,100,2048}, bad3{16,256,100};
        for (auto& b : {bad1,bad2,bad3}) {
            auto sup = NB::run_multiplication(A.data(),Bm.data(),C.data(),256,256,256,b);
            bool untouched = std::all_of(C.begin(),C.end(),[](float v){return v==9.f;});
            cases++; if (sup != NB::Support::BadBlocking || !untouched) { bad++; std::printf("  bad blocking not refused FAIL\n"); }
        }
        std::printf("\nbad blocking refused, C untouched: ok\n");
    }
    std::printf("\n%s  (%d cases, %d failures)\n", bad ? "FAILED" : "ALL PASS", cases, bad);
    return bad != 0;
}
