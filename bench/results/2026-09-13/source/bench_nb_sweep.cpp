// =============================================================================
// Nc/Mc/Kc parameter sweep for sme/v4/sme-1x4-kcout-ncblock-apack4za.
//
// 100 blockings  ( Mc {16,32,64,128} x Nc {64,128,256,512,1024}
//                  x Kc {256,512,1024,2048,4096} )  over 35 shapes.
//
// Two stages, because a single short timing is not enough to eliminate a
// candidate:
//   stage 1  every blocking, short, repeated passes, median of the passes
//   stage 2  each shape's leaders plus the globally strong blockings, longer
//
// Candidate order is rotated per shape and per pass so no blocking is
// systematically measured first into a cold cache. Allocation is inside the
// timed call for every candidate alike -- the kernel allocates its own packed
// buffers, so the scope is identical. Same binary, same flags, single thread.
//
// A candidate whose result does not match the v3 reference is marked bad and
// excluded from every ranking.
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

// Mc is deliberately NOT extended: the 2026-09-13 base sweep found it flat
// (geomean 0.896x-0.902x across 16..128), and its boundary values won 9 and 8
// shapes respectively, which is what a flat axis looks like. Nc and Kc are
// extended upward because their per-shape winners sat on the old top edge --
// Nc=1024 won 19/35 shapes and Kc=4096 won 11/35.
static const size_t kMc[] = {16, 32, 64, 128};
static const size_t kNc[] = {64, 128, 256, 512, 1024, 2048, 4096};
static const size_t kKc[] = {256, 512, 1024, 2048, 4096, 8192, 16384};
// The base sweep's axes, so an extension run can skip what was already measured.
static bool in_base_sweep(const NB::Blocking& b) {
    return b.Nc <= 1024 && b.Kc <= 4096;
}
static const NB::Blocking kRef{16, 256, 2048};

static double med(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n ? ((n % 2) ? v[n/2] : 0.5*(v[n/2-1]+v[n/2])) : 0.0;
}

struct Cand { NB::Blocking b; double ms; double spread; bool ok; };

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const double t1_budget = (argc > 1) ? std::atof(argv[1]) : 0.06;   // s per candidate, stage 1
    const int    t1_passes = (argc > 2) ? std::atoi(argv[2]) : 2;
    const double t2_budget = (argc > 3) ? std::atof(argv[3]) : 0.15;
    const int    t2_runs   = (argc > 4) ? std::atoi(argv[4]) : 5;
    const int    keep      = (argc > 5) ? std::atoi(argv[5]) : 8;
    // 0 = base axes only, 1 = only blockings the base sweep did not cover, 2 = all
    const int    mode      = (argc > 6) ? std::atoi(argv[6]) : 0;

    std::vector<NB::Blocking> combos;
    for (size_t mc : kMc) for (size_t nc : kNc) for (size_t kc : kKc) {
        const NB::Blocking b{mc, nc, kc};
        const bool base = in_base_sweep(b);
        if (mode == 0 && !base) continue;
        if (mode == 1 && base)  continue;
        combos.push_back(b);
    }
    // The fixed reference must be present in every run: every ratio is taken
    // against it, measured in the same pass on the same shape, which is what
    // makes ratios comparable between the base run and an extension run.
    if (std::none_of(combos.begin(), combos.end(), [](const NB::Blocking& b){
            return b.Mc==kRef.Mc && b.Nc==kRef.Nc && b.Kc==kRef.Kc; }))
        combos.push_back(kRef);
    const int NC = (int)combos.size();
    std::fprintf(stderr, "mode=%d, %d blockings per shape\n", mode, NC);

    std::printf("stage,tag,M,K,N,Mc,Nc,Kc,path,packA_MiB,packB_MiB,"
                "a_packs,b_packs,mk_calls,reps,ms,gflops,spread,correct,vs_ref\n");

    std::mt19937 rng(20260913);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    // Global warm-up. Without it the first shape absorbs page-fault and
    // frequency-ramp cost no later shape pays; in the base run that showed up as
    // a 54% spread on the very first candidate measured.
    {
        const size_t w = 512;
        std::vector<float> a(w*w), b(w*w), c(w*w);
        for (auto& v : a) v = dist(rng);
        for (auto& v : b) v = dist(rng);
        for (int i = 0; i < 12; i++)
            NB::run_multiplication(a.data(), b.data(), c.data(), w, w, w, kRef);
    }

    for (int si = 0; si < kNShapes; si++) {
        const Shape& s = kShapes[si];
        const size_t M = s.M, K = s.K, N = s.N;
        std::vector<float> A(M*K), B(K*N), C(M*N), Cref(M*N);
        for (auto& v : A) v = dist(rng);
        for (auto& v : B) v = dist(rng);
        SMEKernels1x4AccKcOut::run_multiplication(A.data(), B.data(), Cref.data(), M, K, N);

        // Per-shape warm-up: the freshly filled A, B and C are cold and their
        // pages are untouched. Paying that once here keeps it out of whichever
        // candidate happens to be measured first.
        for (int i = 0; i < 3; i++)
            NB::run_multiplication(A.data(), B.data(), C.data(), M, K, N, kRef);

        // Strided reference checksum. A full Frobenius norm per candidate would
        // cost more than the measurement on the large shapes; the exhaustive
        // check lives in the standalone correctness program.
        const size_t stride = std::max<size_t>(1, C.size() / 4096);
        auto csum = [&](const std::vector<float>& v) {
            double a = 0; for (size_t i = 0; i < v.size(); i += stride) a += v[i]; return a; };
        const double ref_sum = csum(Cref);

        auto time_one = [&](const NB::Blocking& b, double budget, int runs, double* spread) {
            // one warm call, then `runs` passes of `reps` calls each
            NB::run_multiplication(A.data(), B.data(), C.data(), M, K, N, b);
            auto t0 = Clock::now();
            NB::run_multiplication(A.data(), B.data(), C.data(), M, K, N, b);
            double one = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-t0).count()*1e-9;
            size_t reps = (size_t)std::clamp(budget / std::max(one, 1e-9), 1.0, 5000.0);
            std::vector<double> per;
            for (int r = 0; r < runs; r++) {
                std::vector<double> t; t.reserve(reps);
                for (size_t i = 0; i < reps; i++) {
                    auto a = Clock::now();
                    NB::run_multiplication(A.data(), B.data(), C.data(), M, K, N, b);
                    t.push_back((double)std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-a).count()*1e-6);
                }
                per.push_back(med(t));
            }
            auto mm = std::minmax_element(per.begin(), per.end());
            double m = med(per);
            *spread = 100.0 * (*mm.second - *mm.first) / m;
            return std::make_pair(m, reps);
        };

        auto emit = [&](const char* stage, const NB::Blocking& b, double ms, size_t reps,
                        double spread, bool ok, double vs_ref) {
            size_t pa, pb; NB::capacity(M, K, N, b, &pa, &pb);
            NB::Counts c; NB::counts(M, K, N, b, &c);
            double gf = 2.0*M*K*N / (ms*1e6);
            std::printf("%s,%s,%zu,%zu,%zu,%zu,%zu,%zu,%s,%.3f,%.3f,%zu,%zu,%zu,%zu,%.4f,%.1f,%.2f,%s,",
                stage, s.tag, M, K, N, b.Mc, b.Nc, b.Kc,
                NB::classify(M,K,N,b)==NB::Support::Native ? "native" : "unsupported",
                pa/1048576.0, pb/1048576.0, c.a_panel_packs, c.b_panel_packs, c.microkernel_calls,
                reps, ms, gf, spread, ok ? "ok" : "MISMATCH");
            if (vs_ref > 0) std::printf("%.4f\n", vs_ref); else std::printf("\n");
        };

        // ---------------- stage 1 ----------------
        std::vector<Cand> cands(NC);
        for (int i = 0; i < NC; i++) cands[i] = Cand{combos[i], 1e30, 0, false};
        std::vector<std::vector<double>> passes(NC);
        for (int p = 0; p < t1_passes; p++)
            for (int j = 0; j < NC; j++) {
                const int i = (si*7 + p*13 + j) % NC;      // rotated order
                double sp; auto [ms, reps] = time_one(combos[i], t1_budget, 1, &sp);
                (void)reps;
                passes[i].push_back(ms);
                cands[i].ok = std::fabs(csum(C) - ref_sum) <= 1e-3 * std::fabs(ref_sum) + 1e-6;
            }
        for (int i = 0; i < NC; i++) {
            cands[i].ms = med(passes[i]);
            auto mm = std::minmax_element(passes[i].begin(), passes[i].end());
            cands[i].spread = 100.0*(*mm.second-*mm.first)/cands[i].ms;
        }
        double ref_ms1 = 0;
        for (auto& c : cands) if (c.b.Mc==kRef.Mc && c.b.Nc==kRef.Nc && c.b.Kc==kRef.Kc) ref_ms1 = c.ms;
        for (auto& c : cands) emit("prescan", c.b, c.ms, 0, c.spread, c.ok, ref_ms1 > 0 ? ref_ms1/c.ms : -1);

        // ---------------- stage 2 ----------------
        std::vector<Cand> good;
        for (auto& c : cands) if (c.ok) good.push_back(c);
        std::sort(good.begin(), good.end(), [](const Cand& a, const Cand& b){ return a.ms < b.ms; });
        std::vector<NB::Blocking> finals;
        for (int i = 0; i < (int)good.size() && i < keep; i++) finals.push_back(good[i].b);
        // always re-measure the fixed reference, plus anything within 3% of the
        // stage-1 leader even if it falls outside `keep`
        if (!good.empty())
            for (auto& c : good)
                if (c.ms <= good[0].ms * 1.03 &&
                    std::none_of(finals.begin(), finals.end(), [&](const NB::Blocking& b){
                        return b.Mc==c.b.Mc && b.Nc==c.b.Nc && b.Kc==c.b.Kc; }))
                    finals.push_back(c.b);
        if (std::none_of(finals.begin(), finals.end(), [](const NB::Blocking& b){
                return b.Mc==kRef.Mc && b.Nc==kRef.Nc && b.Kc==kRef.Kc; }))
            finals.push_back(kRef);

        std::vector<double> fms(finals.size()), fsp(finals.size());
        std::vector<size_t> freps(finals.size());
        std::vector<bool> fok(finals.size());
        for (size_t j = 0; j < finals.size(); j++) {
            const size_t i = (si + j) % finals.size();     // rotated
            double sp; auto [ms, reps] = time_one(finals[i], t2_budget, t2_runs, &sp);
            fms[i] = ms; fsp[i] = sp; freps[i] = reps;
            fok[i] = std::fabs(csum(C) - ref_sum) <= 1e-3 * std::fabs(ref_sum) + 1e-6;
        }
        double ref_ms2 = 0;
        for (size_t j = 0; j < finals.size(); j++)
            if (finals[j].Mc==kRef.Mc && finals[j].Nc==kRef.Nc && finals[j].Kc==kRef.Kc) ref_ms2 = fms[j];
        for (size_t j = 0; j < finals.size(); j++)
            emit("final", finals[j], fms[j], freps[j], fsp[j], fok[j], ref_ms2 > 0 ? ref_ms2/fms[j] : -1);
    }
    return 0;
}
