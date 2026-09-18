// =============================================================================
// ARCH-001 PanelFuse experiment 2: v4c (reference) vs PanelFuse with both
// both-large policies, same binary, same session, 35 shapes.
//
// Every entrant allocates its packed buffers inside its own timed call. v4c is
// the reference; PF-B retains B when neither operand fits the budget, PF-A
// retains A. On shapes where one operand fits, the two PF rows are the same
// code path and should agree -- that agreement is a built-in control.
//
// The headline timing runs with phase timing OFF. A second, separate pass runs
// with it ON to split pack from compute; that pass is reported apart and is
// not the headline.
// =============================================================================
#include "sme/v4/sme-1x4-v4c-dispatch.hpp"
#include "sme/v4/sme-1x4-panelfuse.hpp"
#include "sme/v3/sme-1x4-acc-kcout.hpp"
#include "bench_sweep_shapes.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

namespace PF  = SMEKernels1x4PanelFuse;
namespace V4C = SMEKernels1x4V4C;
using Clock = std::chrono::steady_clock;
static double med(std::vector<double> v){std::sort(v.begin(),v.end());size_t n=v.size();return n?((n%2)?v[n/2]:0.5*(v[n/2-1]+v[n/2])):0;}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const int runs = argc > 1 ? std::atoi(argv[1]) : 5;
    PF::Params pB; pB.both_large = PF::BothLarge::RetainB;
    PF::Params pA; pA.both_large = PF::BothLarge::RetainA;

    struct Impl { const char* name; void (*run)(const float*, const float*, float*, size_t, size_t, size_t); };
    Impl impls[3] = {
        {"v4c",  [](const float* a, const float* b, float* c, size_t M, size_t K, size_t N){ V4C::run_multiplication(a,b,c,M,K,N); }},
        {"pf-B", [](const float* a, const float* b, float* c, size_t M, size_t K, size_t N){ PF::Params p; p.both_large=PF::BothLarge::RetainB; PF::run_multiplication(a,b,c,M,K,N,p); }},
        {"pf-A", [](const float* a, const float* b, float* c, size_t M, size_t K, size_t N){ PF::Params p; p.both_large=PF::BothLarge::RetainA; PF::run_multiplication(a,b,c,M,K,N,p); }},
    };
    std::printf("tag,M,K,N,retained,forced,t_budget,t_outer,t_inner,"
                "pfB_apacks,pfB_bpacks,pfB_phases,pfA_apacks,pfA_bpacks,pfA_phases,reps,"
                "v4c_ms,v4c_gf,v4c_sp,pfB_ms,pfB_gf,pfB_sp,pfA_ms,pfA_gf,pfA_sp,"
                "pfB_over_v4c,pfA_over_v4c,err_pfB,err_pfA,"
                "pfB_pack_ms,pfB_comp_ms,pfA_pack_ms,pfA_comp_ms\n");
    std::mt19937 rng(20260916); std::uniform_real_distribution<float> dist(-1.f,1.f);
    {   const size_t w=512; std::vector<float> a(w*w),b(w*w),c(w*w);
        for(auto&v:a)v=dist(rng); for(auto&v:b)v=dist(rng);
        for(int i=0;i<8;i++) for(auto& im:impls) im.run(a.data(),b.data(),c.data(),w,w,w); }

    for (int si = 0; si < kNShapes; si++) {
        const Shape& s = kShapes[si]; const size_t M=s.M,K=s.K,N=s.N;
        std::vector<float> A(M*K),B(K*N),C(M*N),Cref(M*N);
        for(auto&v:A)v=dist(rng); for(auto&v:B)v=dist(rng);
        SMEKernels1x4AccKcOut::run_multiplication(A.data(),B.data(),Cref.data(),M,K,N);
        auto err=[&](int i){ std::fill(C.begin(),C.end(),1234.5f); impls[i].run(A.data(),B.data(),C.data(),M,K,N);
            double n=0,d=0; for(size_t j=0;j<C.size();j++){double e=(double)C[j]-Cref[j];n+=e*e;d+=(double)Cref[j]*Cref[j];} return std::sqrt(n/d); };
        const double eB=err(1), eA=err(2);
        PF::Stats stB, stA; PF::plan(M,K,N,pB);
        PF::run_multiplication(A.data(),B.data(),C.data(),M,K,N,pB,&stB);
        PF::run_multiplication(A.data(),B.data(),C.data(),M,K,N,pA,&stA);
        auto pl = PF::plan(M,K,N,pB);

        for(int i=0;i<2;i++) for(auto& im:impls) im.run(A.data(),B.data(),C.data(),M,K,N);
        double slowest=0; for(auto& im:impls){ auto t0=Clock::now(); im.run(A.data(),B.data(),C.data(),M,K,N);
            slowest=std::max(slowest,(double)std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-t0).count()*1e-9); }
        const size_t reps=(size_t)std::clamp(0.30/std::max(slowest,1e-9),3.0,2000.0);

        std::vector<double> g[3];
        for(int r=0;r<runs;r++) for(int slot=0;slot<3;slot++){ const int i=(si+r+slot)%3;
            std::vector<double> t; t.reserve(reps);
            for(size_t q=0;q<reps;q++){ auto a=Clock::now(); impls[i].run(A.data(),B.data(),C.data(),M,K,N);
                t.push_back((double)std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-a).count()*1e-6); }
            g[i].push_back(med(t)); }
        const double flops=2.0*M*K*N; double ms[3],gf[3],sp[3];
        for(int i=0;i<3;i++){ ms[i]=med(g[i]); gf[i]=flops/(ms[i]*1e6); auto mm=std::minmax_element(g[i].begin(),g[i].end()); sp[i]=100.0*(*mm.second-*mm.first)/ms[i]; }

        // separate timed pass for the pack/compute split -- NOT the headline
        PF::set_phase_timing(true);
        PF::Stats tB, tA;
        PF::run_multiplication(A.data(),B.data(),C.data(),M,K,N,pB,&tB);
        PF::run_multiplication(A.data(),B.data(),C.data(),M,K,N,pA,&tA);
        PF::set_phase_timing(false);
        auto ms_of=[](uint64_t cyc,uint64_t hz){ return hz? 1000.0*cyc/(double)hz : 0.0; };

        std::printf("%s,%zu,%zu,%zu,%s,%d,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,",
            s.tag,M,K,N, pl.retained==PF::Retained::A?"A":"B", pl.forced_by_policy?1:0, pl.t_budget,pl.t_outer,pl.t_inner,
            stB.a_panel_packs,stB.b_panel_packs,stB.phases, stA.a_panel_packs,stA.b_panel_packs,stA.phases, reps);
        for(int i=0;i<3;i++) std::printf("%.4f,%.1f,%.2f,",ms[i],gf[i],sp[i]);
        std::printf("%.4f,%.4f,%.3e,%.3e,%.3f,%.3f,%.3f,%.3f\n", gf[1]/gf[0], gf[2]/gf[0], eB, eA,
            ms_of(tB.pack_cycles,tB.counter_hz), ms_of(tB.compute_cycles,tB.counter_hz),
            ms_of(tA.pack_cycles,tA.counter_hz), ms_of(tA.compute_cycles,tA.counter_hz));
    }
    return 0;
}
