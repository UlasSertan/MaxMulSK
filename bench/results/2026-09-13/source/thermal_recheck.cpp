// Re-measure early shapes with the fixed reference blocking, identical method to
// bench_nb_sweep stage 2. Same shape, same code, different thermal state: the
// shape-order / shape-size confound in the sweep's spread trend does not apply.
#include "sme/v4/sme-1x4-kcout-ncblock-apack4za.hpp"
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include "bench_sweep_shapes.h"
namespace NB = SMEKernels1x4KcOutNcBlockApack4Za;
using Clock = std::chrono::steady_clock;
static double med(std::vector<double> v){std::sort(v.begin(),v.end());size_t n=v.size();
    return n?((n%2)?v[n/2]:0.5*(v[n/2-1]+v[n/2])):0;}
int main(int argc,char**argv){
    std::setvbuf(stdout,nullptr,_IOLBF,0);
    const NB::Blocking ref{16,256,2048};
    std::mt19937 rng(20260913); std::uniform_real_distribution<float> d(-1.f,1.f);
    std::printf("tag,M,K,N,reps,ms,gflops,spread\n");
    for (int a=1;a<argc;a++){
        const Shape& s = kShapes[std::atoi(argv[a])];
        std::vector<float> A(s.M*s.K),B(s.K*s.N),C(s.M*s.N);
        for(auto&v:A)v=d(rng); for(auto&v:B)v=d(rng);
        NB::run_multiplication(A.data(),B.data(),C.data(),s.M,s.K,s.N,ref);
        auto t0=Clock::now(); NB::run_multiplication(A.data(),B.data(),C.data(),s.M,s.K,s.N,ref);
        double one=(double)std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-t0).count()*1e-9;
        size_t reps=(size_t)std::clamp(0.15/std::max(one,1e-9),1.0,5000.0);
        std::vector<double> per;
        for(int r=0;r<5;r++){std::vector<double> t;t.reserve(reps);
            for(size_t i=0;i<reps;i++){auto x=Clock::now();
                NB::run_multiplication(A.data(),B.data(),C.data(),s.M,s.K,s.N,ref);
                t.push_back((double)std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-x).count()*1e-6);}
            per.push_back(med(t));}
        auto mm=std::minmax_element(per.begin(),per.end()); double m=med(per);
        std::printf("%s,%zu,%zu,%zu,%zu,%.4f,%.1f,%.2f\n",s.tag,s.M,s.K,s.N,reps,m,
                    2.0*s.M*s.K*s.N/(m*1e6),100.0*(*mm.second-*mm.first)/m);
    }
    return 0;}
