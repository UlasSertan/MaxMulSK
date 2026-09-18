// =============================================================================
// FP32 LLM inference benchmark: MaxMulSK v4c / Accelerate / MpGEMM.
//
// A real forward pass through llama.cpp -- embeddings, 30 decoder layers,
// attention, RMS norm, SwiGLU, residuals, KV cache and the output head all run.
// The ONLY thing that changes between backends is which implementation serves
// the dense matrix multiplications that ggml's BLAS backend claims, namely the
// seven projections per layer (Q, K, V, O, gate, up, down) plus the output head.
// Attention's own QK^T and attn*V stay in ggml's common implementation for every
// backend; they are not part of the swap.
//
// Timing scope: model load and context creation are OUTSIDE. Everything that
// happens during llama_decode -- packing, the adapter's B transpose, allocation
// and any fallback call -- is INSIDE.
//
// Single thread everywhere: llama n_threads = 1, ggml BLAS backend threads = 1,
// Accelerate pinned single-threaded.
// =============================================================================
#include "llama.h"
#include "gemm_providers.hpp"

#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
using llmbench::Provider;

extern "C" void ggml_backend_blas_set_n_threads(ggml_backend_t, int);

static double med(std::vector<double> v) {
    std::sort(v.begin(), v.end()); size_t n = v.size();
    return n ? ((n % 2) ? v[n/2] : 0.5*(v[n/2-1]+v[n/2])) : 0.0;
}
static double spread(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    auto mm = std::minmax_element(v.begin(), v.end());
    return 100.0 * (*mm.second - *mm.first) / med(v);
}

struct Case { const char* name; int prefill; int decode_ctx; int decode_steps; };

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const char* model_path = argc > 1 ? argv[1] : "";
    const int rounds = argc > 2 ? std::atoi(argv[2]) : 5;
    BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);

    llama_backend_init();
    llmbench::install_hook();

    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(model_path, mp);
    if (!model) { std::fprintf(stderr, "model yuklenemedi: %s\n", model_path); return 1; }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // Fixed pseudo-token stream, identical for every backend, so numerical
    // differences cannot send them down different token paths.
    auto make_tokens = [&](int n, int seed) {
        std::vector<llama_token> t(n);
        t[0] = llama_vocab_bos(vocab);
        uint32_t s = (uint32_t)seed * 2654435761u + 12345u;
        for (int i = 1; i < n; i++) { s = s*1664525u + 1013904223u; t[i] = (llama_token)(s % (uint32_t)(n_vocab - 10) + 5); }
        return t;
    };

    const Provider provs[4] = {Provider::Accelerate, Provider::V4C, Provider::V4CTail, Provider::MpGEMM};
    const int NP = 4;

    // ---- correctness: all three providers must agree on the logits ----------
    // A prefill plus several decode steps, so the KV cache path is exercised
    // too, compared against the Accelerate run as reference.
    {
        auto cp = llama_context_default_params();
        cp.n_ctx = 640; cp.n_batch = 512; cp.n_ubatch = 512;
        cp.n_threads = 1; cp.n_threads_batch = 1;
        cp.type_k = GGML_TYPE_F32; cp.type_v = GGML_TYPE_F32; cp.no_perf = true;
        llama_context* cctx = llama_init_from_model(model, cp);
        llama_memory_t cmem = llama_get_memory(cctx);
        auto pre = make_tokens(512, 7);
        auto con = make_tokens(8, 77);
        std::vector<std::vector<float>> got(NP);
        for (int p = 0; p < NP; p++) {
            llmbench::set_provider(provs[p]);
            llmbench::counters_reset(); llmbench::counters_enable(true);
            llama_memory_clear(cmem, true);
            auto b = llama_batch_get_one(pre.data(), 512);
            llama_decode(cctx, b);
            got[p].assign(llama_get_logits_ith(cctx, -1), llama_get_logits_ith(cctx, -1) + n_vocab);
            for (int s = 0; s < 8; s++) {          // KV cache path
                auto d = llama_batch_get_one(&con[s], 1);
                llama_decode(cctx, d);
                const float* lg = llama_get_logits_ith(cctx, -1);
                got[p].insert(got[p].end(), lg, lg + n_vocab);
            }
            llmbench::counters_enable(false);
            std::printf("#   routing %-11s %ld calls, %ld native, %ld fallback\n",
                        llmbench::provider_name(provs[p]), llmbench::counters_total_calls(),
                        llmbench::counters_native_calls(), llmbench::counters_fallback_calls());
        }
        for (int p = 0; p < NP; p++)
            std::printf("#   ilk 4 logit %-11s % .8f % .8f % .8f % .8f\n",
                        llmbench::provider_name(provs[p]),
                        got[p][0], got[p][1], got[p][2], got[p][3]);
        std::printf("# correctness vs accelerate (prefill 512 + 8 decode steps, %d logits each)\n", n_vocab);
        for (int p = 1; p < NP; p++) {
            double num = 0, den = 0, mx = 0; long bad = 0;
            for (size_t i = 0; i < got[0].size(); i++) {
                if (!std::isfinite(got[p][i])) bad++;
                double e = (double)got[p][i] - got[0][i];
                num += e*e; den += (double)got[0][i]*got[0][i];
                mx = std::fmax(mx, std::fabs(e));
            }
            std::printf("#   %-11s relFrob=%.3e  max|err|=%.3e  nonfinite=%ld\n",
                        llmbench::provider_name(provs[p]), std::sqrt(num/den), mx, bad);
        }
        llama_free(cctx);
    }

    const Case cases[] = {
        {"prefill-128",       128,    0,   0},
        {"prefill-512",       512,    0,   0},
        {"prefill-2048",     2048,    0,   0},
        {"decode-ctx128",       0,  128, 128},
        {"decode-ctx2048",      0, 2048, 128},
        {"decode-ctx8192",      0, 8192, 128},
        {"combined-512+128",  512,    0, 128},
    };


    std::printf("case,provider,phase,n_tokens,ctx,rounds,ms_median,ms_spread_pct,tok_per_s,"
                "calls,native_calls,fallback_calls,native_pct,transpose_ms\n");

    for (int ci = 0; ci < (int)(sizeof(cases)/sizeof(cases[0])); ci++) {
        const Case& c = cases[ci];
        const int need_ctx = std::max(c.prefill + c.decode_steps + 8,
                                      c.decode_ctx + c.decode_steps + 8);

        auto cp = llama_context_default_params();
        // Fixed microbatch for every case and every backend, stated explicitly:
        // a prefix longer than this is fed in chunks of exactly this size.
        const int kChunk = 2048;
        cp.n_ctx = need_ctx; cp.n_batch = kChunk;
        cp.n_ubatch = kChunk;
        cp.n_threads = 1; cp.n_threads_batch = 1;
        cp.type_k = GGML_TYPE_F32; cp.type_v = GGML_TYPE_F32;
        cp.no_perf = true;
        llama_context* ctx = llama_init_from_model(model, cp);
        if (!ctx) { std::fprintf(stderr, "context olusmadi (%s)\n", c.name); return 1; }
        llama_memory_t mem = llama_get_memory(ctx);

        const int prefix_len = c.prefill > 0 ? c.prefill : c.decode_ctx;
        std::vector<llama_token> prefix = make_tokens(std::max(prefix_len, 1), ci + 1);
        std::vector<llama_token> cont   = make_tokens(std::max(c.decode_steps, 1), 900 + ci);

        struct Acc { std::vector<double> pre, dec; long calls=0, nat=0, fb=0; double tr=0; };
        Acc acc[4];

        // common warm-up: every backend, both phases, before any timing
        for (int w = 0; w < 2; w++)
            for (int p = 0; p < NP; p++) {
                llmbench::set_provider(provs[p]); llama_memory_clear(mem, true);
                for (int off = 0; off < prefix_len; off += kChunk) {
                    const int nchunk = std::min(kChunk, prefix_len - off);
                    auto b = llama_batch_get_one(prefix.data() + off, nchunk);
                    llama_decode(ctx, b);
                }
                for (int s = 0; s < std::min(c.decode_steps, 4); s++) {
                    auto b = llama_batch_get_one(&cont[s], 1); llama_decode(ctx, b);
                }
            }

        for (int r = 0; r < rounds; r++) {
            for (int slot = 0; slot < NP; slot++) {
                const int p = (ci + r + slot) % NP;          // Latin square rotation
                llmbench::set_provider(provs[p]);

                // fresh logical KV state for every repetition
                llama_memory_clear(mem, true);
                llmbench::counters_reset(); llmbench::counters_enable(true);

                double pre_ms = 0, dec_ms = 0;
                if (c.prefill > 0) {
                    auto t0 = Clock::now();
                    auto b = llama_batch_get_one(prefix.data(), c.prefill);
                    llama_decode(ctx, b);
                    pre_ms = std::chrono::duration<double, std::milli>(Clock::now()-t0).count();
                } else if (c.decode_ctx > 0) {
                    // Prefix really is processed so the KV cache is populated,
                    // but that preparation is NOT part of the decode timing.
                    llmbench::counters_enable(false);
                    for (int off = 0; off < c.decode_ctx; off += kChunk) {
                        const int nchunk = std::min(kChunk, c.decode_ctx - off);
                        auto b = llama_batch_get_one(prefix.data() + off, nchunk);
                        llama_decode(ctx, b);
                    }
                    llmbench::counters_enable(true);
                    llmbench::counters_reset();
                }
                if (c.decode_steps > 0) {
                    auto t0 = Clock::now();
                    for (int s = 0; s < c.decode_steps; s++) {
                        auto b = llama_batch_get_one(&cont[s], 1);
                        llama_decode(ctx, b);
                    }
                    dec_ms = std::chrono::duration<double, std::milli>(Clock::now()-t0).count();
                }
                llmbench::counters_enable(false);
                if (pre_ms > 0) acc[p].pre.push_back(pre_ms);
                if (dec_ms > 0) acc[p].dec.push_back(dec_ms);
                acc[p].calls = llmbench::counters_total_calls();
                acc[p].nat   = llmbench::counters_native_calls();
                acc[p].fb    = llmbench::counters_fallback_calls();
                acc[p].tr    = llmbench::counters_transpose_seconds();
            }
        }

        for (int p = 0; p < NP; p++) {
            const char* pn = llmbench::provider_name(provs[p]);
            const double npct = acc[p].calls ? 100.0*acc[p].nat/acc[p].calls : 0.0;
            if (!acc[p].pre.empty()) {
                double m = med(acc[p].pre);
                std::printf("%s,%s,prefill,%d,%d,%d,%.3f,%.2f,%.1f,%ld,%ld,%ld,%.1f,%.3f\n",
                    c.name, pn, c.prefill, need_ctx, rounds, m, spread(acc[p].pre),
                    1000.0*c.prefill/m, acc[p].calls, acc[p].nat, acc[p].fb, npct, acc[p].tr*1000);
            }
            if (!acc[p].dec.empty()) {
                double m = med(acc[p].dec);
                std::printf("%s,%s,decode,%d,%d,%d,%.3f,%.2f,%.1f,%ld,%ld,%ld,%.1f,%.3f\n",
                    c.name, pn, c.decode_steps, need_ctx, rounds, m, spread(acc[p].dec),
                    1000.0*c.decode_steps/m, acc[p].calls, acc[p].nat, acc[p].fb, npct, acc[p].tr*1000);
            }
        }
        llama_free(ctx);
    }

    llama_model_free(model); llama_backend_free();
    return 0;
}
