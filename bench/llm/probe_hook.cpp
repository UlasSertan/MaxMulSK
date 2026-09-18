// Feasibility probe: does the patched ggml BLAS hook actually receive the
// model's matrix multiplications, in prefill and in single-token decode?
// Counts calls and records every distinct (m, n, k) shape.
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <map>
#include <tuple>
#include <vector>
#include <string>

extern "C" {
typedef void (*maxmulsk_sgemm_hook_t)(int, int, int, const float*, int, const float*, int, float*, int);
extern maxmulsk_sgemm_hook_t maxmulsk_sgemm_hook;
}
extern "C" void cblas_sgemm(int, int, int, int, int, int, float,
                            const float*, int, const float*, int, float, float*, int);

static std::map<std::tuple<int,int,int>, long> g_shapes;
static long g_calls = 0;
static bool g_record = false;

// C[m x n] = A[m x k] * B^T, B stored n x k row-major. Passes straight through
// to Accelerate; this probe only measures WHICH shapes arrive, not speed.
static void probe_hook(int m, int n, int k, const float* A, int lda,
                       const float* B, int ldb, float* C, int ldc) {
    if (g_record) { g_calls++; g_shapes[{m,n,k}]++; }
    cblas_sgemm(101 /*RowMajor*/, 111 /*NoTrans*/, 112 /*Trans*/,
                m, n, k, 1.0f, A, lda, B, ldb, 0.0f, C, ldc);
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "";
    const int n_prefill = argc > 2 ? atoi(argv[2]) : 128;
    const int n_decode  = argc > 3 ? atoi(argv[3]) : 8;

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(path, mp);
    if (!model) { std::fprintf(stderr, "model yuklenemedi\n"); return 1; }

    auto cp = llama_context_default_params();
    cp.n_ctx = 2048; cp.n_batch = 2048; cp.n_ubatch = 2048;
    cp.n_threads = 1; cp.n_threads_batch = 1;
    cp.type_k = GGML_TYPE_F32; cp.type_v = GGML_TYPE_F32;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "context olusmadi\n"); return 1; }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(n_prefill, 0);
    toks[0] = llama_vocab_bos(vocab);
    for (int i = 1; i < n_prefill; i++) toks[i] = 100 + (i % 500);

    maxmulsk_sgemm_hook = probe_hook;

    // ---- prefill ----
    g_record = true; g_calls = 0; g_shapes.clear();
    llama_batch b = llama_batch_get_one(toks.data(), n_prefill);
    if (llama_decode(ctx, b) != 0) { std::fprintf(stderr, "prefill decode hatasi\n"); return 1; }
    std::printf("PREFILL %d token: %ld hook cagrisi, %zu farkli shape\n", n_prefill, g_calls, g_shapes.size());
    for (auto& [s, c] : g_shapes) {
        auto [m,n,k] = s;
        std::printf("   m=%-6d n=%-6d k=%-6d  x%ld   %s\n", m, n, k, c,
                    (m%16==0 && n%64==0 && k%64==0) ? "v4c-native-uygun" : "v4c-native-DEGIL");
    }

    // ---- decode, tek token ----
    auto prev = g_shapes; long pc = g_calls;
    g_calls = 0; g_shapes.clear();
    llama_token t = 200;
    for (int i = 0; i < n_decode; i++) {
        llama_batch d = llama_batch_get_one(&t, 1);
        if (llama_decode(ctx, d) != 0) { std::fprintf(stderr, "decode hatasi\n"); return 1; }
        t = 201 + i;
    }
    std::printf("\nDECODE %d adim: %ld hook cagrisi (%ld/adim), %zu farkli shape\n",
                n_decode, g_calls, g_calls/n_decode, g_shapes.size());
    for (auto& [s, c] : g_shapes) {
        auto [m,n,k] = s;
        std::printf("   m=%-6d n=%-6d k=%-6d  x%ld   %s\n", m, n, k, c,
                    (m%16==0 && n%64==0 && k%64==0) ? "v4c-native-uygun" : "v4c-native-DEGIL");
    }
    (void)prev; (void)pc;
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
