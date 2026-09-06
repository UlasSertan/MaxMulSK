// bench/ggml_gemm_baseline.cpp — see ggml_gemm_baseline.hpp.

#include "ggml_gemm_baseline.hpp"

#include <chrono>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <sys/resource.h>

#ifndef BENCH_GGML_FLAGS
#define BENCH_GGML_FLAGS "(unknown)"
#endif

namespace {
std::string g_log;
void log_cb(ggml_log_level, const char* text, void*) { if (text) g_log += text; }
}  // namespace

namespace GgmlCpu {

bool available() { return true; }

bool kleidiai_build() {
#ifdef BENCH_GGML_KLEIDIAI
    return true;
#else
    return false;
#endif
}

void start_log_capture() { g_log.clear(); ggml_log_set(log_cb, nullptr); }
std::string ggml_log_capture() { return g_log; }

std::string build_description() { return BENCH_GGML_FLAGS; }

struct Gemm::Impl {
    ggml_backend_t backend = nullptr;
    ggml_context* ctx_data = nullptr;     // owns b (and a, when not KleidiAI)
    ggml_context* ctx_w = nullptr;        // owns a when it lives in CPU_KLEIDIAI
    ggml_backend_buffer_t buf_w = nullptr;
    bool kleidiai_buft = false;
    double pack_ms = 0.0;
    ggml_context* ctx_graph = nullptr;    // owns the op + graph nodes
    ggml_backend_buffer_t buf = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_tensor* a = nullptr;             // [K, M]  == A row-major
    ggml_tensor* b = nullptr;             // [K, N]  == B transposed
    ggml_tensor* c = nullptr;             // [M, N]  == C transposed
    ggml_cgraph* gf = nullptr;
    size_t M = 0, K = 0, N = 0;
    double transpose_ms = 0.0;

    void build_graph() {
        if (ctx_graph) ggml_free(ctx_graph);
        ggml_init_params gp{};
        gp.mem_size   = ggml_tensor_overhead() * 8 + ggml_graph_overhead();
        gp.mem_buffer = nullptr;
        gp.no_alloc   = true;
        ctx_graph = ggml_init(gp);
        c  = ggml_mul_mat(ctx_graph, a, b);
        gf = ggml_new_graph(ctx_graph);
        ggml_build_forward_expand(gf, c);
        ggml_gallocr_alloc_graph(galloc, gf);
    }

    ~Impl() {
        if (galloc)    ggml_gallocr_free(galloc);
        if (ctx_graph) ggml_free(ctx_graph);
        if (buf)       ggml_backend_buffer_free(buf);
        if (buf_w)     ggml_backend_buffer_free(buf_w);
        if (ctx_data)  ggml_free(ctx_data);
        if (ctx_w)     ggml_free(ctx_w);
        if (backend)   ggml_backend_free(backend);
    }
};

Gemm::Gemm() : impl_(std::make_unique<Impl>()) {}
Gemm::~Gemm() = default;

double Gemm::b_transpose_ms() const { return impl_->transpose_ms; }

// Locate ggml's "CPU_KLEIDIAI" extra buffer type through the same public
// proc-address mechanism llama.cpp uses for model weights.
static ggml_backend_buffer_type_t find_kleidiai_buft(ggml_backend_t backend) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (!dev) return nullptr;
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (!reg) return nullptr;
    auto fn = (ggml_backend_dev_get_extra_bufts_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    if (!fn) return nullptr;
    ggml_backend_buffer_type_t* bufts = fn(dev);
    for (; bufts && *bufts; ++bufts) {
        const char* nm = ggml_backend_buft_name(*bufts);
        if (nm && std::string(nm) == "CPU_KLEIDIAI") return *bufts;
    }
    return nullptr;
}

bool Gemm::using_kleidiai_buft() const { return impl_->kleidiai_buft; }
double Gemm::weight_pack_ms() const { return impl_->pack_ms; }

bool Gemm::prepare(size_t M, size_t K, size_t N,
                   const float* A, const float* B, int n_threads,
                   bool use_kleidiai) {
    Impl& s = *impl_;
    s.M = M; s.K = K; s.N = N;

    s.backend = ggml_backend_cpu_init();
    if (!s.backend) return false;
    ggml_backend_cpu_set_n_threads(s.backend, n_threads);

    ggml_init_params dp{};
    dp.mem_size   = ggml_tensor_overhead() * 8;
    dp.mem_buffer = nullptr;
    dp.no_alloc   = true;
    s.ctx_data = ggml_init(dp);
    if (!s.ctx_data) return false;

    // ne0 is the contiguous dimension. src0 = [K, M] is A row-major as-is;
    // src1 = [K, N] must hold B transposed.
    if (use_kleidiai) {
        ggml_backend_buffer_type_t kb = find_kleidiai_buft(s.backend);
        if (!kb) return false;
        ggml_init_params wp{};
        wp.mem_size = ggml_tensor_overhead() * 4;
        wp.no_alloc = true;
        s.ctx_w = ggml_init(wp);
        if (!s.ctx_w) return false;
        s.a = ggml_new_tensor_2d(s.ctx_w, GGML_TYPE_F32, (int64_t)K, (int64_t)M);
        s.buf_w = ggml_backend_alloc_ctx_tensors_from_buft(s.ctx_w, kb);
        if (!s.buf_w) return false;
        s.kleidiai_buft = true;
    } else {
        s.a = ggml_new_tensor_2d(s.ctx_data, GGML_TYPE_F32, (int64_t)K, (int64_t)M);
    }
    s.b = ggml_new_tensor_2d(s.ctx_data, GGML_TYPE_F32, (int64_t)K, (int64_t)N);

    s.buf = ggml_backend_alloc_ctx_tensors(s.ctx_data, s.backend);
    if (!s.buf) return false;

    // Static layout conversion, outside every timed region. Timed here only so
    // its cost can be reported honestly rather than hidden.
    std::vector<float> Bt(K * N);
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t k = 0; k < K; ++k)
        for (size_t n = 0; n < N; ++n)
            Bt[n * K + k] = B[k * N + n];
    s.transpose_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    const auto tp0 = std::chrono::steady_clock::now();
    ggml_backend_tensor_set(s.a, A, 0, M * K * sizeof(float));
    s.pack_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tp0).count();
    ggml_backend_tensor_set(s.b, Bt.data(), 0, K * N * sizeof(float));

    s.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(s.backend));
    if (!s.galloc) return false;
    s.build_graph();
    return s.c != nullptr;
}

void Gemm::run() {
    ggml_backend_graph_compute(impl_->backend, impl_->gf);
}

void Gemm::run_with_setup() {
    impl_->build_graph();
    ggml_backend_graph_compute(impl_->backend, impl_->gf);
}

void Gemm::read_result(float* C) const {
    const Impl& s = *impl_;
    // dst is [M, N] with M contiguous: dst[n*M + m] == C[m][n].
    std::vector<float> tmp(s.M * s.N);
    ggml_backend_tensor_get(s.c, tmp.data(), 0, s.M * s.N * sizeof(float));
    for (size_t n = 0; n < s.N; ++n)
        for (size_t m = 0; m < s.M; ++m)
            C[m * s.N + n] = tmp[n * s.M + m];
}

double Gemm::measure_cpu_to_wall_ratio(int runs) {
    auto cpu_seconds = [] {
        rusage ru{};
        getrusage(RUSAGE_SELF, &ru);
        return double(ru.ru_utime.tv_sec) + double(ru.ru_utime.tv_usec) * 1e-6
             + double(ru.ru_stime.tv_sec) + double(ru.ru_stime.tv_usec) * 1e-6;
    };
    run();  // warm
    const double c0 = cpu_seconds();
    const auto   w0 = std::chrono::steady_clock::now();
    for (int i = 0; i < runs; ++i) run();
    const double wall =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
    const double cpu = cpu_seconds() - c0;
    return (wall > 0.0) ? cpu / wall : 0.0;
}

}  // namespace GgmlCpu
