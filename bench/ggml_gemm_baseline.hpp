// bench/ggml_gemm_baseline.hpp
// Experiment 3: the native ggml CPU FP32 matrix-multiplication path, as used by
// llama.cpp, as a third baseline alongside MaxMulSK and KleidiAI.
//
// WHAT ACTUALLY EXECUTES. `ggml_compute_forward_mul_mat` tries `llamafile_sgemm`
// first; for (F32, F32, F32) with a contiguous src1 and n >= 4 that dispatches
// on AArch64 to ggml's built-in `tinyBLAS<4, float32x4_t, ...>` — a 4-wide NEON
// FMLA kernel compiled into ggml-cpu. It is not an external library, and it does
// not use SME. Metal, Accelerate, BLAS and ggml's KleidiAI integration are all
// disabled in the build; see the flags reported by build_description().
//
// LAYOUT. ggml computes dst[n][m] = dot(src0 row m, src1 row n) with dst stored
// ne0-major. To get row-major C = A(MxK) * B(KxN) the src1 tensor must hold B
// transposed (N rows of length K) and the result must be read back transposed.
// That is not an artificial adapter: it is exactly how llama.cpp stores weights,
// so both conversions are static layout, done outside the timed region. The cost
// of the one-off B transpose is measured and reported separately so it can be
// added back if you disagree.

#pragma once

#include <cstddef>
#include <memory>
#include <string>

namespace GgmlCpu {

bool available();

// Which backend/kernel was selected, and what was compiled out.
std::string build_description();

// True if this binary was linked against a ggml built with GGML_CPU_KLEIDIAI=ON.
bool kleidiai_build();

// Everything ggml itself logged at backend init: "kleidiai: SME enabled ...",
// "kleidiai: primary f32 kernel feature SME2", etc. Captured via ggml_log_set
// so the diagnostics come from ggml rather than from an assumption here.
std::string ggml_log_capture();
void start_log_capture();

class Gemm {
public:
    Gemm();
    ~Gemm();
    Gemm(const Gemm&)            = delete;
    Gemm& operator=(const Gemm&) = delete;

    // Untimed: backend init, context, tensors, graph, and uploading A and B^T.
    // Returns false if ggml could not accept the shape.
    // use_kleidiai=true allocates src0 in ggml's "CPU_KLEIDIAI" extra buffer
    // type, which is the only way the KleidiAI path is reachable: its
    // supports_op() requires src0->buffer->buft to be that buffer type. This is
    // exactly how llama.cpp allocates model weights, so it is the normal ggml
    // operator flow, not a bypass. Returns false if the shape was rejected or
    // the buffer type is unavailable.
    bool prepare(size_t M, size_t K, size_t N,
                 const float* A, const float* B, int n_threads,
                 bool use_kleidiai = false);

    // True if src0 actually landed in the CPU_KLEIDIAI buffer type.
    bool using_kleidiai_buft() const;

    // Wall-clock ms spent in ggml_backend_tensor_set for src0. With the
    // KleidiAI buffer type this is the one-off weight repack that llama.cpp
    // pays at model load, so it is reported rather than timed per call.
    double weight_pack_ms() const;

    // Timed (mandatory mode): execute the already-built graph.
    void run();

    // Timed (optional mode): rebuild the graph, then execute. Excludes tensor
    // allocation and data upload.
    void run_with_setup();

    // Untimed: transpose ggml's [M-major, N] result into row-major C (M x N).
    void read_result(float* C) const;

    // Wall-clock milliseconds spent transposing B during prepare().
    double b_transpose_ms() const;

    // Measured, not assumed: process CPU time / wall time over a run. ~1.0 means
    // a single worker actually did the work.
    double measure_cpu_to_wall_ratio(int runs);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace GgmlCpu
