#pragma once

#include <cstddef>

namespace GEMM {

    void package(const float* A, const float* B, float* C,
                 size_t M, size_t N, size_t K);

    void multiply(const float* A, const float* B, float* C,
                  size_t M, size_t Nc, size_t Kc, size_t N);

    void pack_B_block(const float* B, float* pack_B,
                      size_t j_outer, size_t Nc, size_t Kc, size_t N);

    void pack_A_block(const float* A, float* pack_A,
                      size_t Mc, size_t Kc, size_t K);

    void transpose_8x4(const float* src, float* dest, size_t K);

} // namespace GEMM
