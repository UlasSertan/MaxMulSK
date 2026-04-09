#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <string>
#include <cstddef>

namespace Utils {

    inline void fill_random(std::vector<float>& vec) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        for (auto& val : vec) val = dis(gen);
    }

    inline double compute_gflops(size_t M, size_t N, size_t K, double time_ms) {
        return (2.0 * M * N * K) / (time_ms / 1000.0) / 1e9;
    }

    inline bool check_correctness(const float* ref, const float* target, size_t size,
                                  const std::string& name, float tol = 1e-3f) {
        float max_diff = 0.0f;
        size_t error_count = 0;

        for (size_t i = 0; i < size; ++i) {
            float diff = std::abs(ref[i] - target[i]);
            if (diff > tol) {
                if (error_count < 5) {
                    std::cout << "  [" << name << "] Mismatch @ " << i
                              << " | ref=" << ref[i] << " got=" << target[i]
                              << " diff=" << diff << "\n";
                }
                max_diff = std::max(max_diff, diff);
                error_count++;
            }
        }

        if (error_count > 0) {
            std::cout << "  [" << name << "] FAIL: " << error_count
                      << " errors, max diff=" << max_diff << "\n";
            return false;
        }
        return true;
    }

    inline void multiply_scalar(const float* A, const float* B, float* C,
                                size_t M, size_t N, size_t K) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }

} // namespace Utils
