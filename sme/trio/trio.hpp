#pragma once

#include <cstddef>
#include <cstdint>

namespace trio {

    bool supported(size_t M, size_t K, size_t N);

    class Engine {
    public:
        explicit Engine(uint32_t idle_spin_us = 200);
        ~Engine();
        Engine(const Engine&) = delete;
        Engine& operator=(const Engine&) = delete;

        bool gemm(const float* A, const float* B, float* C, size_t M, size_t K, size_t N);

        uint64_t park_count() const;

    private:
        struct Impl;
        Impl* impl_;
    };

}
