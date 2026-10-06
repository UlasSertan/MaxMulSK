#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace trio::detail {

    constexpr size_t kMc = 64, kNc = 1024, kKc = 1024;
    constexpr size_t kP = 2;
    constexpr int kPackers = 2;

    struct alignas(128) Word { std::atomic<uint64_t> v{0}; char pad[128 - sizeof(std::atomic<uint64_t>)]; };

    struct alignas(128) Cons {
        std::atomic<uint64_t> b{0}, a{0};
        char pad[128 - 2 * sizeof(std::atomic<uint64_t>)];
    };

    struct Unit { const float* src; size_t rows, npan; };

    constexpr size_t kMaxDepth = 64;

    struct Call {
        const float* A = nullptr;
        size_t M = 0, K = 0, N = 0;
        size_t Mc = 0, Nc = 0, Kc = 0, Kc_pad = 0;
        const Unit* units = nullptr; size_t nunits = 0;
        float* ring = nullptr; size_t depth = 0;
        float* aring = nullptr; size_t nA = 0;
        size_t b_first = 1, a_first = 1;
    };

    struct Shared {
        Call call;
        Cons cons;
        Word seq;
        Word done;
        Word ack[kPackers];
        Word ready[kMaxDepth];
        Word a_ready[2];
    };

    bool run_sme(Shared& sh, float* C, float* stage);

    size_t streaming_lanes();

}
