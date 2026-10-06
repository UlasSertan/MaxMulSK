#include "trio.hpp"
#include "trio_internal.hpp"

#include <arm_acle.h>
#include <arm_neon.h>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>
#include <pthread.h>
#include <sys/qos.h>

#define PF_L2KEEP(p) __pldx(0, 1, 0, (const void*)(p))
#define PF_L2STRM(p) __pldx(0, 1, 1, (const void*)(p))

namespace trio {
namespace {
    using namespace detail;

    __attribute__((always_inline)) inline uint64_t ldaxr64(const std::atomic<uint64_t>* a) {
        return __builtin_arm_ldaex(reinterpret_cast<const volatile uint64_t*>(a));
    }
    __attribute__((always_inline)) inline uint64_t ticks() { __builtin_arm_isb(15); return __builtin_arm_rsr64("cntvct_el0"); }

    typedef float v8f __attribute__((ext_vector_type(8), aligned(16)));

    // NEON: B packing starts
    void pack_b_unit(Shared& sh, const Call& c, size_t u) {
        const size_t Kc = c.Kc, N = c.N, PFD = 32;
        const Unit& un = c.units[u];
        float* const dst = c.ring + (u % c.depth) * (Kc * 64 * kP);
        for (size_t r = 0; r < un.rows; r++) {
            const float* src = un.src + r * N;
            if (r + PFD < un.rows) {
                const char* pf = reinterpret_cast<const char*>(un.src + (r + PFD) * N);
                for (size_t l = 0; l < un.npan * 2; l++) PF_L2STRM(pf + 128 * l);
            }
            for (size_t q = 0; q < un.npan; q++) {
                float* const d = dst + q * (Kc * 64) + r * 64;
                const float* const s = src + q * 64;
                for (size_t o = 0; o < 64; o += 8)
                    *reinterpret_cast<v8f*>(d + o) = __builtin_nontemporal_load(reinterpret_cast<const v8f*>(s + o));
            }
        }
        sh.ready[u % c.depth].v.store(u + 1, std::memory_order_release);
    }
    // NEON: B packing ends

    // NEON: A packing (transpose) starts
    void pack_a_unit(Shared& sh, const Call& g, size_t a) {
        const size_t nmm = (g.M + g.Mc - 1) / g.Mc, nnn = (g.N + g.Nc - 1) / g.Nc;
        const size_t mm = (a % nmm) * g.Mc, kk = (a / (nmm * nnn)) * g.Kc;
        const size_t mcl = std::min(g.Mc, g.M - mm), kcl = std::min(g.Kc, g.K - kk), nb = kcl / 16;
        float* const slotbase = g.aring + (a % 2) * (g.Mc * g.Kc_pad);
        for (size_t ir = 0; ir < mcl; ir += 16) {
            const float* const rows = g.A + (mm + ir) * g.K + kk;
            float* const dst = slotbase + (ir / 16) * (g.Kc_pad * 16);
            for (size_t c = 0; c < nb; c++) {
                if (c + 8 < nb)
                    for (int j = 0; j < 16; j++) PF_L2KEEP(rows + (size_t)j * g.K + 16 * (c + 8));
                for (int gq = 0; gq < 4; gq++) {
                    const float* r0 = rows + (size_t)(4 * gq + 0) * g.K + 16 * c;
                    const float* r1 = r0 + g.K; const float* r2 = r1 + g.K; const float* r3 = r2 + g.K;
                    for (int cc = 0; cc < 16; cc += 4) {
                        const float32x4_t a0 = vld1q_f32(r0 + cc), a1 = vld1q_f32(r1 + cc), a2 = vld1q_f32(r2 + cc), a3 = vld1q_f32(r3 + cc);
                        const float32x4_t t0 = vtrn1q_f32(a0, a1), t1 = vtrn2q_f32(a0, a1), t2 = vtrn1q_f32(a2, a3), t3 = vtrn2q_f32(a2, a3);
                        float* const d = dst + 16 * (16 * c + cc) + 4 * gq;
                        vst1q_f32(d + 0,  vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t0), vreinterpretq_f64_f32(t2))));
                        vst1q_f32(d + 16, vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t1), vreinterpretq_f64_f32(t3))));
                        vst1q_f32(d + 32, vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t0), vreinterpretq_f64_f32(t2))));
                        vst1q_f32(d + 48, vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t1), vreinterpretq_f64_f32(t3))));
                    }
                }
            }
        }
        sh.a_ready[a % 2].v.store(a + 1, std::memory_order_release);
    }
    // NEON: A packing ends

    // NEON: packer schedule for one call starts
    void worker_call(Shared& sh, int id, uint64_t seq) {
        const Call c = sh.call;
        const size_t n = kPackers;
        size_t bi = c.b_first + (size_t)id, ai = c.a_first + (size_t)id;
        while (bi < c.nunits || ai < c.nA) {
            const uint64_t cb = ldaxr64(&sh.cons.b);
            const uint64_t ca = sh.cons.a.load(std::memory_order_acquire);
            const bool canB = bi < c.nunits && bi < cb + c.depth;
            const bool canA = ai < c.nA && ai < ca + 2;
            if (canA) {
                __builtin_arm_clrex();
                pack_a_unit(sh, c, ai); ai += n;
                continue;
            }
            if (canB) {
                __builtin_arm_clrex();
                pack_b_unit(sh, c, bi); bi += n;
                continue;
            }
            if (sh.done.v.load(std::memory_order_acquire) == seq) { __builtin_arm_clrex(); break; }
            __wfe();
            __builtin_arm_clrex();
        }
    }
    // NEON: packer schedule ends
}

struct Engine::Impl {
    Shared sh;
    bool usable = false;
    uint64_t idle_ticks = 0;
    std::atomic<bool> quit{false};
    std::atomic<int> sleepers{0}, exited{0};
    std::atomic<uint64_t> parks{0};
    std::mutex m;
    std::condition_variable cv;
    std::thread th[kPackers];
    float* ring = nullptr; size_t ring_floats = 0;
    float* aring = nullptr; size_t aring_floats = 0;
    float* stage = nullptr;
    std::vector<Unit> units;

    static float* grow(float* p, size_t& have, size_t need) {
        if (need <= have) return p;
        std::free(p);
        p = static_cast<float*>(std::aligned_alloc(64, (need * sizeof(float) + 63) / 64 * 64));
        have = p ? need : 0;
        return p;
    }

    // NEON: packer thread, waits between calls, starts
    void loop(int id) {
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
        uint64_t seen = 0;
        for (;;) {
            uint64_t s = 0, t0 = ticks();
            for (;;) {
                s = ldaxr64(&sh.seq.v);
                if (s != seen || quit.load(std::memory_order_acquire)) { __builtin_arm_clrex(); break; }
                if (ticks() - t0 > idle_ticks) {
                    __builtin_arm_clrex();
                    std::unique_lock<std::mutex> lk(m);
                    sleepers.fetch_add(1, std::memory_order_seq_cst);
                    parks.fetch_add(1, std::memory_order_relaxed);
                    cv.wait(lk, [&] { return sh.seq.v.load(std::memory_order_seq_cst) != seen || quit.load(); });
                    sleepers.fetch_sub(1, std::memory_order_seq_cst);
                    t0 = ticks();
                    continue;
                }
                __wfe();
                __builtin_arm_clrex();
            }
            if (quit.load(std::memory_order_acquire)) break;
            seen = s;
            worker_call(sh, id, s);
            sh.ack[id].v.store(s, std::memory_order_release);
        }
        exited.fetch_add(1, std::memory_order_release);
    }
    // NEON: packer thread ends
};

bool supported(size_t M, size_t K, size_t N) {
    return M && N && K && M % 16 == 0 && N % 64 == 0 && K % 64 == 0 && (double)M * N * K >= 256.0 * 256.0 * 256.0;
}

Engine::Engine(uint32_t idle_spin_us) : impl_(new Impl) {
    Impl& I = *impl_;
    I.usable = detail::streaming_lanes() == 16;
    I.idle_ticks = (uint64_t)idle_spin_us * __builtin_arm_rsr64("cntfrq_el0") / 1000000;
    I.stage = static_cast<float*>(std::aligned_alloc(64, 16 * 64 * sizeof(float)));
    for (int i = 0; i < kPackers; i++) I.th[i] = std::thread([&I, i] { I.loop(i); });
}

Engine::~Engine() {
    Impl& I = *impl_;
    I.quit.store(true, std::memory_order_seq_cst);
    { std::lock_guard<std::mutex> lk(I.m); }
    I.cv.notify_all();
    while (I.exited.load(std::memory_order_acquire) < kPackers) __sev();
    for (auto& t : I.th) t.join();
    std::free(I.ring); std::free(I.aring); std::free(I.stage);
    delete impl_;
}

uint64_t Engine::park_count() const { return impl_->parks.load(std::memory_order_relaxed); }

// Call: plan, publish, run the SME part, wait for the packers, starts
bool Engine::gemm(const float* A, const float* B, float* C, size_t M, size_t K, size_t N) {
    Impl& I = *impl_;
    if (!I.usable || !supported(M, K, N) || !I.stage) return false;
    Call& c = I.sh.call;
    const size_t Mc = std::min(kMc, M), Nc = std::min(kNc, N), Kc = std::min(kKc, K);
    const size_t Kc_pad = Kc;
    const size_t upb = (Nc / 64 + kP - 1) / kP, depth = M > Mc ? std::max<size_t>(4, upb) : 4;
    I.ring = Impl::grow(I.ring, I.ring_floats, depth * Kc * 64 * kP);
    I.aring = Impl::grow(I.aring, I.aring_floats, 2 * Mc * Kc_pad);
    if (!I.ring || !I.aring) return false;
    I.units.clear();
    for (size_t kk = 0; kk < K; kk += Kc)
        for (size_t nn = 0; nn < N; nn += Nc) {
            const size_t ncl = std::min(Nc, N - nn);
            for (size_t j = 0; j < ncl; j += 64 * kP)
                I.units.push_back({B + kk * N + nn + j, std::min(Kc, K - kk), std::min<size_t>(kP, (ncl - j) / 64)});
        }
    c.A = A; c.M = M; c.K = K; c.N = N; c.Mc = Mc; c.Nc = Nc; c.Kc = Kc; c.Kc_pad = Kc_pad;
    c.units = I.units.data(); c.nunits = I.units.size();
    c.ring = I.ring; c.depth = depth; c.aring = I.aring;
    c.nA = ((K + Kc - 1) / Kc) * ((N + Nc - 1) / Nc) * ((M + Mc - 1) / Mc);
    c.b_first = 1; c.a_first = 1;
    I.sh.cons.b.store(0, std::memory_order_relaxed); I.sh.cons.a.store(0, std::memory_order_relaxed);
    for (size_t i = 0; i < depth; i++) I.sh.ready[i].v.store(0, std::memory_order_relaxed);
    I.sh.a_ready[0].v.store(0, std::memory_order_relaxed); I.sh.a_ready[1].v.store(0, std::memory_order_relaxed);
    const uint64_t seq = I.sh.seq.v.load(std::memory_order_relaxed) + 1;
    I.sh.seq.v.store(seq, std::memory_order_seq_cst);
    __sev();
    if (I.sleepers.load(std::memory_order_seq_cst)) { { std::lock_guard<std::mutex> lk(I.m); } I.cv.notify_all(); }
    // SME part
    const bool ok = detail::run_sme(I.sh, C, I.stage);
    I.sh.done.v.store(seq, std::memory_order_release);
    for (int i = 0; i < kPackers; i++)
        while (I.sh.ack[i].v.load(std::memory_order_acquire) != seq) __sev();
    return ok;
}
// Call ends

}
