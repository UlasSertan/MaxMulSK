// bench/kleidiai_gemm.cpp
// Implementation of the KleidiAI benchmark wrapper. See kleidiai_gemm.hpp.

#include "kleidiai_gemm.hpp"

#ifdef MAXMULSK_HAVE_KLEIDIAI

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <vector>

extern "C" {
#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/kai_matmul_pack_lhs.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_f32p2vlx1_f32_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme.h"
}

// kai_common.h only declares the SME length helpers under __ARM_FEATURE_SVE2,
// which -mcpu=apple-m4 does not define (the M4 exposes SVE only inside
// streaming mode). The symbol itself is unconditionally present in the library,
// defined in kai/kai_common_sme_asm.S, so declare it here rather than building
// this translation unit with +sve2 — that would let the compiler emit
// non-streaming SVE2 instructions the CPU cannot execute.
extern "C" uint64_t kai_get_sme_vector_length_u8(void);

namespace {

// vscale: SVL in units of 128 bits. 4 on an M4 (SVL = 512 bits).
size_t sme_vscale() { return kai_get_sme_vector_length_u8() / 16; }

}  // namespace

namespace KleidiAI {

struct Gemm::Impl {
    Kernel kind;
    size_t M = 0, N = 0, K = 0;

    std::vector<uint8_t> lhs_packed;
    std::vector<uint8_t> rhs_packed;
    size_t lhs_bytes = 0, rhs_bytes = 0;  // sizes, for run_end_to_end()
    // When non-null these replace the persistent buffers for the duration of a
    // run_end_to_end() call. See that function.
    uint8_t* ext_lhs = nullptr;
    uint8_t* ext_rhs = nullptr;
    uint8_t* lhs_buf() { return ext_lhs ? ext_lhs : lhs_packed.data(); }
    uint8_t* rhs_buf() { return ext_rhs ? ext_rhs : rhs_packed.data(); }
    const uint8_t* lhs_buf() const { return ext_lhs ? ext_lhs : lhs_packed.data(); }
    const uint8_t* rhs_buf() const { return ext_rhs ? ext_rhs : rhs_packed.data(); }
    // C = A*B, so the bias the "…bias/…b" kernels fold in must be zero.
    std::vector<float> bias;

    // Sme2Mopa8VS uses KleidiAI's newer uker_api interface: the entry points
    // are function-pointer tables returned by value, and the packed layout is
    // described by an explicit format config rather than by mr/nr/kr/sr args.
    kai_matmul_uker_api           mm_api{};
    kai_matmul_pack_lhs_uker_api  lhs_api{};
    kai_matmul_pack_rhs_uker_api  rhs_api{};
    kai_matmul_uker_config           mm_cfg{};
    kai_matmul_pack_lhs_uker_config  lhs_cfg{};
    kai_matmul_pack_rhs_uker_config  rhs_cfg{};
    kai_matmul_pack_lhs_uker_lhs_packed_stride_args lhs_packed_stride{};
    kai_matmul_pack_rhs_uker_rhs_packed_stride_args rhs_packed_stride{};
    kai_matmul_pack_rhs_uker_rhs_stride_args        rhs_stride{};

    // Outer N-blocking. KleidiAI ships no cache blocking above the micro-kernel
    // — it expects the calling runtime to supply it, which is what the
    // *_packed_offset / *_dst_offset accessors are for. Without it the kernel
    // re-reads the whole packed B for every m_step row panel, so once B stops
    // fitting in L2 throughput collapses: measured on an M4 (16 MB P-core L2),
    // a monolithic 4096^3 call runs at 887 GFLOPS while the same kernel driven
    // in N blocks runs at 1754. Blocking M was measured to make no difference —
    // A streams through once either way — so only N is blocked here.
    size_t n_step  = 1;
    size_t n_block = 0;  // columns per outer block; set in reshape()
    size_t forced_n_block = 0;  // set_n_block() override; 0 = use the heuristic

    explicit Impl(Kernel k) : kind(k) {
        if (kind == Kernel::Sme2Mopa8VS) {
            mm_api  = kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa();
            lhs_api = kai_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme();
            rhs_api = kai_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme();

            const size_t vs = sme_vscale();
            lhs_cfg.format = {/*mr=*/4 * vs, /*kr=*/1, /*sr=*/1};
            rhs_cfg.format = {/*nr=*/4 * vs, /*kr=*/1, /*sr=*/1, /*bl=*/0};
        }
    }
};

const char* name(Kernel k) {
    switch (k) {
        case Kernel::Sme2Mopa2VL:
            return "kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa";
        case Kernel::Sme2Mopa8VS:
            return "kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa";
        case Kernel::Neon6x8Mla:
            return "kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla";
    }
    return "unknown";
}

const char* label(Kernel k) {
    switch (k) {
        case Kernel::Sme2Mopa2VL: return "2VL";
        case Kernel::Sme2Mopa8VS: return "8VS";
        case Kernel::Neon6x8Mla:  return "6x8";
    }
    return "?";
}

bool available() { return true; }

Gemm::Gemm(Kernel k) : impl_(std::make_unique<Impl>(k)) {}
Gemm::~Gemm() = default;

void Gemm::reshape(size_t M, size_t N, size_t K) {
    Impl& s = *impl_;
    s.M = M; s.N = N; s.K = K;
    s.bias.assign(N, 0.0f);

    switch (s.kind) {
        case Kernel::Sme2Mopa2VL:
            s.n_step = kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
            break;
        case Kernel::Sme2Mopa8VS:
            s.n_step = s.mm_api.get_step(&s.mm_cfg).n;
            break;
        case Kernel::Neon6x8Mla:
            s.n_step = kai_get_n_step_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla();
            break;
    }
    // The panel re-read per row sweep is N_blk*K*4 bytes. 4 MB leaves the rest
    // of the 16 MB L2 for the streaming A and C traffic; measured best or within
    // 5% of best at both 2048^3 and 4096^3. A 16 MB panel is the cliff.
    constexpr size_t kRhsPanelBudget = 4u << 20;
    s.n_block = kRhsPanelBudget / (K * sizeof(float));

    // Overrides, in precedence order: set_n_block(), then the environment
    // escape hatch MAXMULSK_KAI_NBLOCK=<columns> (0 = one monolithic call).
    if (s.forced_n_block != 0) {
        s.n_block = s.forced_n_block;
    } else if (const char* env = std::getenv("MAXMULSK_KAI_NBLOCK")) {
        const long v = std::strtol(env, nullptr, 10);
        s.n_block = (v <= 0) ? N : static_cast<size_t>(v);
    }

    s.n_block = (s.n_block / s.n_step) * s.n_step;   // the accessors require
    if (s.n_block == 0) s.n_block = s.n_step;        // n_step alignment
    if (s.n_block > N) s.n_block = N;                // small N: one block

    switch (s.kind) {
        case Kernel::Sme2Mopa2VL: {
            const size_t mr = kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
            const size_t kr = kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
            const size_t sr = kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
            s.lhs_bytes = kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme(M, K, mr, kr, sr);
            s.rhs_bytes = kai_get_rhs_packed_size_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(N, K);
            s.lhs_packed.assign(s.lhs_bytes, 0);
            s.rhs_packed.assign(s.rhs_bytes, 0);
            break;
        }
        case Kernel::Sme2Mopa8VS: {
            const kai_matmul_pack_lhs_uker_lhs_packed_dim_args ld{M, K};
            s.lhs_packed_stride = s.lhs_api.get_lhs_packed_stride(&s.lhs_cfg, &ld);
            s.lhs_bytes = s.lhs_api.get_lhs_packed_size(&s.lhs_cfg, &ld, &s.lhs_packed_stride);
            s.lhs_packed.assign(s.lhs_bytes, 0);

            const kai_matmul_pack_rhs_uker_rhs_packed_dim_args rd{N, K};
            s.rhs_packed_stride = s.rhs_api.get_rhs_packed_stride(&s.rhs_cfg, &rd);
            s.rhs_bytes = s.rhs_api.get_rhs_packed_size(&s.rhs_cfg, &rd, &s.rhs_packed_stride);
            s.rhs_packed.assign(s.rhs_bytes, 0);

            const kai_matmul_pack_rhs_uker_rhs_dim_args rs{N, K};
            s.rhs_stride = s.rhs_api.get_rhs_stride(&s.rhs_cfg, &rs);
            break;
        }
        case Kernel::Neon6x8Mla: {
            // Left-hand side is consumed row-major; only B is packed.
            s.lhs_bytes = 0;
            s.lhs_packed.clear();
            s.rhs_bytes = kai_get_rhs_packed_size_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(N, K);
            s.rhs_packed.assign(s.rhs_bytes, 0);
            break;
        }
    }
}

void Gemm::pack(const float* A, const float* B) {
    Impl& s = *impl_;
    switch (s.kind) {
        case Kernel::Sme2Mopa2VL: {
            const size_t mr = kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
            const size_t nr = kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
            const size_t kr = kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
            const size_t sr = kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
            kai_run_lhs_pack_f32p2vlx1_f32_sme(s.M, s.K, mr, kr, sr, /*m_idx_start=*/0,
                                               A, s.K * sizeof(float), s.lhs_buf());
            kai_run_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(
                /*num_groups=*/1, s.N, s.K, nr, kr, sr, /*rhs_stride_row=*/s.N * sizeof(float),
                B, s.bias.data(), /*scale=*/nullptr, s.rhs_buf(),
                /*extra_bytes=*/0, /*params=*/nullptr);
            break;
        }
        case Kernel::Sme2Mopa8VS: {
            kai_matmul_pack_lhs_uker_args la{};
            la.shape = {s.M, s.K};
            la.operand.lhs = {A, {s.K * sizeof(float)}};
            la.operand.lhs_packed = {s.lhs_buf(), {s.lhs_packed_stride.m}};
            s.lhs_api.run(&s.lhs_cfg, &la);

            kai_matmul_pack_rhs_uker_args ra{};
            ra.shape = {s.N, s.K};
            ra.operand.rhs = {B, s.rhs_stride};
            ra.operand.rhs_packed = {s.rhs_buf(), {s.rhs_packed_stride.n}};
            ra.operand.bias_n = {s.bias.data()};
            s.rhs_api.run(&s.rhs_cfg, &ra);
            break;
        }
        case Kernel::Neon6x8Mla: {
            (void)A;
            const size_t nr = kai_get_nr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla();
            const size_t kr = kai_get_kr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla();
            const size_t sr = kai_get_sr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla();
            kai_run_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(
                /*num_groups=*/1, s.N, s.K, nr, kr, sr, /*rhs_stride=*/s.N * sizeof(float),
                B, s.bias.data(), /*scale=*/nullptr, s.rhs_buf(),
                /*extra_bytes=*/0, /*params=*/nullptr);
            break;
        }
    }
}

void Gemm::matmul(const float* A, float* C) {
    Impl& s = *impl_;
    const size_t ldc = s.N * sizeof(float);
    auto* const dst  = reinterpret_cast<uint8_t*>(C);

    // One pass per N block, full M and K inside. See the note on n_block above:
    // this outer loop is the caller's responsibility in KleidiAI's design, and
    // omitting it costs ~2x at sizes where the packed B panel outgrows L2.
    for (size_t n0 = 0; n0 < s.N; n0 += s.n_block) {
        const size_t nn = std::min(s.n_block, s.N - n0);

        switch (s.kind) {
            case Kernel::Sme2Mopa2VL:
                (void)A;
                kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(
                    s.M, nn, s.K,
                    s.lhs_buf(),
                    s.rhs_buf() +
                        kai_get_rhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(
                            n0, s.K),
                    dst + kai_get_dst_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(
                              /*m_idx=*/0, n0, ldc),
                    ldc, /*dst_stride_col=*/sizeof(float), -FLT_MAX, FLT_MAX);
                break;

            case Kernel::Sme2Mopa8VS: {
                (void)A;
                const kai_matmul_uker_rhs_dim_args rhs_idx{n0, 0};
                const kai_matmul_uker_dst_dim_args dst_idx{0, n0};
                const kai_matmul_uker_rhs_stride_args rhs_str{s.rhs_packed_stride.n};
                const kai_matmul_uker_dst_stride_args dst_str{ldc};

                kai_matmul_uker_args a{};
                a.flags = 0;  // no clamp: the kernel then uses [-FLT_MAX, FLT_MAX]
                a.shape = {s.M, nn, s.K};
                a.operand.dst = {dst + s.mm_api.get_dst_offset(&s.mm_cfg, &dst_idx, &dst_str),
                                 dst_str};
                a.operand.lhs = {s.lhs_buf(), {s.lhs_packed_stride.m}};
                a.operand.rhs = {s.rhs_buf() +
                                     s.mm_api.get_rhs_offset(&s.mm_cfg, &rhs_idx, &rhs_str),
                                 rhs_str};
                s.mm_api.run(&s.mm_cfg, &a);
                break;
            }

            case Kernel::Neon6x8Mla:
                kai_run_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla(
                    s.M, nn, s.K, A, /*lhs_stride=*/s.K * sizeof(float),
                    s.rhs_buf() +
                        kai_get_rhs_packed_offset_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla(
                            n0, s.K),
                    dst + kai_get_dst_offset_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla(
                              /*m_idx=*/0, n0, ldc),
                    ldc, /*dst_stride_col=*/sizeof(float), -FLT_MAX, FLT_MAX);
                break;
        }
    }
}

void Gemm::set_n_block(size_t columns) {
    Impl& s = *impl_;
    s.forced_n_block = columns;
    size_t nb = columns ? columns : (4u << 20) / (s.K * sizeof(float));
    nb = (nb / s.n_step) * s.n_step;
    if (nb == 0) nb = s.n_step;
    if (nb > s.N) nb = s.N;
    s.n_block = nb;
}

size_t Gemm::n_block() const { return impl_->n_block; }

void Gemm::run_end_to_end(const float* A, const float* B, float* C) {
    Impl& s = *impl_;
    // Fresh, *untouched* buffers every call. std::vector would zero-fill them
    // and pre-fault every page; MaxMulSK's driver uses a bare aligned_alloc and
    // takes the faults during packing, so mirror that exactly.
    struct Scoped {
        uint8_t* p;
        explicit Scoped(size_t n)
            : p(n ? static_cast<uint8_t*>(std::aligned_alloc(64, (n + 63) & ~size_t(63)))
                  : nullptr) {}
        ~Scoped() { std::free(p); }
    } lhs(s.lhs_bytes), rhs(s.rhs_bytes);

    std::vector<uint8_t> saved_lhs, saved_rhs;
    saved_lhs.swap(s.lhs_packed);
    saved_rhs.swap(s.rhs_packed);
    // Point the pack/matmul paths at the scratch allocations for this call.
    s.ext_lhs = lhs.p;
    s.ext_rhs = rhs.p;
    pack(A, B);
    matmul(A, C);
    s.ext_lhs = nullptr;
    s.ext_rhs = nullptr;
    saved_lhs.swap(s.lhs_packed);
    saved_rhs.swap(s.rhs_packed);
}

bool Gemm::supports_panel_blocking() const {
    return impl_->kind == Kernel::Sme2Mopa2VL || impl_->kind == Kernel::Sme2Mopa8VS;
}

void Gemm::run_panel_blocked(const float* A, const float* B, float* C,
                             const PanelConfig& cfg) {
    panel_blocked_impl(A, B, C, cfg, nullptr);
}

PanelBreakdown Gemm::run_panel_blocked_instrumented(const float* A, const float* B,
                                                    float* C, const PanelConfig& cfg) {
    PanelBreakdown bd;
    panel_blocked_impl(A, B, C, cfg, &bd);
    return bd;
}

void Gemm::panel_blocked_impl(const float* A, const float* B, float* C,
                              const PanelConfig& cfg, PanelBreakdown* bd) {
    Impl& s = *impl_;
    const size_t M = s.M, N = s.N, K = s.K;
    const size_t Mc = cfg.Mc ? std::min(cfg.Mc, M) : M;
    const size_t Kc = cfg.Kc ? std::min(cfg.Kc, K) : K;
    const size_t Nc = cfg.Nc ? std::min(cfg.Nc, N) : N;
    const bool k_blocked = Kc < K;
    const bool is2vl = (s.kind == Kernel::Sme2Mopa2VL);

    using Clk = std::chrono::steady_clock;
    const auto t_begin = Clk::now();
    auto stamp = [&](double& slot, Clk::time_point a) {
        if (bd) slot += std::chrono::duration<double, std::milli>(Clk::now() - a).count();
    };

    // 2VL packing parameters.
    const size_t mr = is2vl ? kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa() : 0;
    const size_t nr = is2vl ? kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa() : 0;
    const size_t kr = is2vl ? kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa() : 1;
    const size_t sr = is2vl ? kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa() : 1;

    // Worst-case panel sizes; packed size is monotonic in m/n/k, so buffers
    // sized for (Mc, Kc) / (Nc, Kc) also hold every edge panel.
    size_t lhs_bytes, rhs_bytes;
    if (is2vl) {
        lhs_bytes = kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme(Mc, Kc, mr, kr, sr);
        rhs_bytes = kai_get_rhs_packed_size_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(Nc, Kc);
    } else {
        const kai_matmul_pack_lhs_uker_lhs_packed_dim_args ld{Mc, Kc};
        const auto ls = s.lhs_api.get_lhs_packed_stride(&s.lhs_cfg, &ld);
        lhs_bytes = s.lhs_api.get_lhs_packed_size(&s.lhs_cfg, &ld, &ls);
        const kai_matmul_pack_rhs_uker_rhs_packed_dim_args rd{Nc, Kc};
        const auto rs = s.rhs_api.get_rhs_packed_stride(&s.rhs_cfg, &rd);
        rhs_bytes = s.rhs_api.get_rhs_packed_size(&s.rhs_cfg, &rd, &rs);
    }

    // Panel buffers are allocated per call, exactly as MaxMulSK's driver does.
    struct Scoped {
        void* p;
        explicit Scoped(size_t n)
            : p(n ? std::aligned_alloc(64, (n + 63) & ~size_t(63)) : nullptr) {}
        ~Scoped() { std::free(p); }
    };
    Scoped bufA(lhs_bytes), bufB(rhs_bytes);
    Scoped scratch(k_blocked ? Mc * Nc * sizeof(float) : 0);
    std::vector<float> bias(Nc, 0.0f);
    if (bd) stamp(bd->other_ms, t_begin);

    const size_t ldc = N * sizeof(float);

    for (size_t n0 = 0; n0 < N; n0 += Nc) {
        const size_t nc = std::min(Nc, N - n0);
        for (size_t k0 = 0; k0 < K; k0 += Kc) {
            const size_t kc = std::min(Kc, K - k0);

            // ---- pack the B panel, then consume it while it is still hot ----
            const auto tb = Clk::now();
            size_t rhs_stride_n = 0;
            if (is2vl) {
                kai_run_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(
                    1, nc, kc, nr, kr, sr, N * sizeof(float),
                    B + k0 * N + n0, bias.data(), nullptr, bufB.p, 0, nullptr);
            } else {
                const kai_matmul_pack_rhs_uker_rhs_packed_dim_args rd{nc, kc};
                const auto rs = s.rhs_api.get_rhs_packed_stride(&s.rhs_cfg, &rd);
                rhs_stride_n = rs.n;
                kai_matmul_pack_rhs_uker_args ra{};
                ra.shape = {nc, kc};
                // Sub-panel of a kxn source: the row stride is the parent N,
                // not nc, so these cannot come from get_rhs_stride().
                ra.operand.rhs = {B + k0 * N + n0, {sizeof(float), N * sizeof(float)}};
                ra.operand.rhs_packed = {bufB.p, {rs.n}};
                ra.operand.bias_n = {bias.data()};
                s.rhs_api.run(&s.rhs_cfg, &ra);
            }
            if (bd) stamp(bd->pack_b_ms, tb);

            for (size_t m0 = 0; m0 < M; m0 += Mc) {
                const size_t mc = std::min(Mc, M - m0);

                const auto ta = Clk::now();
                size_t lhs_stride_m = 0;
                if (is2vl) {
                    kai_run_lhs_pack_f32p2vlx1_f32_sme(
                        mc, kc, mr, kr, sr, 0,
                        A + m0 * K + k0, K * sizeof(float), bufA.p);
                } else {
                    const kai_matmul_pack_lhs_uker_lhs_packed_dim_args ld{mc, kc};
                    const auto ls = s.lhs_api.get_lhs_packed_stride(&s.lhs_cfg, &ld);
                    lhs_stride_m = ls.m;
                    kai_matmul_pack_lhs_uker_args la{};
                    la.shape = {mc, kc};
                    la.operand.lhs = {A + m0 * K + k0, {K * sizeof(float)}};
                    la.operand.lhs_packed = {bufA.p, {ls.m}};
                    s.lhs_api.run(&s.lhs_cfg, &la);
                }
                if (bd) stamp(bd->pack_a_ms, ta);

                float* dst = k_blocked ? static_cast<float*>(scratch.p)
                                       : C + m0 * N + n0;
                const size_t dst_stride = k_blocked ? Nc * sizeof(float) : ldc;

                const auto tc = Clk::now();
                if (is2vl) {
                    kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(mc, nc, kc, bufA.p, bufB.p, dst,
                                      dst_stride, sizeof(float), -FLT_MAX, FLT_MAX);
                } else {
                    const kai_matmul_uker_rhs_stride_args rstr{rhs_stride_n};
                    const kai_matmul_uker_dst_stride_args dstr{dst_stride};
                    kai_matmul_uker_args a{};
                    a.flags = 0;
                    a.shape = {mc, nc, kc};
                    a.operand.dst = {dst, dstr};
                    a.operand.lhs = {bufA.p, {lhs_stride_m}};
                    a.operand.rhs = {bufB.p, rstr};
                    s.mm_api.run(&s.mm_cfg, &a);
                }
                if (bd) stamp(bd->compute_ms, tc);

                if (k_blocked) {
                    // Forced by the API: these kernels overwrite C and offer no
                    // beta=1, so a blocked K has to be summed by the caller.
                    const auto tacc = Clk::now();
                    const float* src = static_cast<const float*>(scratch.p);
                    float* out = C + m0 * N + n0;
                    if (k0 == 0) {
                        for (size_t r = 0; r < mc; ++r)
                            std::memcpy(out + r * N, src + r * Nc, nc * sizeof(float));
                    } else {
                        for (size_t r = 0; r < mc; ++r)
                            for (size_t c = 0; c < nc; ++c)
                                out[r * N + c] += src[r * Nc + c];
                    }
                    if (bd) stamp(bd->accumulate_ms, tacc);
                }
            }
        }
    }
}

// ---- Experiment 1: hot micro-kernel tile ------------------------------

struct HotTile::Impl {
    Kernel kind = Kernel::Sme2Mopa2VL;
    size_t Kc = 0, m_step = 0, n_step = 0;
    std::vector<uint8_t> lhs, rhs;
    std::vector<float> tile, bias;

    kai_matmul_uker_api           mm_api{};
    kai_matmul_pack_lhs_uker_api  lhs_api{};
    kai_matmul_pack_rhs_uker_api  rhs_api{};
    kai_matmul_uker_config           mm_cfg{};
    kai_matmul_pack_lhs_uker_config  lhs_cfg{};
    kai_matmul_pack_rhs_uker_config  rhs_cfg{};
    size_t lhs_stride = 0, rhs_stride = 0;
};

HotTile::HotTile() : impl_(std::make_unique<Impl>()) {}
HotTile::~HotTile() = default;
size_t HotTile::m_step() const { return impl_->m_step; }
size_t HotTile::n_step() const { return impl_->n_step; }
const float* HotTile::tile() const { return impl_->tile.data(); }
size_t HotTile::panel_bytes() const {
    return impl_->lhs.size() + impl_->rhs.size() + impl_->tile.size() * sizeof(float);
}

void HotTile::prepare(Kernel k, size_t Kc, const float* A_small, const float* B_small) {
    Impl& s = *impl_;
    s.kind = k;
    s.Kc = Kc;

    if (k == Kernel::Sme2Mopa2VL) {
        s.m_step = kai_get_m_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
        s.n_step = kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
        const size_t mr = kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(), nr = kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
        const size_t kr = kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(), sr = kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
        s.bias.assign(s.n_step, 0.0f);
        s.lhs.assign(kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme(s.m_step, Kc, mr, kr, sr), 0);
        s.rhs.assign(kai_get_rhs_packed_size_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(s.n_step, Kc), 0);
        s.tile.assign(s.m_step * s.n_step, 0.0f);
        kai_run_lhs_pack_f32p2vlx1_f32_sme(s.m_step, Kc, mr, kr, sr, 0,
                                           A_small, Kc * sizeof(float), s.lhs.data());
        kai_run_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(
            1, s.n_step, Kc, nr, kr, sr, s.n_step * sizeof(float),
            B_small, s.bias.data(), nullptr, s.rhs.data(), 0, nullptr);
    } else {
        s.mm_api  = kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa();
        s.lhs_api = kai_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme();
        s.rhs_api = kai_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme();
        const size_t vs = sme_vscale();
        s.lhs_cfg.format = {4 * vs, 1, 1};
        s.rhs_cfg.format = {4 * vs, 1, 1, 0};
        const auto step = s.mm_api.get_step(&s.mm_cfg);
        s.m_step = step.m;
        s.n_step = step.n;
        s.bias.assign(s.n_step, 0.0f);

        const kai_matmul_pack_lhs_uker_lhs_packed_dim_args ld{s.m_step, Kc};
        const auto ls = s.lhs_api.get_lhs_packed_stride(&s.lhs_cfg, &ld);
        s.lhs_stride = ls.m;
        s.lhs.assign(s.lhs_api.get_lhs_packed_size(&s.lhs_cfg, &ld, &ls), 0);

        const kai_matmul_pack_rhs_uker_rhs_packed_dim_args rd{s.n_step, Kc};
        const auto rs = s.rhs_api.get_rhs_packed_stride(&s.rhs_cfg, &rd);
        s.rhs_stride = rs.n;
        s.rhs.assign(s.rhs_api.get_rhs_packed_size(&s.rhs_cfg, &rd, &rs), 0);
        s.tile.assign(s.m_step * s.n_step, 0.0f);

        kai_matmul_pack_lhs_uker_args la{};
        la.shape = {s.m_step, Kc};
        la.operand.lhs = {A_small, {Kc * sizeof(float)}};
        la.operand.lhs_packed = {s.lhs.data(), {ls.m}};
        s.lhs_api.run(&s.lhs_cfg, &la);

        kai_matmul_pack_rhs_uker_args ra{};
        ra.shape = {s.n_step, Kc};
        ra.operand.rhs = {B_small, {sizeof(float), s.n_step * sizeof(float)}};
        ra.operand.rhs_packed = {s.rhs.data(), {rs.n}};
        ra.operand.bias_n = {s.bias.data()};
        s.rhs_api.run(&s.rhs_cfg, &ra);
    }
}

void HotTile::run_batch(size_t reps) const {
    const Impl& s = *impl_;
    float* tile = const_cast<float*>(s.tile.data());
    const size_t ldc = s.n_step * sizeof(float);
    if (s.kind == Kernel::Sme2Mopa2VL) {
        for (size_t r = 0; r < reps; ++r)
            kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(s.m_step, s.n_step, s.Kc, s.lhs.data(), s.rhs.data(),
                              tile, ldc, sizeof(float), -FLT_MAX, FLT_MAX);
    } else {
        const kai_matmul_uker_rhs_stride_args rstr{s.rhs_stride};
        const kai_matmul_uker_dst_stride_args dstr{ldc};
        for (size_t r = 0; r < reps; ++r) {
            kai_matmul_uker_args a{};
            a.flags = 0;
            a.shape = {s.m_step, s.n_step, s.Kc};
            a.operand.dst = {tile, dstr};
            a.operand.lhs = {s.lhs.data(), {s.lhs_stride}};
            a.operand.rhs = {s.rhs.data(), rstr};
            s.mm_api.run(&s.mm_cfg, &a);
        }
    }
}

}  // namespace KleidiAI

#else  // !MAXMULSK_HAVE_KLEIDIAI

namespace KleidiAI {

struct Gemm::Impl {};

const char* name(Kernel)  { return "kleidiai-not-built"; }
const char* label(Kernel) { return "n/a"; }
bool available()          { return false; }

Gemm::Gemm(Kernel) : impl_(nullptr) {}
Gemm::~Gemm() = default;
void Gemm::reshape(size_t, size_t, size_t) {}
void Gemm::pack(const float*, const float*) {}
void Gemm::matmul(const float*, float*) {}
void Gemm::run_end_to_end(const float*, const float*, float*) {}
void Gemm::set_n_block(size_t) {}
void Gemm::run_panel_blocked(const float*, const float*, float*, const PanelConfig&) {}
PanelBreakdown Gemm::run_panel_blocked_instrumented(const float*, const float*, float*,
                                                    const PanelConfig&) { return {}; }
bool Gemm::supports_panel_blocking() const { return false; }
void Gemm::panel_blocked_impl(const float*, const float*, float*,
                              const PanelConfig&, PanelBreakdown*) {}
struct HotTile::Impl {};
HotTile::HotTile() : impl_(nullptr) {}
HotTile::~HotTile() = default;
void HotTile::prepare(Kernel, size_t, const float*, const float*) {}
size_t HotTile::m_step() const { return 0; }
size_t HotTile::n_step() const { return 0; }
size_t HotTile::panel_bytes() const { return 0; }
void HotTile::run_batch(size_t) const {}
const float* HotTile::tile() const { return nullptr; }
size_t Gemm::n_block() const { return 0; }

}  // namespace KleidiAI

#endif  // MAXMULSK_HAVE_KLEIDIAI
