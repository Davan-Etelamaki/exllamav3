// sm_89 flash-style prefill attention for Inkling 2 global attention
// (infer_params.inkling_attn_flashprefill). Kernel body + plan + launcher, torch-free so
// tests/cpu/fa_prefill_shim/ can compile and RUN this same translation unit on a CPU box.
//
// WHICH LAYERS THIS SERVES. The served pack is 20-local / 4-global: local_layer_ids
// [0,1,2,3,4,6,7,8,9,10,12,13,14,15,16,18,19,20,21,22] with sliding_window_size 1024, and only
// layers [5, 11, 17, 23] carry unbounded full-context K/V (parent's pack read of the served
// config.json, 2026-11-27, recorded in agent-teams/sm89-kernel-lane-brief.md §BE). At S=63,387 the
// causal S^2/2 term puts ~86% of prefill attention FLOPs on those four layers, and the (h, rows, S)
// fp32 bias materialisation the shipped fuse eliminated lived in the same global path. This kernel is
// therefore wired to the GLOBAL attention path only: the 20 local layers attend a fixed 1024-key band
// and want a narrow banded path, not a general flash kernel, so a pass that carries a ring-validity
// mask declines by design (INKLING-FLASHDECLINE) and keeps the fused path it already uses.
//
// Geometry is inherited from this tree's own turing/fa75.cu (BM=64 query rows per block, 4 warps x
// 16 rows, XOR-swizzled tiles, ldmatrix fragments, the m16n8 fp32 accumulator doubling as the
// m16n8k8 A fragment), specialised for the served pack's attention — head_dim 128, as its
// config.json states — with head_dim 256 kept as a second instantiation because fa75's choreography
// was written for it and a 256-dim Inkling pack would share this kernel. Four changes are the lane:
//   * the RELATIVE-BIAS BAND folds INSIDE the key-tile loop: one fp32 gather per score element, and
//     only on tiles that can overlap a row's band (tile predicates below). The (h, rows, S) fp32
//     plane stays eliminated; nothing per-key reaches memory except the K/V tiles this kernel
//     streams through shared memory, which is the materialisation traffic the rate is lost to.
//   * the softmax is the SINK-SEEDED online form the shipped arms use: m = sink * log2e, l = 1,
//     O = 0, so the (S+1) sink concat stays gone and an all-masked row returns zeros, as
//     _attend_one_biasfuse and _attend_one_keytile do.
//   * P V accumulates in fp16 only across the FA_PV_FP16_KEYS/8 mma k-steps of ONE key tile (= one
//     FA_BN-wide tile, 32 keys) and is promoted to the fp32 O accumulators at the tile's end:
//     p = exp2(s - m) <= 1, so a tile partial is bounded by FA_PV_FP16_KEYS * max|v|, three orders
//     under fp16's 65504 for a normed V, and the un-normalised growth
//     (which overflowed at ~63k keys and poisoned every logit — 4cb7529a, dsh-keytile-merge) lives
//     only in fp32. The 2x fp16-accumulate HMMA rate is kept.
//   * K and V tiles stream through a cp.async pipeline (2 stages when the SM affords them, 1 when
//     it does not), so the global round trip overlaps the mma work instead of riding through
//     registers as it must on sm_75, where cp.async does not exist.
//
// Tile predicates, once per key tile, from the position extremes (block extremes on the query side,
// one full-warp reduction on the key side, so every thread in the block agrees and the barriers stay
// uniform and legal):
//   skip  — no row of this block sees any key of this tile: the tile's compute is elided
//   mask  — some (row, key) pair breaks causality or lies past S: pay the per-element test
//   bias  — some (row, key) pair has 0 <= dist < rel_extent: pay the per-element gather
// On the served band (rel_extent 1024) over a 63,387-key cache, ceil(1024/32) = 32 of
// ceil(63387/32) = 1981 tiles take the gather (~1.6%), and the mask test is paid on the diagonal
// tiles only — the bias band costs a fraction of a tile's mma work instead of a 12.7 GiB plane.
//
// Refusals never crash: fa_prefill_plan declines, inkling_fa_prefill_launch reports, and the Python
// seam falls back to the fused path behind an INKLING-FLASHDECLINE line.
//
// Interface notes the next lane needs (parent's pack read and census, 2026-11-27):
//   * L2: inkling_attn_l2win is ON in the served config and carves 49.5 MiB of the 4090's 72 MiB of
//     L2 for the decode row span, so this kernel plans on ~22.5 MiB of ordinary L2 and takes nothing
//     on faith from the rest: K and V stream global→shared with cp.async, and correctness does not
//     change when a reuse misses. Re-read traffic is grid_x × grid_y tiles per pass — halving it is a
//     scheduling question (larger BM, or a kv-head-major block), not a correctness one.
//   * launches: exactly ONE cudaLaunchKernel per served pass (grid (ceil(T/64), h)). The only other
//     per-call CUDA work is the caller's four-byte non-finite counter; a fused DECODE launch that
//     must not add kernels should own that counter and have the kernel reset it after reading.
//   * fa_prims.cuh is instruction-only: rows and sequence length travel in FaPrefillArgs, no chunk
//     size or block count is baked in, so the decode lane can reuse the same primitives verbatim.

#if !defined(USE_ROCM)

#include "fa_prefill.cuh"
#include "fa_prims.cuh"

#include <cuda_runtime.h>

#include <cmath>

#if defined(__CUDACC__)
#define FA_FN __device__ __forceinline__
#else
#define FA_FN inline
#endif

namespace
{
constexpr int BM = FA_BM;
constexpr int BN = FA_BN;
constexpr int NWARPS = FA_NWARPS;
constexpr int NTHREADS = NWARPS * 32;
constexpr int NT = BN / 8;                       // 8-key score tiles per key tile
constexpr int NKG = BN / 16;                     // 16-key groups per key tile (one ldmatrix x4 each)

// Swizzled 16-byte chunk inside a region of `chunks` chunks per row: XOR by row&7 puts the 8 rows one
// ldmatrix group reads on distinct banks (the helper shape fa75 uses for these dimensions).
FA_FN uint32_t swz(int r, int c, int chunks)
{
    return uint32_t((r * chunks + (c ^ (r & 7))) * 16);
}
}  // namespace

#if defined(__CUDACC__)
#define FA_SMEM_BASE(name) \
    extern __shared__ __align__(128) uint8_t name##_dyn[]; \
    uint8_t* name = name##_dyn
#else
uint8_t* em_smem_base();
#define FA_SMEM_BASE(name) uint8_t* name = em_smem_base()
#endif

template<int D>
__global__ void __launch_bounds__(NTHREADS)
inkling_fa_prefill_kernel(const FaPrefillArgs A)
{
    constexpr int ROW_CHUNKS = D / 8;                  // 16-byte chunks per tile row
    constexpr int QUP_CHUNKS = D / 16;                 // chunks per row of the resident Q upper half
    constexpr size_t TILE_BYTES = size_t(BN) * D * 2;  // one K tile, or one V tile
    constexpr size_t STAGE_BYTES = 2 * TILE_BYTES;     // K tile + V tile
    constexpr size_t QUP_BYTES = size_t(D / 2) * BM * 2;

    FA_SMEM_BASE(sm);
    const uint32_t sbase = fa_smem_u32(sm);
    const int qh_base = A.stages * int(STAGE_BYTES);
    const int scr_base = qh_base + int(QUP_BYTES);

    const int t_id = int(threadIdx.x);
    const int warp = t_id >> 5;
    const int lane = t_id & 31;
    const int gq = lane >> 2;              // this thread's rows in its warp's 16: gq, gq+8
    const int cq = lane & 3;               // its columns in an 8-wide group: 2*cq, 2*cq+1
    const int lr = lane & 7;               // ldmatrix: row within an 8x8 matrix
    const int lm = lane >> 3;              // ldmatrix: which of the four matrices this lane points at

    const int m0 = int(blockIdx.x) * BM;
    const int hq = int(blockIdx.y);
    const int hk = hq / A.grp;                          // GQA broadcast: kvh divided out at plan time

    const half* q_ = reinterpret_cast<const half*>(A.q) + int64_t(hq) * A.q_head_stride;
    const half* k_ = reinterpret_cast<const half*>(A.k) + int64_t(hk) * A.k_head_stride;
    const half* v_ = reinterpret_cast<const half*>(A.v) + int64_t(hk) * A.k_head_stride;
    const float* rel_ = reinterpret_cast<const float*>(A.rel) + int64_t(hq) * A.rel_head_stride;
    float* scr = reinterpret_cast<float*>(sm + scr_base);
    const int n_tiles = (A.S + BN - 1) / BN;

    // ---- Q: the lower half (dims 0..D/2) stages through buffer 0 and becomes this warp's A
    // fragments; the upper half lives in shared memory for the whole kernel and is read per key tile.
    // The split is fa75's: all D dims of A fragments plus a D-wide fp32 O accumulator do not both fit,
    // and the fp32 O is the one the numerics depend on.
    #pragma unroll
    for (int j = 0; j < (BM * ROW_CHUNKS) / NTHREADS; ++j)
    {
        const int idx = t_id + j * NTHREADS;
        const int r = idx / ROW_CHUNKS;
        const int c = idx % ROW_CHUNKS;
        const int row = m0 + r;
        const bool live = row < A.T;
        const half* src = q_ + int64_t(live ? row : A.T - 1) * A.q_row_stride + c * 8;
        if (c < QUP_CHUNKS) fa_cp_async16(sm + swz(r, c, QUP_CHUNKS), src, live);
        else fa_cp_async16(sm + qh_base + swz(r, c - QUP_CHUNKS, QUP_CHUNKS), src, live);
    }
    fa_cp_async_commit();
    fa_cp_async_wait_all();
    __syncthreads();

    uint32_t qf[D / 16][2];
    #pragma unroll
    for (int kk = 0; kk < D / 16; kk += 2)
    {
        uint32_t r4[4];
        fa_ldsm_x4(r4, sbase + swz(16 * warp + lr + (lm & 1) * 8, kk + (lm >> 1), QUP_CHUNKS));
        qf[kk][0] = r4[0]; qf[kk][1] = r4[1];
        qf[kk + 1][0] = r4[2]; qf[kk + 1][1] = r4[3];
    }

    // ---- this warp's query-position extremes, then the block's: the block extremes drive the tile
    // predicates, so every warp takes the same branches and the barriers below stay uniform.
    // Positions are exactly representable in fp32 up to 2^24, above the 262,144-token window.
    float qmin = 1e30f, qmax = -1e30f;
    {
        const int row = m0 + warp * 16 + (lane < 16 ? lane : 0);
        const float pv = float(row < A.T ? A.qpos[row] : 0);
        qmin = lane < 16 ? pv : 1e30f;
        qmax = lane < 16 ? pv : -1e30f;
        #pragma unroll
        for (int m = 1; m < 16; m <<= 1)
        {
            qmin = fminf(qmin, fa_shfl_xor(qmin, m));
            qmax = fmaxf(qmax, fa_shfl_xor(qmax, m));
        }
        if (lane == 0) { scr[warp * 2] = qmin; scr[warp * 2 + 1] = qmax; }
    }
    __syncthreads();                       // extremes published; buffer 0 free for K/V
    qmin = scr[0]; qmax = scr[1];
    #pragma unroll
    for (int w = 1; w < NWARPS; ++w)
    {
        qmin = fminf(qmin, scr[w * 2]);
        qmax = fmaxf(qmax, scr[w * 2 + 1]);
    }

    // ---- prime the pipeline: key tile 0 stages into buffer 0, the region the Q lower half vacated
    // once its A fragments sat in registers (its bytes sit below qh_base, so the resident upper half
    // survives). Without this prime the loop's first iteration would read Q's bytes as K and V, and
    // the prefetch below only ever covers tile kt+1.
    if (n_tiles > 0)
    {
        #pragma unroll
        for (int j = 0; j < (BN * ROW_CHUNKS) / NTHREADS; ++j)
        {
            const int idx = t_id + j * NTHREADS;
            const int r = idx / ROW_CHUNKS;
            const int c = idx % ROW_CHUNKS;
            const bool live = r < A.S;
            const int krow = live ? r : A.S - 1;
            fa_cp_async16(sm + swz(r, c, ROW_CHUNKS),
                          k_ + int64_t(krow) * A.k_row_stride + c * 8, live);
            fa_cp_async16(sm + TILE_BYTES + swz(r, c, ROW_CHUNKS),
                          v_ + int64_t(krow) * A.k_row_stride + c * 8, live);
        }
        fa_cp_async_commit();
    }

    float oacc[D / 8][4];
    #pragma unroll
    for (int dn = 0; dn < D / 8; ++dn)
        #pragma unroll
        for (int i = 0; i < 4; ++i) oacc[dn][i] = 0.0f;

    // ---- sink-seeded online softmax, exp2 domain: m = sink * log2e, l = 1 (the sink's own
    // denominator term), O = 0 — _attend_one_biasfuse's initialisation, another summation order.
    // The seed goes into ONE lane of each row-group, not into all four: the normalise below sums l
    // across the four lanes that share a row (columns 2*cq of an 8-wide score tile), so seeding every
    // lane would count the sink four times and shrink every output by (Σe+e_sink)/(Σe+4·e_sink).
    float mrow[2], lrow[2];
    #pragma unroll
    for (int hh = 0; hh < 2; ++hh)
    {
        const int row = m0 + warp * 16 + gq + hh * 8;
        mrow[hh] = row < A.T ? float(A.sink[int64_t(row) * A.sink_row_stride + hq]) * FA_LOG2E
                             : -1e30f;
        lrow[hh] = (cq == 0) ? 1.0f : 0.0f;
    }

    const bool early_prefetch = A.stages >= 2;

    for (int kt = 0; kt < n_tiles; ++kt)
    {
        const int n0 = kt * BN;
        const int buf = kt % A.stages;
        fa_cp_async_wait_all();
        __syncthreads();                                  // (A) tile kt resident

        // ---- tile predicates: one full-warp reduction over this tile's key positions. Lanes past BN
        // and keys past S are neutral for the minimum and push the maximum toward "mask", so a
        // partial tile always takes the per-element path.
        bool skip_tile = false, need_mask = false, need_bias = false;
        {
            const int key = n0 + lane;
            const bool in = (lane < BN) && (key < A.S);
            const float kv = in ? float(A.kpos[key]) : 0.0f;
            float kmin = in ? kv : 1e30f;
            float kmax = in ? kv : -1e30f;
            #pragma unroll
            for (int m = 1; m < 32; m <<= 1)
            {
                kmin = fminf(kmin, fa_shfl_xor(kmin, m));
                kmax = fmaxf(kmax, fa_shfl_xor(kmax, m));
            }
            skip_tile = (qmax < kmin) || (n0 >= A.S);
            need_mask = (qmin < kmax) || (n0 + BN > A.S);
            need_bias = (qmin - kmax < float(A.rel_extent));
        }

        if (early_prefetch && kt + 1 < n_tiles)
        {
            const int nb = (kt + 1) % A.stages;
            uint8_t* dst = sm + size_t(nb) * STAGE_BYTES;
            #pragma unroll
            for (int j = 0; j < (BN * ROW_CHUNKS) / NTHREADS; ++j)
            {
                const int idx = t_id + j * NTHREADS;
                const int r = idx / ROW_CHUNKS;
                const int c = idx % ROW_CHUNKS;
                const int key = n0 + BN + r;
                const bool live = key < A.S;
                const int krow = live ? key : A.S - 1;
                fa_cp_async16(dst + swz(r, c, ROW_CHUNKS),
                              k_ + int64_t(krow) * A.k_row_stride + c * 8, live);
                fa_cp_async16(dst + TILE_BYTES + swz(r, c, ROW_CHUNKS),
                              v_ + int64_t(krow) * A.k_row_stride + c * 8, live);
            }
            fa_cp_async_commit();
        }

        const uint32_t sK_u = sbase + uint32_t(buf) * uint32_t(STAGE_BYTES);
        const uint32_t sV_u = sK_u + uint32_t(TILE_BYTES);
        uint8_t* sK = sm + size_t(buf) * STAGE_BYTES;
        uint8_t* sV = sK + TILE_BYTES;

        float s[NT][4];
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) s[nt][i] = 0.0f;

        if (!skip_tile)
        {
            // ---- S = Q K^T, fp32 accumulate: 16 head dims per iteration (two ldmatrix chunks),
            // NKG groups of 16 keys each, two 8-key score tiles per group.
            #pragma unroll
            for (int kk = 0; kk < ROW_CHUNKS; kk += 2)
            {
                uint32_t a[4];
                if (kk < D / 16)
                {
                    a[0] = qf[kk][0]; a[1] = qf[kk][1];
                    a[2] = qf[kk + 1][0]; a[3] = qf[kk + 1][1];
                }
                else
                {
                    fa_ldsm_x4(a, sbase + uint32_t(qh_base) +
                               swz(16 * warp + lr + (lm & 1) * 8,
                                   (kk - D / 16) + (lm >> 1), QUP_CHUNKS));
                }
                #pragma unroll
                for (int kg = 0; kg < NKG; ++kg)
                {
                    uint32_t b[4];
                    fa_ldsm_x4(b, sK_u + swz(16 * kg + 8 * (lm & 1) + lr, kk + (lm >> 1),
                                             ROW_CHUNKS));
                    fa_mma_1688_f32(s[2 * kg + 0], a[0], a[1], b[0]);
                    fa_mma_1688_f32(s[2 * kg + 1], a[0], a[1], b[1]);
                    fa_mma_1688_f32(s[2 * kg + 0], a[2], a[3], b[2]);
                    fa_mma_1688_f32(s[2 * kg + 1], a[2], a[3], b[3]);
                }
            }

            // ---- relative bias + causal mask + online softmax, in place on the score elements this
            // thread owns: rows gq and gq+8, columns 2*cq and 2*cq+1 of every 8-key score tile.
            #pragma unroll
            for (int hh = 0; hh < 2; ++hh)
            {
                const int row_abs = m0 + warp * 16 + gq + hh * 8;
                float mx = -1e30f;
                #pragma unroll
                for (int nt = 0; nt < NT; ++nt)
                    #pragma unroll
                    for (int e = 0; e < 2; ++e)
                    {
                        float x = s[nt][2 * hh + e] * A.scale_log2;
                        if (need_mask || need_bias)
                        {
                            const int key_abs = n0 + 8 * nt + 2 * cq + e;
                            const int64_t qp = row_abs < A.T ? A.qpos[row_abs]
                                                             : -(int64_t(1) << 50);
                            if (key_abs >= A.S) x = -1e30f;
                            else
                            {
                                const int64_t dist = qp - A.kpos[key_abs];
                                if (dist < 0) x = -1e30f;
                                else if (need_bias && dist < A.rel_extent)
                                    x += rel_[int64_t(row_abs) * A.rel_row_stride + dist] * A.log2e;
                            }
                        }
                        s[nt][2 * hh + e] = x;
                        mx = fmaxf(mx, x);
                    }
                mx = fmaxf(mx, fa_shfl_xor(mx, 1));
                mx = fmaxf(mx, fa_shfl_xor(mx, 2));
                const float m_new = fmaxf(mrow[hh], mx);
                const float alpha = exp2f(mrow[hh] - m_new);
                mrow[hh] = m_new;
                float sum = 0.0f;
                #pragma unroll
                for (int nt = 0; nt < NT; ++nt)
                    #pragma unroll
                    for (int e = 0; e < 2; ++e)
                    {
                        const float p = exp2f(s[nt][2 * hh + e] - m_new);
                        s[nt][2 * hh + e] = p;
                        sum += p;
                    }
                lrow[hh] = lrow[hh] * alpha + sum;
                // Unconditional rescale: a branch on alpha == 1 costs more than the multiplies (fa75's
                // call), which is why the sink-seeded first tile still pays one multiply by 1.0.
                #pragma unroll
                for (int dn = 0; dn < D / 8; ++dn)
                {
                    oacc[dn][2 * hh] *= alpha;
                    oacc[dn][2 * hh + 1] *= alpha;
                }
            }

            // ---- O += P V. P comes straight out of the S accumulator (an m16n8 fp32 C fragment,
            // repacked pairwise, IS an m16n8k8 A fragment); V is read transposed. The
            // FA_PV_FP16_KEYS/8 k-steps of one tile accumulate in fp16 and are promoted to the fp32
            // O at the tile's end — that promotion point is the bound on the un-normalised partial
            // (FA_PV_FP16_KEYS * max|v|), and the static_assert in fa_prefill.cuh pins the window to
            // one key tile.
            uint32_t pa[FA_PV_FP16_KEYS / 8][2];
            #pragma unroll
            for (int j = 0; j < FA_PV_FP16_KEYS / 8; ++j)
            {
                pa[j][0] = fa_pack_h2(s[j][0], s[j][1]);
                pa[j][1] = fa_pack_h2(s[j][2], s[j][3]);
            }
            #pragma unroll
            for (int dn = 0; dn < D / 8; dn += 4)
            {
                uint32_t t[4][2];
                #pragma unroll
                for (int x = 0; x < 4; ++x) { t[x][0] = 0u; t[x][1] = 0u; }
                #pragma unroll
                for (int j = 0; j < FA_PV_FP16_KEYS / 8; ++j)
                {
                    uint32_t b[4];
                    fa_ldsm_x4_t(b, sV_u + swz(8 * j + lr, dn + lm, ROW_CHUNKS));
                    #pragma unroll
                    for (int x = 0; x < 4; ++x) fa_mma_1688_f16(t[x], pa[j][0], pa[j][1], b[x]);
                }
                #pragma unroll
                for (int x = 0; x < 4; ++x)
                {
                    const float2 lo = fa_h2_to_f2(t[x][0]);
                    const float2 hi = fa_h2_to_f2(t[x][1]);
                    oacc[dn + x][0] += lo.x; oacc[dn + x][1] += lo.y;
                    oacc[dn + x][2] += hi.x; oacc[dn + x][3] += hi.y;
                }
            }
        }

        __syncthreads();                                  // (B) buffer buf fully consumed
        if (!early_prefetch && kt + 1 < n_tiles)
        {
            #pragma unroll
            for (int j = 0; j < (BN * ROW_CHUNKS) / NTHREADS; ++j)
            {
                const int idx = t_id + j * NTHREADS;
                const int r = idx / ROW_CHUNKS;
                const int c = idx % ROW_CHUNKS;
                const int key = n0 + BN + r;
                const bool live = key < A.S;
                const int krow = live ? key : A.S - 1;
                fa_cp_async16(sK + swz(r, c, ROW_CHUNKS),
                              k_ + int64_t(krow) * A.k_row_stride + c * 8, live);
                fa_cp_async16(sV + swz(r, c, ROW_CHUNKS),
                              v_ + int64_t(krow) * A.k_row_stride + c * 8, live);
            }
            fa_cp_async_commit();
        }
    }

    // ---- normalize, store (h, T, d) fp16, and count non-finite outputs for the caller
    int bad = 0;
    #pragma unroll
    for (int hh = 0; hh < 2; ++hh)
    {
        float l = lrow[hh];
        l += fa_shfl_xor(l, 1);
        l += fa_shfl_xor(l, 2);
        const int row_abs = m0 + warp * 16 + gq + hh * 8;
        if (row_abs >= A.T) continue;
        half* o_ = reinterpret_cast<half*>(A.o) + int64_t(hq) * A.o_head_stride +
                   int64_t(row_abs) * A.o_row_stride;
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        #pragma unroll
        for (int dn = 0; dn < D / 8; ++dn)
        {
            const float x0 = oacc[dn][2 * hh] * inv;
            const float x1 = oacc[dn][2 * hh + 1] * inv;
            if (!fa_is_finite(x0)) ++bad;
            if (!fa_is_finite(x1)) ++bad;
            *reinterpret_cast<uint32_t*>(o_ + 8 * dn + 2 * cq) = fa_pack_h2(x0, x1);
        }
    }
    if (bad && A.nfin)
    {
#if defined(__CUDACC__)
        atomicAdd(A.nfin, int32_t(bad));
#else
        em_atomic_add_i32(A.nfin, int32_t(bad));
#endif
    }
}

// ---------------------------------------------------------------------------
// Device facts, plan, launch
// ---------------------------------------------------------------------------

struct FaDevFacts
{
    size_t smem_optin;         // dynamic shared memory one block may opt into
    size_t smem_per_sm;        // shared memory an SM holds
    int sm_major, sm_minor;
    int ok;
    char why[96];
};

static FaDevFacts fa_dev_facts(int device)
{
    static FaDevFacts cache[16];
    static unsigned char have[16];
    if (device >= 0 && device < 16 && have[device]) return cache[device];

    FaDevFacts f{};
    int major = 0, minor = 0, optin = 0, per_sm = 0, per_block = 0;
    cudaError_t e = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    if (e == cudaSuccess)
        e = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    if (e == cudaSuccess)
        e = cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (e == cudaSuccess)
        e = cudaDeviceGetAttribute(&per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
    if (e == cudaSuccess)
        e = cudaDeviceGetAttribute(&per_block, cudaDevAttrMaxSharedMemoryPerBlock, device);
    if (e != cudaSuccess)
    {
        f.ok = 0;
        fa_detail::set_reason(f.why, sizeof(f.why), "device attributes unavailable");
        (void) cudaGetLastError();          // a failed query must not poison the calling stream
        return f;
    }
    f.sm_major = major;
    f.sm_minor = minor;
    f.smem_optin = size_t(optin > 0 ? optin : per_block);
    f.smem_per_sm = size_t(per_sm > 0 ? per_sm : per_block);
    f.ok = 1;
    if (device >= 0 && device < 16) { cache[device] = f; have[device] = 1; }
    return f;
}

static char g_refusal[160] = "";

static void fa_decline(FaPrefillLaunchReport& rep, const char* reason)
{
    rep.launched = 0;
    fa_detail::set_reason(rep.reason, sizeof(rep.reason), reason);
    fa_detail::set_reason(g_refusal, sizeof(g_refusal), reason);
}

const char* inkling_fa_prefill_refusal() { return g_refusal; }

FaPrefillPlan inkling_fa_prefill_plan(const FaPrefillShapes& s, int T, int S,
                                      size_t smem_budget_bytes, int device)
{
    const FaDevFacts f = fa_dev_facts(device);
    if (!f.ok)
    {
        FaPrefillPlan p{};
        fa_detail::set_reason(p.reason, sizeof(p.reason), f.why);
        return p;
    }
    return fa_prefill_plan(s, T, S, smem_budget_bytes, f.smem_optin, f.smem_per_sm,
                           f.sm_major, f.sm_minor);
}

FaPrefillLaunchReport inkling_fa_prefill_launch(const FaPrefillArgs& a, size_t smem_budget_bytes)
{
    FaPrefillLaunchReport rep{};
    rep.launched = 0;
    g_refusal[0] = '\0';

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
    {
        (void) cudaGetLastError();
        fa_decline(rep, "no current CUDA device");
        return rep;
    }
    const FaPrefillShapes s{a.h, a.kvh, a.d, a.rel_extent};
    const FaPrefillPlan p = inkling_fa_prefill_plan(s, a.T, a.S, smem_budget_bytes, device);
    rep.stages = p.stages;
    rep.threads = p.threads;
    rep.smem_bytes = p.smem_bytes;
    if (!p.ok)
    {
        fa_decline(rep, p.reason);
        return rep;
    }
    rep.grid_x = p.grid_x;
    rep.grid_y = p.grid_y;
    rep.blocks = int64_t(p.grid_x) * int64_t(p.grid_y);

    // Dynamic shared memory above the 48 KiB default needs the kernel's opt-in, and the request
    // travels with the launch. A driver that refuses it is a decline, never a failed launch. The
    // opt-in is tracked per instantiation, because each head dim is its own kernel.
    static size_t opted_bytes[2] = {0, 0};        // [0] head_dim 128, [1] head_dim 256
    const bool wide = a.d == 256;
    const int sel = wide ? 1 : 0;
    const void* kern = wide ? (const void*) inkling_fa_prefill_kernel<256>
                            : (const void*) inkling_fa_prefill_kernel<128>;
    if (p.smem_bytes > opted_bytes[sel])
    {
        const cudaError_t e = cudaFuncSetAttribute(
            kern, cudaFuncAttributeMaxDynamicSharedMemorySize, int(p.smem_bytes));
        if (e != cudaSuccess)
        {
            (void) cudaGetLastError();
            fa_decline(rep, "driver refused the dynamic shared-memory opt-in");
            return rep;
        }
        opted_bytes[sel] = p.smem_bytes;
    }

    FaPrefillArgs args = a;
    args.stages = p.stages;
    args.grp = p.grp;

#if defined(__CUDACC__)
    if (wide)
        inkling_fa_prefill_kernel<256><<<dim3(p.grid_x, p.grid_y), p.threads,
                                         int(p.smem_bytes), (cudaStream_t) args.stream>>>(args);
    else
        inkling_fa_prefill_kernel<128><<<dim3(p.grid_x, p.grid_y), p.threads,
                                         int(p.smem_bytes), (cudaStream_t) args.stream>>>(args);
    const cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess)
    {
        (void) cudaGetLastError();
        fa_decline(rep, "kernel launch rejected by the driver");
        return rep;
    }
#else
    em_launch_kernel(wide ? inkling_fa_prefill_kernel<256> : inkling_fa_prefill_kernel<128>,
                     &args, p.grid_x, p.grid_y, p.threads, p.smem_bytes);
    (void) kern;
#endif
    rep.launched = 1;
    return rep;
}

#endif  // USE_ROCM