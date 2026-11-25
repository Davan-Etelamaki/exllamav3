// Phase-1 grouped expert tier for Inkling-2 decode: one persistent launch per leg, all expert
// selection resolved ON DEVICE (moe_route.cu), weights reached through the load-time pointer
// table (KiB-scale; the 128x3 independent per-expert-qgroup Linears of the served architecture,
// inkling2_loop.py served __init__ — never a stacked copy, VRAM can't pay one).
//
// Leg GU (template GU=true):  gate+up for a 16-entry segment block share one A tile; fused
// swiglu-softcap epilogue writes act_scratch in fp16 — bit-for-bit the same rounding point as
// the served `act.to(torch.half)` before down_proj (served epilogue's rounding line).
// Leg D  (template GU=false): down_proj over act_scratch; epilogue multiplies each row by its
// routing weight and atomically accumulates into the fp32 out (the served index_add_ becomes
// atomicAdd — accumulation order intentionally free, fence 1 prices the reassociation).
//
// sm_89 form: cp.async double-buffered stages, ldmatrix + mma.sync.aligned.m16n8k16; no wgmma,
// no TMA. Bandwidth-bound at M <= 8*top_k: the win is launch/sync removal, measured in the
// dot26 bench-lane window (fence 3), never claimed from launch counts (law 15).
//
// Wire format: the int4+per-(row,qgroup)-scale unpack in moe_sm89_common.cuh::moe_unpack_int4x8
// is the PLACEHOLDER for the exl3 trellis; the integration step substitutes that one function
// and the qbytes/scales fields (see README.md). Restriction carried by the placeholder:
// group_size % MOE_BLOCK_K == 0 (a table-build check will enforce it; draft trusts it).
#include "moe_sm89_common.cuh"

#include <cstdio>

template <bool GU>
__global__ void __launch_bounds__(MOE_THREADS)
moe_group_gemv_kernel(const MoERouting r, const MoEPtrTable* table, const half* __restrict__ x) {
    constexpr int K_DIM = GU ? MOE_HIDDEN : MOE_INTER;
    constexpr int N_DIM = GU ? MOE_INTER : MOE_HIDDEN;
    constexpr int PROJ  = GU ? 2 : 1;
    static_assert(K_DIM % MOE_BLOCK_K == 0 && N_DIM % MOE_BLOCK_N == 0, "tile must divide geometry");
    const int KS = K_DIM / MOE_BLOCK_K;

    __shared__ half    xs[2][MOE_BLOCK_M][MOE_BLOCK_K + MOE_SHARED_PAD];
    __shared__ half    ws[2][PROJ][MOE_BLOCK_N][MOE_BLOCK_K + MOE_SHARED_PAD];
    __shared__ uint8_t raw[2][PROJ][MOE_BLOCK_N][MOE_BLOCK_K / 2];
    __shared__ half    ss[2][PROJ][MOE_BLOCK_N];
    __shared__ int32_t s_entry[MOE_BLOCK_M];   // entry id (token*TOP_K+i) or MOE_INVALID_ENTRY
    __shared__ int32_t s_src[MOE_BLOCK_M];     // GU: token row into x; D: entry row into act_scratch

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int num_blocks = *r.num_blocks_out;   // device-resident: no host readback, no sync
    const int tiles_n    = N_DIM / MOE_BLOCK_N;
    const long total_work = (long)num_blocks * tiles_n;

    for (long work = blockIdx.x; work < total_work; work += gridDim.x) {
        const int blk   = (int)(work / tiles_n);
        const int ntile = (int)(work % tiles_n);
        const int e     = r.expert_for_block[blk];          // on-device branch (vLLM expert_ids shape)
        const int n0    = ntile * MOE_BLOCK_N;

        const MoEWeightRef* wr[PROJ];
        wr[0] = GU ? &table->gate[e] : &table->down[e];
        if (GU) wr[1] = &table->up[e];

        if (tid < MOE_BLOCK_M) {
            const int s = r.sorted_token_ids[blk * MOE_BLOCK_M + tid];
            s_entry[tid] = s;
            const int entry = (s != MOE_INVALID_ENTRY) ? s : 0;
            s_src[tid] = GU ? (entry / MOE_TOP_K) : entry;
        }
        __syncthreads();

        float acc[PROJ][2][4];
        #pragma unroll
        for (int p = 0; p < PROJ; ++p)
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt)
                #pragma unroll
                for (int i = 0; i < 4; ++i) acc[p][nt][i] = 0.0f;

        // ---------- stage: A rows + packed weight rows (16B/thread) + row scales, cp.async ----------
        auto stage = [&](int ks, int buf) {
            if (tid < 64) {                                // A tile: 16 rows x 64B = 64 pieces
                const int row  = tid >> 2;
                const int part = (tid & 3) * 8;            // halfs (16B)
                const half* src = GU
                    ? x + (size_t)s_src[row] * MOE_HIDDEN + (size_t)ks * MOE_BLOCK_K + part
                    : r.act_scratch + (size_t)s_src[row] * MOE_INTER + (size_t)ks * MOE_BLOCK_K + part;
                moe_cp_async16_zfill(&xs[buf][row][part], src, s_entry[row] != MOE_INVALID_ENTRY);
            }
            if (tid < 64) {                                // one BLOCK_K nibble-row per thread = 16B
                #pragma unroll
                for (int p = 0; p < PROJ; ++p) {
                    const int n = n0 + tid;
                    const uint8_t* src = wr[p]->qbytes + (size_t)n * (K_DIM / 2)
                                         + (size_t)ks * (MOE_BLOCK_K / 2);
                    moe_cp_async16(&raw[buf][p][tid][0], src);
                    ss[buf][p][tid] = wr[p]->scales[(size_t)n * (K_DIM / wr[p]->group_size)
                                                    + ((size_t)ks * MOE_BLOCK_K) / wr[p]->group_size];
                }
            }
        };

        // ---------- unpack: int4 nibbles + row scale -> [N][K] half tile (mma .col layout) --------
        auto unpack = [&](int buf) {
            constexpr int GROUPS = (PROJ * MOE_BLOCK_N * (MOE_BLOCK_K / 8)) / MOE_THREADS;
            #pragma unroll
            for (int i = 0; i < GROUPS; ++i) {
                const int gid = i * MOE_THREADS + tid;
                const int p   = gid / (MOE_BLOCK_N * (MOE_BLOCK_K / 8));
                const int rem = gid % (MOE_BLOCK_N * (MOE_BLOCK_K / 8));
                const int n   = rem / (MOE_BLOCK_K / 8);
                const int kb  = (rem % (MOE_BLOCK_K / 8)) * 8;
                uint8_t pk[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) pk[j] = raw[buf][p][n][(kb >> 1) + j];
                half o[8];
                moe_unpack_int4x8(pk, o, ss[buf][p][n]);
                uint4 v;
                unsigned* pv = reinterpret_cast<unsigned*>(&v);
                #pragma unroll
                for (int m = 0; m < 4; ++m) {
                    const half2 hpair = __halves2half2(o[2 * m], o[2 * m + 1]);
                    pv[m] = *reinterpret_cast<const unsigned*>(&hpair);
                }
                *reinterpret_cast<uint4*>(&ws[buf][p][n][kb]) = v;
            }
        };

        stage(0, 0);
        moe_cp_async_commit();

        for (int ks = 0; ks < KS; ++ks) {
            const int buf = ks & 1;
            if (ks + 1 < KS) stage(ks + 1, buf ^ 1);
            moe_cp_async_commit();
            moe_cp_async_wait<1>();                        // stage ks resident (incl. zero-filled rows)
            unpack(buf);
            __syncthreads();

            #pragma unroll
            for (int kk = 0; kk < MOE_BLOCK_K; kk += 16) {
                unsigned a0, a1, a2, a3;
                moe_ldmatrix_x4(a0, a1, a2, a3, &xs[buf][lane & 15][kk + ((lane >> 4) << 3)]);
                #pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const int nb = warp * 16 + nt * 8;
                    #pragma unroll
                    for (int p = 0; p < PROJ; ++p) {
                        unsigned b0, b1;
                        moe_ldmatrix_x2(b0, b1,
                            &ws[buf][p][nb + (lane & 7)][kk + (((lane >> 3) & 1) << 3)]);
                        moe_mma_16x8x16(acc[p][nt][0], acc[p][nt][1], acc[p][nt][2], acc[p][nt][3],
                                        a0, a1, a2, a3, b0, b1);
                    }
                }
            }
            __syncthreads();                               // buffers safe to restage
        }

        // ------------------------------------ epilogues ------------------------------------------
        const int cloc = warp * 16 + (lane & 3) * 2;       // col within BLOCK_N (adjacent pair)
        if (GU) {
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt)
                #pragma unroll
                for (int mh = 0; mh < 2; ++mh) {
                    const int row = (lane >> 2) + mh * 8;
                    if (s_entry[row] != MOE_INVALID_ENTRY) {
                        const int col = n0 + cloc + nt * 8;
                        const half2 v = __halves2half2(
                            moe_swiglu_softcap(acc[0][nt][2 * mh],     acc[1][nt][2 * mh]),
                            moe_swiglu_softcap(acc[0][nt][2 * mh + 1], acc[1][nt][2 * mh + 1]));
                        *reinterpret_cast<half2*>(&r.act_scratch[(size_t)s_entry[row] * MOE_INTER + col]) = v;
                    }
                }
        } else {
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt)
                #pragma unroll
                for (int mh = 0; mh < 2; ++mh) {
                    const int row = (lane >> 2) + mh * 8;
                    const int s = s_entry[row];
                    if (s != MOE_INVALID_ENTRY) {
                        const int token = s / MOE_TOP_K;
                        const float ew = r.w[s];
                        const int col = n0 + cloc + nt * 8;
                        atomicAdd(&r.out[(size_t)token * MOE_HIDDEN + col],     acc[0][nt][2 * mh] * ew);
                        atomicAdd(&r.out[(size_t)token * MOE_HIDDEN + col + 1], acc[0][nt][2 * mh + 1] * ew);
                    }
                }
        }
        __syncthreads();                                   // shared reuse across work items
    }
}

// Launches both legs. Preconditions (checked by the python wiring at integration): DECODE-ONLY
// batch (bsz <= MOE_MAX_DECODE_TOKENS, moe_sm89_common.cuh — act_scratch is [bsz*TOP_K, MOE_INTER]
// fp16, 144 KiB at bsz 1, 4.6 MiB at the cap, absurd for prefill which keeps the fused path);
// routing arena at FIXED addresses sized by moe_max_padded_rows/moe_max_blocks (allocated once per
// graph-capture set by python, never mid-forward); act_scratch [num_entries, MOE_INTER] fp16;
// out [T, MOE_HIDDEN] fp32 PRE-ZEROED (the served out = torch.zeros carries over); every table
// ref group_size divisible by MOE_BLOCK_K; grid <= device SM count for the persistent regime
// (draft fixed value).
extern "C" void moe_launch_group_gemv(const MoERouting* routing, const MoEPtrTable* table,
                                      const half* x, int grid, cudaStream_t stream) {
    moe_group_gemv_kernel<true ><<<grid, MOE_THREADS, 0, stream>>>(*routing, table, x);
    moe_group_gemv_kernel<false><<<grid, MOE_THREADS, 0, stream>>>(*routing, table, x);
}