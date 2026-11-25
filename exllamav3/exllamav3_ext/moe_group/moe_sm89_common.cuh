// Shared types, geometry and sm_89 primitives for the Inkling-2 MoE grouped tier (phase 1 draft).
//
// Standalone-compilable by design: includes only cuda_runtime/cuda_fp16/cuda_fp16.hpp so
// `nvcc -arch=compute_89 -c` verifies it without torch headers (compile-verify runs in
// ~/kernel-scratch on dot26; see README.md — no build in the served tree, law 31/38).
//
// sm_89 constraints honoured: mma.sync (m16n8k16) + cp.async pipelining; NO wgmma, NO TMA
// (those are sm_90+). At the decode regime (entries = T*top_k, so padded rows M <= 16 per
// segment block) the op is bandwidth-bound: the win is one launch with device-side routing
// instead of 24 python-dispatched groups + a per-layer D2H sync, not tensor-core utilisation.
//
// Containers: experts are 128x3 INDEPENDENT quantized Linears with per-expert qgroups
// (exllamav3/modules/arch_specific/inkling2_loop.py:1072-1088). The tier therefore rides a
// POINTER TABLE built at load (MoEPtrTable below, ~12 KiB), never a stacked weight copy —
// NOT modules/block_sparse_mlp.py's stacked ExpertsCFG tensors (different architecture).
#ifndef MOE_SM89_COMMON_CUH
#define MOE_SM89_COMMON_CUH

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Served geometry (dot26 config read 2026-11-25): 128 routed experts, top_k 8, hidden 2304,
// moe_intermediate 9216. Shared expert count is 0 — the shared-express lane does not exist here.
#define MOE_NUM_EXPERTS   128
#define MOE_TOP_K         8
#define MOE_HIDDEN        2304
#define MOE_INTER         9216
// Softcaps, verbatim semantics from the served epilogue (inkling2_loop.py:1155-1158).
#define MOE_GATE_CAP      4.0f
#define MOE_UP_CAP        25.0f

// Kernel tiles. BLOCK_M covers a padded segment slice; BLOCK_K must divide every qgroup size
// the pack uses (draft restriction; the exl3 wire-in substitutes the real group walk).
#define MOE_BLOCK_M       16
#define MOE_BLOCK_N       64
#define MOE_BLOCK_K       32
#define MOE_SHARED_PAD    8                    // halfs of padding per shared row (bank-conflict diet)
#define MOE_THREADS       128                  // 4 warps; each warp owns 16 of the 64 output cols

// Sentinel marking a padded (invalid) slot in the segmented token list, vLLM's masking shape
// (fused_moe.py:69-162: offs_token < num_valid_tokens).
#define MOE_INVALID_ENTRY 0x7fffffff

// ---------------------------------------------------------------------------
// Pointer table: one entry per expert per projection. Built once at model load; the routing
// hot path only reads expert ids, never pointers-to-pointers that move. Wire format here is the
// generic int4 + per-(row,qgroup) fp16 scale placeholder; the exl3 trellis substititution is the
// integration step named in README.md (fields are shaped to carry it: qbytes/scales/group_size).
// ---------------------------------------------------------------------------
struct MoEWeightRef {
    const uint8_t* qbytes;      // packed int4 weights, row-major [N, K/2]
    const half*    scales;      // per (row, qgroup) fp16 scales, row-major [N, K/group_size]
    int32_t        group_size;  // qgroup width in elements
};

struct MoEPtrTable {
    MoEWeightRef gate[MOE_NUM_EXPERTS];   // [INTER, HIDDEN]
    MoEWeightRef up  [MOE_NUM_EXPERTS];   // [INTER, HIDDEN]
    MoEWeightRef down[MOE_INTER];         // [HIDDEN, INTER]
};

// ---------------------------------------------------------------------------
// Device-local routing exchange. Everything below lives at FIXED ADDRESSES allocated once per
// graph-capture region: kernels never allocate, never synchronize, and the host never reads a
// routing value mid-forward — the by-value doorbell lesson from the Strata audit (adopted
// 2026-11-24, campaign §210): routing values travel through fixed addresses, dissolving the
// capture-freeze limitation by construction instead of via stream syncs.
// ---------------------------------------------------------------------------
struct MoERouting {
    // in: sigmoid scores [T, MOE_NUM_EXPERTS] from the existing router GEMV (kept outside this
    // tier so router math stays bit-for-bit ours), plus the additive selection bias [E]
    const float* scores;
    const float* bias;
    float        route_scale;

    // produced by moe_dev_topk: chosen ids [T, TOP_K] and renormalized weights
    // w = scores.gather(sel) / (sum + 1e-20) * route_scale   (inkling2_loop.py:1114-1116)
    int32_t* sel;
    float*   w;

    // produced by moe_build_segments (device-only histogram/scan/scatter, one block, no sync)
    int32_t* sorted_token_ids;    // [capacity] entry ids (token*TOP_K+i) or MOE_INVALID_ENTRY
    int32_t* expert_for_block;    // [blocks] one expert id per BLOCK_M-slice (the on-device branch)
    int32_t* num_blocks_out;      // [1] read ONLY by device kernels (grid-stride bound), never host
    int32_t* counts_scratch;      // [MOE_NUM_EXPERTS]
    int32_t* cursor_scratch;      // [MOE_NUM_EXPERTS]
    int32_t* scan_scratch;        // [2 * MOE_NUM_EXPERTS]

    // A/B scratch: fused epilogue of the gate/up tier writes act[entry][INTER] (fp16 — bit-identical
    // to the served act.to(torch.half) before down_proj); the down tier reads it.
    half*    act_scratch;         // [num_entries, MOE_INTER]
    float*   out;                 // [T tokens, MOE_HIDDEN] fp32 accumulator (served out dtype)

    int32_t  num_tokens;          // T
    int32_t  num_entries;         // T * TOP_K
    int32_t  max_blocks;          // bound on expert_for_block/sorted_token_ids capacity
};

// ---------------------------------------------------------------------------
// sm_89 primitives
// ---------------------------------------------------------------------------
__device__ __forceinline__ void moe_cp_async16(void* smem, const void* gmem) {
    const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(gmem));
}
__device__ __forceinline__ void moe_cp_async16_zfill(void* smem, const void* gmem, bool pred) {
    // cp.async with src_size 0 zero-fills: used for masked/padding rows so garbage never enters.
    const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    const int src_size = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(gmem), "r"(src_size));
}
__device__ __forceinline__ void moe_cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void moe_cp_async_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N)); __syncthreads(); }

__device__ __forceinline__ void moe_ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* smem) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void moe_ldmatrix_x2_trans(unsigned& r0, unsigned& r1, const void* smem) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r0), "=r"(r1) : "r"(addr));
}
// Non-transposing x2: a shared tile stored [N][K] (k contiguous) already IS the mma .col B
// fragment layout (pair of halves along k per thread), so no transpose is needed.
__device__ __forceinline__ void moe_ldmatrix_x2(unsigned& r0, unsigned& r1, const void* smem) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r0), "=r"(r1) : "r"(addr));
}
// D = A(16x16,row) * B(16x8,col) + C ; A/B halves, C/D fp32.
__device__ __forceinline__ void moe_mma_16x8x16(float& d0, float& d1, float& d2, float& d3,
                                                 unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                                 unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float moe_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// Fused swiglu-softcap epilogue, mirroring inkling2_loop.py:1155-1158 elementwise:
// act = (tanh(g/4)*4) * sigmoid(g) * (tanh(u/25)*25)
__device__ __forceinline__ half moe_swiglu_softcap(float g, float u) {
    const float capped_g = tanhf(g / MOE_GATE_CAP) * MOE_GATE_CAP;
    const float gated = capped_g * moe_sigmoid(g);
    const float capped_u = tanhf(u / MOE_UP_CAP) * MOE_UP_CAP;
    return __float2half_rn(gated * capped_u);
}

// int4 (two nibbles per byte, biased 8) x per-row fp16 scale -> half. Placeholder for the exl3
// trellis unpack; the substitution point is a single function on purpose (README wire-in).
__device__ __forceinline__ void moe_unpack_int4x8(uint8_t packed8[4], half out[8], const half scale) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int lo = (packed8[i] & 0x0f) - 8;
        const int hi = (packed8[i] >> 4) - 8;
        out[2 * i]     = __float2half_rn((float)lo * __half2float(scale));
        out[2 * i + 1] = __float2half_rn((float)hi * __half2float(scale));
    }
}

#endif  // MOE_SM89_COMMON_CUH