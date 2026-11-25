// Phase-1 device routing: top-k on device + expert segmentation, replacing the per-MoE-layer
// D2H sync and python sort of inkling2_loop.py:1134-1135. Zero host synchronization: the
// padded block count stays device-resident (MoERouting::num_blocks_out is read only by the
// grouped kernels' grid-stride bound, never by the host — vLLM reads num_tokens_post_padded
// on host only because it sizes its grid dynamically; a persistent grid does not need it).
//
// Numerics mirror the served router exactly (fence 1 pins it): w uses the UNBIASED sigmoid
// score gathered at sel (inkling2_loop.py:1114), renormalized as w/(sum+1e-20)*route_scale.
#include "moe_sm89_common.cuh"

#include <cfloat>

// One warp per token. Each lane owns 4 experts: [lane*4, lane*4+4).
__global__ void moe_dev_topk_kernel(const MoERouting r) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= r.num_tokens) return;

    const float* sc = r.scores + (size_t)warp * MOE_NUM_EXPERTS;
    float own_s[4];   // unbiased sigmoid scores (needed for w)
    float own_b[4];   // scores + selection bias (ranking key)
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int e = lane * 4 + j;
        own_s[j] = sc[e];
        own_b[j] = sc[e] + r.bias[e];
    }

    float w_raw[MOE_TOP_K];
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < MOE_TOP_K; ++k) {
        float bv = -FLT_MAX;
        int   be = MOE_NUM_EXPERTS;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int e = lane * 4 + j;
            if (own_b[j] > bv || (own_b[j] == bv && e < be)) { bv = own_b[j]; be = e; }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, bv, off);
            const int   oe = __shfl_xor_sync(0xffffffffu, be, off);
            if (ob > bv || (ob == bv && oe < be)) { bv = ob; be = oe; }
        }
        // Winner's UNBIASED score travels to every lane (w gathers scores, not scores+bias).
        const int owner = be >> 2;
        const float wb = __shfl_sync(0xffffffffu, own_s[be & 3], owner);
        if (lane == owner) { own_s[be & 3] = -FLT_MAX; own_b[be & 3] = -FLT_MAX; }
        w_raw[k] = wb;
        sum += wb;
        if (lane == 0) r.sel[warp * MOE_TOP_K + k] = be;
    }

    const float inv = (1.0f / (sum + 1e-20f)) * r.route_scale;
    if (lane < MOE_TOP_K) r.w[warp * MOE_TOP_K + lane] = w_raw[lane] * inv;
}

// Single block: histogram -> per-expert block padding -> exclusive scan -> scatter, producing
// the on-device segment list the grouped kernels branch on (vLLM fused_moe.py:69-162 shape,
// but never leaving the GPU). blockDim.x MUST equal MOE_NUM_EXPERTS.
__global__ void moe_build_segments_kernel(const MoERouting r) {
    __shared__ int s_scan[MOE_NUM_EXPERTS];
    const int tid = threadIdx.x;
    const int n = r.num_entries;

    r.counts_scratch[tid] = 0;
    r.cursor_scratch[tid] = 0;
    for (int i = tid; i < r.max_blocks * MOE_BLOCK_M; i += blockDim.x)
        r.sorted_token_ids[i] = MOE_INVALID_ENTRY;
    if (tid == 0) *r.num_blocks_out = 0;
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x)
        atomicAdd(&r.counts_scratch[r.sel[i]], 1);
    __syncthreads();

    const int nb = (r.counts_scratch[tid] + MOE_BLOCK_M - 1) / MOE_BLOCK_M;
    s_scan[tid] = nb;
    __syncthreads();
    #pragma unroll
    for (int off = 1; off < MOE_NUM_EXPERTS; off <<= 1) {
        __syncthreads();
        const int add = (tid >= off) ? s_scan[tid - off] : 0;
        __syncthreads();
        s_scan[tid] += add;                    // inclusive
    }
    const int excl_blocks = s_scan[tid] - nb;  // exclusive
    if (tid == MOE_NUM_EXPERTS - 1) *r.num_blocks_out = s_scan[tid];
    for (int b = 0; b < nb; ++b)
        r.expert_for_block[excl_blocks + b] = tid;
    r.cursor_scratch[tid] = excl_blocks * MOE_BLOCK_M;
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        const int pos = atomicAdd(&r.cursor_scratch[r.sel[i]], 1);
        r.sorted_token_ids[pos] = i;
    }
}

// Launcher: routing exchange only; buffers must already exist at fixed addresses (capturable).
extern "C" void moe_launch_routing(const MoERouting* routing, cudaStream_t stream) {
    const int warps = routing->num_tokens;
    const int blocks = (warps * 32 + 127) / 128;
    moe_dev_topk_kernel<<<blocks, 128, 0, stream>>>(*routing);
    moe_build_segments_kernel<<<1, MOE_NUM_EXPERTS, 0, stream>>>(*routing);
}