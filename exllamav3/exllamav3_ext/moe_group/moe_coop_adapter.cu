// Device-routing adapter for the EXISTING exl3_moe_coop decode tier (decision 2026-11-25,
// see README "Integration decision"). The served Inkling-2 MoE pays its per-layer D2H sync
// (inkling2_loop.py:1134) only because the routing values must reach the python dispatch loop;
// exl3_moe_coop_launch already reads sel/rw as DEVICE arrays through per-expert pointer
// tables, so the group tier itself does not need reinventing. This file adapts moe_route.cu's
// output (int32 sel, float w) to the coop contract (int64 sel, half rw) at FIXED addresses —
// the Strata by-value doorbell rule: the exchange never allocates, never synchronizes.
//
// Compile-verified in ~/kernel-scratch on dot26 (nvcc -c with the venv's torch includes).
// Linked INTO the extension at wheel build time; nothing here launches a model.
#include "moe_exl3_table.cuh"
#include "moe_sm89_common.cuh"
#include "../quant/exl3_moe_coop.cuh"

// int32 sel -> int64 sel, float w -> half rw (the coop struct's declared input types).
// n = bsz * top_k; all buffers preallocated at fixed addresses by the python wiring.
__global__ void moe_route_to_coop_kernel(
    const int32_t* __restrict__ sel, const float* __restrict__ w,
    int64_t* __restrict__ sel64, half* __restrict__ rw16, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        sel64[i] = static_cast<int64_t>(sel[i]);
        rw16[i] = __float2half_rn(w[i]);
    }
}

// Attach the device routing to a prepared parameter block and launch the coop tier.
// Preconditions enforced by the python wiring (sizes pinned by exl3_moe_coop_ctr_len and the
// capacity formula in moe_exl3_table.cuh): bsz <= MAX decode batch of the coop scratch,
// slots <= slots_max, pointer tables built from the 128x3 independent LinearEXL3 experts,
// act selector matches the served softcap-swiglu (README records the selector gap on this
// lineage — the dsh-softcap MOE_COOP_ACT_SOFTCAP must ride the same wheel batch or be
// re-implemented here before this path serves).
extern "C" int moe_coop_launch_with_device_routing
(
    MoeCoopParams* p,             // prepared once by python (exl3_moe_coop_prepare)
    const int32_t* sel_i32,       // MoERouting::sel
    const float*   w_f32,         // MoERouting::w
    int64_t*       sel64_scratch, // fixed-address coop views
    half*          rw16_scratch,
    int            entries,       // bsz * top_k
    float K_gu, float K_d, int cb, int device, cudaStream_t stream
) {
    if (entries > 0) {
        moe_route_to_coop_kernel<<<(entries + 127) / 128, 128, 0, stream>>>(
            sel_i32, w_f32, sel64_scratch, rw16_scratch, entries);
    }
    p->sel = sel64_scratch;
    p->rw = rw16_scratch;
    if (p->bsz <= 0 || p->bsz * p->topk > p->slots_max) {
        return -1;                // batch beyond the coop scratch: python keeps the legacy path
    }
    exl3_moe_coop_launch(*p, K_gu, K_d, cb, device, stream);
    return 0;
}