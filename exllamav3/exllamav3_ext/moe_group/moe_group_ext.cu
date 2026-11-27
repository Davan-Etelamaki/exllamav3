// Grouped multi-token expert tier — kernel bodies (see moe_group_ext.cuh for the contract).
//
// BUILD STAGE, stated plainly: the served Inkling pack quantizes its routed experts
// (LinearEXL3 .trellis/.suh/.svh, K in [1, 8]); this build implements the fp16-dense
// weight layout (K == 0) and REFUSES the quant layout with rc -2 and a named reason, so
// flipping infer_params.inkling_moe_grouped_ext on today's pack keeps the eager chain by
// refusal, never by silent wrong math. The adoption step ports the exl3 dequant stream
// into these two launches (the tile/dequant device functions live in
// ../quant/exl3_moe_coop_kernel.cuh; the fallback int4 unpack in moe_group_gemv.cu must
// NOT feed a real pack — README rule), and flips the refusal into a launch. Until then
// the on-card win of §CJ/§CH attaches to that build, not this one; the python tier, the
// op-count gate and the semantics fence are already final, so adoption is a kernel-only
// change behind an untouched call contract.
//
// Compile proof (CPU-only host, no GPU context): nvcc -std=c++20 -arch=sm_89 -c with the
// wheel-exact flag set (torch's four -D__CUDA_NO_* defines + --expt-relaxed-constexpr,
// -O3 --use_fast_math, -Xcudafe --diag_suppress=177/20012); attribution via
// cuobjdump --dump-elf-symbols naming the two kernels below.
#include "moe_group_ext.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#include <cstdarg>
#include <cstdio>
#include <cstring>

namespace moe_group {

namespace {

constexpr int GROUP_BLOCK = 256;

thread_local char g_reason[256] = {0};

void set_reason (const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_reason, sizeof(g_reason), fmt, ap);
    va_end(ap);
}

__device__ __forceinline__ float softcap_swiglu (const float g, const float u,
                                                 const float gate_cap, const float up_cap) {
    // The served epilogue, one expression: tanh(g/cap)*cap * sigmoid(g) * (tanh(u/cap)*cap)
    // (inkling2_loop.py InklingMoE; MOE_COOP_ACT_SOFTCAP in the coop tier agrees — its
    // elementwise fence priced 7.6e-06 max deviation).
    return tanhf(g / gate_cap) * gate_cap * (1.0f / (1.0f + expf(-g)))
           * (tanhf(u / up_cap) * up_cap);
}

// One block per entry. Expert recovered by scanning the counts histogram in shared
// memory (E ≤ 128: one thread, negligible against the dot products); row gathered via
// order_all — the device order python never round-trips.
template <bool GU>
__global__ void moe_group_launch_kernel (const __half *__restrict__ in_,
                                         const int64_t *__restrict__ order_all,
                                         const int64_t *__restrict__ counts,
                                         const int64_t *__restrict__ tables,
                                         int experts, int top_k,
                                         int K_dim, int N_dim,
                                         __half *__restrict__ out_,
                                         float gate_cap, float up_cap) {
    const int entry = blockIdx.x;                       // 0..entries-1, order_all order
    const int64_t e = order_all[entry];
    const int token = (int)(e / top_k);
    const int col0 = threadIdx.x;

    __shared__ int s_expert;
    if (threadIdx.x == 0) {
        int c = 0, seen = 0;
        while (c < experts) {                           // ascending-expert segment scan
            const int n = (int)counts[c];
            if (entry < seen + n) break;
            seen += n;
            ++c;
        }
        s_expert = c;
    }
    __syncthreads();
    const int expert = s_expert;

    // Stacked [9, experts] pointer table; slot order = moe_coop_plan.TABLE_ORDER
    // (d_suh, d_svh, d_trellis, g_suh, g_svh, g_trellis, u_suh, u_svh, u_trellis).
    // Under MOE_GROUPED_LAYOUT_FP16 the *_trellis slots carry the dense __half bases,
    // each an [N_dim, K_dim] row-major matrix per expert.
    enum : int { SLOT_D = 2, SLOT_G = 5, SLOT_U = 7 };
    const __half *w0, *w1 = nullptr;
    if (GU) {
        w0 = reinterpret_cast<const __half *>(tables[SLOT_G * experts + expert])
             + (int64_t)0;                              // block base; rows offset below
        w1 = reinterpret_cast<const __half *>(tables[SLOT_U * experts + expert]);
    } else {
        w0 = reinterpret_cast<const __half *>(tables[SLOT_D * experts + expert]);
    }
    const __half *xrow = in_ + (int64_t)token * K_dim;

    // Output columns stripe across the block; each thread accumulates fp32 over K_dim
    // with __half2 loads (scalar-per-thread scaffold: correctness stage; the tiled/mma
    // form arrives with the exl3 adoption).
    for (int n = col0; n < N_dim; n += GROUP_BLOCK) {
        const __half2 *wg = reinterpret_cast<const __half2 *>(w0 + (int64_t)n * K_dim);
        const __half2 *xu = reinterpret_cast<const __half2 *>(xrow);
        float acc0 = 0.0f, acc1 = 0.0f;
        if (GU) {
            const __half2 *wu = reinterpret_cast<const __half2 *>(w1 + (int64_t)n * K_dim);
            float g_acc = 0.0f, u_acc = 0.0f;
            for (int k2 = 0; k2 < K_dim / 2; ++k2) {
                const __half2 xv = xu[k2];
                const float2 xf = __half22float2(xv);
                const float2 gv = __half22float2(wg[k2]);
                const float2 uv = __half22float2(wu[k2]);
                g_acc += xf.x * gv.x + xf.y * gv.y;
                u_acc += xf.x * uv.x + xf.y * uv.y;
            }
            acc0 = softcap_swiglu(g_acc, u_acc, gate_cap, up_cap);
        } else {
            for (int k2 = 0; k2 < K_dim / 2; ++k2) {
                const float2 xf = __half22float2(xu[k2]);
                const float2 wv = __half22float2(wg[k2]);
                acc0 += xf.x * wv.x + xf.y * wv.y;
            }
        }
        out_[(int64_t)entry * N_dim + n] = __float2half_rn(acc0);
        (void)acc1;
    }
}

// Shared host-side validation for both launches. Returns false after set_reason().
bool validate_common (const torch::Tensor &t, const char *name, c10::Device dev,
                      c10::ScalarType dt, const char *shapes_hint) {
    if (!t.is_cuda()) { set_reason("%s: not a CUDA tensor", name); return false; }
    if (t.device() != dev) { set_reason("%s: device mismatch", name); return false; }
    if (t.scalar_type() != dt) {
        set_reason("%s: dtype != %s (shapes %s)", name,
                   c10::toString(dt), shapes_hint); return false;
    }
    if (!t.is_contiguous()) { set_reason("%s: not contiguous", name); return false; }
    return true;
}

int64_t slot (const char *name) {                 // matches moe_coop_plan.TABLE_ORDER
    static const char *order[] = {"d_suh", "d_svh", "d_trellis", "g_suh", "g_svh",
                                  "g_trellis", "u_suh", "u_svh", "u_trellis"};
    for (int i = 0; i < 9; ++i) if (strcmp(order[i], name) == 0) return i;
    return -1;
}

}  // namespace

int exl3_moe_group_gu (torch::Tensor x, torch::Tensor order_all, torch::Tensor counts,
                       torch::Tensor tables, int64_t K, bool mcg, bool mul1,
                       torch::Tensor act, double gate_cap, double up_cap) {
    g_reason[0] = 0;
    const c10::Device dev = x.device();
    if (!validate_common(x, "x", dev, at::kHalf, "[rows, h]")
        || !validate_common(order_all, "order_all", dev, at::kLong, "[entries]")
        || !validate_common(counts, "counts", dev, at::kLong, "[experts]")
        || !validate_common(tables, "tables", dev, at::kLong, "[9*experts]")
        || !validate_common(act, "act", dev, at::kHalf, "[cap, I]"))
        return -1;
    const int64_t rows = x.size(0), entries = order_all.numel(), experts = counts.size(0);
    const int h = (int)x.size(1), I = (int)act.size(1);
    if (rows <= 0 || entries % rows != 0 || act.size(0) < entries
        || experts <= 0 || tables.numel() != 9 * experts
        || slot("g_trellis") != 5 || slot("u_trellis") != 7 || slot("d_trellis") != 2) {
        set_reason("geometry or table-order mismatch: rows %lld entries %lld experts %lld "
                   "cap %lld (kernel slots assume TABLE_ORDER)",
                   (long long)rows, (long long)entries, (long long)experts,
                   (long long)act.size(0));
        return -1;
    }
    if (rows > MOE_GROUPED_MAX_ROWS) {
        set_reason("%lld rows above MOE_GROUPED_MAX_ROWS %d (python ladder refuses first)",
                   (long long)rows, MOE_GROUPED_MAX_ROWS);
        return -1;
    }
    if (K != MOE_GROUPED_LAYOUT_FP16) {
        set_reason("layout exl3 (K=%lld) not yet wired into the grouped tier: the trellis/"
                   "suh/svh stream (exl3_moe_coop_kernel.cuh tile functions) is the "
                   "adoption step; keep the eager chain", (long long)K);
        return -2;
    }
    if (mcg || mul1) { set_reason("fp16-dense layout carries no codebooks (mcg/mul1 set)");
                       return -1; }
    if (!(gate_cap > 0.0) || !(up_cap > 0.0)) { set_reason("softcaps must be > 0"); return -1; }
    if ((h & 1) || (I & 1)) { set_reason("odd feature width %d/%d needs the scalar path", h, I);
                              return -1; }
    const int top_k = (int)(entries / rows);
    const c10::cuda::CUDAGuard guard(dev);
    auto stream = at::cuda::getCurrentCUDAStream(dev.index()).stream();
    moe_group_launch_kernel<true><<<(int)entries, GROUP_BLOCK, 0, stream>>>(
        reinterpret_cast<const __half *>(x.data_ptr()), order_all.data_ptr<int64_t>(),
        counts.data_ptr<int64_t>(), tables.data_ptr<int64_t>(), (int)experts, top_k,
        h, I, reinterpret_cast<__half *>(act.data_ptr()),
        (float)gate_cap, (float)up_cap);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int exl3_moe_group_down (torch::Tensor act, int64_t entries, torch::Tensor order_all,
                         torch::Tensor counts, torch::Tensor tables, int64_t K, bool mcg,
                         bool mul1, torch::Tensor y) {
    g_reason[0] = 0;
    const c10::Device dev = act.device();
    if (!validate_common(act, "act", dev, at::kHalf, "[cap, I]")
        || !validate_common(y, "y", dev, at::kHalf, "[cap, h]")
        || !validate_common(order_all, "order_all", dev, at::kLong, "[entries]")
        || !validate_common(counts, "counts", dev, at::kLong, "[experts]")
        || !validate_common(tables, "tables", dev, at::kLong, "[9*experts]"))
        return -1;
    const int64_t experts = counts.size(0);
    const int I = (int)act.size(1), h = (int)y.size(1);
    if (entries != order_all.numel() || entries <= 0 || y.size(0) < entries
        || tables.numel() != 9 * experts || slot("d_trellis") != 2) {
        set_reason("geometry mismatch: entries %lld cap_y %lld experts %lld",
                   (long long)entries, (long long)y.size(0), (long long)experts);
        return -1;
    }
    if (K != MOE_GROUPED_LAYOUT_FP16) {
        set_reason("layout exl3 (K=%lld) not yet wired into the grouped down launch",
                   (long long)K);
        return -2;
    }
    if (mcg || mul1) { set_reason("fp16-dense layout carries no codebooks"); return -1; }
    if ((I & 1) || (h & 1)) { set_reason("odd feature width %d/%d", I, h); return -1; }
    const c10::cuda::CUDAGuard guard(dev);
    auto stream = at::cuda::getCurrentCUDAStream(dev.index()).stream();
    moe_group_launch_kernel<false><<<(int)entries, GROUP_BLOCK, 0, stream>>>(
        reinterpret_cast<const __half *>(act.data_ptr()), order_all.data_ptr<int64_t>(),
        counts.data_ptr<int64_t>(), tables.data_ptr<int64_t>(), (int)experts, 1,
        I, h, reinterpret_cast<__half *>(y.data_ptr()), 0.0f, 0.0f);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

std::string exl3_moe_group_reason() { return std::string(g_reason); }

}  // namespace moe_group