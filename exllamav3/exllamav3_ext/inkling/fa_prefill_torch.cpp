// Tensor layer of the Inkling 2 flash-prefill path: shape/dtype/stride refusal and the call into
// the launcher. Every refusal here carries a reason the Python seam prints as
// INKLING-FLASHDECLINE <reason> and then falls back to the fused path — a pass this kernel was not
// written for is a decline, never an exception and never a silent approximation.

#if !defined(USE_ROCM)

#include "fa_prefill_torch.cuh"
#include "fa_prefill.cuh"

#include <ATen/cuda/CUDAContext.h>

#include <sstream>

namespace { constexpr double LOG2E = 1.4426950408889634; }

namespace
{
std::string g_reason;

void refuse(const std::string& why) { g_reason = why; }

std::string sizes(const at::Tensor& t)
{
    std::ostringstream os;
    os << "[" << t.size(0);
    for (int i = 1; i < t.dim(); ++i) os << "," << t.size(i);
    os << "]";
    return os.str();
}
}  // namespace

std::map<std::string, int64_t> exl3_inkling_fa_plan(int64_t h, int64_t kvh, int64_t d,
                                                    int64_t rel_extent, int64_t T, int64_t S,
                                                    int64_t smem_budget, int64_t device)
{
    const FaPrefillPlan p = inkling_fa_prefill_plan(
        FaPrefillShapes{int(h), int(kvh), int(d), int(rel_extent)}, int(T), int(S),
        size_t(smem_budget), int(device));
    if (!p.ok) g_reason = p.reason;
    else g_reason.clear();

    std::map<std::string, int64_t> out;
    out["ok"] = p.ok;
    out["bm"] = p.bm;
    out["bn"] = p.bn;
    out["warps"] = p.warps;
    out["threads"] = p.threads;
    out["stages"] = p.stages;
    out["grp"] = p.grp;
    out["grid_x"] = p.grid_x;
    out["grid_y"] = p.grid_y;
    out["key_tiles"] = p.key_tiles;
    out["band_tiles"] = p.band_tiles;
    out["smem_bytes"] = int64_t(p.smem_bytes);
    out["smem_q_upper"] = int64_t(p.smem_q_upper);
    out["smem_stage"] = int64_t(p.smem_stage);
    out["blocks_per_sm"] = p.blocks_per_sm;
    out["smem_per_sm"] = int64_t(p.smem_per_sm);
    out["k_steps"] = p.k_steps;
    out["n_steps"] = p.n_steps;
    return out;
}

std::string exl3_inkling_fa_reason() { return g_reason; }

int64_t exl3_inkling_fa_prefill(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor rel,
                                at::Tensor sink, at::Tensor qpos, at::Tensor kpos,
                                at::Tensor o, at::Tensor nfin,
                                double scale, int64_t rel_extent, int64_t smem_budget)
{
    g_reason.clear();

    // ---- dtypes: the kernel reads fp16 Q/K/V/O, fp32 bias and sink, int64 positions
    const bool half_ok = q.dtype() == at::kHalf && k.dtype() == at::kHalf &&
                         v.dtype() == at::kHalf && o.dtype() == at::kHalf;
    if (!half_ok) { refuse("dtype: q/k/v/o must be fp16"); return 0; }
    if (rel.dtype() != at::kFloat || sink.dtype() != at::kFloat)
    {
        refuse("dtype: rel and sink must be fp32");
        return 0;
    }
    if (qpos.dtype() != at::kLong || kpos.dtype() != at::kLong)
    {
        refuse("dtype: qpos and kpos must be int64");
        return 0;
    }
    if (nfin.numel() > 0 && nfin.dtype() != at::kInt)
    {
        refuse("dtype: the non-finite counter must be int32");
        return 0;
    }

    // ---- ranks and agreements
    if (q.dim() != 3 || k.dim() != 3 || v.dim() != 3 || o.dim() != 3)
    {
        refuse("rank: q/k/v/o must be 3-d (got q " + sizes(q) + ")");
        return 0;
    }
    if (rel.dim() != 3 || sink.dim() != 2 || qpos.dim() != 1 || kpos.dim() != 1)
    {
        refuse("rank: rel (h,T,ext), sink (T,h), qpos (T), kpos (S) expected");
        return 0;
    }
    const int64_t h = q.size(0), T = q.size(1), d = q.size(2);
    const int64_t S = k.size(0), kvh = k.size(0) == 0 ? 0 : k.size(1);
    if (v.size(0) != S || v.size(1) != kvh) { refuse("k and v shapes disagree"); return 0; }
    if (o.size(0) != h || o.size(1) != T || o.size(2) != d)
    {
        refuse("output shape must match q");
        return 0;
    }
    if (rel.size(0) != h || rel.size(1) != T || rel.size(2) != rel_extent)
    {
        refuse("rel shape must be (h, T, rel_extent)");
        return 0;
    }
    if (sink.size(0) != T || sink.size(1) != h)
    {
        refuse("sink shape must be (T, h)");
        return 0;
    }
    if (qpos.numel() < T || kpos.numel() < S)
    {
        refuse("position tensor shorter than its axis");
        return 0;
    }

    // ---- strides: the kernel streams 16-byte chunks, so the head dim must be contiguous and the
    // row/head strides multiples of 8 halves (16 B). A cache layout that breaks this is a decline,
    // not a copy: copying the whole KV slab per prefill would cost more than the arm can save.
    if (q.stride(2) != 1 || k.stride(2) != 1 || v.stride(2) != 1 || o.stride(2) != 1 ||
        rel.stride(2) != 1 || sink.stride(1) != 1)
    {
        refuse("last-dim stride not contiguous");
        return 0;
    }
    if (q.stride(0) % 8 || q.stride(1) % 8 || k.stride(0) % 8 || k.stride(1) % 8 ||
        v.stride(0) % 8 || v.stride(1) % 8 || o.stride(0) % 8 || o.stride(1) % 8)
    {
        refuse("row/head strides not 16-byte aligned");
        return 0;
    }
    if (qpos.stride(0) != 1 || kpos.stride(0) != 1)
    {
        refuse("position tensors not contiguous");
        return 0;
    }
    const int dev = q.get_device();
    if (k.get_device() != dev || v.get_device() != dev || rel.get_device() != dev ||
        sink.get_device() != dev || o.get_device() != dev || qpos.get_device() != dev ||
        kpos.get_device() != dev)
    {
        refuse("tensors span more than one device");
        return 0;
    }

    FaPrefillArgs a{};
    a.q = q.data_ptr();
    a.k = k.data_ptr();
    a.v = v.data_ptr();
    a.rel = rel.data_ptr();
    a.sink = static_cast<const float*>(sink.data_ptr());
    a.qpos = static_cast<const int64_t*>(qpos.data_ptr());
    a.kpos = static_cast<const int64_t*>(kpos.data_ptr());
    a.o = o.data_ptr();
    a.nfin = nfin.numel() > 0 ? static_cast<int32_t*>(nfin.data_ptr()) : nullptr;
    a.stream = nullptr;                       // replaced below with the current stream
    a.q_head_stride = q.stride(0);
    a.q_row_stride = q.stride(1);
    a.k_row_stride = k.stride(0);
    a.k_head_stride = k.stride(1);
    a.rel_head_stride = rel.stride(0);
    a.rel_row_stride = rel.stride(1);
    a.sink_row_stride = sink.stride(0);
    a.o_head_stride = o.stride(0);
    a.o_row_stride = o.stride(1);
    a.T = int(T);
    a.S = int(S);
    a.h = int(h);
    a.kvh = int(kvh);
    a.d = int(d);
    a.rel_extent = int(rel_extent);
    // The kernel's softmax runs in the exp2 domain: scale and the additive logits both need the
    // log2(e) factor, and only the scaled product carries the attention scale.
    a.scale_log2 = float(scale * LOG2E);
    a.log2e = float(LOG2E);

    a.stream = (void*) at::cuda::getCurrentCUDAStream().stream();

    const FaPrefillLaunchReport rep =
        inkling_fa_prefill_launch(a, size_t(smem_budget > 0 ? smem_budget : 0));
    if (!rep.launched)
    {
        refuse(rep.reason);
        return 0;
    }
    return 1;
}

#else   // USE_ROCM — the tensor-core path is CUDA-only; the ROCm build declines every pass.

#include "fa_prefill_torch.cuh"

#include <map>
#include <string>

namespace
{
std::string g_reason_rocm = "ROCm build: the Inkling flash-prefill kernel is a CUDA tensor-core path";
}

std::map<std::string, int64_t> exl3_inkling_fa_plan(int64_t, int64_t, int64_t, int64_t,
                                                    int64_t, int64_t, int64_t, int64_t)
{
    return std::map<std::string, int64_t>{{"ok", 0}};
}

int64_t exl3_inkling_fa_prefill(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                double, int64_t, int64_t)
{
    return 0;
}

std::string exl3_inkling_fa_reason() { return g_reason_rocm; }

#endif  // USE_ROCM