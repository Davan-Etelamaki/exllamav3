// Grouped multi-token expert tier for Inkling 2 MoE (sm_89 first) — the two ext launches
// the §CJ absorption map names: gu (gather + gate/up + fused softcap-swiglu into an fp16
// act buffer) and down (weighted rows for the single full-order combine python keeps).
// Served from inkling2_loop.py InklingMoE._grouped_moe behind infer_params.
// inkling_moe_grouped_ext (default OFF); the regime seam (t == 1, capture) keeps the
// python/coop arms verbatim, so nothing here runs on today's decode path.
//
// Row order is the DEVICE order_all (stable argsort of the flattened selection, keyed on
// the expert id), so an entry's expert is recovered from the counts histogram inside the
// kernel — no host roundtrip, no per-group boundaries. The combine stays in python as one
// full-order index_add_: entries processed in order_all order visit each token's
// contributions in exactly the per-group loop's sequence (bit-identical CPU; on-card the
// per-entry rows are fp32-accumulated inside the launch, tests/cpu/
// inkling_moe_grouped_fence.py executes both arms, law 31's installed→patched pair owns
// the on-card verdict).
//
// Argument names below are FENCED: tests/cpu/inkling_moe_grouped_fence.py [G] re-parses
// them against moe_coop_plan.GU_CALL_ARGS / DOWN_CALL_ARGS — a signature change must land
// on both sides or the fence REDs. Constants likewise mirror moe_coop_plan.
#ifndef MOE_GROUP_EXT_CUH
#define MOE_GROUP_EXT_CUH

#include <torch/extension.h>

namespace moe_group {

// mirrors moe_coop_plan.MOE_GROUPED_MAX_ROWS / GROUPED_BUCKETS[-1]: the tier admits rows
// up to the served chunk_size; above the ladder the python dispatch keeps the shape
// (the workspace is sized from the allocation site at this cap — 16384 entries ×
// 1152 fp16 act + 16384 × 2304 fp16 y ≈ 113 MB, held once per device per generation).
constexpr int MOE_GROUPED_MAX_ROWS = 2048;

// Expert weight layouts the tables argument can carry. The served pack quantizes the
// routed experts (LinearEXL3: .trellis/.suh/.svh), which is K in [1, 8]; the fp16-dense
// residency mode passes K == 0 (tables hold __half weight bases, row-major [out, in]).
enum MoEGroupedLayout : int {
    MOE_GROUPED_LAYOUT_FP16 = 0,
    MOE_GROUPED_LAYOUT_EXL3 = 1,   // trellis stream: DECLINED until the unpack adoption
                                   // (see moe_group_ext.cu::exl3_moe_group_gu — the
                                   // refusal names the missing piece; python keeps eager)
};

// Launch-level rc: 0 ok; negative refused (reason via exl3_moe_group_reason()).
// -1 = argument validation failed, -2 = layout not implemented on this build stage.
#define INKLING_MOE_GROUPED_MARKER "INKLING-MOE-GROUPED"

// gu: per entry e (grid-strided over rows*top_k), expert E = scan(counts, e),
// token T = order_all[e] / top_k, row = x[T]; acc_g/acc_u fp32 over in_features,
// act[e] = to_half( tanh(g/gate_cap)*gate_cap * sigmoid(g) * (tanh(u/up_cap)*up_cap) )
// — the same rounding point as the served act.to(torch.half) before down_proj.
int exl3_moe_group_gu(torch::Tensor x, torch::Tensor order_all, torch::Tensor counts,
                      torch::Tensor tables, int64_t K, bool mcg, bool mul1,
                      torch::Tensor act, double gate_cap, double up_cap);

// down: per entry, y[e] = to_half(act[e] @ Wd[E]^T) — the rows the python combine
// weights and scatters. No atomics here: the combine's per-token sequence is the
// bit-identity anchor, so this launch only produces rows.
int exl3_moe_group_down(torch::Tensor act, int64_t entries, torch::Tensor order_all,
                        torch::Tensor counts, torch::Tensor tables, int64_t K, bool mcg,
                        bool mul1, torch::Tensor y);

// Last negative rc's full reason (thread-local, host-side; "" when none).
std::string exl3_moe_group_reason();

}  // namespace moe_group

#endif  // MOE_GROUP_EXT_CUH