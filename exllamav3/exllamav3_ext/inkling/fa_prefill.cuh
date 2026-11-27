#pragma once

// sm_89 flash-style PREFILL attention for Inkling 2 (Inkling2Loop global attention), behind
// infer_params.inkling_attn_flashprefill (default False).
//
// What is new here, relative to the shipped fused epilogue (inkling_attn_biasfuse):
// the fused arm still materialises an (kvh, grp*rows, cols) fp32 scores plane per key block and
// pays a torch matmul per block. This path keeps the whole softmax inside one kernel: scores
// live in tensor-core accumulators, the running max/denominator and the fp32 output partial
// live in registers, and NOTHING per-key ever reaches memory except the K/V tiles the kernel
// streams through shared memory. That is where the remaining rate against the same-card
// tensor-core comparator lives — the compiled-kernel dimension, not the glue.
//
// The architecture's crux is honoured, not regressed: the relative-bias band (rel_extent wide per
// row) is folded INSIDE the key-tile loop, as one gather per score element, for those tiles that
// can overlap a row's band at all. The (h, rows, S) fp32 bias plane never comes back — the defect
// the fuse eliminated (12.7 GiB of mostly-zeros at S=23k) stays eliminated.
//
// Head dim is fixed at 256 (the served geometry) and the key/query tile sizes are DERIVED from
// the served shapes plus the device's shared-memory limit by fa_prefill_plan below; nothing here
// hard-codes a buffer size. Any geometry or budget the derivation cannot honour is a DECLINE with
// a named reason (INKLING-FLASHDECLINE) and the caller falls back to the fused path — refuse,
// never crash.

#include <cstddef>
#include <cstdint>

// ---------------------------------------------------------------------------
// Served geometry (Inkling2Loop on dot26, from the pack's own config.json as read by the parent
// lane on 2026-11-27): h=24 query heads, kvh=8 kv heads (GQA 3:1), head_dim=128, rel_extent=1024,
// window 262,144, chunk_size 2048, max_batch_size 4, 24 layers of which 20 are sliding-local
// (window 1024) and 4 ([5, 11, 17, 23]) are global. The plan reads all of these as ARGUMENTS, never
// as literals: the fence re-plans with each one changed and requires the derived value to move.
// ---------------------------------------------------------------------------

// Primary head dim of the served pack. The tile choreography is written generically in D (row
// chunks = D/8, k-steps of m16n8k8 = D/16, n-steps for P V = D/8) and instantiated for the served
// 128 and for 256, which this tree's turing/fa75.cu was written for; any other head dim declines.
#define FA_HD 128
#define FA_HD_ALT 256

// Query rows per block and keys per key tile. 64 rows is the geometry the same tree's
// turing/fa75.cu runs on hardware; BN is raised to 32 keys for head_dim 128 (a K+V stage lands at
// 16 KiB, so two stages plus the resident Q upper half stay inside an SM's 100 KiB with two blocks
// resident). Warps: 4, one 16-row band each.
#define FA_BM 64
#define FA_BN 32
// Keys one fp16 P·V partial covers before the kernel promotes it into the fp32 O accumulator:
// exactly ONE key tile. fa_prefill.cu's P·V loop runs FA_PV_FP16_KEYS/8 mma k-steps of 8 keys
// against one fp16 accumulator and converts it to fp32 at fa_prefill.cu:416-419, at the tile's end.
// For one output element p = exp2(s - m) <= 1 (m is the running max INCLUDING this tile), so the
// fp16 partial is bounded by FA_PV_FP16_KEYS * max|v| — the ceiling that keeps the estate's fp16-PV
// overflow (65504 at ~63k keys, 4cb7529a) out of this arm. Widening the window widens the bound
// proportionally; the static_assert below pins it to the tile so the two cannot drift apart, and
// tests/cpu/inkling_fa_prefill_fence.py pins every prose site of this number to this constant.
constexpr int FA_PV_FP16_KEYS = 32;
static_assert(FA_PV_FP16_KEYS == FA_BN,
              "the fp16 P.V window is one key tile: change FA_BN and this bound together");
#define FA_NWARPS 4
#define FA_MIN_ROWS FA_BM            // a pass shorter than one query tile stays on the fused path
#define FA_STAGES_PREFERRED 2        // cp.async double buffering of the K and V tiles
#define FA_SCRATCH_BYTES (2 * FA_NWARPS * 4)   // per-warp position extremes, in shared memory

// Head dims this kernel is instantiated for: the served pack's 128, and 256 (fa75's geometry).
inline bool fa_head_dim_supported(int d) { return d == FA_HD || d == FA_HD_ALT; }

// Bytes of one K (or V) tile: BN rows x head_dim halves.
inline size_t fa_tile_bytes(int d) { return size_t(FA_BN) * size_t(d) * 2; }
// Bytes of the resident upper half of Q (dims d/2..d for BM rows), kept in shared memory and read
// with ldmatrix per key tile, exactly as fa75 keeps them.
inline size_t fa_q_upper_bytes(int d) { return size_t(d / 2) * size_t(FA_BM) * 2; }
// Bytes of one pipeline stage: a K tile and a V tile side by side.
inline size_t fa_stage_bytes(int d) { return 2 * fa_tile_bytes(d); }

// Shared memory an Inkling 2 flash-prefill block needs, from the shapes alone. Exposed separately
// so the fence (and a decline message) can state the arithmetic the launch is about to request:
// one pipeline per stage, the resident Q upper half, and the position-extreme scratch.
inline size_t fa_smem_bytes(int d, int stages)
{
    return fa_q_upper_bytes(d) + size_t(stages > 0 ? stages : 1) * fa_stage_bytes(d)
           + size_t(FA_SCRATCH_BYTES);
}

struct FaPrefillShapes
{
    int h;             // query heads
    int kvh;           // kv heads (GQA: h / kvh q heads share one kv head)
    int d;             // head dim
    int rel_extent;    // relative-bias band width
};

struct FaPrefillPlan
{
    int ok;                        // 1 = launch this plan, 0 = decline
    int bm, bn, warps, threads;    // block geometry
    int stages;                    // cp.async pipeline stages actually afforded
    int grp;                       // h / kvh, the GQA broadcast factor
    int grid_x, grid_y;            // query tiles x heads
    int key_tiles;                 // ceil(S / bn) key tiles one block walks
    int band_tiles;                // key tiles a row's bias band can touch: ceil(rel_extent / bn)
    size_t smem_q_upper;           // resident Q upper half
    size_t smem_stage;             // one K+V stage
    size_t smem_bytes;             // what the launch asks the driver for
    int blocks_per_sm;             // occupancy the smem arithmetic affords (SM has smem_per_sm)
    size_t smem_per_sm;            // the SM budget the arithmetic was done against
    int k_steps;                   // m16n8k8 k-steps of Q K^T: d / 8
    int n_steps;                   // m16n8k8 n-steps of P V: d / 8
    char reason[112];              // named decline reason, empty when ok
};

namespace fa_detail
{
inline void set_reason(char* buf, size_t cap, const char* text)
{
    size_t i = 0;
    for (; text[i] && i + 1 < cap; ++i) buf[i] = text[i];
    buf[i] = '\0';
}
}  // namespace fa_detail

// Derive the plan. `smem_budget_bytes` is the caller's cap for one block's dynamic shared memory
// (0 = no caller cap); `device_smem_optin` is the device's opt-in maximum per block;
// `device_smem_per_sm` is the SM's shared-memory capacity; `sm_major`/`sm_minor` the compute
// capability. Declines, in order: unsupported head dim, non-integral GQA ratio, a pass too short
// to fill a query tile, a cache shorter than one key tile, a nonsense bias band, an architecture
// without cp.async, and a shared-memory request the device (or the caller's budget) will not hold.
// Every derived number comes out of the shapes and the device limits — never out of a remembered
// constant for the served card.
inline FaPrefillPlan fa_prefill_plan(const FaPrefillShapes& s, int T, int S,
                                     size_t smem_budget_bytes,
                                     size_t device_smem_optin, size_t device_smem_per_sm,
                                     int sm_major, int sm_minor)
{
    FaPrefillPlan p{};
    p.bm = FA_BM; p.bn = FA_BN; p.warps = FA_NWARPS; p.threads = FA_NWARPS * 32;
    p.ok = 0; p.reason[0] = '\0';
    p.grp = s.kvh > 0 ? s.h / s.kvh : 0;

    if (!fa_head_dim_supported(s.d))
    {
        fa_detail::set_reason(p.reason, sizeof(p.reason),
                              "head_dim has no kernel instantiation (built for 128 and 256)");
        return p;
    }
    if (s.kvh <= 0 || s.h % s.kvh != 0)
    {
        fa_detail::set_reason(p.reason, sizeof(p.reason), "GQA ratio h/kvh not integral");
        return p;
    }
    if (T < FA_MIN_ROWS)
    {
        fa_detail::set_reason(p.reason, sizeof(p.reason),
                              "query rows below one 64-row tile (decode stays on the fused path)");
        return p;
    }
    if (S < FA_BN)
    {
        // The threshold is FA_BN keys (32 at every instantiated head dim); the number is left out of
        // the sentence on purpose, because this string must not disagree with the constant above.
        fa_detail::set_reason(p.reason, sizeof(p.reason), "cache shorter than one key tile");
        return p;
    }
    if (s.rel_extent <= 0)
    {
        fa_detail::set_reason(p.reason, sizeof(p.reason), "relative-bias extent is not positive");
        return p;
    }
    if (sm_major < 8)
    {
        fa_detail::set_reason(p.reason, sizeof(p.reason),
                              "cp.async needs sm_80+; this device predates it");
        return p;
    }

    p.k_steps = s.d / 8;
    p.n_steps = s.d / 8;
    p.grid_x = (T + p.bm - 1) / p.bm;
    p.grid_y = s.h;
    p.key_tiles = (S + p.bn - 1) / p.bn;
    p.band_tiles = (s.rel_extent + p.bn - 1) / p.bn;
    p.smem_q_upper = fa_q_upper_bytes(s.d);
    p.smem_stage = fa_stage_bytes(s.d);

    // Largest pipeline the SM can hold. Stages shrink before the plan declines, because a
    // single-buffered flash loop still beats the eager plane it replaces — it just waits more.
    const size_t per_sm = device_smem_per_sm ? device_smem_per_sm : (100 * 1024);
    const size_t optin = device_smem_optin ? device_smem_optin : per_sm;
    size_t cap = optin < per_sm ? optin : per_sm;
    if (smem_budget_bytes && smem_budget_bytes < cap) cap = smem_budget_bytes;
    int stages = 0;
    for (int cand = FA_STAGES_PREFERRED; cand >= 1; --cand)
    {
        if (fa_smem_bytes(s.d, cand) <= cap) { stages = cand; break; }
    }
    if (stages == 0)
    {
        p.smem_bytes = fa_smem_bytes(s.d, 1);
        p.smem_per_sm = cap;
        fa_detail::set_reason(p.reason, sizeof(p.reason),
                              "shared memory above the device opt-in limit or the caller budget");
        return p;
    }
    p.stages = stages;
    p.smem_bytes = fa_smem_bytes(s.d, stages);
    p.smem_per_sm = per_sm;
    p.blocks_per_sm = int(per_sm / p.smem_bytes);
    if (p.blocks_per_sm < 1) p.blocks_per_sm = 1;
    p.ok = 1;
    return p;
}

// ---------------------------------------------------------------------------
// Launch surface (torch-free). The tensor layer in fa_prefill_torch.cpp fills this and calls
// inkling_fa_prefill_launch; tests/cpu/fa_prefill_shim/ compiles and runs this same code on a CPU.
// ---------------------------------------------------------------------------

struct FaPrefillArgs
{
    const void* q;        // (h, T, d) fp16, contiguous
    const void* k;        // (S, kvh, d) fp16
    const void* v;        // (S, kvh, d) fp16
    const void* rel;      // (h, T, rel_extent) fp32
    const float* sink;    // (T, h) fp32
    const int64_t* qpos;  // (T,) absolute query positions
    const int64_t* kpos;  // (S,) absolute key positions
    void* o;              // (h, T, d) fp16, contiguous
    int32_t* nfin;        // running count of non-finite outputs, 0 on entry (may be null)
    void* stream;         // the stream the launch goes down (torch's current stream)
    int stages;           // set by the launcher, from the plan
    int grp;              // h / kvh, set by the launcher
    int64_t q_head_stride;   // elements between consecutive q heads of q
    int64_t q_row_stride;    // elements between consecutive q rows of one head
    int64_t k_row_stride;    // elements between consecutive keys (kvh * d for a packed cache)
    int64_t k_head_stride;   // elements between consecutive kv heads
    int64_t rel_head_stride;
    int64_t rel_row_stride;
    int64_t sink_row_stride;       // elements between consecutive sink rows (>= h)
    int64_t o_head_stride;
    int64_t o_row_stride;
    int T, S, h, kvh, d, rel_extent;
    float scale_log2;    // scaling * log2(e): the kernel's softmax runs in the exp2 domain
    float log2e;         // for the bias band and the sink, which are additive logits, not scaled
};

// Decisions the launcher makes and the fence reads back: the plan it honoured, plus what the
// launch actually asked the driver for. Zeroed by the launcher on entry.
struct FaPrefillLaunchReport
{
    int launched;          // 1 = the kernel went down the stream, 0 = declined
    int grid_x, grid_y, threads, stages;
    size_t smem_bytes;     // dynamic shared memory the launch requested
    int64_t blocks;        // grid_x * grid_y
    char reason[112];      // named decline reason when launched == 0
};

// Plan for a pass against the current device's own limits (compute capability, opt-in shared
// memory and per-SM shared memory, read through the CUDA runtime and cached per device). This is
// the entry the Python seam queries before it allocates anything.
FaPrefillPlan inkling_fa_prefill_plan(const FaPrefillShapes& s, int T, int S,
                                      size_t smem_budget_bytes, int device);

// Validate + launch. Plans through inkling_fa_prefill_plan, opts the kernel into the dynamic
// shared memory it asked for, and either launches with the planned configuration or declines with
// a reason. Returns the report; the stream comes from FaPrefillArgs::stream.
FaPrefillLaunchReport inkling_fa_prefill_launch(const FaPrefillArgs& a, size_t smem_budget_bytes);

// Last decline reason of this process, "" if none (for the Python INKLING-FLASHDECLINE line).
const char* inkling_fa_prefill_refusal();