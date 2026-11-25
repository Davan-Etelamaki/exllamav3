// The REAL per-expert weight reference for the Inkling-2 grouped tier, from the served exl3
// tensor representation (exllamav3/modules/quant/exl3.py, LinearEXL3): each quantized Linear
// holds trellis int16 [16, out_features/16, k_width] (rotated basis; k_width = 16*K with K the
// bitrate, half-integral only for the mul1 rates), suh half [in_features], svh half
// [out_features], optional mcg/mul1 codebooks. NO int4/qgroup fields — the draft placeholder
// struct in moe_sm89_common.cuh stays for the mma-machinery smoke path only, until the trellis
// stream replaces it.
//
// Hadamard consequence recorded from LinearEXL3.forward (non-fused path): the effective operand
// chain is  x -> had_r_128(with suh) -> GEMV(rotated trellis) -> had_r_128(with svh) -> out,
// so leg GU must consume hadamard+suh-transformed activations and leg D must apply the svh-side
// hadamard to its accumulator tile (128-wide output blocks; 2304 and 9216 are both 128-divisible).
// This header fixes the TABLE and its ABI fences; the stream substitution is the next commit.
//
// Pure-host header: stdint/cstddef only, so the probe compiles and RUNS with no CUDA runtime
// contact (no cudaMalloc, no kernel launch — the dot26 serving context stays untouched).
#ifndef MOE_EXL3_TABLE_CUH
#define MOE_EXL3_TABLE_CUH

#include <cstdint>
#include <cstddef>

// --- ABI expectation literals (authoritative numbers). The python crosscheck
// tests/cpu/moe_abi_crosscheck.py parses exactly these lines; the static_asserts below tie them
// to the compiled layout. Drift is impossible without the C++ compile or the python test going
// RED. ---
#define MOE_REF_EXPECT_SIZEOF          64
#define MOE_REF_EXPECT_OFF_TRELLIS      0
#define MOE_REF_EXPECT_OFF_SUH          8
#define MOE_REF_EXPECT_OFF_SVH         16
#define MOE_REF_EXPECT_OFF_MCG         24
#define MOE_REF_EXPECT_OFF_MUL1        32
#define MOE_REF_EXPECT_OFF_IN_FEATURES 40
#define MOE_REF_EXPECT_OFF_OUT_FEATURES 44
#define MOE_REF_EXPECT_OFF_K_WIDTH     48
#define MOE_REF_EXPECT_OFF_HAS_MCG     52

struct MoEExl3Ref {
    const int16_t* trellis;    // [16, out_features/16, k_width] rotated-basis trellis (int16)
    const void*    suh;        // half [in_features]  (void* keeps this header free of cuda_fp16)
    const void*    svh;        // half [out_features]
    const void*    mcg;        // uint16 codebook or null
    const void*    mul1;       // uint16 codebook or null (half-integral bitrate only)
    int32_t        in_features;   // 2304 gate/up, 9216 down (served geometry)
    int32_t        out_features;  // 9216 gate/up, 2304 down
    int32_t        k_width;       // trellis.shape[-1]; bitrate K = k_width / 16.0 (may be x.5)
    uint8_t        has_mcg;       // flags bytes at MOE_REF_EXPECT_OFF_HAS_MCG
    uint8_t        has_mul1;
    uint8_t        pad_[6];
};

// Per-projection table: one MoEExl3Ref per expert, three projections — built once at model
// load, 128*3*64 B = 24 KiB per layer. A pointer table, never a stacked weight copy.
#define MOE_EXL3_LAYER_BYTES (128 * 3 * (int)MOE_REF_EXPECT_SIZEOF)

static_assert(sizeof(MoEExl3Ref) == MOE_REF_EXPECT_SIZEOF,              "MoEExl3Ref size drifted from ABI literals");
static_assert(offsetof(MoEExl3Ref, trellis)      == MOE_REF_EXPECT_OFF_TRELLIS,     "trellis offset drift");
static_assert(offsetof(MoEExl3Ref, suh)          == MOE_REF_EXPECT_OFF_SUH,         "suh offset drift");
static_assert(offsetof(MoEExl3Ref, svh)          == MOE_REF_EXPECT_OFF_SVH,         "svh offset drift");
static_assert(offsetof(MoEExl3Ref, mcg)          == MOE_REF_EXPECT_OFF_MCG,         "mcg offset drift");
static_assert(offsetof(MoEExl3Ref, mul1)         == MOE_REF_EXPECT_OFF_MUL1,        "mul1 offset drift");
static_assert(offsetof(MoEExl3Ref, in_features)  == MOE_REF_EXPECT_OFF_IN_FEATURES, "in_features offset drift");
static_assert(offsetof(MoEExl3Ref, out_features) == MOE_REF_EXPECT_OFF_OUT_FEATURES, "out_features offset drift");
static_assert(offsetof(MoEExl3Ref, k_width)      == MOE_REF_EXPECT_OFF_K_WIDTH,     "k_width offset drift");
static_assert(offsetof(MoEExl3Ref, has_mcg)      == MOE_REF_EXPECT_OFF_HAS_MCG,       "flags offset drift");

// Routing-arena capacity formula, shared single source of truth: python sizes the buffers with
// the SAME expression it re-derives on CPU in moe_abi_crosscheck.py, and the probe prints the
// compiled value for the diff. Padding every non-empty expert slice up to BLOCK_M=16 wastes at
// most 15 rows per touched expert:
//     max_padded_rows(entries) = entries + 15 * min(entries, 128)
// Worst case at the decode cap (entries = MOE_MAX_DECODE_ENTRIES = 256): 256 + 15*128 = 2176
// rows, 136 blocks. ALLOCATION OWNER: the python wiring allocates ONE fixed-address arena per
// graph-capture set (per device per layer region: sel/w/sorted_token_ids/expert_for_block/
// counters + act_scratch) at capture-set creation, sized by these formulas; the kernels and
// this lane never allocate mid-forward — Strata's recall lesson: the captured address set
// must be stable across replays (campaign §210 doorbell law).
inline int32_t moe_max_padded_rows(int32_t entries) {
    const int32_t touched = entries < 128 ? entries : 128;
    return entries + 15 * touched;
}
inline int32_t moe_max_blocks(int32_t entries) {
    return (moe_max_padded_rows(entries) + 15) / 16;
}

#endif  // MOE_EXL3_TABLE_CUH