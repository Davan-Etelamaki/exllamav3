#pragma once

// Tensor layer of the Inkling 2 sm_89 flash-prefill path (infer_params.inkling_attn_flashprefill).
// This is the only part of the lane that knows about at::Tensor: it reads shapes, dtypes and
// strides out of the caller's tensors, refuses anything the kernel cannot serve with a named
// reason (never a throw — the caller falls back to the fused path), and hands the plain-pointer
// argument struct to the launcher in fa_prefill.cu.

#include <ATen/Tensor.h>

#include <cstdint>
#include <map>
#include <string>

// Derive the plan for one pass against the given device's own limits. The returned map carries every
// number the launcher would use — ok, bm, bn, warps, threads, stages, grp, grid_x, grid_y, key_tiles,
// band_tiles, smem_bytes, smem_q_upper, smem_stage, blocks_per_sm, smem_per_sm — so the caller can
// decline BEFORE it allocates anything, and so a fence can pin each derived value. When ok is 0 the
// reason is in exl3_inkling_fa_reason().
std::map<std::string, int64_t> exl3_inkling_fa_plan(int64_t h, int64_t kvh, int64_t d,
                                                    int64_t rel_extent, int64_t T, int64_t S,
                                                    int64_t smem_budget, int64_t device);

// Validate and launch one pass: q (h, T, d) fp16, k/v (S, kvh, d) fp16, rel (h, T, rel_extent) fp32,
// sink (T, h) fp32, qpos (T,) / kpos (S,) int64 absolute positions, o (h, T, d) fp16 written, nfin
// int32 accumulator of non-finite outputs (may be an empty tensor). Returns 1 when the kernel went
// down the current stream, 0 when this pass was declined (reason: exl3_inkling_fa_reason()).
int64_t exl3_inkling_fa_prefill(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor rel,
                                at::Tensor sink, at::Tensor qpos, at::Tensor kpos,
                                at::Tensor o, at::Tensor nfin,
                                double scale, int64_t rel_extent, int64_t smem_budget);

// Last decline reason of this process (plan or launch), "" when the last call succeeded.
std::string exl3_inkling_fa_reason();