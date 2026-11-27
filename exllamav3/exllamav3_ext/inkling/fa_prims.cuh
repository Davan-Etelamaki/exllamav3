#pragma once

// Tensor-core and async-copy primitives of the Inkling 2 flash-prefill kernel (sm_89), and the
// host emulations that let tests/cpu/inkling_fa_prefill_fence.py EXECUTE the production kernel on
// a CPU box. One source, two back ends: nvcc compiles the PTX forms, the fence's g++ build takes
// the emulated forms, and the kernel body between them is the same code.
//
// The emulations are not abbreviations of the instructions: they model the fragment layouts the
// PTX ISA defines for mma.sync.aligned.m16n8k8 (A row-major 16x8, B column-major 8x8, D 16x8 with
// lane = (row>>0 …) ownership as written out in fa_emu.cpp), the 8x8 matrix distribution of
// ldmatrix(.trans), and cp.async's 16-byte global->shared copy. Cross-lane movement is done with
// the shim's cooperative scheduler, so a host run of the kernel is a real SIMT execution, not a
// scalar rewrite of the algorithm.

#include <cstdint>

#if defined(__CUDACC__)

#include <cuda_fp16.h>

#define FA_DEV __device__ __forceinline__

FA_DEV uint32_t fa_smem_u32(const void* p)
{
    return (uint32_t) __cvta_generic_to_shared(p);
}

// 16-byte global -> shared copy. With pred false the copy still lands 16 bytes, but with
// src-size 0, which the ISA defines as ZERO-FILLED: a tile row that falls off the end of the
// tensor therefore arrives as zeros rather than as shared-memory garbage, and a garbage NaN could
// otherwise reach the fp16 P V product as 0 * NaN = NaN and poison the output.
FA_DEV void fa_cp_async16(void* smem_dst, const void* src, bool pred)
{
    const uint32_t dst = fa_smem_u32(smem_dst);
    asm volatile("{ .reg .pred p; setp.ne.b32 p, %2, 0;"
                 " @p cp.async.cg.shared.global [%0], [%1], 16, 16;"
                 " @!p cp.async.cg.shared.global [%0], [%1], 16, 0; }" ::
                 "r"(dst), "l"(src), "r"((int) pred));
}

FA_DEV void fa_cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
FA_DEV void fa_cp_async_wait_all() { asm volatile("cp.async.wait_all;\n" ::); }

FA_DEV void fa_ldsm_x4(uint32_t r[4], uint32_t saddr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(saddr));
}

FA_DEV void fa_ldsm_x4_t(uint32_t r[4], uint32_t saddr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(saddr));
}

// S = Q K^T: fp32 accumulate, fp16 inputs (fp32-accumulate HMMA peak is the rate this kernel bills
// against; the same instruction fa75 runs on hardware for these shapes).
FA_DEV void fa_mma_1688_f32(float c[4], uint32_t a0, uint32_t a1, uint32_t b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(b));
}

// P V: fp16 accumulate over ONE 16-key slice, promoted to fp32 afterwards. The slice bound is the
// contract that keeps the estate's fp16-PV overflow (65504 at ~63k keys, fixed by 4cb7529a) from
// coming back: p = exp2(s - m) <= 1, so a 16-key partial is bounded by 16 * max|v|, two orders
// under fp16's ceiling for a normed V, and the un-normalised growth lives only in the fp32 O.
FA_DEV void fa_mma_1688_f16(uint32_t c[2], uint32_t a0, uint32_t a1, uint32_t b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%0,%1};\n"
        : "+r"(c[0]), "+r"(c[1]) : "r"(a0), "r"(a1), "r"(b));
}

FA_DEV uint32_t fa_pack_h2(float x, float y)
{
    half2 h = __floats2half2_rn(x, y);
    return *reinterpret_cast<uint32_t*>(&h);
}

FA_DEV float2 fa_h2_to_f2(uint32_t h)
{
    return __half22float2(*reinterpret_cast<half2*>(&h));
}

FA_DEV float fa_shfl_xor(float v, int lane_mask)
{
    return __shfl_xor_sync(0xffffffffu, v, lane_mask);
}

FA_DEV void fa_zero_h2(uint32_t* p, uint32_t v) { *p = v; }

// Finiteness of a stored output. Device: CUDA's isfinite; host: std::isfinite.
FA_DEV bool fa_is_finite(float x) { return isfinite(x); }

#else  // ---------------------------------------------------------- host emulation

 #include <cmath>
#include <cstddef>
#include <cuda_runtime.h>   // the fence shim's: float2 and the fp16 typedefs (its -I comes first)

struct FaPrefillArgs;

// Provided by tests/cpu/fa_prefill_shim/fa_emu.cpp. Each call executes on one simulated thread of
// the cooperative block scheduler (ucontext), so the cross-lane ones publish and yield.
uint8_t* em_smem_base();
void em_launch_kernel(void (*fn)(FaPrefillArgs), const FaPrefillArgs* args,
                      int grid_x, int grid_y, int threads, size_t smem_bytes);
uint32_t em_smem_addr(const void* p);
void em_cp_async16(void* smem_dst, const void* src, bool pred);
void em_ldsm_x4(uint32_t out[4], uint32_t saddr, int transpose);
void em_mma_1688_f32(float c[4], uint32_t a0, uint32_t a1, uint32_t b);
void em_mma_1688_f16(uint32_t c[2], uint32_t a0, uint32_t a1, uint32_t b);
uint32_t em_pack_h2(float x, float y);
float2 em_h2_to_f2(uint32_t h);
float em_shfl_xor(float v, int lane_mask);
void em_syncthreads();
void em_syncwarp();
void em_atomic_add_i32(int32_t* p, int32_t v);

#define FA_DEV inline

inline uint32_t fa_smem_u32(const void* p) { return em_smem_addr(p); }
inline void fa_cp_async16(void* d, const void* s, bool pred) { em_cp_async16(d, s, pred); }
inline void fa_cp_async_commit() {}
inline void fa_cp_async_wait_all() {}
inline void fa_ldsm_x4(uint32_t r[4], uint32_t a) { em_ldsm_x4(r, a, 0); }
inline void fa_ldsm_x4_t(uint32_t r[4], uint32_t a) { em_ldsm_x4(r, a, 1); }
inline void fa_mma_1688_f32(float c[4], uint32_t a0, uint32_t a1, uint32_t b)
{
    em_mma_1688_f32(c, a0, a1, b);
}
inline void fa_mma_1688_f16(uint32_t c[2], uint32_t a0, uint32_t a1, uint32_t b)
{
    em_mma_1688_f16(c, a0, a1, b);
}
inline uint32_t fa_pack_h2(float x, float y) { return em_pack_h2(x, y); }
inline float2 fa_h2_to_f2(uint32_t h) { return em_h2_to_f2(h); }
inline float fa_shfl_xor(float v, int lane_mask) { return em_shfl_xor(v, lane_mask); }
inline void fa_zero_h2(uint32_t* p, uint32_t v) { *p = v; }
inline bool fa_is_finite(float x) { return std::isfinite(x); }

#endif  // __CUDACC__

constexpr float FA_LOG2E = 1.4426950408889634f;