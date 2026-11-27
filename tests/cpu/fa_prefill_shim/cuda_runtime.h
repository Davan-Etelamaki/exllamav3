#pragma once

// Host shim for <cuda_runtime.h>, so tests/cpu/inkling_fa_prefill_fence.py can COMPILE AND RUN the
// production translation unit exllamav3/ext/inkling/fa_prefill.cu on a CPU box. Two layers live
// here: the SIMT execution model (thread indices, barriers, shared memory, the fp16 type) and the
// handful of runtime entry points the launcher calls.
//
// The surface is deliberately small. If a future edit makes the kernel or the launcher depend on
// another CUDA entry point or type, this build fails loudly instead of quietly judging less than
// the code does. Names and enumerator values follow the CUDA 12/13 headers, so the shim cannot
// drift from the real ones unnoticed.

#include <cstddef>
#include <cstdint>
#include <cmath>

// ------------------------------------------------------------------ execution model

#define __global__
#define __device__
#define __host__
#define __forceinline__
#define __restrict__
#define __shared__ static
#define __launch_bounds__(...)
#define __align__(n) alignas(n)

typedef uint16_t half;               // the kernel only ever does pointer arithmetic on it; the
typedef uint16_t __half;             // fp16 arithmetic lives in the instruction models in fa_emu.cpp
typedef uint16_t half2;

struct uint4 { uint32_t x, y, z, w; };
struct float2 { float x, y; };
struct Dim3 { uint32_t x, y, z; };
typedef Dim3 dim3;

typedef void* cudaStream_t;

Dim3 em_thread_idx();
Dim3 em_block_idx();
Dim3 em_block_dim();
Dim3 em_grid_dim();
#define threadIdx (::em_thread_idx())
#define blockIdx (::em_block_idx())
#define blockDim (::em_block_dim())
#define gridDim (::em_grid_dim())

void em_syncthreads();
void em_syncwarp();
#define __syncthreads() ::em_syncthreads()
#define __syncwarp(...) ::em_syncwarp()
#define __syncthreads_and() (::em_syncthreads(), 1)

inline uint4 make_uint4(uint32_t a, uint32_t b, uint32_t c, uint32_t d)
{
    uint4 r; r.x = a; r.y = b; r.z = c; r.w = d; return r;
}

// exp2f / fminf / fmaxf / isfinite come from <cmath> on the host; alias them into the names the
// kernel uses.
#ifndef exp2f
#define exp2f exp2f
#endif

// ------------------------------------------------------------------ runtime API surface

typedef enum
{
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1
} cudaError_t;

typedef enum
{
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81
} cudaDeviceAttr;

typedef enum
{
    cudaFuncAttributeMaxDynamicSharedMemorySize = 8
} cudaFuncAttribute;

cudaError_t cudaGetDevice(int* device);
cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device);
cudaError_t cudaFuncSetAttribute(const void* func, cudaFuncAttribute attr, int amount);
cudaError_t cudaGetLastError();