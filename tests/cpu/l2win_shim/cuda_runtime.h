#pragma once

// Host shim for the CUDA runtime API. It exists so tests/cpu/inkling_l2win_fence.py can COMPILE
// AND RUN the production translation unit quant/exl3_l2win.cu on a CPU box: every call that file
// makes is recorded in order and answered from a scenario the driver picks, so the fence judges the
// real call sequence rather than a transcription of it.
//
// Only the surface the two persisting-L2 sites touch is declared here. That is deliberate: when a
// future edit makes exl3_l2win.cu depend on another CUDA entry point, this build fails loudly
// instead of silently judging less than the code does. Field names and enum spellings follow
// CUDA 11/12 (cuda_runtime.h) so the shim cannot drift from the real header unnoticed.

#include <cstddef>
#include <cstdint>

typedef enum
{
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorUnsupportedLimit = 16
} cudaError_t;

typedef enum
{
    cudaLimitStackSize = 0,
    cudaLimitPersistingL2CacheSize = 6
} cudaLimit;

typedef enum
{
    cudaDevAttrL2CacheSize = 38,
    cudaDevAttrMaxAccessPolicyWindowSize = 109
} cudaDeviceAttr;

typedef enum
{
    cudaMemoryTypeUnregistered = 0,
    cudaMemoryTypeHost = 1,
    cudaMemoryTypeDevice = 2,
    cudaMemoryTypeManaged = 3
} cudaMemoryType;

typedef struct
{
    cudaMemoryType type;
    int device;
    void* devicePointer;
    void* hostPointer;
} cudaPointerAttributes;

typedef void* cudaStream_t;

typedef enum
{
    cudaStreamAttributeAccessPolicyWindow = 1
} cudaStreamAttrID;

typedef enum
{
    cudaAccessPropertyNormal = 0,
    cudaAccessPropertyStreaming = 1,
    cudaAccessPropertyPersisting = 2
} cudaAccessProperty;

typedef struct
{
    void* base_ptr;
    size_t num_bytes;
    float hitRatio;
    cudaAccessProperty hitProp;
    cudaAccessProperty missProp;
} cudaAccessPolicyWindow;

typedef union
{
    cudaAccessPolicyWindow accessPolicyWindow;
} cudaStreamAttrValue;

cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device);
cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value);
cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* attributes, const void* ptr);
cudaError_t cudaStreamSetAttribute(cudaStream_t stream, cudaStreamAttrID attr,
                                   const cudaStreamAttrValue* value);
cudaError_t cudaGetLastError();