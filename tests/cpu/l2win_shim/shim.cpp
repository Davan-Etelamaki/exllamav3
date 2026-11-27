// Implementation of the CUDA runtime shim. Answers from g_shim, records every call in order.

#include "cuda_runtime.h"
#include "shim.h"

#include <cstdio>
#include <cstring>

ShimCtl g_shim;
ShimRange g_shim_ranges[SHIM_MAX_RANGES];
static int g_shim_range_n = 0;

static cudaError_t g_pending = cudaSuccess;

#define SHIM_LOG_MAX 64
static char g_calls[SHIM_LOG_MAX][256];
static int g_call_n = 0;

void shim_record(const char* call)
{
    if (g_call_n < SHIM_LOG_MAX)
        std::snprintf(g_calls[g_call_n++], sizeof(g_calls[0]), "%s", call);
    std::printf("CALL %s\n", call);
    std::fflush(stdout);
}

void shim_register(uintptr_t base, size_t bytes)
{
    if (g_shim_range_n < SHIM_MAX_RANGES)
    {
        g_shim_ranges[g_shim_range_n++] = ShimRange { base, bytes };
    }
}

void shim_clear_ranges()
{
    g_shim_range_n = 0;
}

int shim_call_count(const char* name)
{
    int n = 0;
    for (int i = 0; i < g_call_n; i++)
        if (std::strncmp(g_calls[i], name, std::strlen(name)) == 0)
            n++;
    return n;
}

const char* shim_last_call(const char* name)
{
    const char* found = "";
    for (int i = 0; i < g_call_n; i++)
        if (std::strncmp(g_calls[i], name, std::strlen(name)) == 0)
            found = g_calls[i];
    return found;
}

cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device)
{
    const char* name = attr == cudaDevAttrL2CacheSize ? "L2CacheSize"
                       : attr == cudaDevAttrMaxAccessPolicyWindowSize ? "MaxAccessPolicyWindowSize" : "?";
    size_t bytes = attr == cudaDevAttrL2CacheSize ? g_shim.l2_bytes
                   : attr == cudaDevAttrMaxAccessPolicyWindowSize ? g_shim.max_window_bytes : 0;
    char line[256];
    std::snprintf(line, sizeof(line), "cudaDeviceGetAttribute attr=%s dev=%d -> %zu", name, device, bytes);
    shim_record(line);
    *value = (int) bytes;
    return cudaSuccess;
}

cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value)
{
    char line[256];
    std::snprintf(line, sizeof(line), "cudaDeviceSetLimit limit=%d bytes=%zu%s",
                  (int) limit, value, g_shim.setlimit_fails ? " -> UNSUPPORTED_LIMIT" : " -> ok");
    shim_record(line);
    if (g_shim.setlimit_fails)
    {
        g_pending = cudaErrorUnsupportedLimit;
        return g_pending;
    }
    return cudaSuccess;
}

cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* attributes, const void* ptr)
{
    uintptr_t addr = (uintptr_t) ptr;
    char line[256];
    for (int i = 0; i < g_shim_range_n; i++)
    {
        if (addr >= g_shim_ranges[i].base && addr < g_shim_ranges[i].base + g_shim_ranges[i].bytes)
        {
            std::snprintf(line, sizeof(line), "cudaPointerGetAttributes 0x%lx -> device=%d",
                          (unsigned long) addr, g_shim.device);
            shim_record(line);
            attributes->type = cudaMemoryTypeDevice;
            attributes->device = g_shim.device;
            attributes->devicePointer = (void*) ptr;
            attributes->hostPointer = nullptr;
            return cudaSuccess;
        }
    }
    std::snprintf(line, sizeof(line), "cudaPointerGetAttributes 0x%lx -> INVALID_VALUE",
                  (unsigned long) addr);
    shim_record(line);
    g_pending = cudaErrorInvalidValue;
    return cudaErrorInvalidValue;
}

cudaError_t cudaStreamSetAttribute(cudaStream_t stream, cudaStreamAttrID attr,
                                   const cudaStreamAttrValue* value)
{
    char line[256];
    std::snprintf(line, sizeof(line),
                  "cudaStreamSetAttribute stream=%p base=0x%lx num_bytes=%zu hit=%.1f "
                  "hitProp=%d missProp=%d%s",
                  stream, (unsigned long) value->accessPolicyWindow.base_ptr,
                  value->accessPolicyWindow.num_bytes, (double) value->accessPolicyWindow.hitRatio,
                  (int) value->accessPolicyWindow.hitProp, (int) value->accessPolicyWindow.missProp,
                  g_shim.setattr_rejects ? " -> INVALID_VALUE" : " -> ok");
    shim_record(line);
    if (g_shim.setattr_rejects)
    {
        g_pending = cudaErrorInvalidValue;
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

cudaError_t cudaGetLastError()
{
    cudaError_t e = g_pending;
    g_pending = cudaSuccess;
    return e;
}