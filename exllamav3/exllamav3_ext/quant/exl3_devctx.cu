#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cstdio>
namespace cg = cooperative_groups;
#include "exl3_devctx.cuh"
#include "exl3_l2win.cuh"
#include "../util.h"
#include "../util.cuh"

//DevCtx::DevCtc()
//{
//    int num_sms[MAX_DEVICES] = {};
//    int cc[MAX_DEVICES] = {};
//    void* locks[MAX_DEVICES] = {};
//    std::mutex mtx;
//}

DevCtx& DevCtx::instance()
{
    static DevCtx ctx;
    return ctx;
}

int DevCtx::get_num_sms(int device)
{
    std::lock_guard<std::mutex> lock(mtx);
    if (!num_sms[device])
        cuda_check(cudaDeviceGetAttribute(&num_sms[device], cudaDevAttrMultiProcessorCount, device));
    return num_sms[device];
}

int DevCtx::get_cc(int device)
{
    std::lock_guard<std::mutex> lock(mtx);
    if (!cc[device])
    {
#if defined(USE_ROCM)
        // HIP reports the gfx generation as the major version (gfx11 -> 11), which the NVIDIA thresholds
        // below would read as Blackwell. Until RDNA devices get classes of their own, they take the
        // conservative host-side choices of the oldest class (the device code is unaffected: the
        // EXL3_SM75 gate in arch.cuh is compile-time and off for HIP)
        cc[device] = CC_OLD;
#else
        cudaDeviceProp prop;
        cuda_check(cudaGetDeviceProperties(&prop, device));
        if (prop.major >= 10) cc[device] = CC_BLACKWELL;
        else if (prop.major >= 9) cc[device] = CC_HOPPER;
        else if (prop.major >= 8 && prop.minor >= 9) cc[device] = CC_ADA;
        else if (prop.major >= 8) cc[device] = CC_AMPERE;
        else cc[device] = CC_OLD;
#endif
    }
    return cc[device];
}

int DevCtx::get_smem_max(int device)
{
    std::lock_guard<std::mutex> lock(mtx);
    if (!smem_max[device])
    {
        // The driver value, unclamped: this is a device capability, and callers that only need
        // "what may I request for an EXL3 kernel" clamp to SMEM_MAX themselves (see
        // get_smem_request). Clamping here would have made an sm_86 device indistinguishable
        // from a 90 KB one and dragged it onto the Turing code paths.
        int optin = 0;
        cuda_check(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
        smem_max[device] = optin;
    }
    return smem_max[device];
}

int DevCtx::get_smem_request(int device)
{
    return MIN(get_smem_max(device), EXL3_SMEM_MAX_DEFAULT);
}

size_t DevCtx::get_l2win_carve(int device)
{
    std::lock_guard<std::mutex> lock(mtx);
    // One owner for the carve. The l2win state memoizes a single SetLimit per device and clears that
    // memo when a genuine re-arm changes the shapes; caching a second copy here let layer 2's re-arm
    // zero the live carve while this cache kept answering "carved", so every decode round on the card
    // declined (dot26 boot 190495, 2026-11-27: 24 armed lines, zero windows). Read from the owner.
    l2win_carve[device] = exl3_l2win_ctx_carve(device);
    return l2win_carve[device];
}

void* DevCtx::get_ws(int device)
{
    std::lock_guard<std::mutex> lock(mtx);
    if (!ws[device])
    {
        c10::cuda::CUDAGuard guard(device);
        cudaError_t e = cudaMalloc(&ws[device], WORKSPACE_SIZE);
        TORCH_CHECK(e == cudaSuccess, "exl3 workspace allocation failed on device ", device, ": ", cudaGetErrorString(e));
    }
    return ws[device];
}

int* DevCtx::get_locks(int device)
{
    std::lock_guard<std::mutex> lock(mtx);
    if (!locks[device])
    {
        c10::cuda::CUDAGuard guard(device);
#if defined(USE_ROCM)
        size_t size = (MAX_TILES_C + MAX_BARRIERS * 2 + MOE_SCHED_INTS + EXL3_GRID_BARRIER_INTS) * sizeof(int);
#else
        size_t size = (MAX_TILES_C + MAX_BARRIERS * 2 + MOE_SCHED_INTS) * sizeof(int);
#endif
        cudaError_t e = cudaMalloc(&locks[device], size);
        TORCH_CHECK(e == cudaSuccess, "exl3 lock buffer allocation failed on device ", device, ": ", cudaGetErrorString(e));
        e = cudaMemset(locks[device], 0, size);
        TORCH_CHECK(e == cudaSuccess, "exl3 lock buffer memset failed: ", cudaGetErrorString(e));
    }
    return (int*) locks[device];
}

int g_get_cc(int device)
{
    return DevCtx::instance().get_cc(device);
}

int g_get_num_sms(int device)
{
    return DevCtx::instance().get_num_sms(device);
}

int g_get_smem_max(int device)
{
    return DevCtx::instance().get_smem_max(device);
}

size_t g_get_l2win_carve(int device)
{
    return DevCtx::instance().get_l2win_carve(device);
}

void prepare_ctx(int device)
{
    DevCtx::instance().get_num_sms(device);
    DevCtx::instance().get_cc(device);
    DevCtx::instance().get_smem_max(device);
    DevCtx::instance().get_locks(device);
    // Site 1 of the persisting-L2 window: the carve is a per-device limit taken against this
    // context, so it belongs beside the buffers it protects. While inkling_attn_l2win is off the
    // policy is unarmed and exl3_l2win_ctx_carve returns before its first CUDA call — no attribute
    // query, no SetLimit, no served-path delta.
    DevCtx::instance().get_l2win_carve(device);
}

namespace
{

// Capability query for the banner: -1 when the device cannot answer, and the failed call's error is
// consumed here so a later cudaGetLastError() (torch's launch checks) does not attribute it to an
// unrelated call. Deliberately not cuda_check: a banner must never abort the import it runs in.
long long banner_attr(int device, cudaDeviceAttr attr)
{
    int value = 0;
    if (cudaDeviceGetAttribute(&value, attr, device) != cudaSuccess)
    {
        cudaGetLastError();
        return -1;
    }
    return (long long) value;
}

}   // namespace

void exl3_device_banner(void)
{
    static bool printed = false;
    if (printed)
        return;
    printed = true;

    // The device this process is bound to. At module init no model has been placed yet, so this is
    // the ordinal CUDA defaults to; a box that has already chosen one gets its own card's numbers.
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
    {
        cudaGetLastError();
        device = 0;
    }

    long long l2_bytes = -1;
    int major = -1, minor = -1;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) == cudaSuccess)
    {
        l2_bytes = (long long) prop.l2CacheSize;
        major = prop.major;
        minor = prop.minor;
    }
    else
        cudaGetLastError();

    // The two ceilings the persisting-L2 carve and the access-policy window are bounded by. Asked
    // of the device rather than assumed: a carve arithmetic claim is only as good as these, and
    // until they were read nothing on this card printed them. HIP exposes neither, so there they
    // report -1 rather than failing the compile.
#if defined(USE_ROCM)
    const long long persisting_max = -1;
    const long long policy_window_max = -1;
#else
    const long long persisting_max = banner_attr(device, cudaDevAttrMaxPersistingL2CacheSize);
    const long long policy_window_max = banner_attr(device, cudaDevAttrMaxAccessPolicyWindowSize);
#endif
    // cudaDevAttrMaxSharedMemoryPerBlockOptin is the driver's own spelling of the opt-in shared
    // memory per block (there is no cudaDevAttrSharedMemPerBlockOptin); it is what get_smem_max reads.
    const long long smem_optin = banner_attr(device, cudaDevAttrMaxSharedMemoryPerBlockOptin);
    const long long regs_per_block = banner_attr(device, cudaDevAttrMaxRegistersPerBlock);

    printf("INKLING-DEVICEPROP l2CacheSize=%lld persistingMax=%lld accessPolicyWindowMax=%lld sm=%d%d sharedMemPerBlockOptin=%lld regsPerBlock=%lld\n",
           l2_bytes, persisting_max, policy_window_max, major, minor, smem_optin, regs_per_block);
    // The banner's whole purpose is boot-time grounding in the journal. stdout is fully
    // buffered when the engine runs under systemd (not a tty), so without this flush the
    // line sat in the C-runtime buffer and NEVER surfaced (measured: INKLING-DEVICEPROP
    // 0 lines since boot on pid 394346 while the .so provably carries the printf).
    fflush(stdout);
}
