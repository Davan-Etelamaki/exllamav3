#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <mutex>

#include "exl3_l2win.cuh"
#include "exl3_devctx.cuh"

// The two persisting-L2 calls and everything that decides whether to make them. Kept free of
// torch and of the kernel headers so a host compiler can build this file against a recording shim
// (tests/cpu/inkling_l2win_fence.py), which is how the call ORDER is judged off-card: the fence
// executes this translation unit, not a transcription of it.

namespace
{

struct L2WinState
{
    bool armed = false;
    L2WinShapes shapes { 0, 0, 0, 0 };
    bool carve_tried = false;         // one SetLimit attempt per device, success or not
    size_t carve = 0;                 // what the device accepted, 0 = none
    bool max_window_read = false;
    int max_window = 0;               // cudaDevAttrMaxAccessPolicyWindowSize
    uintptr_t set_stream = 0;         // the (stream, base, bytes) this device last carried
    uintptr_t set_base = 0;
    size_t set_bytes = 0;
    char refusal[160] = "";
};

L2WinState g_l2win[MAX_DEVICES];
std::mutex g_l2win_mtx;

void set_refusal(L2WinState& st, const char* reason)
{
    std::snprintf(st.refusal, sizeof(st.refusal), "%s", reason);
}

}   // namespace

#if !defined(USE_ROCM)
// RDNA has no persisting-L2 carve to request, so its entry points at the bottom of this file keep
// the policy unarmed and report a refusal instead of issuing calls the runtime would reject.

void exl3_l2win_arm(int device, const L2WinShapes& shapes)
{
    std::lock_guard<std::mutex> lock(g_l2win_mtx);
    L2WinState& st = g_l2win[device];
    st.armed = true;
    st.shapes = shapes;
    st.carve_tried = false;           // a re-arm (different shapes) re-carves
    st.carve = 0;
    st.set_stream = st.set_base = 0;
    st.set_bytes = 0;
    st.refusal[0] = '\0';
}

bool exl3_l2win_armed(int device)
{
    std::lock_guard<std::mutex> lock(g_l2win_mtx);
    return g_l2win[device].armed;
}

size_t exl3_l2win_carve(int device)
{
    std::lock_guard<std::mutex> lock(g_l2win_mtx);
    return g_l2win[device].carve;
}

const char* exl3_l2win_refusal(int device)
{
    std::lock_guard<std::mutex> lock(g_l2win_mtx);
    return g_l2win[device].refusal;
}

size_t exl3_l2win_ctx_carve(int device)
{
    std::lock_guard<std::mutex> lock(g_l2win_mtx);
    L2WinState& st = g_l2win[device];
    if (!st.armed)
        return 0;                     // gate OFF: no attribute query, no SetLimit, no print
    if (st.carve_tried)
        return st.carve;              // one attempt per device: a refusal is not retried every ctx
    st.carve_tried = true;

    int l2_bytes = 0;
    cudaError_t e = cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, device);
    if (e != cudaSuccess || l2_bytes <= 0)
    {
        cudaGetLastError();           // a failed probe must not leave the context poisoned
        set_refusal(st, "cudaDevAttrL2CacheSize unavailable");
        printf("INKLING-L2WIN DECLINED device=%d: %s\n", device, st.refusal);
        return 0;
    }

    // The ceiling is a request the device may refuse by being smaller; clamping is why the
    // SetLimit below cannot fail on size. A card with 96 MB of L2 still gets 49.5 MiB, which is
    // the comparator's carve, so the two arms stay comparable.
    size_t carve = l2win_clamp_carve(L2_CARVE_CEILING_BYTES, size_t(l2_bytes));
    e = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, carve);
    if (e != cudaSuccess)
    {
        cudaGetLastError();
        set_refusal(st, "cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize) rejected");
        printf("INKLING-L2WIN DECLINED device=%d: %s\n", device, st.refusal);
        return 0;
    }
    st.carve = carve;
    return carve;
}

size_t exl3_l2win_planned_bytes(int device, int rows)
{
    std::lock_guard<std::mutex> lock(g_l2win_mtx);
    L2WinState& st = g_l2win[device];
    if (!st.armed || !st.carve)
        return 0;
    return l2win_window_bytes(st.shapes, rows, st.carve, size_t(st.max_window > 0 ? st.max_window : ~0u));
}

int64_t exl3_l2win_stream(int device, uintptr_t stream, uintptr_t hot_base, int rows)
{
    std::lock_guard<std::mutex> lock(g_l2win_mtx);
    L2WinState& st = g_l2win[device];
    if (!st.armed || !st.carve)
    {
        set_refusal(st, st.armed ? "device carve declined" : "policy not armed");
        return -1;
    }
    if (!stream || !hot_base || rows <= 0)
    {
        set_refusal(st, "missing stream handle, hot base or rows");
        printf("INKLING-L2WIN DECLINED device=%d: %s\n", device, st.refusal);
        return -1;
    }

    if (!st.max_window_read)
    {
        // The device's own limit on window length; a zero or a failed read means "no bound known",
        // which the sizing treats as unbounded rather than as a reason to give up.
        st.max_window_read = true;
        if (cudaDeviceGetAttribute(&st.max_window, cudaDevAttrMaxAccessPolicyWindowSize, device) != cudaSuccess)
        {
            cudaGetLastError();
            st.max_window = 0;
        }
    }

    size_t bytes = l2win_window_bytes(st.shapes, rows, st.carve,
                                      st.max_window > 0 ? size_t(st.max_window) : ~size_t(0));
    if (!bytes || hot_base > ~size_t(0) - bytes)
    {
        set_refusal(st, "window span is empty or overflows the address space");
        printf("INKLING-L2WIN DECLINED device=%d: %s\n", device, st.refusal);
        return -1;
    }
    if (stream == st.set_stream && hot_base == st.set_base && bytes == st.set_bytes)
        return 0;                     // this stream already carries exactly this window

    // Both ends of the span must be device memory on this device. cudaPointerGetAttributes answers
    // cudaErrorInvalidValue for an address in no allocation at all, which is how a window whose
    // tail walks past the end of the live VA gets refused here rather than at the driver. The full
    // containment check (cuMemGetAddressRange) is deliberately NOT used: with expandable segments
    // the driver reports no address range for pointers it is happy to serve, so it declines windows
    // that are actually fine.
    for (uintptr_t probe : { hot_base, hot_base + bytes - 1 })
    {
        cudaPointerAttributes attr;
        cudaError_t e = cudaPointerGetAttributes(&attr, (const void*) probe);
        if (e != cudaSuccess || attr.type != cudaMemoryTypeDevice || attr.device != device)
        {
            cudaGetLastError();
            set_refusal(st, "window span crosses unallocated VA");
            printf("INKLING-L2WIN DECLINED device=%d: %s\n", device, st.refusal);
            return -1;
        }
    }

    cudaStreamAttrValue val;
    std::memset(&val, 0, sizeof(val));
    val.accessPolicyWindow.base_ptr = (void*) hot_base;
    val.accessPolicyWindow.num_bytes = bytes;
    val.accessPolicyWindow.hitRatio = 1.0f;
    val.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    val.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;   // everything outside: evict-first

    cudaError_t e = cudaStreamSetAttribute((cudaStream_t) stream, cudaStreamAttributeAccessPolicyWindow, &val);
    if (e != cudaSuccess)
    {
        // A window the driver will not take leaves the stream as it found it: the round runs with
        // ordinary L2 behaviour, the engine stays eager, and the reason is printed once per shape.
        cudaGetLastError();
        set_refusal(st, "cudaStreamSetAttribute(accessPolicyWindow) rejected");
        printf("INKLING-L2WIN DECLINED device=%d: %s\n", device, st.refusal);
        return -1;
    }
    st.set_stream = stream;
    st.set_base = hot_base;
    st.set_bytes = bytes;
    return 1;
}

#else   // USE_ROCM

void exl3_l2win_arm(int device, const L2WinShapes&)
{
    std::lock_guard<std::mutex> lock(g_l2win_mtx);
    g_l2win[device].armed = false;
    set_refusal(g_l2win[device], "persisting L2 is not implemented for this backend");
    printf("INKLING-L2WIN DECLINED device=%d: %s\n", device, g_l2win[device].refusal);
}

bool exl3_l2win_armed(int) { return false; }
size_t exl3_l2win_carve(int) { return 0; }
size_t exl3_l2win_ctx_carve(int) { return 0; }
size_t exl3_l2win_planned_bytes(int, int) { return 0; }
const char* exl3_l2win_refusal(int) { return ""; }
int64_t exl3_l2win_stream(int, uintptr_t, uintptr_t, int) { return -1; }

#endif  // USE_ROCM