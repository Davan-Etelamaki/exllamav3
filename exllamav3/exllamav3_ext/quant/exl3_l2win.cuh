#pragma once

#include <cstddef>
#include <cstdint>

// Persisting-L2 window for the decode row span (Inkling 2, sm_89 resident).
//
// Two CUDA calls do the whole job: cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize) reserves a
// slice of L2 that persisting accesses may hold, and cudaStreamSetAttribute with
// cudaStreamAttributeAccessPolicyWindow steers one stream's accesses inside one VA range onto
// that slice. Nothing here guesses at a card: the window length comes from the model's served
// shapes, and a carve or an address the device cannot honour is refused with a printed reason
// rather than approximated.
//
// Why the row span deserves a slice: a decode step reads its q/k/v/o rows, the norm row, the
// router row and the PLE row once each, while the routed-expert weights (≈32 MB of weight reads
// per decode token on the served pack) stream through the same L2 and evict it. Splitting the two
// reuse classes — rows persisting on a carved slice, experts evict-first on the default stream —
// is the Xolotl same-card comparator's split-by-reuse-class; the bytes are the ones a decode step
// touches again on its next token, not the ones it discards.
//
// The gate is infer_params.inkling_attn_l2win (default False). Unarmed, exl3_l2win_ctx_carve
// returns 0 before its first CUDA call and exl3_l2win_stream is never reached, so neither call
// happens and the served path keeps its own L2 behaviour.

// Xolotl's carve on the same card (RTX 4090: 72 MB L2 total). A ceiling, not a request:
// l2win_clamp_carve takes the smaller of this and the device's own L2, so a card with less L2
// keeps what it has instead of failing the SetLimit. Written as 99/2 MiB to stay an integer
// constant (49.5 MiB = 51904512 B).
#define L2_CARVE_CEILING_BYTES (size_t(99) * 1024 * 1024 / 2)

// The served shapes the window is sized from: q heads, kv heads, head dim, relative-bias extent.
struct L2WinShapes
{
    int h;
    int kvh;
    int d;
    int rel_extent;
};

// Bytes of the hot span for ONE row: the q/k/v/o projections plus the row's relative-bias band,
// plus the three hidden-wide rows a decode step also revisits (norm, router, PLE). Every term is
// a product of the served shapes and a dtype width; there is no buffer size in here to rot when
// the served geometry changes, which is what makes the sizing safe to compute at ctx time.
inline size_t l2win_row_bytes(const L2WinShapes& s)
{
    const size_t attn_proj = (size_t(s.h) + 2 * size_t(s.kvh)) * size_t(s.d) * 2;  // q,k,v: fp16
    const size_t o_proj = size_t(s.h) * size_t(s.d) * 4;                           // wo.forward: fp32
    const size_t hidden_rows = 3 * size_t(s.h) * size_t(s.d) * 2;                  // norm, router, PLE: fp16
    const size_t rel_band = size_t(s.h) * size_t(s.rel_extent) * 4;                // fp32 bias band
    return attn_proj + o_proj + hidden_rows + rel_band;
}

// Window length for `rows` decode rows: the span the round actually touches, cut down to what the
// carve can hold and to the device's maximum window size. A window longer than the carve would
// advertise persistence for bytes the slice cannot keep.
inline size_t l2win_window_bytes(const L2WinShapes& s, int rows, size_t carve_bytes, size_t max_window_bytes)
{
    size_t span = l2win_row_bytes(s) * size_t(rows > 0 ? rows : 1);
    size_t bytes = span < carve_bytes ? span : carve_bytes;
    return bytes < max_window_bytes ? bytes : max_window_bytes;
}

// Carve the device will accept: the lesser of the requested bytes, the comparator ceiling and the
// device's own L2. Clamping is the whole contract — a request above L2 must not fail the SetLimit.
inline size_t l2win_clamp_carve(size_t requested_bytes, size_t device_l2_bytes)
{
    size_t bytes = requested_bytes < L2_CARVE_CEILING_BYTES ? requested_bytes : L2_CARVE_CEILING_BYTES;
    return bytes < device_l2_bytes ? bytes : device_l2_bytes;
}

// Policy: set once per device by exl3_l2win_arm (Python: gate ON at module load). Armed means the
// device carve is wanted; the shapes are what every later window is sized from.
void exl3_l2win_arm(int device, const L2WinShapes& shapes);
bool exl3_l2win_armed(int device);

// Site 1, called once per device from DevCtx::get_l2win_carve (so it runs at ctx creation).
// Returns the carve actually applied, 0 when unarmed or declined; prints the decline.
size_t exl3_l2win_ctx_carve(int device);
size_t exl3_l2win_carve(int device);                      // applied carve, 0 if none
const char* exl3_l2win_refusal(int device);               // last decline reason, "" if none

// Site 2, called per decode round on the attention/aux stream. Sets the access-policy window over
// `hot_base`'s row span; idempotent per (device, stream, base, length). Returns 1 when the window
// was set, 0 when this stream+span already carries it, -1 when declined (reason via exl3_l2win_refusal).
int64_t exl3_l2win_stream(int device, uintptr_t stream, uintptr_t hot_base, int rows);

// Diagnostics for the Python arm line: the length the next round of `rows` would get.
size_t exl3_l2win_planned_bytes(int device, int rows);