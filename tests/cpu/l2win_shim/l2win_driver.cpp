// Drives quant/exl3_l2win.cu under the CUDA runtime shim. One scenario per process, chosen by
// argv[1]; controls come as key=value arguments. Prints `CALL ...` (from the shim) and
// `RESULT key=value` lines that tests/cpu/inkling_l2win_fence.py parses. Nothing here reimplements
// the policy: every decision it reports is one the production file made.

#include "exl3_l2win.cuh"
#include "shim.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static uintptr_t g_base = 0x700000000000ull;    // stand-in for the hot row span's device address
static size_t g_span = 128ull * 1024 * 1024;     // the allocation that range covers
static int g_rows = 4;
static int g_h = 24, g_kvh = 2, g_d = 256, g_rel = 1024;

static void* STREAM = (void*) 0x12345ull;        // torch owns stream creation; the shim never unwraps it

static size_t kb(const char* v) { return strtoull(v, nullptr, 10); }

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::printf("RESULT error=no-scenario\n");
        return 2;
    }
    const std::string scene = argv[1];
    for (int i = 2; i < argc; i++)
    {
        const char* a = argv[i];
        if (!strncmp(a, "l2=", 3)) g_shim.l2_bytes = kb(a + 3);
        else if (!strncmp(a, "maxwin=", 7)) g_shim.max_window_bytes = kb(a + 7);
        else if (!strcmp(a, "setlimit_fail")) g_shim.setlimit_fails = true;
        else if (!strcmp(a, "setattr_reject")) g_shim.setattr_rejects = true;
        else if (!strncmp(a, "base=", 5)) g_base = (uintptr_t) strtoull(a + 5, nullptr, 16);
        else if (!strncmp(a, "span=", 5)) g_span = kb(a + 5);
        else if (!strncmp(a, "rows=", 5)) g_rows = (int) kb(a + 5);
        else if (!strncmp(a, "h=", 2)) g_h = (int) kb(a + 2);
        else if (!strncmp(a, "kvh=", 3)) g_kvh = (int) kb(a + 3);
        else if (!strncmp(a, "d=", 2)) g_d = (int) kb(a + 2);
        else if (!strncmp(a, "rel=", 4)) g_rel = (int) kb(a + 4);
    }

    const L2WinShapes shapes { g_h, g_kvh, g_d, g_rel };

    if (scene == "gate_off")
    {
        // Exactly what prepare_ctx does on a served boot with the gate off, and what a decode round
        // would do: the policy was never armed, so neither CUDA call may appear.
        size_t carve = exl3_l2win_ctx_carve(0);
        int64_t rc = exl3_l2win_stream(0, (uintptr_t) STREAM, g_base, g_rows);
        std::printf("RESULT carve=%zu stream_rc=%lld refusal=%s setlimit=%d setattr=%d\n",
                    carve, (long long) rc, exl3_l2win_refusal(0),
                    shim_call_count("cudaDeviceSetLimit"), shim_call_count("cudaStreamSetAttribute"));
        return 0;
    }

    if (scene == "arm_ok")
    {
        exl3_l2win_arm(0, shapes);
        size_t carve = exl3_l2win_ctx_carve(0);
        shim_register(g_base, g_span);                  // the allocation the hot rows live in
        int64_t first = exl3_l2win_stream(0, (uintptr_t) STREAM, g_base, g_rows);
        int64_t again = exl3_l2win_stream(0, (uintptr_t) STREAM, g_base, g_rows);
        int64_t grew = exl3_l2win_stream(0, (uintptr_t) STREAM, g_base, g_rows * 2);
        std::printf("RESULT carve=%zu first=%lld again=%lld grew=%lld setlimit=%d setattr=%d "
                    "planned=%zu\n",
                    carve, (long long) first, (long long) again, (long long) grew,
                    shim_call_count("cudaDeviceSetLimit"), shim_call_count("cudaStreamSetAttribute"),
                    exl3_l2win_planned_bytes(0, g_rows));
        std::printf("RESULT last_window=%s\n", shim_last_call("cudaStreamSetAttribute"));
        return 0;
    }

    if (scene == "rearm_layers")
    {
        // The field sequence, layer for layer: a model arms the SAME window once per attention
        // layer (24 times on the served pack) before any decode round runs. An earlier revision
        // reset the carve on every arm while the caller kept the bytes it had already been
        // promised, so the card printed 24 "armed" lines and then refused every round with
        // "device carve declined" (dot26 boot 190495, 2026-11-27). The window must survive the
        // re-arms, and a genuinely different shape must still re-carve.
        exl3_l2win_arm(0, shapes);
        size_t first = exl3_l2win_ctx_carve(0);
        int layers_zeroed = 0;
        for (int layer = 1; layer < 24; layer++)
        {
            exl3_l2win_arm(0, shapes);
            if (exl3_l2win_ctx_carve(0) == 0)
                layers_zeroed++;
        }
        shim_register(g_base, g_span);
        int64_t rc = exl3_l2win_stream(0, (uintptr_t) STREAM, g_base, g_rows);
        const char* refusal_after_round = exl3_l2win_refusal(0);
        char saved[160];
        std::snprintf(saved, sizeof(saved), "%s", refusal_after_round);
        // Counted HERE, before the deliberate different-shape re-arm below: 24 arms of one model must
        // cost exactly one cudaDeviceSetLimit. That count is the discriminator for the reset-on-every-
        // arm defect — without the idempotence guard each layer's arm clears carve_tried and the next
        // carve request goes back to the driver, so a served boot would spend 24 SetLimit calls.
        const int setlimit_after_24 = shim_call_count("cudaDeviceSetLimit");
        const L2WinShapes other { g_h, g_kvh * 2, g_d, g_rel };
        exl3_l2win_arm(0, other);
        size_t after_diff = exl3_l2win_ctx_carve(0);
        std::printf("RESULT carve=%zu layers_zeroed=%d stream_rc=%lld refusal=%s "
                    "carve_rearm=%zu setlimit=%d setlimit_after_24=%d setattr=%d\n",
                    first, layers_zeroed, (long long) rc, saved, after_diff,
                    shim_call_count("cudaDeviceSetLimit"), setlimit_after_24,
                    shim_call_count("cudaStreamSetAttribute"));
        return 0;
    }

    if (scene == "va_crossing")
    {
        // The window's tail walks past the end of the live allocation: the span is registered only
        // over its first row's worth of bytes.
        exl3_l2win_arm(0, shapes);
        size_t carve = exl3_l2win_ctx_carve(0);
        shim_register(g_base, 1024);
        int64_t rc = exl3_l2win_stream(0, (uintptr_t) STREAM, g_base, g_rows);
        std::printf("RESULT carve=%zu stream_rc=%lld refusal=%s setattr=%d\n",
                    carve, (long long) rc, exl3_l2win_refusal(0),
                    shim_call_count("cudaStreamSetAttribute"));
        return 0;
    }

    if (scene == "setattr_rejects")
    {
        g_shim.setattr_rejects = true;              // the driver answers cudaErrorInvalidValue
        exl3_l2win_arm(0, shapes);
        size_t carve = exl3_l2win_ctx_carve(0);
        shim_register(g_base, g_span);
        int64_t rc = exl3_l2win_stream(0, (uintptr_t) STREAM, g_base, g_rows);
        size_t after = exl3_l2win_ctx_carve(0);          // the refusal must not poison the device
        std::printf("RESULT carve=%zu stream_rc=%lld refusal=%s carve_after=%zu setattr=%d\n",
                    carve, (long long) rc, exl3_l2win_refusal(0), after,
                    shim_call_count("cudaStreamSetAttribute"));
        return 0;
    }

    if (scene == "setlimit_fails")
    {
        g_shim.setlimit_fails = true;               // the device refuses the carve outright
        exl3_l2win_arm(0, shapes);
        size_t carve = exl3_l2win_ctx_carve(0);
        shim_register(g_base, g_span);
        int64_t rc = exl3_l2win_stream(0, (uintptr_t) STREAM, g_base, g_rows);
        size_t again = exl3_l2win_ctx_carve(0);           // a refusal is remembered, not retried
        std::printf("RESULT carve=%zu stream_rc=%lld refusal=%s carve_again=%zu setlimit=%d setattr=%d\n",
                    carve, (long long) rc, exl3_l2win_refusal(0), again,
                    shim_call_count("cudaDeviceSetLimit"), shim_call_count("cudaStreamSetAttribute"));
        return 0;
    }

    if (scene == "clamp_pure")
    {
        // The sizing helpers the CUDA sites call, exercised directly: they are inline in
        // exl3_l2win.cuh, so this executes the same code the device path runs.
        const size_t ceiling = L2_CARVE_CEILING_BYTES;
        std::printf("RESULT ceiling=%zu\n", ceiling);
        std::printf("RESULT clamp_64mib_on_72mib=%zu\n", l2win_clamp_carve(64ull << 20, 72ull << 20));
        std::printf("RESULT clamp_ceiling_on_32mib=%zu\n", l2win_clamp_carve(ceiling, 32ull << 20));
        std::printf("RESULT clamp_8mib_on_72mib=%zu\n", l2win_clamp_carve(8ull << 20, 72ull << 20));
        return 0;
    }

    if (scene == "sizing")
    {
        L2WinShapes a { 24, 2, 256, 1024 }, h_only { 48, 2, 256, 1024 }, kvh_only { 24, 4, 256, 1024 },
                    d_only { 24, 2, 128, 1024 }, rel_only { 24, 2, 256, 2048 };
        const size_t carve = L2_CARVE_CEILING_BYTES;
        std::printf("RESULT row_bytes_served=%zu\n", l2win_row_bytes(a));
        std::printf("RESULT row_bytes_h_only=%zu\n", l2win_row_bytes(h_only));
        std::printf("RESULT row_bytes_kvh_only=%zu\n", l2win_row_bytes(kvh_only));
        std::printf("RESULT row_bytes_d_only=%zu\n", l2win_row_bytes(d_only));
        std::printf("RESULT row_bytes_rel_only=%zu\n", l2win_row_bytes(rel_only));
        std::printf("RESULT window_rows1=%zu window_rows2=%zu window_rows8=%zu\n",
                    l2win_window_bytes(a, 1, carve, (size_t) ~0ll),
                    l2win_window_bytes(a, 2, carve, (size_t) ~0ll),
                    l2win_window_bytes(a, 8, carve, (size_t) ~0ll));
        std::printf("RESULT window_capped=%zu\n",
                    l2win_window_bytes(a, 64, 1ull << 20, (size_t) ~0ll));
        std::printf("RESULT window_maxwin=%zu\n",
                    l2win_window_bytes(a, 64, 1ull << 40, 4096));
        return 0;
    }

    std::printf("RESULT error=unknown-scenario %s\n", scene.c_str());
    return 2;
}