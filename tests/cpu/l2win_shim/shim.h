#pragma once

// Controls and recorder for the CUDA runtime shim (see cuda_runtime.h in this directory).

#include <cstddef>

struct ShimCtl
{
    size_t l2_bytes = 72ull * 1024 * 1024;   // what cudaDevAttrL2CacheSize reports
    size_t max_window_bytes = 128ull * 1024 * 1024;
    bool setlimit_fails = false;             // device refuses the carve outright
    bool setattr_rejects = false;            // driver answers cudaErrorInvalidValue
    int device = 0;                          // device the registered ranges belong to
};

extern ShimCtl g_shim;

struct ShimRange { uintptr_t base; size_t bytes; };
#define SHIM_MAX_RANGES 8
extern ShimRange g_shim_ranges[SHIM_MAX_RANGES];

void shim_register(uintptr_t base, size_t bytes);        // device allocation the VA probe accepts
void shim_clear_ranges();
void shim_record(const char* call);                       // echoed as `CALL <call>` on stdout
int shim_call_count(const char* name);                    // how many recorded lines start with name
const char* shim_last_call(const char* name);             // last recorded line for that call, "" if none