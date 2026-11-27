#pragma once

// Controls and recorder for the Inkling 2 flash-prefill shim (see cuda_runtime.h in this
// directory). The driver in api_driver.cpp sets the controls per scenario; every CUDA call and
// every kernel launch the production code makes is echoed as a `CALL ...` line the fence parses.

#include <cstddef>
#include <cstdint>

struct ShimCtl
{
    int device = 0;
    int sm_major = 8;              // what the compute-capability queries answer
    int sm_minor = 9;
    size_t smem_optin = 99 * 1024;             // per-block opt-in maximum, as on sm_89
    size_t smem_per_sm = 100 * 1024;           // per-SM shared memory, as on sm_89
    size_t smem_per_block = 48 * 1024;         // the non-opt-in default
    bool attrs_fail = false;                   // device attribute query fails
    bool setattr_refuses = false;              // driver refuses the smem opt-in
};

ShimCtl& shim_ctl();
void shim_record(const char* call);            // echoes `CALL <call>`
int shim_call_count(const char* prefix);       // recorded lines starting with prefix
const char* shim_last_call(const char* prefix);