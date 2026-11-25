// Pure-host ABI probe for the moe_group tier. Compiled AND run on dot26 in ~/kernel-scratch:
// it makes no CUDA runtime call (no cudaMalloc, no kernel launch, no context), so the serving
// GPU stays untouched. Prints the compiled struct layout, geometry constants and the arena
// formula table; tests/cpu/moe_abi_crosscheck.py pins its output against the python-side
// ctypes layout, the CPU simulation constants and the committed fixture moe_abi_probe_output.txt.
//
// Build:  nvcc -arch=compute_89 moe_abi_probe.cu -o moe_abi_probe
// (the .cu suffix is only to share nvcc with the kernel TUs; the code is plain C++.)
#include "moe_exl3_table.cuh"
#include "moe_sm89_common.cuh"

#include <cstdio>

#define PRINT_OFF(field, UPPER) \
    printf("offsetof %-14s %3zu  expect %3d\n", #field, offsetof(MoEExl3Ref, field), MOE_REF_EXPECT_OFF_##UPPER);

int main() {
    printf("sizeof MoEExl3Ref      %3zu  expect %3d\n", sizeof(MoEExl3Ref), MOE_REF_EXPECT_SIZEOF);
    PRINT_OFF(trellis, TRELLIS)
    PRINT_OFF(suh, SUH)
    PRINT_OFF(svh, SVH)
    PRINT_OFF(mcg, MCG)
    PRINT_OFF(mul1, MUL1)
    PRINT_OFF(in_features, IN_FEATURES)
    PRINT_OFF(out_features, OUT_FEATURES)
    PRINT_OFF(k_width, K_WIDTH)
    PRINT_OFF(has_mcg, HAS_MCG)
    printf("layer_bytes            %3d\n", (int)MOE_EXL3_LAYER_BYTES);

    printf("geom num_experts       %d\n", MOE_NUM_EXPERTS);
    printf("geom top_k             %d\n", MOE_TOP_K);
    printf("geom hidden            %d\n", MOE_HIDDEN);
    printf("geom inter             %d\n", MOE_INTER);
    printf("geom gate_cap          %.1f\n", (double)MOE_GATE_CAP);
    printf("geom up_cap            %.1f\n", (double)MOE_UP_CAP);
    printf("geom block_m           %d\n", MOE_BLOCK_M);
    printf("geom block_n           %d\n", MOE_BLOCK_N);
    printf("geom block_k           %d\n", MOE_BLOCK_K);
    printf("geom max_decode_tokens %d\n", MOE_MAX_DECODE_TOKENS);
    printf("geom max_decode_entries %d\n", MOE_MAX_DECODE_ENTRIES);
    printf("sizeof MoEPtrTable     %3zu  expect %3d\n", sizeof(MoEPtrTable), MOE_PTR_TABLE_EXPECT_SIZEOF);

    const int32_t samples[] = {8, 64, 512, 2048};
    for (unsigned i = 0; i < sizeof(samples) / sizeof(samples[0]); ++i) {
        printf("capacity entries %-5d padded_rows %-6d max_blocks %-5d\n",
               samples[i], (int)moe_max_padded_rows(samples[i]), (int)moe_max_blocks(samples[i]));
    }
    return 0;
}