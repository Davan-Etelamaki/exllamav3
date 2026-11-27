# Grouped multi-token MoE expert tier — kernel contract and design (Inkling 2, sm_89)

Lane: grouped-expert CUDA kernel (dsh-moe-kernel). Date 2026-11-27. Governing gate: §CJ
(`tests/cpu/moe_fused_dispatch_repro.py`), bar = installed-eager/10 dispatcher ops per
executed MoE layer-pass. Measurement regime: CPU dispatcher-ops (law 38 — no CUDA loads on
the workstation); on-card confirmation is law-31's installed-wheel → patched-source pair
plus one nsys census on dot26 (§CY rows), owed by the parent.

## What the tier is

`InklingMoE._grouped_moe` (behind `infer_params.inkling_moe_grouped_ext`, DEFAULT OFF)
serves the eager MULTI-TOKEN regime (t > 1, not capturing). Two ext launches absorb the
per-group fan-out the §CJ map names; python keeps exactly the residue the FLOOR arm
priced: the router, the device grouping, and one full-order combine.

    router (torch, verbatim)  →  order_all = argsort(sel_flat, stable=True)
                                 counts    = bincount(sel_flat, minlength=num_experts)
    ext.exl3_moe_group_gu(x, order_all, counts, tables, K, mcg, mul1, act, gate_cap, up_cap)
    ext.exl3_moe_group_down(act, entries, order_all, counts, tables, K, mcg, mul1, y)
    out.index_add_(0, tokens_all, y[:entries].float()
                   * w_flat.index_select(0, order_all).unsqueeze(-1))

No host roundtrip: the kernel recovers an entry's expert by scanning `counts` in shared
memory (E ≤ 128, one thread per block) — `order_all` is the stable ascending-expert device
order, identical to the total order the python loop sorts into, so group membership and
the per-token accumulation sequence are unchanged.

## Absorption map — measured, not claimed

Same-run §CJ gate against the INSTALLED baseline (`/tmp/installed_baseline/inkling2_loop.py`,
sha1(fe944038617c), fetched from dot26): dispatcher ops per executed MoE layer-pass, bar 51.5.

| arm | ops | note |
|---|---|---|
| EAGER (installed wheel) | 515 | RED baseline, same-run twice (515/515) |
| GLUED (swiglu hoist only) | 221 | > bar: glue alone cannot close §CJ |
| GROUPED python skeleton | 132 | > bar: folding the launch without absorbing the glue buys nothing |
| GROUPED-LAUNCH (modeled) | 19 | ≤ bar — the absorbed shape |
| FLOOR | 22 | reachability control |
| **FUSED (tree dispatch, grouped ON)** | **24** | **≤ 51.5 → gate GREEN** |

Family contraction (eager → fused): arith 247→12, copies/casts 97→3, ext-tier 72→2,
index_select 48→1, activations 25→1, scatter_add 24→1, cat 0→0; NO-TELEMETRY rule: no
counted family vanished without a control. Bit-identity arms Δ = 0.000e+00 (fused vs
installed, glued/grouped vs installed). Planted extra-op controls: +10 → 142 > bar
(modeled arm), +40 → 64 > bar (executed tree arm) — the threshold bites on both.

## Memory layout

Experts are device-resident per-expert `Linear`s (residency ledger verdict
`all_device_resident_at_load`). The launches read them through the STACKED int64 table the
coop tier already builds: `tables` = 9 × num_experts row-major in
`moe_coop_plan.TABLE_ORDER` (d_suh, d_svh, d_trellis, g_suh, g_svh, g_trellis, u_suh,
u_svh, u_trellis) — `_grouped_moe` reuses `_coop_ptr_lists()`/`_coop_ptrs_probe()`
unchanged. Upload: pinned staging buffer → ONE `.to(device)`; refill IN PLACE (`copy_`)
only when the pointer fingerprint moves (the coop arena's doorbell rule; no pageable H2D,
though the tier is eager-only).

Workspace lives in `params["inkling2_moe_grouped_ws"]` keyed by device (MoE instances run
sequentially on one stream → one pool per generation, not per layer), bucketed over
`GROUPED_BUCKETS = (64…2048)`; `MOE_GROUPED_MAX_ROWS = 2048` = the served chunk_size.
At the top bucket: act 16384 × 1152 × 2 B ≈ 37.7 MB + y 16384 × 2304 × 2 B ≈ 75.5 MB
≈ 113 MB total (law 23: sized from the allocation site at the deployed shape).

## Numerics

* Combine order: entries processed in order_all order visit each token's contributions in
  the per-group loop's sequence → CPU bit-identical (fence [A], three executed routings;
  gate bit-identity arms). No atomics in either launch.
* Act rounding: fp16 cast INSIDE the gu epilogue — the same point as the served
  `act.to(torch.half)` before down_proj (`torch.half`, not `g_all.dtype` — the defect class
  the §CJ fence caught). Softcap chain = the served triplet, fenced to 7.6e-06 against
  `MOE_COOP_ACT_SOFTCAP`.
* Inside-kernel accumulation: fp32 accumulators over in_features; the summation ORDER
  differs from cuBLAS per-group GEMMs — an accumulation-order decision adjudicated
  on-card by law 31's pair + fence-3 greedy parity, never waived here.
* Routing math unchanged: bias selects, unbiased sigmoid scores weight (router untouched).

## Regime seams (fence sections [B]/[C], executed)

t == 1 → decline "rows ≤ 32 belong to the coop tier"; capture → decline "keeps the
recorded routing"; rows > 2048 → decline at the ladder; ext rc < 0 → decline with the
kernel's reason. Every decline path executes today's chain verbatim (outputs bit-compared
against the knob-off arm). Knob OFF = served path byte-identical (config AST check).

## Build-stage honesty (maximal defensible slice, stated plainly)

`moe_group_ext.cu` implements the two launches for the **fp16-dense weight layout
(K == 0)** and REFUSES the served pack's exl3 quant layout with rc −2 + a named reason, so
flipping the knob on today's quant pack keeps the eager chain by refusal, never by wrong
math. The adoption step ports the trellis/suh/svh dequant stream into these launches
(tile/dequant device functions already exist in `../quant/exl3_moe_coop_kernel.cuh`;
`moe_group_gemv.cu`'s int4 placeholder must not feed a real pack — README rule). The
python contract (arguments, tables, workspace, seams) is final — adoption changes only
the kernel interior and lifts the refusal. Compile evidence (workstation CPU, no GPU
context): nvcc 13.4 `-std=c++20 -arch=sm_89 -c` with the wheel-exact flag set → rc 0;
symbols attributed by `cuobjdump --dump-elf-symbols`:
`moe_group_launch_kernel<true/false>` in the `moe_group_ext_cu` TU. `bindings.cpp` passes
`g++ -fsyntax-only` with the three new defs.

## Fences

* `tests/cpu/moe_fused_dispatch_repro.py` — §CJ gate, GREEN at this tip (numbers above).
* `tests/cpu/inkling_moe_grouped_fence.py` — semantics: [A] equivalence ×3 routings,
  [B]/[C] seams, [D] default OFF + production values at the launch boundary, [E] ladder,
  [F] planted defects (division, softcap swap, launch-row shift), [G] header-drift vs
  `moe_group_ext.cuh` (constants + positional argument order).
* Standing quartet re-run on the merged tree (fork/dsh-stack 8882610e1536 included):
  devdispatch 26/26, expertx 35/35, fusedglue 13/13, coop-plan GREEN, opcensus rc 0.

## Owed by others at merge

On-card legs (parent/dot26): installed-wheel RED → patched-source GREEN for the grouped
launch itself, one nsys census vs §CY, and the §CM step-5 read-back of the
`INKLING-MOE-GROUPED` marker + the grouped-decline witness line when the knob stays off.