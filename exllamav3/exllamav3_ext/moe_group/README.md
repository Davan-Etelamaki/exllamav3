# moe_group — phase-1 MoE grouped expert tier (Inkling-2, sm_89), DRAFT

Branch `dsh-moe` (base `d48fe4c1`, dsh-triple tip = wheel-C lineage). Status: **draft,
compile-verified only** — nothing here is wired into the build or the served path yet.

## What it replaces

The served python dispatch at `exllamav3/modules/arch_specific/inkling2_loop.py:1134-1160`:
a per-MoE-layer D2H sync (`host_values(...).tolist()`), a python sort + 8-iteration grouping
loop, and 24 launches per layer through independent per-expert `Linear` modules
(`experts_g/u/d[e]`, built at :1072-1088 — per-expert qgroups, NOT the stacked `ExpertsCFG`
containers of `modules/block_sparse_mlp.py`, which belong to a different architecture).

## ABI fence (instant, CPU + dot26 probe — added 2026-11-25)

`moe_exl3_table.cuh` defines the REAL per-expert reference `MoEExl3Ref` (trellis/suh/svh/mcg/
mul1 + dims, 64 B — fields taken from the served `LinearEXL3`, exllamav3/modules/quant/exl3.py)
with expectation literals; `moe_abi_probe.cu` is a PURE-HOST probe (no CUDA runtime call) built
and run on dot26; its raw output is the fixture `tests/cpu/moe_abi_probe_output.txt`.
`tests/cpu/moe_abi_crosscheck.py` pins compiled layout == literals == python ctypes mirror ==
simulation geometry, checks the arena capacity formula against the compiled values, builds the
table from exl3-shaped mocks and carries the swap positive control (names the pair). GREEN
2026-11-25: 9 offsets, 4 geometry ties, 4 capacity points, control flagged exactly (5, 97)
across 24 field comparisons. Regenerate the fixture after any header change:
`nvcc -arch=compute_89 moe_abi_probe.cu -o moe_abi_probe && ./moe_abi_probe`.

## Integration decision (2026-11-25, supersedes "write our own grouped decode")

Reading `exllamav3/exllamav3_ext/quant/exl3_moe_coop.{cuh,cu}` showed the tree ALREADY has the grouped
decode tier phase 1 needs, in service for stacked-expert architectures:

- `MoeCoopParams` is driven by PER-EXPERT POINTER TABLES (nine int64 device arrays:
  {g,u,d}_{trellis,suh,svh} + bias tables) — the same design conclusion this branch reached
  independently; bsz≤scratch decode regime; two ordinary launches (rot kernel only when bsz>1)
  with device-side completion counters (ctr_a/ctr_b), run table and ksplit — no stream syncs,
  graph-recordable like the rest of ext.
- `exl3_moe_coop_launch` reads `sel` (int64) and `rw` (half) as DEVICE arrays — the served
  Inkling-2 D2H at `inkling2_loop.py:1134` exists only because python dispatches, not because
  the kernel needs host values.

Primary path therefore: `moe_route.cu` (device top-k/renorm, this branch) ->
`moe_coop_adapter.cu` (int32 sel -> int64, float w -> half at fixed addresses; compile-verified
symbol-attributed in the Verification record below: `nvcc -arch=sm_89 -std=c++20 -c` exit 0,
entry `moe_route_to_coop_kernel`) ->
`exl3_moe_coop_launch`. The hand-written `moe_group_gemv.cu` (int4 placeholder stream) is
DEMOTED to the fallback/measurement comparator for fence 3; it must not serve a real pack.

GATE FOUND WHILE READING, NOW LANDED (wheel-batch coupling RESOLVED 2026-11-25): this
lineage originally lacked the softcap-swiglu coop selector (`MOE_COOP_ACT_SOFTCAP` +
`up_cap`, landed on `dsh-softcap` `aae2681f`, not in the `d48fe4c1` lineage). `aae2681f` is
now cherry-picked here as `020ce47c` (clean merge of all seven source files; the lane's test
file `tests/kernels/moe/test_softcap_swiglu_lane.py` never existed in this lineage — branch
divergence, not a deletion — and was adopted whole with its fence suite). Elementwise fence
`tests/cpu/moe_softcap_elementwise.py` extracts the `MOE_COOP_ACT_SOFTCAP` return expression
FROM THE TREE and compares it against the served epilogue (the `inkling2_loop.py` act
triplet, bound by `SERVED_ANCHORS` strings — bare line numbers decay with edits) at
N=2^20 fp32: GREEN, max|delta| 7.6e-06 against an effective bound 9.1e-04; positive control (the classic
single-cap mistake, `up_cap := gate cap`) breaches at 7.5e+01. No wheel-batch dependency on
`dsh-softcap` remains for the act selector.

## Shape (updated after reading exl3.py 2026-11-25)

- `moe_sm89_common.cuh` — served geometry (128 experts, top_k 8, hidden 2304, inter 9216,
  gate cap 4 / up cap 25), the load-time `MoEPtrTable` (exactly 9,216 B per layer — pinned by
  `MOE_PTR_TABLE_EXPECT_SIZEOF` static_assert and probe; supervisor correction 2026-11-25:
  an earlier `down[MOE_INTER]` slip meant 228 KiB and is fixed to `down[MOE_NUM_EXPERTS]`),
  the fixed-address `MoERouting` exchange, the DECODE-ONLY cap (`MOE_MAX_DECODE_TOKENS 32`,
  see contract below), and the sm_89 primitives (cp.async, ldmatrix,
  mma.sync m16n8k16 — no wgmma/TMA, those are sm_90+).
- `moe_route.cu` — device top-k (one warp/token, sigmoid scores + additive bias, renorm
  `w/(sum+1e-20)*route_scale` exactly as :1114-1116) and the vLLM-shaped segment build
  (histogram → pad each expert's slice to `BLOCK_M` → one `expert_for_block` id per kernel
  block → scatter; the kernel BRANCHES ON-DEVICE, `fused_moe.py:69-162` shape borrowed from
  `/data/Repos/musicman89/vllm-src` — reference only, vLLM is not installed and never will be).
- `moe_group_gemv.cu` — persistent grouped GEMV, two capture-safe launches: leg GU computes
  gate+up from one gathered A tile with the fused swiglu-softcap epilogue into an fp16
  `act_scratch` (same rounding point as the served `act.to(torch.half)` before down_proj);
  leg D reads it and atomically accumulates `w_entry * y` into the fp32 out (serves the
  `index_add_` semantics with free accumulation order). Grid-stride bound read from
  `num_blocks_out` ON DEVICE — the padded block count never crosses to the host, so the
  whole lane is stream-sync-free (the D2H at :1134 dies with this file's integration).

## Numerics stance

TIE-BREAK SPEC (supervisor charge C): on equal biased router scores the DEVICE rule is
AUTHORITATIVE — lowest expert id wins (`moe_route.cu`'s butterfly reduces ties to the
minimum index). `torch.topk`'s tie order is UNSPECIFIED and is never a parity reference:
fence 1 asserts its seeded cases tie-free (`moe_grouped_sim.route()` sorted-neighbor
scan), so a torch-specific tie can never be read as a grouped-vs-served parity failure,
and at runtime the device selection — not any host-side topk — defines membership.
PARITY SURFACE (supervisor §Q): ranking uses score+bias but the EMITTED weights gather the
UNBIASED scores renormalised (`moe_route.cu` broadcasts the unbiased score; served lines
do the same via `scores.gather`), so the fence-3 greedy arm SHALL assert on the emitted
weights per token, not only on the selected ids — the biased-rank/unbiased-emit asymmetry
is where a routing bug can hide behind equal id sets.

Expert-order changes reorder the atomic accumulates, so a non-byte greedy result after
integration is honest-to-FAIL and gets adjudicated (campaign card row 5), never waived.
Fence 1 (this branch, `tests/cpu/test_moe_grouped_dispatch.py`) already prices the dispatch
shape + reassociation on identical dequant operands and carries the pointer-table positive
control. Fence 2 (trace: zero host syncs per MoE layer, RED-arm on the pre-change code) and
fence 3 (dot26 bench-lane decode delta + greedy parity, owner window) attach at integration.

## Decode-only contract and arena sizing (supervisor order 2026-11-25)

This tier serves DECODE ONLY: `bsz <= MOE_MAX_DECODE_TOKENS = 32` entries <= 256 = the coop
tier's `MAX_SLOTS`. `act_scratch` is `[bsz*TOP_K, MOE_INTER]` fp16 — 144 KiB at bsz 1, 4.6 MiB
at the cap — and prefill (which would ask gigabytes) is REFUSED by the host gate in
`moe_coop_adapter.cu` (`return -1`, python keeps the fused/mgemm path), never silently
allocated.

Worst-case segment capacity (named rule, header `moe_exl3_table.cuh`):
`max_padded_rows(entries) = entries + 15 * min(entries, 128)` — each of at most
`min(entries, 128)` non-empty experts wastes at most `BLOCK_M-1 = 15` padding rows;
`max_blocks = ceil(max_padded_rows/16)`. At the cap: 256 entries -> 2176 rows -> 136 blocks.
Compiled and python values are pinned equal by the ABI crosscheck at {8, 64, 512, 2048}.

Allocation owner: the PYTHON WIRING allocates one fixed-address arena per graph-capture set
(per device, per layer: sel/w/sorted_token_ids/expert_for_block/counters + act_scratch) at
capture-set creation, sized by the formulas above; kernels never allocate mid-forward and the
address set is stable across replays (Strata in-place-recall lesson, campaign §210).

## Verification record (corrected 2026-11-25 after a supervisor attribution refutation)

The first "compile-verified" claim (milestone-1 report, commit `56428c48` message) rested on
EXIT codes plus `cuobjdump --list-elf` in `~/kernel-scratch/dsh-moe/` and did NOT dump symbols:
the objects there turned out to be FOREIGN (`finalize`, `gemm1_segmented`, `gemm2_segmented`,
`moe_sort_launch`, `moe_gemv_test_probe`), i.e. someone else's draft under the same filenames
in the shared scratch dir. Claim withdrawn for that evidence class — an unnamed-symbol cubin
proves nothing about attribution (law 7b: a fence that cannot tell whose code it ran is no fence).

Corrected fresh builds, sources extracted with `git show <sha>:<path>` (md5 recorded), one
directory PER COMMIT, symbol dump as part of the check:

- `56428c48` -> `~/kernel-scratch/dsh-moe-56428c48/` (md5 da87dbad… gemv, 7c575d9c… route,
  71c221a7… common):
  `nvcc -arch=sm_89 -c moe_route.cu -o moe_route.o` exit 0 → cubin `moe_route.1.sm_89.cubin`,
  entries `_Z25moe_build_segments_kernel10MoERouting`, `_Z19moe_dev_topk_kernel10MoERouting`;
  `nvcc -arch=sm_89 -c moe_group_gemv.cu -o moe_group_gemv.o` exit 0 → `moe_group_gemv.1.sm_89.cubin`,
  entries `_Z21moe_group_gemv_kernelILb0EEv…`, `_Z21moe_group_gemv_kernelILb1EEv…`. Exact
  kernel roster of the committed sources — attribution holds for `56428c48`.
- `a387ec5b` -> `~/kernel-scratch/dsh-moe-a387ec5b/` (adapter + headers + vendored
  `exl3_moe_coop.cuh` from the commit):
  `nvcc -arch=sm_89 -std=c++20 -c moe_group/moe_coop_adapter.cu -o moe_coop_adapter.o
  -I<torch inc> -I<torch api inc>` exit 0 → `moe_coop_adapter.1.sm_89.cubin`, entry
  `_Z24moe_route_to_coop_kernelPKiPKfPlP6__halfi`.
- The earlier ABI-probe execution (pure host) and both CPU fences were regenerated from
  working-tree files in the same turn as their GREEN outputs and are unaffected.

Protocol going forward: scratch paths are named `<branch>-<sha12>`; every compile-verify ends
with `cuobjdump --dump-elf-symbols` and the expected kernel-entry names in the report.
- `fc7450d4` -> `~/kernel-scratch/dsh-moe-fc7450d4/` (six TUs + pure-host probe from
  `git show`): route/gemv/adapter/probe all exit 0; entries
  `_Z25moe_build_segments_kernel10MoERouting`, `_Z19moe_dev_topk_kernel10MoERouting`,
  `_Z21moe_group_gemv_kernelILb0EE/ILb1EEv…`, `_Z24moe_route_to_coop_kernel…`; probe output
  diffs EMPTY against the committed fixture.
- `020ce47c` -> `~/kernel-scratch/dsh-moe-020ce47c/` (whole `exllamav3_ext` extracted with
  `git archive 020ce47c`): `exl3_moe_coop.cu` exit 0 (entries `exl3_moe_coop_rot_kernel`,
  `exl3_moe_coop_empty_kernel`), `moe_coop_adapter.cu` exit 0 (entry
  `_Z24moe_route_to_coop_kernel…`), representative instantiation TU
  `quant/comp_units/exl3_moe_coop_inst_k1.cu` exit 0 → entries
  `exl3_moe_coop_ns::exl3_moe_coop_a/b_kernel<…>` (the templates that embed `act_gate`'s
  softcap branch); `cuobjdump -sass` of one instantiation shows MUFU.TANH instructions —
  the softcap math is really emitted, not dead-stripped. Standalone coop-TU compiles must
  pass `--expt-relaxed-constexpr` (from torch `cpp_extension.py` `COMMON_NVCC_FLAGS`,
  line 388 of 2.14.1+cu130; the repo's own `cuda_flags.cuda_cflags` does not list it) —
  without it CUDA 13 rejects the plain-`constexpr` tile helpers. `--expt-extended-lambda`
  is NOT needed (recompiled the instantiation TU without it: exit 0 — it was my hand
  addition). Wheel-EXACT scratch recipe verified: torch's four `-D__CUDA_NO_*HALF/BFLOAT*`
  defines + `--expt-relaxed-constexpr` + repo `-O3 --use_fast_math` +
  `-Xcudafe --diag_suppress=177/20012`, `-std=c++20 -arch=sm_89` → instantiation TU
  exit 0; scratch and wheel flag sets agree.

## Deferred to dot26 / the owner's wheel window (laws 13/31/38)

- Compile verification runs as a compiler invocation only, in a per-commit `~/kernel-scratch/
  <branch>-<sha12>/` dir on dot26 (RTX 4090, sm_89) — a `.cu` → `.o` compile does not touch the
  serving GPU context and is not a wheel build; no build happens in the exllamav3 tree on dot26
  and its engine is never restarted by this lane. Repro (with the attribution step):
  `nvcc -arch=sm_89 -c moe_route.cu -o moe_route.o && nvcc -arch=sm_89 -c moe_group_gemv.cu -o moe_group_gemv.o && cuobjdump --dump-elf-symbols moe_route.o moe_group_gemv.o` — the symbol
  entries MUST name the committed kernels; see Verification record above.
- Runtime behaviour (fence 2 GPU arm, fence 3) waits for the owner's wheel window; no
  `setup.py build_ext` on the workstation (law 31 — build slot is after wheel-1/wheel-2).

## Integration TODOs (next commits on this branch, before any wheel ride)

1. **Act-selector alignment — LANDED** (`aae2681f` cherry-picked as `020ce47c`; elementwise
   fence vs the served act triplet (anchor-bound; line numbers decay) GREEN with single-cap control RED; see Integration
   decision + Verification record above). Python callers that want softcap must set
   `p.act = MOE_COOP_ACT_SOFTCAP`, `p.act_limit = 4.0`, `p.up_cap = 25.0` — done in TODO 2's
   wiring, where the values come from the served module, not literals.
2. Python wiring for the coop drive — **LANDED 2026-11-25**:
   `exllamav3/modules/arch_specific/moe_coop_plan.py` (pure-python plan layer: constants,
   `ctr_len` mirror, eligibility, positional `CALL_ARGS`) + `_coop_decode`/`_coop_state`
   in `inkling2_loop.py` drive `ext.exl3_moe_coop` (already exposed, bindings.cpp:322 —
   zero C++ changes). Nine int64 POINTER tables built from `experts_g/u/d[e]`
   LinearEXL3 `.trellis/.suh/.svh` `data_ptr()` at first eligible decode step — the coop
   `prepare` consumes tables directly (its `*_trellis` args ARE int64 tables,
   `n_local = size(0)`), so no stacked copy ever exists. Fixed-address arena per
   (decode-rows, load-epoch): `had_g/had_u/gu_g/gu_u/act_out/d_out/ctr/out`, cleared on
   `unload()`; a weight-pointer fingerprint is re-checked per call and refilled IN PLACE
   on drift so captured addresses stay meaningful (doorbell rule). Router stays torch
   (byte-identical routing; `sel` is already int64, `rw = w.half()`). NUMERICS NOTE:
   `rw` rides the coop boundary as HALF while the python path multiplied fp32 weights —
   fence 3's greedy parity prices this honestly; `gu` scratch is half, which MATCHES the
   served rounding (`expert.forward` returns half, `act` computed after `.float()`).
   Legacy `host_values`/`stage_host` path RETAINED (prefill + ineligible shapes still
   ride it); its retirement behind the fence-2 trace arm is a wheel-window decision.
   Drift fence: `tests/cpu/moe_coop_plan_fence.py` re-parses the headers (constants,
   37-parameter call order, ctr formula text) with a planted-rename control.
3. Fallback comparator only (`moe_group_gemv.cu`): host grid sizing
   (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`) and, if it is ever promoted to primary,
   the trellis stream substitution (replace the int4 placeholder unpack; fold x*suh pre-GU and
   the svh hadamard into leg D's epilogue). Until then its `MoEWeightRef` table must not feed
   a real pack.
4. Shared-memory opt-in in the fallback only if BLOCK_K grows; the draft fits 48 KB static.