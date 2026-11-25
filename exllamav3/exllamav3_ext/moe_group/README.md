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

GATE FOUND WHILE READING (wheel-batch coupling): this lineage's coop act selector has only
SILU/GELU/RELU2/SILU_OAI (`exl3_moe_coop_kernel.cuh:119-123`) — the softcap-swiglu selector
(`MOE_COOP_ACT_SOFTCAP` + `up_cap`) landed on `dsh-softcap` (`aae2681f`), which is NOT in the
`d48fe4c1` lineage. The coop path cannot serve Inkling-2 until that selector rides the same
wheel batch or is reproduced on this branch; driving coop with a wrong act would be WORSE than
today's python path (it was exactly this silent-SiLU bug class that aae2681f fixed).

## Shape (updated after reading exl3.py 2026-11-25)

- `moe_sm89_common.cuh` — served geometry (128 experts, top_k 8, hidden 2304, inter 9216,
  gate cap 4 / up cap 25), the load-time `MoEPtrTable` (~12 KiB of pointers per layer — a
  pointer table, never a stacked weight copy: VRAM can't pay ~16 GB of dequantized duplicates),
  the fixed-address `MoERouting` exchange, and the sm_89 primitives (cp.async, ldmatrix,
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

Expert-order changes reorder the atomic accumulates, so a non-byte greedy result after
integration is honest-to-FAIL and gets adjudicated (campaign card row 5), never waived.
Fence 1 (this branch, `tests/cpu/test_moe_grouped_dispatch.py`) already prices the dispatch
shape + reassociation on identical dequant operands and carries the pointer-table positive
control. Fence 2 (trace: zero host syncs per MoE layer, RED-arm on the pre-change code) and
fence 3 (dot26 bench-lane decode delta + greedy parity, owner window) attach at integration.

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

1. **Act-selector alignment** (gate above): bring `MOE_COOP_ACT_SOFTCAP` + `up_cap` into this
   lineage (cherry-pick `aae2681f` from `dsh-softcap` — same act family, `cg*tanh(g/cg)*
   sigmoid(g) * cu*tanh(u/cu)`) and verify the selector matches `inkling2_loop.py:1155-1158`
   elementwise before the coop path is allowed to serve Inkling-2.
2. Python wiring for the coop drive: build the nine int64 pointer tables from `experts_g/u/d[e]`
   LinearEXL3 tensors at load; scratch via `exl3_moe_coop_ctr_len` + this branch's capacity
   formula; fixed-address `sel64/rw16` + x/out slots per capture round (doorbell rule); expose
   `moe_coop_launch_with_device_routing` through `bindings.cpp`; retire
   `host_values`/`stage_host` in `inkling2_loop.py:1131-1160` behind the fence-2 trace arm.
3. Fallback comparator only (`moe_group_gemv.cu`): host grid sizing
   (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`) and, if it is ever promoted to primary,
   the trellis stream substitution (replace the int4 placeholder unpack; fold x*suh pre-GU and
   the svh hadamard into leg D's epilogue). Until then its `MoEWeightRef` table must not feed
   a real pack.
4. Shared-memory opt-in in the fallback only if BLOCK_K grows; the draft fits 48 KB static.