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

## Deferred to dot26 / the owner's wheel window (laws 13/31/38)

- Compile verification runs as a compiler invocation only, in `~/kernel-scratch/` on dot26
  (RTX 4090, sm_89) — a `.cu` → `.o` compile does not touch the serving GPU context and is
  not a wheel build; no build happens in the exllamav3 tree on dot26 and its engine is never
  restarted by this lane. Repro:
  `nvcc -arch=compute_89 -c moe_route.cu -o /dev/null && nvcc -arch=compute_89 -c moe_group_gemv.cu -o /dev/null`
- Runtime behaviour (fence 2 GPU arm, fence 3) waits for the owner's wheel window; no
  `setup.py build_ext` on the workstation (law 31 — build slot is after wheel-1/wheel-2).

## Integration TODOs (next commits on this branch, before any wheel ride)

1. **Trellis stream substitution** (was: exl3 wire-in): the table side is DONE (MoEExl3Ref +
   ABI fence). Remaining: replace the int4 placeholder unpack with the rotated-basis trellis
   decode (bit-exact against ext's `reconstruct`/`BC_LinearEXL3` machinery), and fold the
   Hadamard chain the exl3 forward path implies: x -> `had_r_128`(suh) before leg GU;
   accumulator -> `had_r_128`(svh) in leg D's epilogue (128-wide output blocks; 2304/9216 are
   128-divisible). The old `MoEWeightRef` (qbytes/scales/group_size) remains only as the mma
   smoke-path placeholder until then and must not feed a real pack.
2. Python wiring: table construction at load, buffer arena sized per max batch, `params`
   plumbing, and retirement of the `host_values`/`stage_host` pinned-slot dance (census item A —
   the by-value fixed-address exchange in `MoERouting` is the Strata-doorbell lesson adopted
   2026-11-24, campaign §210; whole-forward graph capture becomes a by-product witness, fence 4,
   never the gate: law 15).
3. Host-side grid sizing (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`), replacing the
   draft's fixed `grid` argument.
4. Shared-memory opt-in only if BLOCK_K grows; the draft fits 48 KB static.