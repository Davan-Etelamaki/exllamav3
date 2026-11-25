# moe_group — phase-1 MoE grouped expert tier (Inkling-2, sm_89), DRAFT

Branch `dsh-moe` (base `d48fe4c1`, dsh-triple tip = wheel-C lineage). Status: **draft,
compile-verified only** — nothing here is wired into the build or the served path yet.

## What it replaces

The served python dispatch at `exllamav3/modules/arch_specific/inkling2_loop.py:1134-1160`:
a per-MoE-layer D2H sync (`host_values(...).tolist()`), a python sort + 8-iteration grouping
loop, and 24 launches per layer through independent per-expert `Linear` modules
(`experts_g/u/d[e]`, built at :1072-1088 — per-expert qgroups, NOT the stacked `ExpertsCFG`
containers of `modules/block_sparse_mlp.py`, which belong to a different architecture).

## Shape

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

1. **exl3 wire-in**: replace `moe_unpack_int4x8` + the raw-nibble stream with the real exl3
   trellis decode (quants machinery), keeping `MoEWeightRef` as the carrier (it already holds
   the fields a pack needs: `qbytes`, `scales`, `group_size`). Until then the int4+qgroup-scale
   placeholder also imposes `group_size % MOE_BLOCK_K == 0`, which the table builder must assert.
2. Python wiring: table construction at load, buffer arena sized per max batch, `params`
   plumbing, and retirement of the `host_values`/`stage_host` pinned-slot dance (census item A —
   the by-value fixed-address exchange in `MoERouting` is the Strata-doorbell lesson adopted
   2026-11-24, campaign §210; whole-forward graph capture becomes a by-product witness, fence 4,
   never the gate: law 15).
3. Host-side grid sizing (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`), replacing the
   draft's fixed `grid` argument.
4. Shared-memory opt-in only if BLOCK_K grows; the draft fits 48 KB static.