"""Packed-resident NVFP4 linear-algebra lane (original implementation).

Keeps weights nibble-packed at rest (4.5 bits/element: 4 weight bits + 8 block-
scale bits per 16 + a scalar) and executes y = x @ W^T without ever storing a
full-width copy resident:

  direct lane   small M — a Triton split-K matvec that decodes nibbles to FP16
                inside the kernel and feeds tensor-core dots (F32 accumulation).
  scratch lane  large M — one-pass block decode to FP16 in transient scratch
                storage, then the ordinary wide-GEMM path (tensor cores fed).

The decode order is the spec order proven by the fenced CPU oracle in
nvfp4_fmt: value = (e2m1_code_value * E4M3_block_scale) * global_scale.
The nibble->F16 lattice decode in the kernel (PRMT byte-select from two packed
magnitude registers, sign by XOR) was validated exhaustively (all 256 codes)
against that oracle before adoption.

Lane choice is a pure function (pick_lane) so dispatch is testable without a
GPU; DIRECT_MAX_M is the initial small-M bound — decode-shape measurements
show the direct lane winning at M=1..8 in the near-production streaming regime;
the crossover is a tuning constant, not a hard semantic boundary.
"""

from __future__ import annotations

import torch

try:  # the direct lane is the only consumer; the rest of exllamav3 must load
    import triton          # without a hard triton dependency.
    import triton.language as tl
    _HAVE_TRITON = True
except ImportError:       # pragma: no cover — CPU-only installs
    triton = None
    tl = None
    _HAVE_TRITON = False

from . import nvfp4_fmt

DIRECT_MAX_M = 256        # inclusive bound: M <= 256 takes the direct lane;
                          # swept 2026-10-23 (engine-live, 4 projection classes,
                          # M=1..256): direct wins every cell, worst ratio 0.67
                          # at M=256; larger M falls to the scratch lane
DEFAULT_BK = 128          # tile depth in columns (scale groups per tile: BK // 16)
DEFAULT_BN = 32           # tile width in output rows
DEFAULT_SK = 4            # K splits (atomically reduced F32 partials)


def pick_lane(m: int) -> str:
    """Dispatch policy, pure: 'direct' below the small-M bound, else 'scratch'."""
    return "direct" if m <= DIRECT_MAX_M else "scratch"


if _HAVE_TRITON:

    # E2M1 magnitude lattice, eight f16 patterns in two b32 registers:
    # 0x00 0x38 0x3c 0x3e -> 0, .5, 1, 1.5 ... 0x40 0x42 0x44 0x46 -> 2, 3, 4, 6.
    # prmt.b32 data-driven byte-select yields the magnitude in one instruction;
    # the nibble sign bit becomes an XOR into the f16 sign (bit 15).
    _E2M1_PRMT_ASM = tl.constexpr("""
    {
    .reg .b32 cc, aa, bb, dd, ss, v1, v2;
    .reg .b16 l1, h2;
    mov.u32 cc, $2;
    mov.u32 aa, 0x3E3C3800;
    mov.u32 bb, 0x46444240;
    and.b32 ss, cc, 7;
    prmt.b32 dd, aa, bb, ss;
    shl.b32 v1, dd, 8;
    and.b32 dd, cc, 8;
    shl.b32 dd, dd, 12;
    xor.b32 v1, v1, dd;
    shr.u32 ss, cc, 4;
    and.b32 ss, ss, 7;
    prmt.b32 dd, aa, bb, ss;
    shl.b32 v2, dd, 8;
    and.b32 dd, cc, 0x80;
    shl.b32 dd, dd, 8;
    xor.b32 v2, v2, dd;
    mov.b32 {l1, h2}, v1;
    mov.b16 $0, l1;
    mov.b32 {l1, h2}, v2;
    mov.b16 $1, l1;
    }
    """)

    @triton.jit
    def _direct_splitk_kernel(A, WQ, WS, Out, g, M, N, K, SK: tl.constexpr,
                              stride_am, stride_wn, BM: tl.constexpr,
                              BN: tl.constexpr, BK: tl.constexpr):
        pid_n = tl.program_id(0)
        pid_s = tl.program_id(1)
        pid_m = tl.program_id(2)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK // 2)
        mmask = rm < M
        KB = K // BK
        kb_lo = pid_s * (KB // SK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for kb in range(kb_lo, kb_lo + KB // SK):
            ko = kb * BK
            a = tl.load(A + rm[:, None] * stride_am + ko + tl.arange(0, BK)[None, :],
                        mask=mmask[:, None], other=0.0)
            wq = tl.load(WQ + rn[:, None] * stride_wn + ko // 2 + rk[None, :])
            ws = tl.load(WS + rn[:, None] * (K // 16) + kb * (BK // 16)
                         + tl.arange(0, BK // 16)[None, :])
            vlo, vhi = tl.inline_asm_elementwise(
                _E2M1_PRMT_ASM, "=h,=h,r", [wq],
                dtype=(tl.float16, tl.float16), is_pure=True, pack=1)
            v = tl.interleave(vlo.to(tl.float32), vhi.to(tl.float32)).to(tl.float16)
            sf = ws.to(tl.uint8).to(tl.float8e4nv, bitcast=True).to(tl.float32)
            sf = sf[:, :, None].broadcast_to(sf.shape[0], BK // 16, 16).reshape(BN, BK)
            w = (v * sf * g).to(tl.float16)
            acc += tl.dot(a, tl.trans(w))
        tl.atomic_add(Out + rm[:, None] * N + rn[None, :], acc, mask=mmask[:, None])


def validate_triple(packed, scales, global_scale):
    """Full admission check (runs E4M3 illegal-byte scan; call once at load).

    Returns (rows, cols, g). Hot paths below trust already-validated tensors;
    this scan touches every scale byte and must not run per forward.
    """
    rows, cols, g = _triple_dims(packed, scales, global_scale)
    bad = bool(((scales & 0x7F) == 0x7F).any().item()) or bool((scales & 0x80).any().item())
    if bad:
        raise ValueError("illegal E4M3 scale byte (NaN pattern or sign bit) in scales")
    return rows, cols, g


def _triple_dims(packed, scales, global_scale):
    """Cheap host-side consistency checks only (dtypes/geometry); no device sync."""
    if packed.dtype is not torch.uint8 or packed.ndim != 2:
        raise ValueError("packed must be a 2-D uint8 tensor [rows, cols//2]")
    rows, half = packed.shape
    if half % 8:
        raise ValueError("packed row length must be a multiple of 8")
    cols = half * 2
    if scales.dtype is not torch.uint8 or scales.ndim != 2 or scales.shape != (rows, cols // 16):
        raise ValueError(f"scales must be uint8 {(rows, cols // 16)}")
    g = float(global_scale)
    if not (g > 0.0) or not (g == g and g < float("inf")):
        raise ValueError("global scale must be a positive finite scalar")
    return rows, cols, g


_validate_triple = _triple_dims   # hot paths: geometry-only, scan-free


def _direct_shape_ok(x, rows, cols, bn, bk, sk):
    return (x.is_cuda and x.ndim == 2 and x.shape[1] == cols and x.dtype is torch.float16
            and cols % bk == 0 and rows % bn == 0 and (cols // bk) % sk == 0)


def matmul_direct(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor,
                  global_scale, *, sk: int = DEFAULT_SK, bn: int = DEFAULT_BN,
                  bk: int = DEFAULT_BK, bm: int = 16, num_warps: int = 4,
                  num_stages: int = 3) -> torch.Tensor:
    """Direct packed-resident lane: x fp16 [M,K] @ decode(packed,...)^T -> x.dtype [M,N].

    Raises ValueError when shapes/dtypes fall outside the kernel's domain — call
    matmul() for the shape-tolerant dispatch that falls back to the scratch lane.
    """
    if not _HAVE_TRITON:
        raise RuntimeError("triton is required for the direct w4a16 lane")
    rows, cols, g = _validate_triple(packed, scales, global_scale)
    packed = packed.contiguous()
    scales = scales.contiguous()
    if not _direct_shape_ok(x, rows, cols, bn, bk, sk):
        raise ValueError("shape/dtype outside direct-lane domain (need cuda fp16 "
                         "[M,K] with K%%BK==0, N%%BN==0, K//BK %% SK == 0)")
    out = torch.zeros((x.shape[0], rows), dtype=torch.float32, device=x.device)
    _direct_splitk_kernel[(rows // bn, sk, (x.shape[0] + bm - 1) // bm)](
        x, packed, scales, out, g, x.shape[0], rows, cols, sk,
        x.stride(0), packed.stride(0),
        BM=bm, BN=bn, BK=bk, num_warps=num_warps, num_stages=num_stages)
    return out.to(x.dtype)


def matmul(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor,
           global_scale) -> torch.Tensor:
    """Dispatching entry: pick_lane(M) selects direct (small M) or scratch (wide GEMM).

    Scratch lane decodes into transient FP storage sized to one layer (freed on
    return — weights never live full-width resident) and uses the ordinary
    matrix product for tensor-core throughput. Shape domains the direct kernel
    cannot express also fall back to scratch, which has no divisibility bounds.
    """
    rows, cols, g = _validate_triple(packed, scales, global_scale)
    lane = pick_lane(x.shape[0]) if x.ndim == 2 else "scratch"
    if (lane == "direct" and _HAVE_TRITON
            and _direct_shape_ok(x, rows, cols, DEFAULT_BN, DEFAULT_BK, DEFAULT_SK)):
        return matmul_direct(x, packed, scales, g)
    if x.is_cuda:
        dec = nvfp4_fmt.decode_weights_gpu(packed, scales, g, torch.float16)
    else:
        dec = nvfp4_fmt.decode_weights(packed, scales, g, x.dtype)
    return x @ dec.t()