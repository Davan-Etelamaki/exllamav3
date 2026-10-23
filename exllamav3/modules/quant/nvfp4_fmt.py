"""NVFP4 microscaling format: decode + quantize primitives (original implementation).

Pure-torch, dependency-light. All numeric grids are DERIVED at import time from the
element-format bit definitions (E2M1: 1/2/1 sign/exp/mant, bias 1; E4M3: 1/4/3, bias 7,
NaN pattern exp=15&man=7); nothing is a transcribed constant table.

Design decisions (ours, deliberate):
  * ONE primitive, `snap_codes`, serves both magnitudes (E2M1 and E4M3): values are snapped
    to the nearest lattice point, ties resolved to the EVEN lattice index. Because the finite
    nonnegative code points of an IEEE-style format increase monotonically with the raw byte,
    lattice index == code, so "even index" is exactly tie-to-even on the code.
  * Decoding is table-driven (tables generated from the bit definitions).
  * Quantizer policy (ours): the per-block E4M3 scale is snapped UPWARD on the lattice so the
    block's largest-magnitude element can never be clipped by the scale's own rounding; the
    global fp32 scale is chosen as the smallest power of two that keeps every block scale
    representable (<= E4M3 max). Power-of-two preserves operand mantissas exactly.
  * Element decode order: (element * block_scale) * global — documented consumer contract.

Nothing here is wired into model dispatch; activation is a separate decision.
"""
from __future__ import annotations

import torch

try:  # soft dependency: the CPU paths and conformance tests work without triton
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:      # pragma: no cover - exercised only on triton-less hosts
    _HAS_TRITON = False

# ---------------------------------------------------------------------------
# Lattice generation from the FORMAT DEFINITION (generic IEEE-style float bits)
# ---------------------------------------------------------------------------

def _ieee_lattice(exp_bits: int, man_bits: int, bias: int, *, no_inf_nan: bool) -> tuple[list[float], int]:
    """Return sorted finite nonnegative values of an IEEE-style narrow float, plus the NaN code
    (or -1 when the format has no NaN pattern). Enumerated from bit fields, not hardcoded."""
    exp_max = (1 << exp_bits) - 1
    nan_code = -1
    vals: list[float] = []
    for code in range(1 << (exp_bits + man_bits)):  # nonnegative half only
        exp = (code >> man_bits) & exp_max
        man = code & ((1 << man_bits) - 1)
        if no_inf_nan and exp == exp_max and man == (1 << man_bits) - 1:
            nan_code = code
            continue
        if exp == 0:
            v = (man / (1 << man_bits)) * (2.0 ** (1 - bias))  # subnormal
        else:
            v = (1.0 + man / (1 << man_bits)) * (2.0 ** (exp - bias))
        vals.append(v)
    assert len(vals) == len(set(vals)), "lattice must be strictly increasing with code"
    return vals, nan_code


_E2M1_VALS, _ = _ieee_lattice(2, 1, 1, no_inf_nan=False)   # {0,.5,1,1.5,2,3,4,6}
_E4M3_VALS, _E4M3_NAN = _ieee_lattice(4, 3, 7, no_inf_nan=True)
_E2M1_MAX = _E2M1_VALS[-1]          # 6.0
_E4M3_MAX = _E4M3_VALS[-1]          # 448.0
_E4M3_MIN_SUB = _E4M3_VALS[1]       # 2**-9


def _boundaries(vals: list[float]) -> torch.Tensor:
    """Midpoint thresholds between successive lattice points (exact halves of neighbor sums)."""
    lo = torch.tensor(vals[:-1], dtype=torch.float64)
    hi = torch.tensor(vals[1:], dtype=torch.float64)
    return ((lo + hi) / 2).to(torch.float32)


_E2M1_BOUNDS = _boundaries(_E2M1_VALS)
_E4M3_BOUNDS = _boundaries(_E4M3_VALS)
_E2M1_TABLE = torch.tensor(_E2M1_VALS, dtype=torch.float32)
_E4M3_TABLE = torch.tensor(_E4M3_VALS, dtype=torch.float32)


def snap_codes(abs_vals: torch.Tensor, bounds: torch.Tensor, table_len: int) -> torch.Tensor:
    """Nearest-lattice code with ties-to-even-index. `abs_vals` finite, >= 0, pre-clipped."""
    idx = torch.bucketize(abs_vals.contiguous().reshape(-1), bounds.reshape(-1))
    idx = idx.clamp_(max=table_len - 1)
    # exact-midpoint corrections: at a boundary, pick the even code
    at = bounds[idx.clamp(max=bounds.numel() - 1)] == abs_vals.reshape(-1) if bounds.numel() else torch.zeros_like(idx, dtype=torch.bool)
    odd = (idx % 2) == 1
    idx = idx + (at & odd).int()
    return idx.view_as(abs_vals).to(torch.int16)


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def decode_e4m3(scale_bytes: torch.Tensor) -> torch.Tensor:
    """u8 -> fp32 block scales. Rejects the NaN pattern and negative-sign bytes (scales are
    defined nonnegative in this container); a scale of 0 is legal (all-zero block)."""
    b = scale_bytes.to(torch.uint8)
    unsigned = b & 0x7F
    if torch.any(unsigned == _E4M3_NAN):
        raise ValueError("E4M3 NaN pattern in block scale")
    if torch.any(b >> 7):
        raise ValueError("negative block scale byte")
    return _E4M3_TABLE.to(unsigned.device)[unsigned.long()]


def decode_packed_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """u8[..., k] -> fp32[..., 2k], LOW nibble first, sign from each nibble's bit 3."""
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    def nib(n):
        mag = _E2M1_TABLE.to(n.device)[(n & 0x07).long()]
        return mag.where((n >> 3) == 0, -mag)
    return torch.stack((nib(lo), nib(hi)), dim=-1).flatten(start_dim=-2)


def decode_weights(packed: torch.Tensor, scale_bytes: torch.Tensor, global_scale: float,
                   out_dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """packed u8 [out, in/2] + E4M3 u8 [out, in/16] + fp32 scalar -> [out, in] `out_dtype`.
    Order: (element * block_scale) * global."""
    if packed.ndim != 2 or scale_bytes.ndim != 2:
        raise ValueError("2-D tensors expected")
    out_, inc2 = packed.shape
    if scale_bytes.shape != (out_, inc2 // 8):  # 8 packed bytes == 16 elements per block
        raise ValueError("scale/weight geometry mismatch")
    if not (global_scale > 0):
        raise ValueError("global scale must be positive")
    elems = decode_packed_e2m1(packed)
    sf = decode_e4m3(scale_bytes).repeat_interleave(16, dim=1)
    w = (elems * sf) * float(global_scale)
    return w.to(out_dtype)


# ---------------------------------------------------------------------------
# Quantize
# ---------------------------------------------------------------------------

def _pow2_ceiling(x: float) -> float:
    import math
    if x <= 0:
        return 1.0
    return 2.0 ** math.ceil(math.log2(x))


def quantize_weights(w: torch.Tensor, global_scale: float | None = None) -> tuple[torch.Tensor, torch.Tensor, float]:
    """fp tensor [out, in] (in %% 16 == 0) -> (packed u8, scale u8, global fp32). See module
    docstring for the scale policies."""
    if w.ndim != 2 or w.shape[1] % 16:
        raise ValueError("shape must be [out, in] with in divisible by 16")
    wf = w.to(torch.float32)
    if not torch.isfinite(wf).all():
        raise ValueError("non-finite weights")
    out_, inc = wf.shape
    absg = wf.abs().amax().item()
    if global_scale is None:
        global_scale = _pow2_ceiling(absg / (_E4M3_MAX * _E2M1_MAX)) if absg > 0 else 1.0
    blocks = wf.view(out_, inc // 16, 16)
    amax = blocks.abs().amax(dim=-1)                                    # [out, nb]
    sf_real = amax / (global_scale * _E2M1_MAX)                         # ideal scale
    # snap UPWARD: first lattice point >= sf_real (never below it) => no clipping of the block max.
    # Search the LATTICE itself (side='left'), not midpoints: first value >= x.
    idx = torch.bucketize(sf_real.contiguous().reshape(-1), _E4M3_TABLE.to(sf_real.device), right=False)
    idx = idx.clamp_(min=1).view_as(sf_real).to(torch.uint8)            # 0 would blank a nonzero block
    sf = _E4M3_TABLE.to(wf.device)[idx.long()]                          # [out, nb]
    scaled = (blocks / sf.unsqueeze(-1)) / global_scale                 # want |.| <= 6
    mag = snap_codes(scaled.abs().clamp(max=_E2M1_MAX), _E2M1_BOUNDS.to(wf.device), len(_E2M1_VALS))
    codes = mag.to(torch.uint8) | ((scaled < 0).to(torch.uint8) << 3)
    codes = codes.view(out_, inc)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed, idx, float(global_scale)

# ---------------------------------------------------------------------------
# GPU decode (fused). v1 is deliberately table-gather and arithmetically
# identical to decode_weights ((elem * sf) * g, fp32, RN to out dtype) so the
# conformance suite can demand BITWISE equality against the CPU oracle.
# Multiply-free lattice variants land only after this fence holds.
# ---------------------------------------------------------------------------

_GPU_TABLE_CACHE: dict = {}


def _gpu_tables(device) -> tuple:
    key = getattr(device, "index", None)
    t = _GPU_TABLE_CACHE.get(key)
    if t is None:
        t = (_E2M1_TABLE.to(device).contiguous(), _E4M3_TABLE.to(device).contiguous())
        _GPU_TABLE_CACHE[key] = t
    return t


if _HAS_TRITON:

    @triton.jit
    def _decode_kernel(packed_ptr, sf_ptr, e2m1_ptr, e4m3_ptr, out_ptr,
                       g, cols, in_bytes, nb_sf, n_blk,
                       BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        row = pid // n_blk
        blk = pid % n_blk
        offs = blk * BLOCK + tl.arange(0, BLOCK)
        mask = offs < cols
        p = tl.load(packed_ptr + row * in_bytes + offs // 2, mask=mask, other=0).to(tl.int32)
        nib = tl.where((offs % 2) == 0, p & 0xF, (p >> 4) & 0xF)   # low nibble first
        mag = tl.load(e2m1_ptr + (nib & 0x07), mask=mask, other=0.0)
        val = tl.where((nib >> 3) != 0, -mag, mag)
        sb = tl.load(sf_ptr + row * nb_sf + offs // 16, mask=mask, other=0).to(tl.int32) & 0x7F
        sf = tl.load(e4m3_ptr + sb, mask=mask, other=0.0)
        w = (val * sf) * g
        tl.store(out_ptr + row * cols + offs, w.to(out_ptr.dtype.element_ty), mask=mask)


if _HAS_TRITON:

    @triton.jit
    def _decode_kernel_arith(packed_ptr, sf_ptr, e2m1_ptr, e4m3_ptr, out_ptr,
                             g, cols, in_bytes, nb_sf, n_blk,
                             BLOCK: tl.constexpr):
        """Decode without table gathers: E2M1 and E4M3 magnitudes are assembled from
        exact f32 bit patterns (finite lattice ⇒ integer-assembled significands).
        Table args kept for the shared launch signature; intentionally unused."""
        pid = tl.program_id(0)
        row = pid // n_blk
        blk = pid % n_blk
        offs = blk * BLOCK + tl.arange(0, BLOCK)
        mask = offs < cols
        p = tl.load(packed_ptr + row * in_bytes + offs // 2, mask=mask, other=0).to(tl.int32)
        nib = tl.where((offs % 2) == 0, p & 0xF, (p >> 4) & 0xF)   # low nibble first
        m = nib & 0x07
        e2m1_bits = ((126 + (m >> 1)) << 23) | ((m & 1) << 22)     # codes>=2: exact normals
        mag = tl.where(m >= 2, e2m1_bits.to(tl.float32, bitcast=True),
              tl.where(m == 1, 0.5, 0.0))                          # 0.5 subnormal; zero
        val = tl.where((nib >> 3) != 0, -mag, mag)
        sb = tl.load(sf_ptr + row * nb_sf + offs // 16, mask=mask, other=0).to(tl.int32) & 0x7F
        e4m3_bits = ((120 + (sb >> 3)) << 23) | ((sb & 7) << 20)   # codes>=8: exact normals
        sf = tl.where(sb < 8, (sb & 7).to(tl.float32) * 0.001953125,   # subnormals c*2^-9
                      e4m3_bits.to(tl.float32, bitcast=True))
        w = (val * sf) * g
        tl.store(out_ptr + row * cols + offs, w.to(out_ptr.dtype.element_ty), mask=mask)



def _decode_launch(packed: torch.Tensor, scale_bytes: torch.Tensor, global_scale: float,
                   out_dtype: torch.dtype, kernel, num_warps: int, block_cols: int) -> torch.Tensor:
    """Shared validation + launch contract for every GPU decode kernel variant."""
    if not _HAS_TRITON:
        raise RuntimeError("triton unavailable on this host")
    if packed.ndim != 2 or scale_bytes.ndim != 2:
        raise ValueError("2-D tensors expected")
    out_, inc2 = packed.shape
    if scale_bytes.shape != (out_, inc2 // 8):
        raise ValueError("scale/weight geometry mismatch")
    if not (global_scale > 0):
        raise ValueError("global scale must be positive")
    u = scale_bytes.to(torch.uint8)
    if torch.any((u & 0x7F) == _E4M3_NAN):
        raise ValueError("E4M3 NaN pattern in block scale")
    if torch.any(u >> 7):
        raise ValueError("negative block scale byte")
    packed = packed.contiguous()
    u = u.contiguous()
    cols = inc2 * 2
    e2m1_tab, e4m3_tab = _gpu_tables(packed.device)
    out = torch.empty((out_, cols), dtype=out_dtype, device=packed.device)
    n_blk = (cols + block_cols - 1) // block_cols
    kernel[(out_ * n_blk,)](
        packed, u, e2m1_tab, e4m3_tab, out,
        float(global_scale), cols, inc2, u.shape[1], n_blk,
        BLOCK=block_cols, num_warps=num_warps)
    return out


def decode_weights_gpu(packed: torch.Tensor, scale_bytes: torch.Tensor, global_scale: float,
                       out_dtype: torch.dtype = torch.float16, num_warps: int = 4,
                       block_cols: int = 512) -> torch.Tensor:
    """Fused GPU twin of decode_weights; same geometry/scale validation, bitwise-equal
    output for identical inputs (guaranteed by the conformance suite, not assumed)."""
    return _decode_launch(packed, scale_bytes, global_scale, out_dtype,
                          _decode_kernel, num_warps, block_cols)


def decode_weights_gpu_arith(packed: torch.Tensor, scale_bytes: torch.Tensor, global_scale: float,
                             out_dtype: torch.dtype = torch.float16, num_warps: int = 4,
                             block_cols: int = 512) -> torch.Tensor:
    """Gather-free variant: both lattices assembled arithmetically from exact f32 bit
    patterns instead of two dependent table lookups. Same bitwise output contract."""
    return _decode_launch(packed, scale_bytes, global_scale, out_dtype,
                          _decode_kernel_arith, num_warps, block_cols)
