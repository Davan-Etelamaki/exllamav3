"""Packed-resident weight container for the NVFP4 seam.

Stores the on-disk/in-GPU representation the loader produces and the compute lanes
consume: ``packed`` u8 [out, in/2] (two E2M1 nibbles per byte, low nibble = even
column), ``scales`` u8 [out, in/16] (one E4M3 byte per 16-element block), and one
fp32 ``global_scale``.  This is the residency win of the seam: 4.5 bits per weight
element instead of 16 for fp16.

Numerics are delegated entirely to :mod:`nvfp4_fmt` (spec-derived oracle on CPU,
fused kernels on GPU, all three pairwise BITWISE-equal), so this class carries no
arithmetic of its own — only validation and movement.  Construction is the trust
boundary for deserialized tensors: geometry, dtypes, positivity and the illegal
E4M3 scale encodings (NaN pattern 0x7F under mask 0x7F; a set sign bit 0x80 is
not a legal block scale) all raise :class:`ValueError` here, before any kernel
sees the data.
"""
from __future__ import annotations

import math

import torch

try:                                    # normal package import
    from . import nvfp4_fmt as fmt
except ImportError:                     # bare file load by the conformance runner
    import importlib.util
    import pathlib

    _spec = importlib.util.spec_from_file_location(
        "nvfp4_fmt", pathlib.Path(__file__).with_name("nvfp4_fmt.py"))
    fmt = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(fmt)

_E4M3_POISON_MASK = 0x7F          # sign-agnostic pattern of the illegal NaN code


class Nvfp4PackedWeight:
    """Immutable packed (weight, scale, global) triple with spec-exact decode."""

    __slots__ = ("packed", "scales", "global_scale")

    def __init__(self, packed: torch.Tensor, scales: torch.Tensor, global_scale: float):
        if packed.dtype is not torch.uint8 or scales.dtype is not torch.uint8:
            raise ValueError("packed and scales must be uint8 byte buffers")
        if packed.ndim != 2 or scales.ndim != 2:
            raise ValueError("2-D tensors expected")
        rows, half_cols = packed.shape
        if half_cols % 8:
            raise ValueError("packed columns must be a multiple of 8 (16 elements)")
        if scales.shape != (rows, half_cols // 8):
            raise ValueError("scale/weight geometry mismatch")
        if not (isinstance(global_scale, float) and math.isfinite(global_scale) and global_scale > 0):
            raise ValueError("global scale must be a positive finite float")
        if bool(((scales & _E4M3_POISON_MASK) == _E4M3_POISON_MASK).any()):
            raise ValueError("illegal E4M3 scale byte: NaN encoding")
        if bool((scales & 0x80).any()):
            raise ValueError("illegal E4M3 scale byte: sign bit set")
        self.packed = packed
        self.scales = scales
        self.global_scale = float(global_scale)

    # ------------------------------------------------------------------ build

    @classmethod
    def from_float(cls, w: torch.Tensor, global_scale: float | None = None) -> "Nvfp4PackedWeight":
        packed, scales, g = fmt.quantize_weights(w, global_scale)
        return cls(packed, scales, g)

    # ---------------------------------------------------------------- inspect

    @property
    def shape(self) -> tuple[int, int]:
        return self.packed.shape[0], self.packed.shape[1] * 2

    @property
    def nbytes(self) -> int:
        return self.packed.numel() + self.scales.numel()

    @property
    def device(self):
        return self.packed.device

    # ---------------------------------------------------------------- movement

    def to(self, device) -> "Nvfp4PackedWeight":
        return Nvfp4PackedWeight(self.packed.to(device), self.scales.to(device),
                                 self.global_scale)

    # ------------------------------------------------------------------ decode

    def decode(self, out_dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """(element * block_scale) * global, BITWISE-equal across CPU oracle and
        GPU kernels; kernel selection follows where the buffers live."""
        if self.packed.is_cuda:
            return fmt.decode_weights_gpu(self.packed, self.scales, self.global_scale, out_dtype)
        return fmt.decode_weights(self.packed, self.scales, self.global_scale, out_dtype)