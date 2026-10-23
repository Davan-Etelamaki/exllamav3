"""Packed-resident NVFP4 inner for Linear — engine duck-type of the fp16 inner.

Holds the weight as the packed triple (nibble payload, E4M3 block scales, f32
global scale) — 4.5 bits per element resident — and evaluates y = x @ W^T
through the dispatching lane (packed-direct matvec at small batch, block-decode
scratch + wide GEMM at large batch). The triple is validated once at
construction; forwards take the scan-free hot path.

Storage orientation is checkpoint orientation (out_features, in_features), the
lane's natural domain, so the loader never transposes packed bytes. Decoding
helpers reproduce the fp16 inner's tensor surface on demand for consumers that
need full-width tensors (save/quantize/importance paths) — transient, freed on
return; nothing here parks a decoded copy resident.
"""

from __future__ import annotations

import torch

from . import nvfp4_fmt
from . import nvfp4_w4a16 as lane


def pad_triple(packed: torch.Tensor, scales: torch.Tensor,
               rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Grow the triple to (rows, cols) in packed space with exact-zero padding.

    Payload bytes 0x00 decode to value 0 and scale bytes 0x00 are legal E4M3
    zero, so padded rows/columns contribute exactly zero. Targets must satisfy
    the format geometry (cols a multiple of 16); never shrinks.
    """
    r0, half = packed.shape
    c0 = half * 2
    if cols % 16 or half * 2 % 16:
        raise ValueError("padding targets must respect the 16-element scale block")
    if rows < r0 or cols < c0:
        raise ValueError("pad_triple grows only; slicing is a separate operation")
    if (rows, cols) == (r0, c0):
        return packed, scales
    p = torch.zeros((rows, cols // 2), dtype=torch.uint8, device=packed.device)
    s = torch.zeros((rows, cols // 16), dtype=torch.uint8, device=scales.device)
    p[:r0, :half] = packed
    s[:r0, :c0 // 16] = scales
    return p, s


class LinearNVFP4Packed:

    quant_type: str = "nvfp4"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        packed: torch.Tensor,
        scales: torch.Tensor,
        global_scale,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        key: str | None = None,
    ):
        rows, cols, g = lane.validate_triple(packed, scales, global_scale)
        if (rows, cols) != (out_features, in_features):
            raise ValueError(f"triple geometry {(rows, cols)} != module "
                             f"{{out,in}} {(out_features, in_features)}; pad at load")
        if bias is not None and bias.dtype == torch.float:
            bias = bias.to(torch.half)
        self.in_features = in_features
        self.out_features = out_features
        self.packed = packed.contiguous()
        self.scales = scales.contiguous()
        self.global_scale = g
        self.bias = bias
        self.out_dtype = out_dtype
        self.key = key
        self.swap_device = None
        self.bc = None       # no block-cache handle: the lane owns its tiling

    # ---- hot path -------------------------------------------------------------

    def forward(self, x: torch.Tensor, params: dict,
                out_dtype: torch.dtype | None = None) -> torch.Tensor:
        out_shape = x.shape[:-1] + (self.out_features,)
        x2 = x.view(-1, x.shape[-1])
        dtype = out_dtype or self.out_dtype or torch.half
        xs = x2 if x2.dtype is torch.half else x2.half()
        y = lane.matmul(xs, self.packed, self.scales, self.global_scale)
        if self.bias is not None:
            y = y + self.bias
        return y.to(dtype).view(out_shape)

    # ---- on-demand full-width views (transient; nothing parked resident) ------

    def decode(self, dtype: torch.dtype = torch.half) -> torch.Tensor:
        if self.packed.is_cuda:
            return nvfp4_fmt.decode_weights_gpu(self.packed, self.scales,
                                                self.global_scale, dtype)
        return nvfp4_fmt.decode_weights(self.packed, self.scales,
                                        self.global_scale, dtype)

    def get_weight_tensor(self) -> torch.Tensor:
        # fp16 inner stores [in, out]; match that layout for importance/save paths
        return self.decode(torch.half).t().contiguous()

    def get_bias_tensor(self) -> torch.Tensor | None:
        return self.bias

    def get_tensors(self, key: str):
        t = {f"{key}.weight": self.decode(torch.half)}   # checkpoint orientation
        if self.bias is not None:
            t[f"{key}.bias"] = self.bias
        return t

    # ---- device parking (mirrors the fp16 inner used by conversion paths) -----

    def unload(self):
        pass

    def swap_cpu(self):
        if self.swap_device is not None:
            return
        self.swap_device = self.packed.device
        self.packed = self.packed.cpu()
        self.scales = self.scales.cpu()
        if self.bias is not None:
            self.bias = self.bias.cpu()

    def unswap_cpu(self):
        if self.swap_device is None:
            return
        self.packed = self.packed.to(self.swap_device)
        self.scales = self.scales.to(self.swap_device)
        if self.bias is not None:
            self.bias = self.bias.to(self.swap_device)
        self.swap_device = None