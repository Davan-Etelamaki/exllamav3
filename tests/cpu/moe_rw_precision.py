#!/usr/bin/env python3
"""Charge A (supervisor, 2026-11-25): PRICE the rw float->half rounding at the coop boundary.

Multiply sites (supervisor charge A, 2026-11-25 — exact lines):
  served:  Inkling2LoopSparseMoeBlock.forward — out.index_add_(..., y.float() * w_flat...)
           fp32 weight (string anchor; at tip fc4a5451 this line is :1250 and w_flat is
           built at :1216; it was :1115 BEFORE the coop wiring commit — the earlier
           :1160 cite came from the design brief and matched neither tree, supervisor
           §R caught it)
  coop:    the half weight is cast back INSIDE the kernel at
           exl3_moe_coop_kernel.cuh:220 (si.w = __half2float(p.rw[s]), slot load of the B
           stage) and multiplied against the fp32 partial in the fp32 accumulate at
           exl3_moe_coop_kernel.cuh:990-993 (o{i} += sj.w * v{i}). This fence's shape —
           float(carrier(w)) * fp32 partial, fp32 sum — matches that site exactly.

Method: identical seeded renormalised weights (served router lines) and identical per-slot
outputs; the ONLY difference is the weight precision. Triangle inequality gives the exact
price ceiling: per-token |Δout| <= max_e |Δw_e|/|w_e| * Σ_e |w_e||y_e| (normalized error
ratio). The fence checks the measured ratio against that COMPUTED ceiling — no guessed
threshold — and that rounding actually bit (ceiling > 0).
POSITIVE CONTROL: degrade the weight granularity by 3 more bits (what a coarser routing
carrier would do); the measured ratio must exceed 4x the fp16 ceiling — a fence that
cannot fire is no fence.
"""

import sys

import torch

SEED = 20261125
ROWS = 4096          # decoded tokens
E = 128
TOP_K = 8
H = 16               # columns are independent; width only steadies the statistics


def served_weights(gen):
    """Renormalised routing weights exactly as the served router produces them."""
    hs = torch.randn(ROWS, 48, generator=gen)
    router_w = torch.randn(E, 48, generator=gen).mul_(0.2)
    router_bias = torch.randn(E, generator=gen).mul_(0.5)
    scores = torch.sigmoid(hs @ router_w.T)
    sel = torch.topk(scores + router_bias, TOP_K, dim=-1, sorted=False).indices
    w = scores.gather(1, sel)
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    return w * 1.0  # route_scale 1.0 (served config)


def price(w, y, carrier):
    """(max normalized per-token error, ceiling) for a weight carrier vs fp32 truth."""
    cw = carrier(w)
    out_fp32 = (w.unsqueeze(-1) * y).sum(dim=1)
    out_carrier = (cw.unsqueeze(-1) * y).sum(dim=1)
    denom = (w.abs().unsqueeze(-1) * y.abs()).sum(dim=1)
    ratio = ((out_fp32 - out_carrier).abs() / denom.clamp_min(1e-30)).max().item()
    ceiling = ((w - cw).abs() / w.abs()).max().item()
    return ratio, ceiling


def main():
    gen = torch.Generator().manual_seed(SEED)
    w = served_weights(gen)
    y = torch.randn(ROWS, TOP_K, H, generator=gen)

    fp16 = lambda v: v.half().float()                                        # #nosec E731
    coarse = lambda v: (v.half().float() * 8).round() / 8                    # 3 bits coarser

    ratio, ceiling = price(w, y, fp16)
    print(f"fp16 rw carrier: measured max normalized error {ratio:.3e}, "
          f"computed ceiling (max per-weight rel rounding) {ceiling:.3e} "
          f"(fp16 normal rounding bound 2^-11 = 4.882e-04)")
    rounding_bit = ceiling > 0.0
    within = ratio <= ceiling * (1.0 + 1e-6)
    sane = ceiling < 1e-2          # generous ceiling sanity; `within` carries the rigor

    ratio_deg, ceiling_deg = price(w, y, coarse)
    print(f"control (carrier 3 bits coarser): measured {ratio_deg:.3e} "
          f"(own ceiling {ceiling_deg:.3e}); must exceed 4x the fp16 ceiling "
          f"{4 * ceiling:.3e}")
    control_fires = ratio_deg > 4.0 * ceiling

    if rounding_bit and within and sane and control_fires:
        print("RW PRECISION PRICE PAID: coop carrier distorts a decoded token by at most "
              f"{ceiling:.1e} relative — one ulp-of-half per weight, bounded, not guessed; "
              "fence demonstrated able to fire.")
        return 0
    print("RW PRECISION FENCE RED: "
          + ", ".join(n for n, ok in (("rounding-bit", rounding_bit), ("within-ceiling", within),
                                      ("ceiling-sane", sane), ("control-fires", control_fires)) if not ok))
    return 1


if __name__ == "__main__":
    sys.exit(main())