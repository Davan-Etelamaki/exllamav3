#!/usr/bin/env python3
"""Elementwise fence: the merged coop SOFTCAP act branch == served inkling2_loop.py epilogue.

TODO-1 gate from the Integration decision: MOE_COOP_ACT_SOFTCAP (cherry-picked as
020ce47c from dsh-softcap aae2681f) must compute EXACTLY the served epilogue in
Inkling2LoopSparseMoeBlock.forward — gate tanh-cap line, sigmoid line, up tanh-cap line
(numbers decay with edits: 1246-1248 at tip 355f7530; the SERVED_ANCHORS strings below
are the authoritative binding — supervisor note 2026-11-25):
    act = tanh(g/cg)*cg * sigmoid(g) * (tanh(u/cu)*cu)

The compared C++ expression is EXTRACTED from the tree (exll3_moe_coop_kernel.cuh's
MOE_COOP_ACT_SOFTCAP branch), not from my memory of it: the fence tracks the code.
Both sides run fp32 torch on CPU. Positive control: feeding the gate cap for BOTH
caps (the classic single-cap mistake) must go RED.
"""

import re
import sys

import torch

GATE_CAP = 4.0        # served gate_softcap (anchor: "act = torch.tanh(g / self.gate_softcap)")
UP_CAP = 25.0         # served up softcap  (anchor: "act = act * (torch.tanh(u / self.up_softcap)")
SEED = 20261125
N = 1 << 20
ATOL = 1e-6           # fp32 reassociation only
RTOL = 1e-5

KERNEL = "exllamav3/exllamav3_ext/quant/exl3_moe_coop_kernel.cuh"
SERVED = "exllamav3/modules/arch_specific/inkling2_loop.py"
SERVED_ANCHORS = (
    "act = torch.tanh(g / self.gate_softcap) * self.gate_softcap",
    "act = act * torch.sigmoid(g)",
    "act = act * (torch.tanh(u / self.up_softcap) * self.up_softcap)",
)


def extract_kernel_expr(path):
    """Return a python callable for the MOE_COOP_ACT_SOFTCAP return expression."""
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    m = re.search(
        r"if \(act == MOE_COOP_ACT_SOFTCAP\)\s*\n\s*return ([^;]+);", src
    )
    if m is None:
        raise AssertionError(f"no MOE_COOP_ACT_SOFTCAP return found in {path}")
    expr = m.group(1)
    py = (expr.replace("tanhf(", "torch.tanh(")
              .replace("expf(", "torch.exp(")
              .replace("1.0f", "1.0"))
    return py, expr


def served_act(g, u):
    """Verbatim the served epilogue triplet (SERVED_ANCHORS below) with the Inkling caps."""
    act = torch.tanh(g / GATE_CAP) * GATE_CAP
    act = act * torch.sigmoid(g)
    act = act * (torch.tanh(u / UP_CAP) * UP_CAP)
    return act


def main():
    for anchor in SERVED_ANCHORS:
        if anchor not in open(SERVED, encoding="utf-8").read():
            print(f"FENCE RED: served epilogue moved; missing anchor: {anchor!r}")
            return 1

    py_expr, raw_expr = extract_kernel_expr(KERNEL)
    print("kernel branch (extracted from tree):", raw_expr)

    gen = torch.Generator().manual_seed(SEED)
    g = torch.randn(N, generator=gen, dtype=torch.float32) * 8.0
    u = torch.randn(N, generator=gen, dtype=torch.float32) * 8.0

    env = {"torch": torch, "g": g, "u": u, "limit": GATE_CAP, "up_cap": UP_CAP}
    kernel_act = eval(py_expr, env)  # noqa: S307 - expression extracted above
    ref = served_act(g, u)
    ok = torch.allclose(kernel_act, ref, atol=ATOL, rtol=RTOL)
    diff = (kernel_act - ref).abs()
    md = diff.max().item()
    at_worst = int(diff.argmax())
    eff = ATOL + RTOL * ref.abs().max().item()
    print(f"elementwise N={N}: max|delta| = {md:.3e} at |ref| = {ref.abs().max().item():.3e}")
    print(f"EFFECTIVE allclose bound at that magnitude: atol {ATOL} + rtol {RTOL} * |ref| = {eff:.3e}"
          f" (max|delta| is {eff / md:.1f}x under)")

    # POSITIVE CONTROL: single-cap mistake (gate cap used for both caps) must breach.
    bad_env = dict(env)
    bad_env["up_cap"] = GATE_CAP
    bad = eval(py_expr, bad_env)  # noqa: S307
    ctrl_ok = torch.allclose(bad, ref, atol=ATOL, rtol=RTOL)
    ctrl_md = (bad - ref).abs().max().item()
    print(f"control (up_cap:=gate cap): max|delta| = {ctrl_md:.3e}, "
          f"{'BREACHED as required' if not ctrl_ok else 'NOT BREACHED'}")

    if ok and not ctrl_ok:
        print("SOFTCAP ELEMENTWISE FENCE GREEN: coop branch == served epilogue; "
              "single-cap control RED as required.")
        return 0
    print("SOFTCAP ELEMENTWISE FENCE RED")
    return 1


if __name__ == "__main__":
    sys.exit(main())