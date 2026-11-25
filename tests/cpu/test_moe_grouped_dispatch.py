"""
FENCE 1 (unit, CPU, instant) — sm89 MoE grouped-dispatch lane, branch dsh-moe.

Spec (Launchers/.wip/moe_device_routing_design_2026-11-25.md, fence spec 1):
    the current python dispatch loop moved verbatim into the test as reference, vs the grouped
    kernel at seeded routing, T in {1,8,64}, expert set shuffled twice; outputs match within
    per-dtype bounds on identical dequant operands. POSITIVE CONTROL: swap two experts in the
    kernel's pointer table -> the test MUST go RED naming the swapped pair (a green here means
    the test never exercised >=2 experts).

Runnable two ways:
- pytest (dot26 CI): plain test_* functions, marker nogpu.
- bare python on the workstation (no pytest installed there): python3 tests/cpu/test_moe_grouped_dispatch.py
  — this module never imports exllamav3 (importing it initializes CUDA; the workstation GPU serves prod).

TIE-BREAK SPEC (supervisor charge C, 2026-11-25): where biased router scores tie, the DEVICE
rule is authoritative — LOWEST EXPERT ID WINS (moe_route.cu's butterfly reduces ties to the
minimum index). torch.topk's tie order is unspecified, so route() ASSERTS each seeded case is
tie-free: membership is then deterministic and a parity failure can never be an unspecified
torch tie posing as one.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_grouped_sim import (  # noqa: E402
    compare_outputs, grouped_dispatch, make_case, reference_dispatch,
)

try:  # pytest exists in CI, not in the workstation's venv; bare-python runs must not need it
    import pytest
    pytestmark = [pytest.mark.nogpu]
except ImportError:
    pytestmark = []

# Per-dtype bounds for fp32 accumulation over <= top_k contributions of O(10) magnitude.
# Calibrated 2026-11-25 at SEED: observed max |ref-grouped| across all six cases = 2.861e-06
# (printed by main()); the pinned bounds sit ~35x above that, while the positive-control swap
# disagrees at 2.64e+04 x these bounds — the test is sensitive to a wrong pointer table and
# insensitive only to fp reassociation.
ATOL = 1e-4
RTOL = 1e-4

SEED = 20261125
TOKENS = (1, 8, 64)
SERVED_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "exllamav3", "modules", "arch_specific", "inkling2_loop.py")

# Single-line, wrap-proof anchors into the served dispatch (memory law: line-independent markers).
SERVED_ANCHORS = (
    'sel_cpu = host_values(sel_flat, ("inkling2", "moe_sel")).tolist()',
    'order = sorted(range(len(sel_cpu)), key = lambda i: sel_cpu[i])',
    "tokens = torch.div(entries, self.top_k, rounding_mode = \"floor\")",
    "act = torch.tanh(g / self.gate_softcap) * self.gate_softcap",
    "out.index_add_(0, tokens, y.float() * w_flat.index_select(0, entries).unsqueeze(-1))",
)


def _cases():
    """Seeded routing for T in {1,8,64}; expert set shuffled twice (two distinct permutations)."""
    gen = torch.Generator().manual_seed(SEED)
    perms = [None]
    for _ in range(2):
        perms.append(torch.randperm(128, generator = gen).long())
    for t in TOKENS:
        for round_id in (0, 1):
            perm = perms[round_id + 1] if round_id else None
            yield t, round_id, make_case(SEED + t, t, perm = perm)


def test_served_dispatch_markers_present():
    """Attribution guard: the reference loop in moe_grouped_sim.py is verbatim against THIS tree.
    If the served dispatch changes shape, this goes RED and the fence must be re-derived, not inherited."""
    with open(SERVED_FILE, "r", encoding = "utf-8") as f:
        src = f.read()
    missing = [a for a in SERVED_ANCHORS if a not in src]
    assert not missing, f"served dispatch moved (markers absent from {SERVED_FILE}): {missing}"


def test_grouped_matches_reference_and_positive_control():
    lines = []
    sets_by_round = {}
    for t, round_id, case in _cases():
        flat = case["x"].reshape(-1, case["x"].shape[-1])
        sel_flat = case["sel"].reshape(-1)
        w_flat = case["w"].reshape(-1)
        support = set(int(e) for e in sel_flat.tolist())
        # Positive-control precondition: more than one expert is actually exercised.
        assert len(support) >= 2, (
            f"T={t} round={round_id}: routing selected only {sorted(support)} — the fence would "
            f"be inert to a swapped expert pair")
        if round_id not in sets_by_round:
            sets_by_round[round_id] = set()
        sets_by_round[round_id] |= support

        ref = reference_dispatch(flat, sel_flat, w_flat, case["table"])
        got = grouped_dispatch(flat, sel_flat, w_flat, case["table"])
        ok, info = compare_outputs(ref, got, case["sel"], w_flat, ATOL, RTOL)
        max_abs = float((ref - got).abs().max())
        lines.append(f"T={t:3d} shuffle={round_id} support={len(support):3d} experts  "
                     f"max|delta|={max_abs:.3e}  {'MATCH' if ok else 'MISMATCH'}")
        assert ok, f"T={t} shuffle={round_id}: grouped dispatch disagrees with served loop. {info}"

    a = sets_by_round[0]
    b = sets_by_round[1]
    assert a != b, f"expert-set shuffle did not change the exercised set: {sorted(a)}"

    # ---------------- POSITIVE CONTROL ----------------
    # Swap two experts INSIDE THE GROUPED IMPLEMENTATION'S POINTER TABLE ONLY. The reference keeps
    # its array indexing. A table dereferenced per expert must change the grouped output, and the
    # breach must land on tokens routed through BOTH swapped experts.
    _, _, case = [c for c in _cases() if c[0] == 8][-1]
    flat = case["x"].reshape(-1, case["x"].shape[-1])
    sel_flat = case["sel"].reshape(-1)
    w_flat = case["w"].reshape(-1)
    ref = reference_dispatch(flat, sel_flat, w_flat, case["table"])
    support = sorted(set(int(e) for e in sel_flat.tolist()))
    ea, eb = support[0], support[-1]
    swapped = list(case["table"])
    swapped[ea], swapped[eb] = swapped[eb], swapped[ea]
    assert swapped[ea] is not case["table"][ea], "swap is a no-op: table slots share one bundle"
    got = grouped_dispatch(flat, sel_flat, w_flat, swapped)
    ok, info = compare_outputs(ref, got, case["sel"], w_flat, ATOL, RTOL)
    if ok:
        raise AssertionError(
            f"POSITIVE CONTROL INERT: swapping pointer-table experts ({ea}, {eb}) left the grouped "
            f"output unchanged — the test never exercised >=2 experts")
    diff = (ref - got).abs()
    breach = diff > (ATOL + RTOL * ref.abs())
    tok_a = [(case["sel"] == ea).any(dim = -1)]
    tok_b = [(case["sel"] == eb).any(dim = -1)]
    hit_a = bool((breach & tok_a[0].unsqueeze(-1)).any())
    hit_b = bool((breach & tok_b[0].unsqueeze(-1)).any())
    assert hit_a and hit_b, (
        f"POSITIVE CONTROL MISATTRIBUTED: swapped pair ({ea}, {eb}) — tokens routed to "
        f"expert {ea} affected: {hit_a}, expert {eb} affected: {hit_b}. {info}")
    lines.append(f"POSITIVE CONTROL RED AS REQUIRED: pointer-table swap of experts ({ea}, {eb}) "
                 f"disagreed on {int(breach.any(dim=-1).sum())} tokens; {info}")
    return lines


def main():
    try:
        lines = test_grouped_matches_reference_and_positive_control()
        for ln in lines:
            print(ln)
        test_served_dispatch_markers_present()
        print("served-dispatch markers present: 5/5 anchors in inkling2_loop.py")
        print("FENCE 1 GREEN")
    except AssertionError as exc:
        print(f"FENCE 1 RED: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())