#!/usr/bin/env python3
"""Fence: the coop-drive wiring plan tracks the C++ interfaces it will call.

CPU-only (workstation law: never import exllamav3 here). moe_coop_plan is imported BY
PATH — it is pure python with no package deps. Checks:
  1. Activation/cap constants == the #defines/constexprs parsed FROM THE HEADERS.
  2. CALL_ARGS (positional order for pybind) == the exl3_moe_coop parameter names,
     IN ORDER, parsed from exl3_moe_coop.cuh — arity and order, not just count.
  3. ctr_len mirrors the C++ inline body verbatim (whitespace-normalised) + samples.
  4. classify_experts/eligible semantics on fake module-likes.
POSITIVE CONTROL: a planted rename in the parsed signature must trip check 2.
"""

import importlib.util
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
PLAN = ROOT / "exllamav3/modules/arch_specific/moe_coop_plan.py"
COOP_CUH = ROOT / "exllamav3/exllamav3_ext/quant/exl3_moe_coop.cuh"
KERNEL_CUH = ROOT / "exllamav3/exllamav3_ext/quant/exl3_moe_coop_kernel.cuh"
COMMON_CUH = ROOT / "exllamav3/exllamav3_ext/moe_group/moe_sm89_common.cuh"

_spec = importlib.util.spec_from_file_location("moe_coop_plan", PLAN)
plan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plan)

fails = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{(': ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


def parse_define(text, name):
    m = re.search(r"#define\s+" + name + r"\s+(\S+)", text)
    return m.group(1) if m else None


def coop_signature(text):
    m = re.search(r"void exl3_moe_coop\s*\((.*?)\)\s*;", text, re.S)
    if m is None:
        raise AssertionError("exl3_moe_coop signature not found in " + str(COOP_CUH))
    params = []
    for chunk in m.group(1).split(","):
        chunk = re.sub(r"/\*.*?\*/", "", chunk).strip()
        if not chunk:
            continue
        name = re.split(r"[\s*&]+", chunk)[-1].strip()
        params.append(name)
    return params


class FakeLin:
    def __init__(self, K, mcg=False, mul1=False, bias=None):
        self.K, self.mcg, self.mul1, self.bias = K, mcg, mul1, bias


def main():
    coop = COOP_CUH.read_text()
    kern = KERNEL_CUH.read_text()
    common = COMMON_CUH.read_text()

    print("[1] constants mirror the headers")
    check("MOE_COOP_ACT_SOFTCAP",
          parse_define(coop, "MOE_COOP_ACT_SOFTCAP") == str(plan.MOE_COOP_ACT_SOFTCAP),
          parse_define(coop, "MOE_COOP_ACT_SOFTCAP") or "missing")
    check("MOE_MAX_DECODE_TOKENS",
          parse_define(common, "MOE_MAX_DECODE_TOKENS") == str(plan.MOE_MAX_DECODE_TOKENS),
          parse_define(common, "MOE_MAX_DECODE_TOKENS") or "missing")
    m = re.search(r"constexpr int MAX_SLOTS\s*=\s*(\d+)", kern)
    check("MOE_COOP_MAX_SLOTS == coop MAX_SLOTS", m and m.group(1) == str(plan.MOE_COOP_MAX_SLOTS),
          m.group(1) if m else "missing")

    print("[2] positional call order == the pybind signature (parsed from header)")
    sig = coop_signature(coop)
    check("arity", len(sig) == len(plan.CALL_ARGS), f"header {len(sig)} vs plan {len(plan.CALL_ARGS)}")
    check("order", sig == list(plan.CALL_ARGS),
          "first divergence: " + (next((f"{a}!={b}" for a, b in zip(sig, plan.CALL_ARGS) if a != b), "-")))
    planted = list(sig)
    planted[6], planted[9] = planted[9], planted[6]  # pretend g_trellis and u_trellis swapped in C++
    check("POSITIVE CONTROL: planted rename caught", planted != list(plan.CALL_ARGS))

    print("[3] ctr_len mirrors the C++ inline")
    m = re.search(r"inline int64_t exl3_moe_coop_ctr_len\([^)]*\)\s*\{\s*return ([^;]+);", coop)
    body = re.sub(r"\s+", " ", m.group(1)).replace("(int64_t)", "").strip() if m else None
    mine = "slots_max * (I / 128) + bsz_max * (Ho / 128) + 2 + (slots_max + 1) + slots_max"
    check("formula text", body == mine, body or "signature not found")
    check("sample rows=1", plan.ctr_len(8, 1, 9216, 2304) == 8 * 72 + 18 + 2 + 9 + 8)
    check("sample cap", plan.ctr_len(256, 32, 9216, 2304) == 256 * 72 + 32 * 18 + 2 + 257 + 256)

    print("[4] eligibility semantics on fake modules")
    uni = [(FakeLin(3), FakeLin(3), FakeLin(3))] * 128
    desc = plan.classify_experts(uni)
    check("uniform pack classifies", desc == {"Kg": 3, "Ku": 3, "Kd": 3, "mcg": False, "mul1": False})
    check("fractional bitrate", plan.classify_experts([(FakeLin(3.05, mul1=True), FakeLin(3.05, mul1=True), FakeLin(3.05, mul1=True))]) is not None)
    check("gate/up bit split refused", plan.classify_experts([(FakeLin(3), FakeLin(4), FakeLin(3))]) is None)
    mixed = [(FakeLin(3), FakeLin(3), FakeLin(3))] * 4 + [(FakeLin(3), FakeLin(3), FakeLin(4))]
    check("per-expert bitrate drift refused", plan.classify_experts(mixed) is None)
    cb_mix = [(FakeLin(3), FakeLin(3), FakeLin(3, mcg=True))]
    check("mixed codebooks refused", plan.classify_experts(cb_mix) is None)
    biased = [(FakeLin(3, bias=object()), FakeLin(3), FakeLin(3))]
    check("biased pack refused", plan.classify_experts(biased) is None)
    check("prefill refused by cap", not plan.eligible(plan.MOE_MAX_DECODE_TOKENS + 1, 8, True, True, True))
    check("slot cap enforced", not plan.eligible(32, 9, True, True, True))
    check("fp32 input refused", not plan.eligible(1, 8, True, False, True))
    check("no ext entry refused", not plan.eligible(1, 8, True, True, False))
    check("decode bs=1 admitted", plan.eligible(1, 8, True, True, True))

    print("[5] build_call_args pins order and fails loudly on a missing key")
    vals = {n: n for n in plan.CALL_ARGS}
    check("round-trip order", plan.build_call_args(vals) == tuple(plan.CALL_ARGS))
    vals.pop("mcg")
    try:
        plan.build_call_args(vals)
        check("missing key raises", False)
    except KeyError:
        check("missing key raises", True)

    if fails:
        print("MOE COOP PLAN FENCE RED:", ", ".join(fails))
        return 1
    print("MOE COOP PLAN FENCE GREEN: wiring mirrors the headers; planted drift caught.")
    return 0


if __name__ == "__main__":
    sys.exit(main())