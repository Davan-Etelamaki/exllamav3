"""Plan layer for driving the exl3_moe_coop decode tier from Inkling2LoopSparseMoeBlock.

Pure python + duck typing: NO exllamav3.ext import, so the CPU fence (tests/cpu/
moe_coop_plan_fence.py) runs on the workstation. All CUDA-facing calls live in the
served module; this file decides eligibility, sizes the fixed-address arena and fixes
the positional argument order of the pybind call. The argument order and the activation
constant are FENCED against the C++ header by tests/cpu/moe_coop_plan_fence.py.
"""
from __future__ import annotations

# mirrors #define MOE_COOP_ACT_SOFTCAP 4 (exl3_moe_coop.cuh) — fence re-parses the header
MOE_COOP_ACT_SOFTCAP = 4
# mirrors #define MOE_MAX_DECODE_TOKENS 32 (moe_sm89_common.cuh) — fence re-parses the header
MOE_MAX_DECODE_TOKENS = 32
# mirrors constexpr int MAX_SLOTS = 256 (exl3_moe_coop_kernel.cuh) — fence re-parses
MOE_COOP_MAX_SLOTS = 256

# Positional argument names of the pybind entry, IN ORDER (arity is fenced against the
# header signature): the call is assembled through this tuple so a signature change on
# either side trips the fence instead of silently shifting a pointer table into K_gu.
CALL_ARGS = (
    "x", "sel", "rw", "min_expert", "max_expert", "Hi",
    "g_trellis", "g_suh", "g_svh", "u_trellis", "u_suh", "u_svh",
    "d_trellis", "d_suh", "d_svh", "g_bias", "u_bias", "d_bias",
    "Kg", "Ku", "Kd", "mcg", "mul1", "act", "act_limit", "up_cap", "gated",
    "had_g", "had_u", "gu_g", "gu_u", "act_out", "d_out", "ctr", "out",
    "sh_out", "sh_gate_w",
)


def ctr_len(slots_max: int, bsz_max: int, I: int, Ho: int) -> int:
    """Mirror of inline exl3_moe_coop_ctr_len (exl3_moe_coop.cuh): completion counters
    + run table. Int division matches the C++ ints (I, Ho are multiples of 128)."""
    return slots_max * (I // 128) + bsz_max * (Ho // 128) + 2 + (slots_max + 1) + slots_max


def classify_experts(groups) -> dict | None:
    """Collapse 3*num_experts LinearEXL3 modules to one (K, cb) description.

    groups: iterable of (g, u, d) triples of module-likes exposing .K/.mcg/.mul1/.bias.
    Returns {"Kg":.., "Ku":.., "Kd":.., "mcg":bool, "mul1":bool} or None when the pack
    is not homogeneous enough for one coop call (the served path treats None as
    'not eligible: keep the python dispatch').
    """
    desc = None
    for g, u, d in groups:
        if g.bias is not None or u.bias is not None or d.bias is not None:
            return None  # the served Inkling pack carries no expert bias; biased packs
                         # keep the python dispatch until bias tables are wired
        if getattr(g, "K", None) != getattr(u, "K", None):
            return None  # coop requires gate/up to share a bit width (prepare enforces)
        cb_flags = {(bool(getattr(m, "mcg", False)), bool(getattr(m, "mul1", False)))
                    for m in (g, u, d)}
        if len(cb_flags) != 1:
            return None  # mixed codebook families across projections/experts
        flags = cb_flags.pop()
        cur = {"Kg": g.K, "Ku": u.K, "Kd": d.K, "mcg": flags[0], "mul1": flags[1]}
        if desc is None:
            desc = cur
            for k in ("Kg", "Ku", "Kd"):
                if not (1 <= desc[k] <= 8):
                    return None
        elif desc != cur:
            return None  # per-expert bitrate drift: one coop call cannot carry it
    return desc


def eligible(rows: int, top_k: int, is_cuda: bool, x_half: bool,
             ext_has_fn: bool) -> bool:
    """Decode-only contract + hard preconditions (everything cheap and host-side)."""
    return (is_cuda and x_half and ext_has_fn
            and 1 <= rows <= MOE_MAX_DECODE_TOKENS
            and rows * top_k <= MOE_COOP_MAX_SLOTS)


def build_call_args(values: dict) -> tuple:
    """Assemble the positional tuple for exl3_moe_coop in CALL_ARGS order. Raises on a
    missing key so wiring mistakes land loudly instead of defaulting."""
    return tuple(values[name] for name in CALL_ARGS)