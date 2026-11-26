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
# The decode batch universe BOTH lanes key on: a piecewise-captured decode pads its batch to
# one of these, so the arena key and the capture padding are the same number and one captured
# graph replays for every live count at or below its bucket (graph lane §U). Fenced equal to
# the ladder and capped by MOE_MAX_DECODE_TOKENS in tests/cpu/moe_coop_plan_fence.py.
DECODE_BUCKETS = (1, 2, 4, 8, 16, 32)

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


def padded_bucket_rows(rows: int) -> int | None:
    """SHARED ARENA RULE (graph lane §U): the arena is keyed and sized by the padded decode
    bucket, never by live rows. A captured graph baked at one bucket replays for every live
    count at or below it; keying by live rows would take a dict miss, a fresh allocation and
    different addresses than the graph baked — a silent replay against the wrong buffer. A
    bucket above the ladder (prefill, mixed, > MOE_MAX_DECODE_TOKENS) returns None: no coop."""
    for bucket in DECODE_BUCKETS:
        if rows <= bucket:
            return bucket
    return None


def unwrap_linear(m):
    """Quant tensors of a Linear WRAPPER (modules/linear.py) live on .inner once loaded; the
    coop plan and pointer tables must see the inner QuantLinear*, never the wrapper (whose
    attribute set lacks .bias/.K/.trellis outright). Modules already exposing the quant attrs
    pass through unchanged."""
    inner = getattr(m, "inner", None)
    return inner if inner is not None else m


def classify_experts(groups) -> tuple[dict | None, str]:
    """Collapse 3*num_experts LinearEXL3 modules to one (K, cb) description.

    groups: iterable of (g, u, d) triples of module-likes exposing .K/.mcg/.mul1/.bias.
    Returns ({"Kg":.., "Ku":.., "Kd":.., "mcg":bool, "mul1":bool}, "") when the pack is
    homogeneous enough for one coop call, else (None, reason) — the reason names the leg that
    refused, so a fence-3 A/B can prove which path the coop arm took instead of reading a
    zero delta as a null result. The served path treats None as 'keep the python dispatch'.
    """
    desc = None
    for e, (g, u, d) in enumerate(groups):
        # InklingMoE passes Linear WRAPPERS whose quant tensors (bias/K/mcg/mul1) live on
        # .inner after load; a bare wrapper has no .bias attribute at all and used to raise
        # AttributeError here, which aborted every request on the first coop eligibility check
        g, u, d = (unwrap_linear(m) for m in (g, u, d))
        if g.bias is not None or u.bias is not None or d.bias is not None:
            return None, f"expert {e} carries a bias (bias tables are not wired)"
        if getattr(g, "K", None) != getattr(u, "K", None):
            return None, f"expert {e}: gate K {getattr(g, 'K', None)} != up K " \
                         f"{getattr(u, 'K', None)} (coop requires one bit width for gate/up)"
        cb_flags = {(bool(getattr(m, "mcg", False)), bool(getattr(m, "mul1", False)))
                    for m in (g, u, d)}
        if len(cb_flags) != 1:
            return None, f"expert {e}: mixed codebook families {sorted(cb_flags)}"
        flags = cb_flags.pop()
        cur = {"Kg": g.K, "Ku": u.K, "Kd": d.K, "mcg": flags[0], "mul1": flags[1]}
        if desc is None:
            desc = cur
            for k in ("Kg", "Ku", "Kd"):
                if not (1 <= desc[k] <= 8):
                    return None, f"{k} = {desc[k]} outside [1, 8]"
        elif desc != cur:
            return None, f"expert {e} bitrate/codebook {cur} != pack baseline {desc}"
    return desc, ""


def refusal(rows: int, top_k: int, is_cuda: bool, x_half: bool,
            ext_has_fn: bool) -> str | None:
    """Why the coop tier must not run for this call, or None when it may. Every leg is cheap
    and host-side. Decode-only contract: rows <= 32, slots <= 256
    (exllamav3/exllamav3_ext/moe_group/README.md)."""
    if not is_cuda:
        return "x is not a CUDA tensor"
    if not x_half:
        return "x is not fp16"
    if not ext_has_fn:
        return "the built extension has no exl3_moe_coop entry point"
    if padded_bucket_rows(rows) is None:
        return f"{rows} rows is above the decode ladder {DECODE_BUCKETS} (prefill/mixed)"
    if rows * top_k > MOE_COOP_MAX_SLOTS:
        return f"{rows} x top_k {top_k} exceeds MOE_COOP_MAX_SLOTS"
    return None


def eligible(rows: int, top_k: int, is_cuda: bool, x_half: bool,
             ext_has_fn: bool) -> bool:
    """Boolean front of refusal() — the hard preconditions, everything host-side."""
    return refusal(rows, top_k, is_cuda, x_half, ext_has_fn) is None


def build_call_args(values: dict) -> tuple:
    """Assemble the positional tuple for exl3_moe_coop in CALL_ARGS order. Raises on a
    missing key so wiring mistakes land loudly instead of defaulting."""
    return tuple(values[name] for name in CALL_ARGS)