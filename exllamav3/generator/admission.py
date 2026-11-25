"""
VRAM admission gate: may this job start now, given the device's free bytes?

Decision only. This module deliberately imports nothing (no torch, no CUDA, no exllamav3)
so it unit-tests on a machine without a GPU or even without torch installed; the caller in
Generator.iterate_start_jobs supplies the live figure from torch.cuda.mem_get_info().

Calibration (dot26, RTX 4090, 23.59 GiB usable, owner cliff measurement 2026-11-24): n
concurrent large prefills peaked at 21.63 + n * 1.55 GiB and busted the card at n >= 2;
gating to one large prefill at a time holds the worst case at 23.18 GiB. The tiers below
are that cliff expressed in the allocator's own unit, a ~20 MiB granule.
"""

# The CUDA caching allocator's expandable-segment granule, as measured on the deployed
# card (2026-11-24: an allocation failed at ~2 MiB free because the 20 MiB granule could
# not be expanded). Every figure in this module is counted in granules of this size.
GRANULE_BYTES = 20 * 1024 * 1024

# Headroom demanded on top of a job's tier cost, in granules. Engine default; a
# deployment overrides it through the model section key admission_margin_granules.
DEFAULT_ADMISSION_MARGIN_GRANULES = 8

# Peak cost of ingesting a prompt, in granules, by prompt size:
# (inclusive token ceiling, cost) for the measured tiers, and the tier above the last
# ceiling. A prompt of 0 tokens has nothing left to ingest -- it is a decode-only
# admission (a resumed job) -- and costs nothing.
PROMPT_COST_TIERS = ((8192, 20), (16384, 41))
LARGE_PROMPT_COST_GRANULES = 87

# An admission this cheap is never a prefill, and a decoder must not wait on VRAM: it
# holds its pages and its state already, so it always proceeds.
DECODE_EXEMPT_GRANULES = 2


def prompt_cost_granules(prompt_tokens: int) -> int:
    """Peak VRAM cost, in granules, of ingesting a prompt of prompt_tokens tokens"""
    if prompt_tokens <= 0:
        return 0
    for ceiling, cost in PROMPT_COST_TIERS:
        if prompt_tokens <= ceiling:
            return cost
    return LARGE_PROMPT_COST_GRANULES


def admits(free_bytes: int, prompt_tokens: int, margin_granules: int) -> bool:
    """
    True when a job that still has to ingest prompt_tokens may start on a device that
    currently reports free_bytes free.

    Requirement: free_granules >= cost_granules + margin_granules, with both sides counted
    in GRANULE_BYTES units (free floors down, so a partial granule is not spendable).
    Equality admits. A decode-only admission (cost within DECODE_EXEMPT_GRANULES) is
    never refused.
    """
    cost = prompt_cost_granules(prompt_tokens)
    if cost <= DECODE_EXEMPT_GRANULES:
        return True
    free_granules = int(free_bytes) // GRANULE_BYTES
    return free_granules >= cost + int(margin_granules)