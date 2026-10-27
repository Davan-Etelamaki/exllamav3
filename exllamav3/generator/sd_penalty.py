"""
SD fast-path penalty support (marker: penalty-fast1).

The speculative-decoding fast path (generator.py, EXL3_SD_FASTSYNC) pre-launches all
Wp per-position sampler forwards asynchronously, then reads every sampled token back
through one batched pinned copy. That needs each position's past_ids row on device
BEFORE any sample exists: position i must see committed_ids ++ draft[j, :i], the row
the serial path feeds position i when every earlier draft token matched (the only
condition under which position i's sample is used).

The first implementation fed each position a sliding window of the last K =
sustain_range + decay_range tokens. That is exact only while the penalty kernels
actually truncate: rep_pen.cu skips past_ids indices older than
past_len - sustain_range - decay_range (rep_pen.cu:51, :169) and measures distance
from the row end (rep_pen.cu:58, :176). But fleet-default penalties carry
sustain_range = max_seq_len or int(10e7) — tabbyAPI maps penalty_range there
(backends/exllamav3/sampler.py from_params) and presets.py keeps the same default —
so K exceeded the old 4096 window cap, the probe declined, and EVERY request with a
presence/frequency penalty fell to the serial sampling path with its per-position
next_token.cpu() (job.py receive_sample) and draft_tokens[j, i].item()
(generator.py) D2H syncs.

Fix: instead of widening the window, feed each position its EXACT full row. One
device buffer holds committed_ids ++ draft[j, :Wp-1]; position i receives the
contiguous prefix view buf[:n_now + i]. The kernels read past_ids linearly from
data_ptr() with past_len = size(1) (rep_pen.cu apply_pres_freq_pens /
apply_rep_pens), so a prefix view is a valid row, and it is byte-identical to the
serial path's row at position i for ANY sustain_range: same token multiset, same
distance-from-end for every token, hence bitwise-identical penalty application and
identical sampler RNG consumption (one rand draw per position, ascending). No window
cap, no minimum-history guard, no per-position sync — one O(n_now) async D2D copy
plus one O(Wp) draft copy per round, both enqueued before the sampler launches.

probe() decides; build_rows() constructs. Both pure and unit-tested CPU-only:
tests/generator/test_sd_penalty_fast.py.
"""

import torch

from .sampler.custom import SS_RepP, SS_PresFreqP


def probe(sampler):
    """
    Decide whether the SD fast path may use this sampler. Returns (ok, needs_rows).

    ok: every past-ID-consuming step is a penalty kernel (SS_RepP / SS_PresFreqP),
        whose semantics build_rows() reproduces exactly. Any other past-ID consumer
        (DRY, fused chains reading past IDs) keeps the request on the serial path.
    needs_rows: at least one such step exists, so per-position past-ID rows must be
        built; otherwise all positions share the committed row and none are needed.
    """
    needs_rows = False
    for st in getattr(sampler, "steps", ()):
        if not st.reqs_past_ids():
            continue
        if not isinstance(st, (SS_RepP, SS_PresFreqP)):
            return False, False
        needs_rows = True
    return True, needs_rows


def build_rows(current_device_ids, draft_tokens, j, Wp, buf):
    """
    Build the Wp per-position past-ID rows for one SD fast-path round.

    Row i (0-based logit position; Wp-1 is the bonus) is committed_ids ++
    draft_tokens[j, :i], a contiguous (1, n_now + i) int64 view into one device
    buffer — byte-identical to the row the serial path would feed position i.

    Returns (rows, buf). rows is None (caller keeps the serial path) when the draft
    row is shorter than the logit window needs; buf is the (possibly grown) scratch
    buffer the caller must store back. All copies are async device-to-device; no
    synchronization happens here.
    """
    n_now = current_device_ids.shape[1]
    if draft_tokens.shape[1] < Wp - 1:
        return None, buf
    need = n_now + Wp - 1
    if buf is None or buf.shape[0] < need:
        buf = torch.empty(max(2 * need, 8192), dtype = torch.long,
                          device = current_device_ids.device)
    buf[:n_now].copy_(current_device_ids[0, :n_now])
    buf[n_now:need].copy_(draft_tokens[j, :Wp - 1])
    rows = [buf[:n_now + i].view(1, -1) for i in range(Wp)]
    return rows, buf