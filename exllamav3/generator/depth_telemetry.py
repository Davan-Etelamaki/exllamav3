"""exllamav3-03: per-step draft-depth telemetry (echo-only counters).

Records, for every RESOLVED draft verify step, the pair (proposed_len,
accepted_len) into two fixed histograms owned by the job. Exported at
request end; rides the existing draft_accept plumbing (job result dict ->
backend finish_chunk -> gen_logging stats line), so depth lands on the
same log line as ctx length and accept rate — no new log stream.

Gate: EXL3_DEPTH_TELE=1, read ONCE at import. When off, new_rec() returns
None and every call site costs one `is not None` check — no allocation,
no import of anything heavier than os.

Ownership/threading: the recorder is job-owned and touched only by the
generator thread that runs the accept loop; no locks by design.

Semantics mirror record_draft_stats at the generator call site:
rewind-abandoned windows are skipped; checkpoint-boundary truncations
count as ordinary rejections. alen = accepted DRAFT tokens (excludes the
bonus/sampled token); plen = drafted positions for the step.

Scope (law 11e): instruments the LINEAR draft chain of the installed
1.6.1 wheel (MTP / ngram / hybrid lanes all emit linear draft_tokens).
The dev-tree tree-verify path is NOT instrumented here; extending this
patch to tree geometry is a follow-up when that wheel ships.

Clamp: lengths above MAXD-1 land in the last bucket (disclosed); draft
widths in service today are <= 8, so the clamp bucket is never hit.
"""
import os

MAXD = 513  # testbed probe: replay widths exceed 32
ON = os.environ.get("EXL3_DEPTH_TELE", "0") == "1"


class Rec:
    __slots__ = ("steps", "prop", "acc")

    def __init__(self, steps=0, prop=None, acc=None):
        self.steps = steps
        self.prop = prop if prop is not None else [0] * MAXD
        self.acc = acc if acc is not None else [0] * MAXD


def _pad(v):
    h = [0] * MAXD
    if v:
        n = min(len(v), MAXD)
        h[:n] = list(v[:n])
    return h


def new_rec(rq_state=None):
    """Fresh recorder, or None when the gate is off. Restores counts from
    a requeue state so a segmented generation keeps one histogram."""
    if not ON:
        return None
    prev = (rq_state or {}).get("draft_depth")
    if prev:
        return Rec(int(prev.get("s", 0)), _pad(prev.get("p")), _pad(prev.get("a")))
    return Rec()


def record(rec, plen, alen):
    """Hot path: two list bumps. Invariant alen <= plen enforced by clamp;
    negative plen means 'no draft this step' and is skipped."""
    if plen > MAXD - 1:
        plen = MAXD - 1
    elif plen < 0:
        return
    if alen > plen:
        alen = plen
    elif alen < 0:
        alen = 0
    rec.steps += 1
    rec.prop[plen] += 1
    rec.acc[alen] += 1


def _trim(h):
    n = MAXD
    while n > 1 and h[n - 1] == 0:
        n -= 1
    return h[:n]


def export(rec):
    """Plain JSON-safe dict, trailing-zero trimmed. None passes through."""
    if rec is None:
        return None
    return {"s": rec.steps, "p": _trim(rec.prop), "a": _trim(rec.acc)}


def survival(hist, steps):
    """Parser-side derivation: out[k-1] = #{observations with length >= k}.
    Shared by the log parser and the fence suite so analysis code and
    proof code cannot disagree about the definition."""
    out = []
    ge = steps
    for k in range(1, len(hist) + 1):
        ge -= hist[k - 1]
        out.append(ge)
    return out