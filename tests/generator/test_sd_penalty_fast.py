"""
SD fast-path penalty rows (marker: penalty-fast1) — CPU-only unit tests.

Regime: CUDA_VISIBLE_DEVICES="" stub world, no model, no GPU process. Run:

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=<worktree> python tests/generator/test_sd_penalty_fast.py

Covers:
  * POSITIVE CONTROL: the verbatim 5c64691e probe (window cap 4096) puts a
    fleet-shaped sampler (presence_penalty 1.5, default-range penalty) on the
    SERIAL path, while the new sd_penalty.probe puts it on the FAST path.
  * EQUIVALENCE: over three speculative rounds (full accept, immediate reject,
    penalty-flipped argmax) a stub sampler whose penalty replica mirrors
    rep_pen.cu (skip older than sustain+decay, distance from row end, presence
    binary via max) receives BYTE-IDENTICAL past_ids rows and produces identical
    accepted tokens / counters on both paths.
  * SYNC COUNTS: a SyncCountTensor subclass tallies .cpu()/.item() — the serial
    simulation pays per-position syncs, the fast round pays zero (batched
    readbacks use .tolist(), which the generator also uses).
  * build_rows contracts: exact rows, contiguity/dtype/shape for the kernel,
    buffer growth, short-draft fallback, Wp=1 bonus-only round.
"""

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from exllamav3.generator import sd_penalty
from exllamav3.generator.sampler.custom import SS_PresFreqP, SS_RepP, CustomSampler, SS_Argmax
from exllamav3.generator.sampler.presets import ComboSampler

try:
    import pytest
    pytestmark = pytest.mark.nogpu
except ImportError:  # standalone runner below
    pass

LEGACY_CAP = 4096


def legacy_probe(sampler):
    """Decision logic VERBATIM from 5c64691e generator.py:1995-2011 (window cap 4096)."""
    _ok, _kk = True, 0
    for st in getattr(sampler, "steps", ()):
        if not st.reqs_past_ids():
            continue
        if isinstance(st, (SS_RepP, SS_PresFreqP)):
            _sk = int(st.sustain_range) + int(st.decay_range)
            if _sk > LEGACY_CAP:
                _ok = False
                break
            _kk = max(_kk, _sk)
        else:
            _ok = False
            break
    return _ok, _kk


# ---------------------------------------------------------------- stub world

class SyncCountTensor(torch.Tensor):
    """Stand-in for a CUDA tensor: tallies the calls that would be D2H syncs."""
    count = 0

    def item(self):
        type(self).count += 1
        return super().item()

    def cpu(self, *args, **kwargs):
        type(self).count += 1
        return super().cpu(*args, **kwargs)


class DummyShapingStep:
    """Temperature/Sample-like step: never reads past IDs."""
    def reqs_past_ids(self):
        return False


class DummyPastIdStep:
    """DRY-like step: reads past IDs but is not a penalty kernel."""
    def reqs_past_ids(self):
        return True


class StubPresFreqSampler:
    """
    Pure-python replica of SS_PresFreqP + argmax. Penalty semantics mirror
    rep_pen.cu apply_pres_freq_pens_kernel: indices older than past_len -
    sustain - decay are skipped, distance is measured from the row end, presence
    is binary (atomicMax), then argmax. Records every past_ids row it is fed.
    """
    reqs_torch_seed = False

    def __init__(self, pres_p, sustain, decay):
        self.pen = SS_PresFreqP(pres_p, 0.0, sustain, decay)
        self.steps = [self.pen, DummyShapingStep()]
        self.rows_seen = []

    def forward(self, logits, past_ids, rand_u32=None, tokenizer=None, logit_mask=None):
        row = past_ids.view(-1).tolist()
        self.rows_seen.append(row)
        lo = max(0, len(row) - self.pen.sustain_range - self.pen.decay_range)
        penalized = set(row[lo:])  # presence: binary; decay weight folded to 1 here
        scored = logits.view(-1).clone()
        for t in penalized:
            scored[t] -= self.pen.pres_p
        tok = int(scored.argmax())
        return torch.tensor([[tok]], dtype=torch.long).as_subclass(SyncCountTensor)


VOCAB = 8
PRES_P = 1.5
SUSTAIN = int(10e7)   # fleet default (penalty_range unset/-1)
DECAY = 1             # tabbyAPI passes max(repetition_decay, 1)


def _logits(scores):
    row = torch.zeros(1, 1, VOCAB)
    for t, s in scores.items():
        row[0, 0, t] = s
    return row


# Three rounds: full accept with a penalty-flipped bonus, immediate reject,
# penalty-suppressed-but-still-winning argmax. Scores are tuned so presence
# penalty (subtract 1.5 once) changes or confirms the winner.
ROUNDS = [
    dict(draft=[[4, 5, 6]],
         logits=[{4: 2.0, 5: 1.9}, {5: 2.0, 4: 0.1}, {6: 2.0, 4: 0.1}, {2: 2.0, 3: 1.6}]),
    dict(draft=[[1, 2, 3]],
         logits=[{1: 2.0, 7: 1.8}, {2: 2.0, 7: 0.1}, {3: 2.0, 7: 0.1}, {6: 2.0, 7: 0.1}]),
    dict(draft=[[7, 0, 4]],
         logits=[{7: 2.0, 6: 0.45}, {0: 2.0, 5: 0.2}, {1: 2.0, 2: 0.4}, {5: 2.0, 4: 0.1}]),
]


def _drive_serial(initial_committed):
    """Mirror of the 5c64691e serial SD branch: per-position sampler + .cpu()/.item()."""
    committed = list(initial_committed)
    sampler = StubPresFreqSampler(PRES_P, SUSTAIN, DECAY)
    SyncCountTensor.count = 0
    outputs = []
    for rd in ROUNDS:
        Wp = len(rd["logits"])
        draft = rd["draft"]
        draft_t = torch.tensor(draft, dtype=torch.long).as_subclass(SyncCountTensor)
        j, i = 0, 0
        for i in range(Wp):
            tok = sampler.forward(_logits(rd["logits"][i]), torch.tensor([committed], dtype=torch.long))
            tok_i = tok.cpu().item()          # job.py:803-804 receive_sample
            committed.append(tok_i)
            if i < Wp - 1:
                if draft_t[j, i].item() != tok_i:   # generator.py:2152
                    break                            # reject_remainder + break
        outputs.append(i + 1)  # tokens committed this round
    return dict(committed=committed, rows=sampler.rows_seen, syncs=SyncCountTensor.count,
                round_tokens=outputs)


def _drive_fast(initial_committed):
    """Mirror of the sd_fast branch with sd_penalty rows: pre-launch all, batched readback."""
    committed = list(initial_committed)
    sampler = StubPresFreqSampler(PRES_P, SUSTAIN, DECAY)
    SyncCountTensor.count = 0
    outputs = []
    used_rows = []
    for rd in ROUNDS:
        Wp = len(rd["logits"])
        draft = rd["draft"]
        draft_t = torch.tensor(draft, dtype=torch.long).as_subclass(SyncCountTensor)
        ok, need = sd_penalty.probe(sampler)
        assert ok and need, "fleet-shaped sampler must take the fast path"
        rows, _ = sd_penalty.build_rows(torch.tensor([committed], dtype=torch.long),
                                        draft_t, 0, Wp, None)
        assert rows is not None
        # "launch" every position on its exact row; no readback yet
        toks = [sampler.forward(_logits(rd["logits"][i]), rows[i]) for i in range(Wp)]
        sd_tokens = torch.stack([t.view(1, -1) for t in toks]).view(-1).tolist()  # one D2H
        draft_cpu = draft_t.tolist()[0]                                           # one D2H
        i = 0
        for i in range(Wp):
            committed.append(sd_tokens[i])
            if i < Wp - 1 and draft_cpu[i] != sd_tokens[i]:
                break
        outputs.append(i + 1)
        # only the rows of USED positions can be compared against the serial path;
        # rows past the first mismatch were speculative (their samples are dropped)
        used_rows.extend(rows[k].view(-1).tolist() for k in range(i + 1))
    return dict(committed=committed, rows=used_rows, syncs=SyncCountTensor.count,
                round_tokens=outputs)


# ------------------------------------------------------------------ tests

def test_positive_control_fleet_penalty_old_serial_new_fast():
    for sampler in (
        ComboSampler(pres_p=1.5, rep_sustain_range=SUSTAIN, rep_decay_range=DECAY),
        CustomSampler([SS_PresFreqP(1.5, 0.0, SUSTAIN, DECAY), SS_Argmax()]),
        StubPresFreqSampler(PRES_P, SUSTAIN, DECAY),
    ):
        old_ok, _ = legacy_probe(sampler)
        new_ok, need = sd_penalty.probe(sampler)
        assert old_ok is False, "old probe must decline the fleet penalty sampler (serial)"
        assert new_ok is True and need is True, "new probe must take the fast path"


def test_positive_control_small_window_and_non_penalty_paths():
    # Small-window penalty: fast on BOTH probes (no regression for the old regime)
    s = CustomSampler([SS_PresFreqP(1.5, 0.0, 512, 0), SS_Argmax()])
    assert legacy_probe(s)[0] is True and sd_penalty.probe(s) == (True, True)
    # Repetition penalty shape (SS_RepP) with default range: old serial, new fast
    r = ComboSampler(rep_p=1.15, rep_sustain_range=int(10e7), rep_decay_range=0)
    assert legacy_probe(r)[0] is False and sd_penalty.probe(r)[0] is True
    # DRY-like past-ID consumer: BOTH stay serial
    d = StubPresFreqSampler(PRES_P, SUSTAIN, DECAY)
    d.steps = [DummyPastIdStep()]
    assert legacy_probe(d)[0] is False and sd_penalty.probe(d) == (False, False)
    # No past-ID consumer at all: fast, no rows needed
    n = CustomSampler([SS_Argmax()])
    assert sd_penalty.probe(n) == (True, False)


def test_equivalence_accepted_tokens_counters_and_rows():
    ser = _drive_serial([0, 1, 2, 3, 0, 1])
    fast = _drive_fast([0, 1, 2, 3, 0, 1])
    assert ser["round_tokens"] == fast["round_tokens"] == [4, 1, 3], \
        f"scenario broke: {ser['round_tokens']}"
    assert ser["committed"] == fast["committed"], "accepted token streams diverged"
    # The rows the fast path fed the sampler at every USED position must be
    # byte-identical to the rows the serial path fed at the same positions.
    assert fast["rows"] == ser["rows"], "penalty rows differ from serial rows"


def test_sync_counts_serial_vs_fast():
    ser = _drive_serial([0, 1, 2, 3, 0, 1])
    fast = _drive_fast([0, 1, 2, 3, 0, 1])
    assert ser["syncs"] > 0, "positive control: serial path must show per-position syncs"
    assert fast["syncs"] == 0, f"fast path took {fast['syncs']} per-position syncs"


def test_build_rows_exact_content_and_kernel_contract():
    committed = torch.tensor([[10, 11, 12, 13]], dtype=torch.long)
    draft = torch.tensor([[20, 21, 22]], dtype=torch.long)
    rows, buf = sd_penalty.build_rows(committed, draft, 0, 4, None)
    assert [r.view(-1).tolist() for r in rows] == [
        [10, 11, 12, 13],
        [10, 11, 12, 13, 20],
        [10, 11, 12, 13, 20, 21],
        [10, 11, 12, 13, 20, 21, 22],
    ]
    for r in rows:
        assert r.shape[0] == 1 and r.dtype == torch.long and r.is_contiguous(), \
            "kernel reads past_ids linearly from data_ptr with past_len=size(1)"
    # rows are views into one buffer (no per-row allocation)
    assert all(r.data_ptr() == buf.data_ptr() for r in rows)


def test_build_rows_edges():
    committed = torch.tensor([[1, 2, 3]], dtype=torch.long)
    # draft shorter than Wp-1 -> decline (caller keeps serial path)
    rows, buf = sd_penalty.build_rows(committed, torch.tensor([[9]], dtype=torch.long), 0, 4, None)
    assert rows is None and buf is None
    # bonus-only round (Wp=1): every position sees just the committed row
    rows, _ = sd_penalty.build_rows(committed, torch.zeros(1, 0, dtype=torch.long), 0, 1, None)
    assert [r.view(-1).tolist() for r in rows] == [[1, 2, 3]]
    # buffer growth: bigger n_now reallocates and still returns exact rows
    big = torch.arange(9000, dtype=torch.long).view(1, -1)
    _, small_buf = sd_penalty.build_rows(committed, torch.tensor([[7]], dtype=torch.long), 0, 2, None)
    rows, grown = sd_penalty.build_rows(big, torch.tensor([[8]], dtype=torch.long), 0, 2, small_buf)
    assert grown is not small_buf, "buffer must grow"
    assert rows[1].view(-1).tolist() == list(range(9000)) + [8]


def test_generator_integration_markers():
    import inspect
    from exllamav3.generator import generator as gen_mod
    src = inspect.getsource(gen_mod)
    assert "penalty-fast1" in src, "fix marker missing from generator.py"
    assert "sd_penalty.build_rows" in src and "sd_penalty.probe" in src
    assert "sd_pen_K" not in src, "stale sliding-window variable still referenced"


# ---------- penalty-tail1: append-only tail copy ----------

def _ct(ids):
    return torch.tensor([ids], dtype=torch.long)


def test_tail_byte_equal_to_full():
    committed = [1, 2, 3, 4, 5]
    draft = torch.tensor([[7, 8, 9, 10]], dtype=torch.long)
    _, buf = sd_penalty.build_rows(_ct(committed), draft, 0, 4, None)
    c2 = _ct(committed + [6])
    ref, _ = sd_penalty.build_rows(c2, draft, 0, 4, buf.clone())
    tail, _ = sd_penalty.build_rows(c2, draft, 0, 4, buf, n_prev=len(committed))
    assert tail is not None and ref is not None
    for a, b in zip(tail, ref):
        assert torch.equal(a, b), "tail path diverged from full copy"


def test_tail_poison_proves_prefix_copy_skipped():
    committed = [1, 2, 3, 4]
    draft = torch.tensor([[5, 6, 7]], dtype=torch.long)
    _, buf = sd_penalty.build_rows(_ct(committed), draft, 0, 3, None)
    buf[:4] = -999
    assert not torch.equal(buf[:4], _ct(committed)[0]), "poison did not land"
    rows, buf2 = sd_penalty.build_rows(_ct(committed + [8]), draft, 0, 3, buf, n_prev=4)
    assert buf2 is buf
    assert torch.equal(buf2[:4], torch.tensor([-999] * 4)), "tail path re-copied the prefix"
    assert torch.equal(buf2[4], torch.tensor(8)), "tail token missing"


def test_growth_forces_full_copy():
    committed = list(range(1, 101))
    draft = torch.tensor([[500, 501, 502]], dtype=torch.long)
    _, buf = sd_penalty.build_rows(_ct(committed), draft, 0, 3, None)
    big = list(range(10000, 10000 + 9000))
    buf[:100] = -7
    rows, buf2 = sd_penalty.build_rows(_ct(big), draft, 0, 3, buf, n_prev=100)
    assert buf2 is not buf, "buffer should have grown"
    assert torch.equal(rows[-1][0, :100], _ct(big[:100])[0]), "grown buffer kept stale prefix"


def test_stale_n_prev_forces_full_copy():
    committed = [1, 2, 3]
    draft = torch.tensor([[4, 5]], dtype=torch.long)
    _, buf = sd_penalty.build_rows(_ct(committed), draft, 0, 3, None)
    buf[:3] = -5
    rows, _ = sd_penalty.build_rows(_ct(committed), draft, 0, 3, buf, n_prev=10)
    assert torch.equal(rows[-1][0, :3], _ct(committed)[0]), "n_prev>n_now must force full copy"


def test_generator_integration_tail_marker():
    import inspect
    from exllamav3.generator import generator as gen_mod
    src = inspect.getsource(gen_mod)
    assert "penalty-tail1" in src, "tail marker missing from generator.py"
    assert "self.sd_pen_tail" in src and "n_prev =" in src, "tail wiring missing"


if __name__ == "__main__":
    tests = sorted((k, v) for k, v in globals().items()
                   if k.startswith("test_") and callable(v))
    assert len(tests) >= 6, f"inline collector found only {len(tests)} tests"
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:
            failed += 1
            import traceback
            print(f"FAIL {name}: {exc!r}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)