"""Fence for chunked-verify16 (wide-round sub-round iteration in iterate_gen).

A verify round wider than the cap pads every row to the global max and computes the
whole rectangle even though acceptance is a prefix per row. The fix iterates such a
round as sequential ≤16-token sub-rounds inside one pass. This fence proves the
chunked path commits exactly what the single-pass rectangular path commits: same
accepted prefix, same correction/bonus token, same counters, same recurrent-state
net advance, and every model.forward width at or below the cap. No attention-tier
claim: generic-tier NVFP4 staging is not width-triggered at batch>1 (lane C arm).

Regime: pure-CPU stubs (no model, no CUDA); run with CUDA_VISIBLE_DEVICES="" and
EXL3_SD_FASTSYNC=0 (set below before import). GDN/page rewind here is the stub's
counter arithmetic — the real recurrent-state equivalence is boot-pending (the 0.6B
testbed is dense and cannot exercise batch>1 GDN).
"""
import os
os.environ["EXL3_SD_FASTSYNC"] = "0"
import types
import torch

from exllamav3.generator import generator as genmod
from exllamav3.generator.generator import Generator, PAGE_SIZE

CAP = genmod._CHUNK_VERIFY_CAP
VOCAB = 4096
FILLER = 4000


class _SeqIds:
    def __init__(self, toks):
        self.toks = list(toks)

    def torch(self):
        return torch.tensor([self.toks], dtype=torch.int64)

    def torch_slice(self, a, b):
        return torch.tensor([self.toks[a:b] if b is not None else self.toks[a:]], dtype=torch.int64)

    def __len__(self):
        return len(self.toks)


class _Page:
    def __init__(self):
        self.kv_position = 0


class _Seq:
    def __init__(self, ids):
        self.sequence_ids = _SeqIds(ids)
        self.kv_position = len(ids) - 1
        # Real lifecycle (chunked-verify16b repro): prefill cached the prefix
        # contiguously, so page fill must START consistent with kv_position — page p
        # holds min(PAGE_SIZE, max(0, kv_position - p*PAGE_SIZE)) tokens. Drift from a
        # verify round is only visible against primed pages.
        self.allocated_pages = []
        n_cached = len(ids) - 1
        for p in range(64):                       # generous: index errors must not mask bugs
            page = _Page()
            page.kv_position = max(0, min(PAGE_SIZE, n_cached - p * PAGE_SIZE))
            self.allocated_pages.append(page)
        self.block_index_tensor = torch.arange(1, 65, dtype=torch.int32).view(1, 64)


class _State:
    """Recurrent-state stand-in: forward advances it by every row it wrote; rewind rolls
    back (and records) — net position must equal len(sequence) after every path."""

    def __init__(self, pos):
        self.pos = pos
        self.rewind_log = []

    def rewind(self, n):
        self.rewind_log.append(n)
        self.pos -= n


class _Job:
    def __init__(self, ids, oracle):
        self.sequences = [_Seq(ids)]
        self.oracle = list(oracle)   # token sampled at global logits row j of the round; last = bonus
        self.recurrent_state = types.SimpleNamespace(target=_State(len(ids)), draft=None)
        self.filters = []
        self.filter_futures = []
        self.logit_masks = []
        self.new_tokens = 0
        self.forced_ids = None
        self.device_logit_mask = None
        self.return_probs = False
        self.return_top_tokens = 0
        self.sampler = types.SimpleNamespace(reqs_torch_seed=False, steps=())
        self.rng = types.SimpleNamespace(randint=lambda a, b: 0)
        self.checkpoint_rewound = False
        self.accepted_draft_tokens = 0
        self.rejected_draft_tokens = 0
        self.draft_stats = []
        self._depth_rec = None
        self._replay_pending = None
        self.time_first_token = 1.0
        self.embeddings = []
        self.alt_rope_offset = 0
        self.mtp_last_hidden = None

    def is_prefill_done(self):
        return True

    def get_max_seq_len(self):
        return len(self.sequences[0].sequence_ids)

    def get_input_ids_list(self, draft_tokens=None, idx=0, add_to_cache=False):
        # Mirrors job.Job.get_input_ids_list: [uncommitted tail] ++ draft row, pages reserved
        out = []
        for seq in self.sequences:
            tail = seq.sequence_ids.torch_slice(seq.kv_position, None)
            ids = tail if draft_tokens is None else torch.cat((tail, draft_tokens[idx:idx + 1, :]), dim=-1)
            out.append(ids)
            if add_to_cache:
                n = ids.shape[-1]
                skvp = seq.kv_position
                guard = 0
                while n:
                    # Loop bound: the real job.py:401-406 loop has none, which is why the
                    # drift bug wedged the engine for 25 min instead of failing. Here a
                    # no-progress spin raises so the fence can assert on it.
                    guard += 1
                    if guard > 1000:
                        raise RuntimeError(
                            f"LIVELOCK: get_input_ids_list add_to_cache made no progress "
                            f"(seq.kv_position={seq.kv_position} page.kv_position="
                            f"{page.kv_position} tokens_left={n}); page fill is ahead of "
                            f"kv_position")
                    page = seq.allocated_pages[skvp // PAGE_SIZE]
                    tp = min(n, PAGE_SIZE - page.kv_position)
                    page.kv_position += tp
                    n -= tp
                    skvp += tp
        return out

    def receive_logits(self, token_logits):
        return token_logits[0, 0].argmax().view(1, 1), None, None, None

    def receive_sample(self, token_logits, next_token, kt, kp, p, results):
        tok = int(next_token.view(-1)[0])
        seq = self.sequences[0]
        seq.sequence_ids.toks.append(tok)
        seq.kv_position += 1
        self.new_tokens += 1
        results.append({"job": self, "eos": False, "token": tok})
        return False, torch.tensor([[tok]], dtype=torch.int64), False

    def prepare_logit_mask(self):
        pass

    def prepare_sampling_past_ids(self):
        pass

    def is_checkpoint_boundary(self):
        return False

    def deallocate_pages(self):
        pass

    def maybe_stash_recurrent(self, rc, ps):
        pass

    def prepare_for_requeue(self):
        return None


class _Model:
    """Stub target model. logits row i of a job row one-hots that job's oracle token at
    global round index (kv_advance + i); rows past the oracle get FILLER (only reached by
    padded/dead rows, whose logits are never committed). Advances each job's recurrent
    state by every row written, like a speculative pass. Asserts every forward width is
    within the BC cap when cap_guard is set (the chunked path's whole point)."""

    def __init__(self, jobs, round_start, cap_guard):
        self.caps = set()
        self.jobs = jobs
        self.round_start = round_start
        self.cap_guard = cap_guard
        self.forward_widths = []
        self.prefetched = []
        self.fwd_modules = []

    def forward(self, input_ids=None, params=None):
        rows, width = input_ids.shape
        self.forward_widths.append(width)
        assert self.cap_guard is None or width <= self.cap_guard, \
            f"chunked-verify16: forward width {width} exceeds BC cap {self.cap_guard}"
        out = torch.zeros(rows, width, VOCAB, dtype=torch.float32)
        r = 0
        for job in self.jobs:
            seq = job.sequences[0]
            adv = seq.kv_position - self.round_start
            for i in range(width):
                gi = adv + i
                tok = job.oracle[gi] if gi < len(job.oracle) else FILLER
                out[r, i, tok] = 1.0
            job.recurrent_state.target.pos += width
            r += 1
        return out

    def prefetch_tokens(self, toks):
        self.prefetched.extend(toks)


def _staging(self, name, rows, width=None, dtype=torch.int32):
    shape = (rows,) if width is None else (rows, width)
    return torch.zeros(shape, dtype=dtype)


def _gen(job, model):
    g = Generator.__new__(Generator)
    g.active_jobs = [job]
    g.pending_jobs = []
    g.staging_buffers = {}
    g._staging = types.MethodType(_staging, g)
    g.model = model
    g.cache = None
    g.target_recurrent = True
    g.draft_model = None
    g.draft_recurrent = False
    g.dflash_draft = False
    g.mtp_draft = False
    g.draft_cache = None
    g.draft_calibrator = None
    g._draft_conf_round = None
    g.record_draft_stats = False
    g.recurrent_cache = None
    g.visualizer = None
    g.disk_page_cache = None
    g.filter_pool = None
    g.num_remaining_jobs = lambda: 0
    g.on_queue_drained = lambda: None
    g.num_draft_tokens = 1
    g.chunk_rounds = 0
    g.chunk_subrounds = 0
    g.chunk_rows_verified = 0
    g.chunk_accepted = 0
    g._chunk_witness = 0
    g.replay_max_tokens = 0
    g.hybrid_draft_tokens = 0
    g._replay_warm = 0
    g.replay_rejects = 0
    g.sample_pinned = None
    g.sd_pen_cache = {}
    g.sd_fast_pinned = None
    g.sd_pen_buf = None
    g.sd_fast_rounds = 0
    g.sd_fast_diag = 0
    return g


def _world(W, mismatch, cap_guard):
    prefix = [1000 + i for i in range(8)]           # committed context, kv = len-1
    draft = [2000 + (i % 37) for i in range(W)]     # proposals
    oracle = list(draft) + [3999]                   # greedy samples; last = bonus
    if mismatch is not None:
        oracle[mismatch] = 3888                     # correction: != any draft token
    job = _Job(prefix, oracle)
    model = _Model([job], len(prefix) - 1, cap_guard)
    g = _gen(job, model)
    dt = torch.tensor([draft], dtype=torch.int64)
    return job, g, dt


def _outcome(job, g):
    seq = job.sequences[0]
    return (
        list(seq.sequence_ids.toks),
        seq.kv_position,
        job.recurrent_state.target.pos,
        job.accepted_draft_tokens,
        job.rejected_draft_tokens,
    )


def _run_pair(W, mismatch):
    """Run the same scripted round through the chunked path and (cap lifted) the
    single-pass rectangular path; return both outcomes plus the chunked generator."""
    job_c, g_c, dt = _world(W, mismatch, CAP)
    g_c.iterate_gen([], dt)
    job_s, g_s, dt2 = _world(W, mismatch, None)
    saved = genmod._CHUNK_VERIFY_CAP
    genmod._CHUNK_VERIFY_CAP = 10 ** 6
    try:
        g_s.iterate_gen([], dt2)
    finally:
        genmod._CHUNK_VERIFY_CAP = saved
    return _outcome(job_c, g_c), _outcome(job_s, g_s), job_c, g_c, g_s


def _expect(W, mismatch):
    seq = [1000 + i for i in range(8)]
    draft = [2000 + (i % 37) for i in range(W)]
    if mismatch is None:
        seq += draft + [3999]
        return seq, len(seq) - 1, len(seq), W, 0
    seq += draft[:mismatch] + [3888]
    return seq, len(seq) - 1, len(seq), mismatch, W - mismatch


def test_full_accept_across_three_subrounds():
    W = 40
    oc, os_, job_c, g_c, g_s = _run_pair(W, None)
    exp_seq, exp_kv, exp_state, exp_acc, exp_rej = _expect(W, None)
    assert oc == os_, f"chunked {oc} != single-pass {os_}"
    assert oc[0] == exp_seq and oc[1] == exp_kv
    assert oc[2] == exp_state == oc[1] + 1        # state net == committed sequence
    assert (oc[3], oc[4]) == (exp_acc, exp_rej)
    # chunked path actually ran: every forward width ≤ cap, witnesses nonzero
    assert g_c.chunk_rounds == 1 and g_c.chunk_subrounds == -(-W // (CAP - 1))
    assert g_c.chunk_accepted == W
    assert g_c.chunk_rows_verified == g_c.chunk_subrounds * CAP
    assert all(w <= CAP for w in g_c.model.forward_widths)
    assert g_s.chunk_rounds == 0                  # positive control: reference stayed single-pass
    assert g_s.model.forward_widths == [W + 1]


def test_reject_at_first_position():
    oc, os_, job_c, g_c, _ = _run_pair(40, 0)
    exp_seq, exp_kv, exp_state, exp_acc, exp_rej = _expect(40, 0)
    assert oc == os_
    assert oc[0] == exp_seq and oc[1] == exp_kv and oc[2] == exp_state
    assert (oc[3], oc[4]) == (exp_acc, exp_rej)
    assert g_c.chunk_subrounds == 1               # all rows dead -> later sub-rounds skipped
    assert job_c.recurrent_state.target.rewind_log == [CAP - 1]


def test_reject_mid_round_second_subround():
    oc, os_, job_c, g_c, _ = _run_pair(40, 20)
    exp_seq, exp_kv, exp_state, exp_acc, exp_rej = _expect(40, 20)
    assert oc == os_
    assert oc[0] == exp_seq and oc[1] == exp_kv and oc[2] == exp_state
    assert (oc[3], oc[4]) == (exp_acc, exp_rej)
    # sub-round 0 committed (rewind 0), sub-round 1 rejected at local i=4
    assert job_c.recurrent_state.target.rewind_log == [0, CAP - 1 - 4]
    assert g_c.chunk_subrounds == 2


def test_reject_at_last_draft_column():
    W = 40
    oc, os_, job_c, g_c, _ = _run_pair(W, W - 1)
    exp_seq, exp_kv, exp_state, exp_acc, exp_rej = _expect(W, W - 1)
    assert oc == os_
    assert oc[0] == exp_seq and oc[1] == exp_kv and oc[2] == exp_state
    assert (oc[3], oc[4]) == (exp_acc, exp_rej)   # rejected == 1, no bonus committed


def test_boundary_column_reject_and_two_subround_bonus():
    # W=17: sub-round 0 verifies columns 0..14 and the boundary column 15; rejecting at
    # column 15 rolls back nothing (all 16 written rows are accepted) and commits the
    # correction as the round's last token.
    oc, os_, job_c, _, _ = _run_pair(17, 15)
    exp_seq, exp_kv, exp_state, exp_acc, exp_rej = _expect(17, 15)
    assert oc == os_
    assert oc[0] == exp_seq and oc[1] == exp_kv and oc[2] == exp_state
    assert (oc[3], oc[4]) == (exp_acc, exp_rej)
    assert job_c.recurrent_state.target.rewind_log == [0]     # commit, not rollback
    # full accept of the same shape: bonus lands in sub-round 1, padded rows roll back
    oc2, os2, job2, g2, _ = _run_pair(17, None)
    exp2 = _expect(17, None)
    assert oc2 == os2 == (exp2[0], exp2[1], exp2[2], exp2[3], exp2[4])
    assert g2.chunk_subrounds == 2
    assert job2.recurrent_state.target.rewind_log == [0, CAP - 1 - 1]   # 14 padded rows


def test_narrow_rounds_take_singlepass_untouched():
    # W+1 <= CAP must not dispatch to the chunked path at all
    job, g, dt = _world(CAP - 1, None, None)
    saved = genmod._CHUNK_VERIFY_CAP
    genmod._CHUNK_VERIFY_CAP = CAP                # real cap, narrow round
    try:
        g.iterate_gen([], dt)
    finally:
        genmod._CHUNK_VERIFY_CAP = saved
    assert g.chunk_rounds == 0
    assert g.model.forward_widths == [CAP]


def test_replay_bandit_runs_once_per_round():
    W = 40
    job_c, g_c, dt = _world(W, None, CAP)
    job_c._replay_pending = 30
    g_c.replay_max_tokens = 64
    g_c.hybrid_draft_tokens = 6
    g_c.iterate_gen([], dt)
    assert job_c._replay_pending is None
    assert job_c.replay_streak == 1
    assert job_c.replay_width == min(64, max(60, 6))
    assert job_c.replay_cap == min(64, max(6 * 4, 60))


def test_witness_counters_are_nonzero_on_fire():
    # law 21: prove the fence's counters actually move (non-empty extraction)
    _, _, _, g_c, g_s = _run_pair(40, 20)
    assert g_c.chunk_rounds == 1 and g_c.chunk_rows_verified > 0 and g_c.chunk_subrounds >= 2
    assert g_s.chunk_rounds == 0 and g_s.chunk_rows_verified == 0


def _world_batch(W, mismatches, cap_guard, prefix_len=247):
    """Multi-row world sharing one prefix, like the real batch: a row that is done in an
    early sub-round keeps feeding its pending token + filler columns in later sub-rounds
    so the batch stays rectangular. prefix_len places the early-rejecting row's pending
    token at the LAST slot of a PAGE_SIZE page, so any page-fill drift bites the very
    next round's add_to_cache."""
    prefix = [1000 + (i % 900) for i in range(prefix_len)]
    jobs, drafts = [], []
    for k, mm in enumerate(mismatches):
        draft = [2000 + ((i + 7 * k) % 37) for i in range(W)]
        oracle = list(draft) + [3999]
        if mm is not None:
            oracle[mm] = 3888 + k             # correction: != any draft token
        jobs.append(_Job(prefix, oracle))
        drafts.append(draft)
    model = _Model(jobs, prefix_len - 1, cap_guard)
    g = _gen(jobs[0], model)
    g.active_jobs = jobs
    dt = torch.tensor(drafts, dtype=torch.int64)
    return jobs, g, dt


def test_wide_chunked_round_leaves_no_page_drift_for_next_round():
    # Reproducer for the dot26 livelock (chunked-verify16b). Two rows, W=24 (wide):
    # row A rejects at sub-round 0 row 8 (done), row B accepts all of sub-round 0 and
    # rejects at sub-round 1 row 0 — so sub-round 1 runs with A still in the batch.
    # A's done-row rollback (reject_window(j, 0)) releases cap-1 of the cap tokens
    # staged for that window (its pending correction token + 15 filler columns),
    # leaving page.kv_position one ahead of seq.kv_position. The next iterate_gen —
    # ANY width — then spins in get_input_ids_list's add_to_cache loop
    # (job.py:403<->406): the pending slot's page is full, tokens_page = 0 forever,
    # tokens_to_add never decrements. Pre-fix this raises LIVELOCK (the stub bounds
    # the loop the real code leaves unbounded); post-fix both rounds complete with
    # page fill == kv_position throughout.
    jobs, g, dt = _world_batch(24, [8, 16], CAP)
    g.iterate_gen([], dt)                                   # wide chunked round
    assert g.chunk_rounds == 1 and g.chunk_subrounds == 2   # the incident's shape
    dt2 = torch.tensor([[2500 + i for i in range(8)] for _ in jobs], dtype=torch.int64)
    try:
        g.iterate_gen([], dt2)                              # narrow round: spun here pre-fix
        spin = None
    except RuntimeError as exc:                             # stub's loop bound caught the spin
        spin = str(exc)
    assert spin is None, f"second iterate_gen livelocked in add_to_cache: {spin}"
    for job in jobs:                                        # the invariant the bug breaks
        seq = job.sequences[0]
        fill = sum(p.kv_position for p in seq.allocated_pages)
        assert fill == seq.kv_position, (
            f"page fill {fill} != kv_position {seq.kv_position}: page ahead -> next "
            f"round's add_to_cache spins (job.py:401-406)")
        assert job.recurrent_state.target.pos == len(seq.sequence_ids), \
            "recurrent state drifted ahead of the committed sequence"
    assert g.chunk_rounds == 1                              # positive control: round 2 was single-pass
    for job in jobs:
        seq = job.sequences[0]
        assert sum(p.kv_position for p in seq.allocated_pages) == seq.kv_position
        assert len(seq.sequence_ids) == seq.kv_position + 1