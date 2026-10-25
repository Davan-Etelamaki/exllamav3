"""Fence for the iterate_draftmodel_mtp_gen glue cut (campaign phase 2, 2026-10-24).

py-spy prod showed 3.48% self-time in the MTP draft loop: three scans of
active_jobs (is_prefill_done() x3 per job per call) and an os.environ.get per
call. The cut collapses to one scan and caches the env flag. Semantics must be
identical: same batch rows, same order, same early-return paths.

Regime: pure-CPU stubs (no model), pinned staging via the real _staging method;
prod-share delta boot-pending.
"""
import os
import types
import torch

from exllamav3.generator.generator import Generator, PAGE_SIZE


class _Seq:
    def __init__(self, pages, pos):
        self.block_index_tensor = torch.tensor([pages], dtype=torch.int32)
        self.kv_position = pos


class _Job:
    def __init__(self, done, pages, pos, seq_len, tid, hidden=True):
        self.done = done
        self.sequences = [_Seq(pages, pos)]
        self.seq_len = seq_len
        self.mtp_last_hidden = torch.zeros(1, 4) if hidden else None
        self.time_first_token = 1.0
        self.tid = tid

    def is_prefill_done(self):
        return self.done

    def get_max_seq_len(self):
        return self.seq_len

    def get_input_ids_list(self):
        return [torch.tensor([[self.tid]], dtype=torch.int64)]


class _DraftModel:
    def __init__(self):
        self.forwards = 0

    def forward(self, ids, params):
        self.forwards += 1
        return {"ids": ids}

    def sample_from_state(self, state, params):
        return torch.tensor([[99]], dtype=torch.int64)


def _gen(jobs, window=1):
    g = Generator.__new__(Generator)
    g.active_jobs = jobs
    g.num_draft_tokens = window
    g.staging_buffers = {}
    g.draft_input_ids_pinned = torch.zeros(8, 1, dtype=torch.int64, pin_memory=True)
    g.draft_ids_pinned = torch.zeros(8, 8, dtype=torch.int64, pin_memory=True)
    g.draft_cache = None
    g.draft_calibrator = None
    g.draft_model = _DraftModel()
    g.model = types.SimpleNamespace(
        modules={0: types.SimpleNamespace(prepare_for_device=lambda s, p: s)},
        logit_layer_idx=0)
    return g


def test_single_scan_preserves_rows_order_and_excludes_pending():
    j1 = _Job(True,  [1, 2, 3], 111, 100, 5)
    jp = _Job(False, [9, 9, 9], 999, 900, 7)      # positive control: must NOT appear
    j2 = _Job(True,  [4, 5], 222, 60, 6)
    g = _gen([j1, jp, j2])
    out = g.iterate_draftmodel_mtp_gen([])
    assert out is not None
    bi = g.staging_buffers[("draft_block_index", bi_width(g))]
    # rows in active_jobs order, pending excluded
    assert bi[0, :3].tolist() == [1, 2, 3]
    assert bi[1, :2].tolist() == [4, 5]
    assert bi[1, 2:].tolist() == [0] * (bi.shape[1] - 2)
    assert 9 not in bi[0].tolist() and bi[1, 2:].tolist() != [9]  # pending table absent
    cs = g.staging_buffers[("draft_cache_seqlens", None)]
    assert cs[:2].tolist() == [111, 222]
    assert out[:, 0].tolist()[:2] == [99, 99]
    assert g.draft_model.forwards == 1


def bi_width(g):
    return next(k[1] for k in g.staging_buffers if k[0] == "draft_block_index")


def test_missing_mtp_hidden_still_returns_none():
    j1 = _Job(True, [1], 10, 50, 5, hidden=False)
    g = _gen([j1])
    assert g.iterate_draftmodel_mtp_gen([]) is None
    assert g.draft_model.forwards == 0


def test_no_ready_jobs_returns_none():
    g = _gen([_Job(False, [1], 1, 1, 1)])
    assert g.iterate_draftmodel_mtp_gen([]) is None


def test_env_flag_cached_after_first_read(monkeypatch):
    monkeypatch.delenv("EXL3_DRAFT_SYNC_CONF", raising=False)
    g = Generator.__new__(Generator)
    assert g._draft_sync_conf() is False
    monkeypatch.setenv("EXL3_DRAFT_SYNC_CONF", "1")
    assert g._draft_sync_conf() is False          # first read wins (documented)
    g2 = Generator.__new__(Generator)
    assert g2._draft_sync_conf() is True           # fresh generator sees the flag