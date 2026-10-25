"""
QSA selection memos must survive inference tensors.

Regression fence for a prod load abort (2026-10-24, wheel 05089e71d8a8a): the load-time
autosplit dummy forward runs under torch.inference_mode(), and inference tensors raise on
``_version`` ("Inference tensors do not track version counter"), so the memo keys in
_bt_tables/_job_arrays killed `load_gen` outright. The fix reads the version through
_memo_ver(), which returns None for an inference tensor and makes the caller skip the memo.

The positive control asserts the tensors used here DO raise on a bare _version read -- without
it a green would only prove the fence was fed a tensor that never exercised the failing regime.
"""
import pytest
import torch

from exllamav3.modules.qsa_indexer import QSAIndexer, _memo_ver


def _stub(compress_ratio = 4):
    s = QSAIndexer.__new__(QSAIndexer)
    s.compress_ratio = compress_ratio
    return s


def _block_table(bsz, npr, device, inference):
    # block_table is 2D (jobs, pages-per-row), as ext.dsa_topk/qsa_sparse_attend_rows require
    src = torch.arange(bsz * npr * 4, dtype = torch.int64, device = device).view(bsz, npr * 4)
    if inference:
        with torch.inference_mode():
            return src.clone()
    return src.clone()


def test_memo_ver_positive_control(device):
    """The tensors this test feeds the memos are inference tensors, and a bare _version read
    on them raises -- i.e. the old key expressions are the ones that aborted the boot."""
    with torch.inference_mode():
        t = torch.zeros(8, dtype = torch.int64, device = device)
    assert torch.is_inference(t)
    with pytest.raises(RuntimeError, match = "do not track version counter"):
        _ = t._version
    # _memo_ver is the guarded read: no raise, and it reports "no version"
    assert _memo_ver(t) is None
    # ... while a normal tensor still yields a version, so the memo stays live outside inference
    plain = torch.zeros(8, dtype = torch.int64, device = device)
    assert not torch.is_inference(plain)
    assert _memo_ver(plain) == plain._version


def test_bt_tables_inference_tensor(device):
    """_bt_tables (block-table memo) under inference_mode: no raise, correct shapes."""
    bsz, npr, seq = 3, 5, 4
    s = _stub()
    bt = _block_table(bsz, npr, device, inference = True)
    with torch.inference_mode():
        bt_job, bt_rows = s._bt_tables(bt, bsz, seq)
    assert bt_job.shape == (bsz, npr * 4) and bt_job.dtype == torch.int32
    assert bt_rows.shape == (bsz * seq, npr * 4) and bt_rows.dtype == torch.int32
    # values still mirror the source table (unchanged math, only the key changed)
    assert torch.equal(bt_job, bt.reshape(bsz, npr * 4).int())
    assert torch.equal(bt_rows, bt.reshape(bsz, npr * 4).int()
                       .unsqueeze(1).expand(bsz, seq, -1).reshape(bsz * seq, -1))


def test_job_arrays_inference_tensor(device):
    """_job_arrays (cache_seqlens memo) under inference_mode: no raise, correct pos/T."""
    bsz, seq, cr = 3, 4, 4
    s = _stub(compress_ratio = cr)
    cs = torch.tensor([7, 11, 0], dtype = torch.int32)
    with torch.inference_mode():
        cs = cs.clone()
    with torch.inference_mode():
        pos, t = s._job_arrays(cs, bsz, seq, device)
    assert pos.tolist()[:bsz] == [7, 11, 0]
    assert t.tolist()[:bsz] == [(7 + seq) // cr, (11 + seq) // cr, (0 + seq) // cr]
    assert pos.dtype == torch.int32 and t.dtype == torch.int32


def test_memo_skipped_for_normal_input_inside_inference_mode(device):
    """Handing a NORMAL tensor to the memo during an InferenceMode pass must still keep
    nothing: what the memo would store (bt_job/bt_rows, the pos/T pair) is produced inside
    the pass and is therefore an inference tensor itself."""
    bsz, npr, seq = 2, 3, 2
    s = _stub()
    bt = _block_table(bsz, npr, device, inference = False)
    cs = torch.tensor([3, 5], dtype = torch.int32)
    assert not torch.is_inference(bt) and not torch.is_inference(cs)
    with torch.inference_mode():
        out = s._bt_tables(bt, bsz, seq)
        assert torch.is_inference(out[0])              # derived inside the pass
        assert not hasattr(s, "_bt_rows_cache")        # ... and so never kept
        pair = s._job_arrays(cs, bsz, seq, device)
        assert torch.is_inference(pair[0])
        assert not hasattr(s, "_job_arrays_cache")

    # The serving path then gets normal tensors, which are writable in place.
    a = s._bt_tables(bt, bsz, seq)
    assert not torch.is_inference(a[0]) and not torch.is_inference(a[1])
    a[0].zero_()
    b = s._job_arrays(cs, bsz, seq, device)
    assert not torch.is_inference(b[0])
    b[0].zero_()


def test_memo_still_caches_outside_inference(device):
    """The guard must not disable the optimization: plain tensors still hit the memo, and an
    in-place rewrite (version bump) still misses it."""
    bsz, npr, seq = 2, 3, 2
    s = _stub()
    bt = _block_table(bsz, npr, device, inference = False)
    a1 = s._bt_tables(bt, bsz, seq)
    a2 = s._bt_tables(bt, bsz, seq)
    assert a2[0] is a1[0] and a2[1] is a1[1]        # memo hit returns the same tensors
    bt[0, 0] += 1                                   # in-place rewrite bumps _version
    a3 = s._bt_tables(bt, bsz, seq)
    assert a3[0] is not a1[0]                       # ... and misses
    assert torch.equal(a3[0], bt.reshape(bsz, npr * 4).int())

    cs = torch.tensor([3, 5], dtype = torch.int32)
    j1 = s._job_arrays(cs, bsz, seq, device)
    j2 = s._job_arrays(cs, bsz, seq, device)
    assert j2[0] is j1[0] and j2[1] is j1[1]
    cs[0] += 1
    j3 = s._job_arrays(cs, bsz, seq, device)
    assert j3[0] is not j1[0]
    assert j3[0].tolist()[0] == 4