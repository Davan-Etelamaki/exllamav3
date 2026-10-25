"""
QSAIndexer._job_arrays class cache (value-keyed).

Live py-spy (20 s, prod decode, 2026-10-24) put _job_arrays at 7.12% of all samples: the
generator hands every QSA layer a FRESH CPU tensor per step, so the old (data_ptr, version)
key missed on every layer of every token (12 layers x 2 pageable H2D copies per token).
The arrays are a pure function of (seqlens values, bsz, seq, compress_ratio), so the cache
is now class-level and value-keyed: one build per token, shared by every same-cr layer.
"""
import math

import torch

from exllamav3.modules.qsa_indexer import QSAIndexer


def _stub(cr):
    s = QSAIndexer.__new__(QSAIndexer)
    s.compress_ratio = cr
    return s


def _fresh(vals):
    # A new tensor object each call, like the generator's per-step CPU seqlens
    return torch.tensor(vals, dtype = torch.int32)


def test_value_keyed_sharing_across_instances(device):
    dev = torch.device(device)
    QSAIndexer._JOB_CACHE.clear()
    a = _stub(4)
    b = _stub(4)
    p1, t1 = a._job_arrays(_fresh([6144, 6144]), 2, 1, dev)
    p2, t2 = b._job_arrays(_fresh([6144, 6144]), 2, 1, dev)
    assert p1 is p2 and t1 is t2, "same values on a fresh tensor must hit the shared cache"
    assert torch.equal(p1.cpu(), torch.tensor([6144, 6144], dtype = torch.int32))
    assert torch.equal(t1.cpu(), torch.tensor([(6144 + 1) // 4] * 2, dtype = torch.int32))


def test_keys_segregate(device):
    dev = torch.device(device)
    QSAIndexer._JOB_CACHE.clear()
    a = _stub(4)
    p1, t1 = a._job_arrays(_fresh([6144]), 1, 1, dev)
    p2, t2 = a._job_arrays(_fresh([6145]), 1, 1, dev)      # different values
    p3, t3 = a._job_arrays(_fresh([6144]), 1, 5, dev)      # different seq
    p4, t4 = _stub(8)._job_arrays(_fresh([6144]), 1, 1, dev)  # different compress_ratio
    assert p1 is not p2 and p1 is not p3 and p1 is not p4
    assert torch.equal(t2.cpu(), torch.tensor([6145 // 4], dtype = torch.int32))
    assert torch.equal(t3.cpu(), torch.tensor([(6144 + 5) // 4], dtype = torch.int32))
    assert torch.equal(t4.cpu(), torch.tensor([6144 // 8], dtype = torch.int32))


def test_positive_control_no_sharing_without_cache(device):
    # The sharing assertion is real: with the cache emptied, identical values on fresh
    # tensors must produce DISTINCT objects (otherwise test 1 could pass vacuously).
    dev = torch.device(device)
    QSAIndexer._JOB_CACHE.clear()
    a = _stub(4)
    p1, _ = a._job_arrays(_fresh([4096]), 1, 1, dev)
    QSAIndexer._JOB_CACHE.clear()
    p2, _ = a._job_arrays(_fresh([4096]), 1, 1, dev)
    assert p1 is not p2
    QSAIndexer._JOB_CACHE.clear()


def test_eviction_bounds_memory(device):
    dev = torch.device(device)
    QSAIndexer._JOB_CACHE.clear()
    a = _stub(4)
    for v in range(QSAIndexer._JOB_CACHE_CAP + 10):
        a._job_arrays(_fresh([v]), 1, 1, dev)
    assert len(QSAIndexer._JOB_CACHE) <= QSAIndexer._JOB_CACHE_CAP
    p, t = a._job_arrays(_fresh([0]), 1, 1, dev)   # earliest key was evicted, rebuild correct
    assert torch.equal(p.cpu(), torch.tensor([0], dtype = torch.int32))
    assert torch.equal(t.cpu(), torch.tensor([0], dtype = torch.int32))
    QSAIndexer._JOB_CACHE.clear()