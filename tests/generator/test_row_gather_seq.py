"""
rowgather-seq1: sequential gather flag — CPU-only unit tests.

Regime: CUDA_VISIBLE_DEVICES="" stub world, temp table file, no GPU. The ext
signature change (7th arg bool sequential) is exercised directly; run with a
build of this branch (lane wheel or in-place ext):

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=<worktree> python tests/generator/test_row_gather_seq.py

Covers:
  * POSITIVE CONTROLS: non-sequential batch and per-row loop agree byte-for-byte
    with the file ground truth (guards against the float64-NaN false-mismatch trap).
  * EQUIVALENCE: sequential=True output is byte-identical to sequential=False and
    to the file truth, for scattered uids (many runs), one contiguous run, and
    mixed runs.
  * EDGES: U=0 no-op; base_offset + uid_base offset arithmetic under sequential.
"""

import os
import sys
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from exllamav3.ext import exllamav3_ext as ext

ROW_BYTES = 32768
ITEMS = ROW_BYTES // 8
NROWS = 4096


def _table(path):
    rng = torch.Generator().manual_seed(7)
    data = torch.randint(0, 256, (NROWS * ROW_BYTES,), dtype=torch.uint8, generator=rng).numpy().tobytes()
    with open(path, "wb") as f:
        f.write(data)
    return data


def _truth(data, uids):
    return torch.frombuffer(bytearray(b"".join(data[u * ROW_BYTES:(u + 1) * ROW_BYTES] for u in uids)),
                            dtype=torch.uint8).view(len(uids), ROW_BYTES)


def _gather(path, uids, sequential, base=0, base_offset=0):
    fd = os.open(path, os.O_RDONLY)
    try:
        out = torch.zeros(len(uids), ITEMS, dtype=torch.float64)
        ext.ngram_gather_cpu(fd, base_offset, ROW_BYTES,
                             torch.tensor([u + base for u in uids], dtype=torch.int64), base, out, sequential)
        return out.view(torch.uint8)
    finally:
        os.close(fd)


def test_sequential_equals_parallel_and_truth():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "t.bin")
        data = _table(path)
        uids = [3, 4, 5, 17, 18, 900, 1024, 1025, 2000, 4095]
        par = _gather(path, uids, False)
        seq = _gather(path, uids, True)
        gt = _truth(data, uids)
        assert torch.equal(par, gt), "batch path drifted from file truth"   # positive control
        assert torch.equal(seq, gt), "sequential path drifted from file truth"
        assert torch.equal(seq, par), "sequential != batch"


def test_single_run_and_edges():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "t.bin")
        data = _table(path)
        run = list(range(100, 116))                      # one coalesced run
        assert torch.equal(_gather(path, run, True), _truth(data, run))
        assert torch.equal(_gather(path, [], True), torch.zeros(0, ROW_BYTES, dtype=torch.uint8))


def test_offsets_under_sequential():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "t.bin")
        data = _table(path)
        shard = [10, 2000, 2001]
        base_offset = 7 * ROW_BYTES                       # shard starts at row 7 in the file
        seq = _gather(path, shard, True, base=7, base_offset=base_offset)
        gt = _truth(data, [s + 7 for s in shard])
        assert torch.equal(seq, gt), "sequential broke offset arithmetic"


def test_row_table_wiring_marker():
    import inspect
    from exllamav3.modules import row_table as rt
    src = inspect.getsource(rt)
    assert "rowgather-seq1" in src, "marker missing"
    assert "out[i0 : i1], True)" in src, "hinted path does not pass sequential=True"
    assert "out[i0 : i1], False)" in src, "batch path does not pass sequential=False"
    assert "for i in range(i0, i1)" not in src, "per-row Python loop still present"


if __name__ == "__main__":
    tests = sorted((k, v) for k, v in globals().items()
                   if k.startswith("test_") and callable(v))
    assert len(tests) >= 4, f"inline collector found only {len(tests)} tests"
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