"""
SlicedMultiLinear.c_ptrs miss path: pinned staging ring + async copy, fresh GPU table.

Live py-spy (20 s, prod decode, wheel b75a161880c0) parked 5.81% of all samples in the
old torch.tensor(ptrs, device=...) pageable-sync H2D: the pointer-tuple memo misses
nearly every token because output blocks churn. The miss path must not sync, and a
table handed to a caller (attn.prealloc_qkv_cptrs_1 holds one across steps) must stay
correct across later misses.
"""
import pytest
import torch

from exllamav3.modules.multilinear import SlicedMultiLinear


class _Lin:
    def __init__(self, out_features):
        self.out_features = out_features


def _stub(device, widths):
    """SlicedMultiLinear with one slice per linear, no column offset."""
    s = SlicedMultiLinear.__new__(SlicedMultiLinear)
    s.device = torch.device(device)
    s.num_src = len(widths)
    s.num_slices = len(widths)
    s.slice_targets = list(range(len(widths)))
    s.slice_offsets = [0] * len(widths)
    s.linears = [_Lin(w) for w in widths]
    return s


def test_values_on_gpu(device):
    s = _stub(device, [8, 4])
    outs = [torch.zeros(3, 8, device = device), torch.zeros(3, 4, device = device)]
    got = s.c_ptrs(outs).cpu()
    assert torch.equal(got, torch.tensor([outs[0].data_ptr(), outs[1].data_ptr()], dtype = torch.long))


def test_memo_hit_returns_same_object(device):
    s = _stub(device, [8])
    outs = [torch.zeros(3, 8, device = device)]
    a = s.c_ptrs(outs)
    assert s.c_ptrs(outs) is a


def test_positive_control_staleness_detected(device):
    # Change the outputs -> the table must change; a stale memo would pass test_memo_hit vacuously.
    s = _stub(device, [8])
    o1 = torch.zeros(3, 8, device = device)
    a = s.c_ptrs([o1])
    o2 = torch.zeros(3, 8, device = device)
    b = s.c_ptrs([o2])
    assert a is not b
    assert b.cpu().item() == o2.data_ptr()


def test_held_table_survives_later_misses(device):
    # attn.prealloc_qkv_cptrs_1 holds a table across decode steps: 8 later misses must
    # not mutate it (each miss returns a FRESH GPU tensor).
    s = _stub(device, [8])
    held_out = torch.zeros(3, 8, device = device)
    held = s.c_ptrs([held_out])
    want = held.cpu().clone()
    assert want.item() == held_out.data_ptr()
    for _ in range(8):
        churn = torch.zeros(3, 8, device = device)
        s.c_ptrs([churn])
        del churn
    assert torch.equal(held.cpu(), want)


def test_offset_applied(device):
    s = _stub(device, [8])
    s.slice_offsets = [2]
    s.linears = [_Lin(6)]
    s.num_src = 1
    big = torch.zeros(3, 6, device = device)
    got = s.c_ptrs([big]).cpu().item()
    assert got == big.data_ptr() + 2 * big.element_size()