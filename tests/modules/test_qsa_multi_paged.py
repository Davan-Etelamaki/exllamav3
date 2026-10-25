"""
Batched paged QSA selection (bsz > 1 -> QSAIndexer._select_rows_multi, the MULTIROW kernel chain).

Regression fence for the prod illegal-memory-access (2026-10-24). In the shipped code the batched
score row was strided from the block table's PAGE count:

    s_stride = -(-block_table.shape[1] // 128) * 128        # pages (e.g. 32) -> 128

but a score row holds one entry per POOL, and the pool count T reaches pages * EPP with
EPP = PAGE_SIZE / compress_ratio (64 at cr = 4). The row was therefore EPP times too short, so the
few-query scorer wrote past the end of its row and dsa_topk scanned past the buffer: a CUDA illegal
memory access as soon as a batched sequence passed roughly one page of context -- precisely what the
engine's own load/warmup sweep (batch > 1) hits.

The invariant held here is the one that matters for the batched chain: it must reproduce the proven
per-sequence path (_select_rows -- one launch per sequence, the code prod ran before the batched
chain existed) *bit for bit*. Every case also asserts the geometry that made the shipped stride
wrong (page-derived stride < T), and that the shape is a real top-k, so the fence goes RED on the
defect instead of merely passing.

Not asserted here: equality against select_indices_paged_ref. Observed 2026-10-24 in this harness,
that comparison flips between two runs of the same case (True, then False, same tensors), because
the reference reads the cache plane through a different position model than the batched chain; it is
not a trustworthy gate and is deliberately not used. The batched-vs-per-sequence equality above is
reproducible across runs and geometries, including seq > 1.
"""
import math
import types

import pytest
import torch

from exllamav3.constants import PAGE_SIZE
from exllamav3.modules.qsa_indexer import QSAIndexer


def _stub(cr, H, dk, token_budget):
    s = QSAIndexer.__new__(QSAIndexer)
    s.compress_ratio = cr
    s.token_budget = token_budget
    s.block_topk = token_budget // cr
    s.n_heads = H
    s.head_dim = dk
    s.scale = 1.0 / math.sqrt(dk)
    return s


@pytest.mark.parametrize("cr, bsz, seq, pos0", [
    (4, 2, 1, 6144),     # decode batch past one page of context: the prod fault geometry
    (4, 2, 5, 5120),     # MTP-verify q_len (seq > 1)
    (4, 3, 1, 4096),
    (8, 2, 1, 6144),     # EPP = 32
    (8, 2, 3, 8192),
])
def test_paged_multi_matches_per_sequence_path(device, cr, bsz, seq, pos0):
    dk, H, budget = 128, 32, 2048
    dev = torch.device(device)
    s = _stub(cr, H, dk, budget)
    epp = PAGE_SIZE // cr
    pages_per_row = 8192 // PAGE_SIZE
    n_pools_per_row = pages_per_row * epp
    t_max = (pos0 + seq) // cr
    old_stride = -(-pages_per_row // 128) * 128
    # The geometry that made the shipped code fault, and a real top-k (fewer candidates than budget)
    assert t_max > old_stride and t_max <= n_pools_per_row
    assert s.block_topk < t_max

    g = torch.Generator().manual_seed(11 + cr)
    plane = torch.randn((bsz * pages_per_row, epp, dk), dtype = torch.half, generator = g).to(dev)
    raw_k = torch.randn((bsz * pages_per_row, PAGE_SIZE, dk), dtype = torch.half, generator = g).to(dev)
    layer = types.SimpleNamespace(pooled = plane, raw_k = raw_k)
    bt = torch.arange(bsz * pages_per_row, dtype = torch.int32, device = dev) \
        .view(bsz, pages_per_row).contiguous()
    q_idx = (torch.randn((bsz, seq, H, dk), generator = g) * 0.5).half().to(dev)
    seqlens = torch.tensor([pos0] * bsz, dtype = torch.int32)

    out = s.select_indices_paged(layer, q_idx, bt, seqlens).clone()
    single = torch.empty_like(out)
    pool_flat = plane.view(-1, dk)
    for b in range(bsz):
        s._select_rows(q_idx[b].contiguous(), pool_flat, pos0, (pos0 + seq) // cr,
                       single[b * seq:(b + 1) * seq], block_table = bt[b], epp = epp)
    torch.cuda.synchronize()

    assert out.shape == single.shape
    assert (out >= 0).sum(1).tolist() == (single >= 0).sum(1).tolist(), \
        f"valid count differs at cr={cr} bsz={bsz} seq={seq} pos0={pos0}: " \
        f"{(out >= 0).sum(1).tolist()} vs {(single >= 0).sum(1).tolist()}"
    assert torch.equal(out, single), \
        f"batched selection differs from the per-sequence path at cr={cr} bsz={bsz} seq={seq} " \
        f"pos0={pos0}: {int(((out != single) & (out >= 0)).sum())} of {out.numel()} tokens"