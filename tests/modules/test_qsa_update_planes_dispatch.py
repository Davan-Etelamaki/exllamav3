"""
Differential fence for cut 10: QSAIndexer.update_planes now launches the stage / plane-update /
pool-update kernels through _fast_dispatch. The dispatcher must be semantically identical to
fn[grid](*args, **consts): run update_planes once per dispatch mode and require bit-identical
planes, pools and projected q. Positive control (law 5): changing x must change the outputs.
Regime: dev GPU (shared sm_120); prod-scale correctness remains boot-pending.
"""
import math
import types

import pytest
import torch

from exllamav3.constants import PAGE_SIZE
from exllamav3.modules.qsa_indexer import QSAIndexer
from exllamav3.modules.attention_fn import triton_paged


def _stub(cr, H, dk, hs, dev):
    s = QSAIndexer.__new__(QSAIndexer)
    s.compress_ratio = cr
    s.n_heads = H
    s.head_dim = dk
    s.scale = 1.0 / math.sqrt(dk)
    g = torch.Generator(device = "cpu").manual_seed(7)
    lin = torch.randn(((H + 1) * dk, hs), generator = g).half().to(dev)
    s.index_qk_proj = types.SimpleNamespace(forward = lambda x, params: x.view(-1, hs) @ lin.T)
    nw_q = torch.ones(dk).half().to(dev)
    nw_k = torch.ones(dk).half().to(dev)
    s._norm_w_half = lambda name, dev_: nw_q if name == "q" else nw_k
    s.q_layernorm = types.SimpleNamespace(rms_norm_eps = 1e-6)
    s.k_layernorm = types.SimpleNamespace(rms_norm_eps = 1e-6)
    s._workspace = lambda R, numel, dt, tag, dev_: torch.empty((numel,), dtype = dt, device = dev_)
    return s


def _rope(dk, dev):
    return types.SimpleNamespace(
        inv_freq = torch.rand(dk // 2, generator = torch.Generator().manual_seed(3)).to(dev),
        attn_factor = 1.0,
        apply = lambda q, k, a, b, c, d: None,
    )


def _run(s, layer, x, rope, bt, seqlens_cpu, params):
    q = s.update_planes(layer, x, rope, bt, seqlens_cpu, params)
    torch.cuda.synchronize()
    return q.clone(), layer.raw_k.clone(), layer.pooled.clone()


@pytest.mark.parametrize("cr, bsz, seqlen", [(4, 1, 4), (8, 2, 8)])
def test_fast_dispatch_matches_direct_launch(device, cr, bsz, seqlen):
    dk, H, hs = 128, 32, 256
    dev = torch.device(device)
    s = _stub(cr, H, dk, hs, dev)
    epp = PAGE_SIZE // cr
    pages = 4
    g = torch.Generator(device = "cpu").manual_seed(11)
    layer = types.SimpleNamespace(
        raw_k = (torch.randn((pages, PAGE_SIZE, dk), generator = g) * 0.3).half().to(dev),
        pooled = (torch.randn((pages * epp, dk), generator = g) * 0.3).half().to(dev),
    )
    bt = torch.arange(pages, dtype = torch.int32, device = dev)[:bsz * (pages // bsz)].view(bsz, -1).contiguous()
    x = (torch.randn((bsz, seqlen, hs), generator = g) * 0.5).half().to(dev)
    seqlens = torch.tensor([0] * bsz, dtype = torch.int32)
    rope = _rope(dk, dev)
    params = {}

    orig = triton_paged._fast_dispatch
    try:
        q_fast, rk_f, pl_f = _run(s, layer, x, rope, bt, seqlens, params)
        # direct-launch mode: the fallback path _fast_dispatch itself uses
        triton_paged._fast_dispatch = lambda fn, grid, args, consts: fn[grid](*args, **consts)
        layer.raw_k.copy_(rk_f); layer.pooled.copy_(pl_f)
        q_dir, rk_d, pl_d = _run(s, layer, x, rope, bt, seqlens, params)
    finally:
        triton_paged._fast_dispatch = orig

    assert torch.equal(q_fast, q_dir)
    assert torch.equal(rk_f, rk_d)
    assert torch.equal(pl_f, pl_d)
    assert torch.isfinite(q_fast.float()).all() and q_fast.abs().sum() > 0

    # positive control: different input must change the projected q
    x2 = x.clone(); x2[:, :, :4] += 1.0
    q2, _, _ = _run(s, layer, x2, rope, bt, seqlens, params)
    assert not torch.equal(q_fast.view(q2.shape), q2)