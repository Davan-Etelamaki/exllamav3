"""
GatedResidual._gr_mix_bufs view memo (fused decode path).

Live py-spy (20 s, prod decode, wheel b75a161880c0) put the ws closure -- three
g_tensor_cache.get_bucketed lookups per GatedResidual site per token -- at ~9.7% of all
samples (hyperconnections.py:568/569 leaves). The memo returns the same view tuple per
(dev, R, M, use_combine) and its entries are built under inference_mode(False) so a
warmup-pass view can never poison serving.
"""
import torch

from exllamav3.modules.hyperconnections import GatedResidual


def _stub(use_combine=True):
    g = GatedResidual.__new__(GatedResidual)
    g.use_combine = use_combine
    return g


def test_identity_and_shapes(device):
    dev = torch.device(device)
    g = _stub()
    R, M, H, Dh = 2, 5, 4, 8
    d1, p1, m1 = g._gr_mix_bufs(dev, R, M, H, Dh)
    assert d1.shape == (R, M, H) and d1.dtype == torch.float
    assert p1.shape == (R, H) and p1.dtype == torch.float
    assert m1.shape == (R, Dh) and m1.dtype == torch.half
    d2, p2, m2 = g._gr_mix_bufs(dev, R, M, H, Dh)
    assert (d1, p1, m1) == (d2, p2, m2)      # memo hit: identical views


def test_positive_control_no_memo_without_cache(device):
    # Clearing the memo must yield fresh view objects, so the identity assertion above
    # cannot pass vacuously.
    dev = torch.device(device)
    g = _stub()
    a = g._gr_mix_bufs(dev, 1, 3, 4, 8)[0]
    g._gr_mix_views.clear()
    b = g._gr_mix_bufs(dev, 1, 3, 4, 8)[0]
    assert a is not b


def test_keys_segregate(device):
    dev = torch.device(device)
    g = _stub()
    r2 = g._gr_mix_bufs(dev, 2, 3, 4, 8)
    r3 = g._gr_mix_bufs(dev, 3, 3, 4, 8)
    nc = _stub(use_combine = False)
    n = nc._gr_mix_bufs(dev, 2, 3, 4, 8)
    assert r2[0] is not r3[0]
    assert n[1] is None and n[0] is not r2[0]


def test_entries_are_normal_tensors_under_inference_mode(device):
    dev = torch.device(device)
    g = _stub()
    with torch.inference_mode():
        d, p, m = g._gr_mix_bufs(dev, 2, 3, 4, 8)
    assert not torch.is_inference(d) and not torch.is_inference(p) and not torch.is_inference(m)
    d.zero_()          # writable from outside the pass


def test_cap_clears(device):
    dev = torch.device(device)
    g = _stub()
    for r in range(1, 18):
        g._gr_mix_bufs(dev, r, 3, 4, 8)
    assert len(g._gr_mix_views) <= 16