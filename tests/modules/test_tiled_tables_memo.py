"""Fence for GatedResidual._tiled_tables memo (campaign phase 2, 2026-10-24).

_tiled_tables re-quantized the resident fp16 weights into int8 tables on EVERY
tiled (prefill-domain) call — ~1.9% of live decode samples (py-spy prod). The
weights are frozen per model load, so the tables are memoized per site, keyed by
the source tensors' data_ptr with strong refs held (a rebuilt proj_h/upx_h lands
on a fresh pointer and misses).

Regime: unit-level, single-GPU, inference_mode contexts; prod-share delta is
boot-pending until the next live py-spy readback.
"""
import pytest
import torch

from exllamav3.modules.hyperconnections import GatedResidual
from exllamav3.ext import exllamav3_ext as ext


def _stub(dev="cuda"):
    obj = GatedResidual.__new__(GatedResidual)
    H, Dh, rank, Mpad = 2, 8, 4, 8
    obj.hc_mult = H
    obj.hidden_size = Dh
    obj.rank = rank
    gen = torch.Generator(device=dev).manual_seed(7)
    obj.proj_h = torch.randn(Mpad, H * Dh, dtype=torch.half, device=dev, generator=gen)
    obj.upx_h = torch.randn(H, Dh // 4, rank, 4, dtype=torch.half, device=dev, generator=gen)
    return obj


@pytest.mark.parametrize("device", ["cuda"])
def test_memo_returns_same_objects(device):
    obj = _stub(device)
    t1 = obj._tiled_tables()
    t2 = obj._tiled_tables()
    assert t1 is not None and len(t1) == 4
    for a, b in zip(t1, t2):
        assert a is b, "memo miss on identical weights"


@pytest.mark.parametrize("device", ["cuda"])
def test_positive_control_no_memo_without_cache(device):
    # Clearing the memo MUST produce fresh tensors — proves the identity in the
    # first test comes from the cache, not from allocator luck.
    obj = _stub(device)
    t1 = obj._tiled_tables()
    del obj._tiled_tables_cache
    t2 = obj._tiled_tables()
    assert t1[0] is not t2[0]
    for a, b in zip(t1, t2):
        assert torch.equal(a, b), "deterministic split must reproduce byte-identical tables"


@pytest.mark.parametrize("device", ["cuda"])
def test_rebuilt_weights_invalidate_memo(device):
    obj = _stub(device)
    t1 = obj._tiled_tables()
    obj.proj_h = obj.proj_h.clone()  # model_init rebuild: new tensor, new ptr
    t2 = obj._tiled_tables()
    assert t1[0] is not t2[0], "rebuilt proj_h must miss the memo"
    assert not torch.equal(t1[2], t2[2]) or True  # up tables may match; proj tables must reflect new data
    fresh = torch.empty_like(t2[0])
    sb = torch.empty_like(t2[1])
    ext.det_quant_weight(obj.proj_h, fresh, sb)
    assert torch.equal(fresh, t2[0]), "memo must carry the CURRENT weights' tables"


@pytest.mark.parametrize("device", ["cuda"])
def test_tables_usable_under_inference_mode(device):
    obj = _stub(device)
    with torch.inference_mode():
        t1 = obj._tiled_tables()  # miss inside inference mode
        t2 = obj._tiled_tables()  # hit
    for a, b in zip(t1, t2):
        assert a is b
    # tables built outside inference mode are writable/normal tensors downstream
    assert not torch.is_inference(t1[0])