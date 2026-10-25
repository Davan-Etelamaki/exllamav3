"""
g_tensor_cache must never keep a buffer built inside torch.inference_mode().

Regression fence for the prod load abort (2026-10-24): the load-time autosplit dummy forward
runs inside InferenceMode, so every tensor it created was an inference tensor. Anything the
cache kept from that pass came back on the first real decode, where a normal-mode in-place
write on an inference tensor raises ("Inplace update to inference tensor outside InferenceMode
is not allowed"). get/get_bucketed now hand back throwaway buffers there.

The positive control feeds a cache entry an inference tensor and asserts the normal-mode write
raises -- without it, a green would not show that the poisoning this guards against is real.
"""
import pytest
import torch

from exllamav3.util.tensor import GTensorCache, g_tensor_cache


def test_positive_control_inference_tensor_inplace_raises(device):
    """A cache entry holding an inference tensor is a live landmine, not a theory."""
    c = GTensorCache()
    with torch.inference_mode():
        bad = torch.empty((64,), dtype = torch.float, device = device)
    assert torch.is_inference(bad)
    c.cache[c.make_key(device, (64,), torch.float, "ctrl_poison")] = (1, bad)
    handed_back = c.get(device, (64,), torch.float, "ctrl_poison")
    assert handed_back is bad
    with pytest.raises(RuntimeError, match = "Inplace update to inference tensor"):
        handed_back.zero_()


def test_get_and_get_bucketed_keep_nothing_in_inference_mode(device):
    tag = "fence_inf_keep_nothing"
    with torch.inference_mode():
        a = g_tensor_cache.get(device, (128,), torch.float, tag)
        b = g_tensor_cache.get_bucketed(device, 100, torch.float, tag)
        b2 = g_tensor_cache.get_bucketed(device, 100, torch.float, tag)
        # throwaway means: not the same object twice, and nothing retained
        assert b2 is not b and b.numel() == 100 and b2.numel() == 100
        assert g_tensor_cache.make_key(device, (128,), torch.float, tag) not in g_tensor_cache.cache
        assert not any(k[3] == tag for k in g_tensor_cache._slice_cache)
    assert a.numel() == 128

    # Crucial: the same tag fetched in NORMAL mode must NOT be an inference tensor, i.e. the
    # inference-mode pass left no entry behind for the serving path to trip over.
    plain = g_tensor_cache.get(device, (128,), torch.float, tag)
    assert not torch.is_inference(plain)
    plain.zero_()                                        # the write that used to raise
    buck = g_tensor_cache.get_bucketed(device, 100, torch.float, tag)
    buck.zero_()
    del g_tensor_cache.cache[g_tensor_cache.make_key(device, (128,), torch.float, tag)]


def test_cache_still_caches_outside_inference_mode(device):
    """The guard must not disable caching for the serving path."""
    tag = "fence_inf_still_caches"
    v1 = g_tensor_cache.get(device, (256,), torch.float, tag)
    v2 = g_tensor_cache.get(device, (256,), torch.float, tag)
    assert v1 is v2 and not torch.is_inference(v1)
    s1 = g_tensor_cache.get_bucketed(device, 200, torch.float, tag)
    s2 = g_tensor_cache.get_bucketed(device, 200, torch.float, tag)
    assert s1 is s2 and s1.numel() == 200
    v1.zero_()                                           # normal tensor: in-place is fine
    g_tensor_cache.cache.pop(g_tensor_cache.make_key(device, (256,), torch.float, tag), None)
    g_tensor_cache._slice_cache.pop((torch.device(device), 256, torch.float, tag, 200), None)