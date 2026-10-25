"""
g_tensor_cache must stay usable inside torch.inference_mode() AND never hand out an inference tensor.

Regression fence for the prod load abort (2026-10-24): the load-time autosplit dummy forward runs
inside InferenceMode, so every tensor it created was an inference tensor. Anything the cache kept
from that pass came back on the first real decode, where a normal-mode in-place write on an
inference tensor raises ("Inplace update to inference tensor outside InferenceMode is not allowed").

The fix is that the cache allocates its entries with InferenceMode disabled, so a kept entry is a
NORMAL tensor: a call inside a pass still reuses its buffer (caching is the whole point of the
cache and the engine's serving path runs in a pass), and the same entry written outside the pass
is legal. Disabling caching inside a pass, the earlier sledgehammer, is RED against
tests/kernels/dsa/test_dsa_kernels.py::test_workspaces_kept_only_for_decode_rows.

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


def test_cache_keeps_normal_buffers_inside_inference_mode(device):
    """Caching must survive a pass, and the kept entry must not be an inference tensor."""
    tag = "fence_inf_keeps"
    with torch.inference_mode():
        v1 = g_tensor_cache.get(device, (128,), torch.float, tag)
        v2 = g_tensor_cache.get(device, (128,), torch.float, tag)
        assert v1 is v2, "caching disabled inside InferenceMode: the whole cache is dead there"
        assert not torch.is_inference(v1), "an inference tensor was kept as a cache entry"
        s1 = g_tensor_cache.get_bucketed(device, 100, torch.float, tag)
        s2 = g_tensor_cache.get_bucketed(device, 100, torch.float, tag)
        assert s1 is s2 and s1.numel() == 100 and not torch.is_inference(s1)
        v1.zero_()                       # legal: v1 is a normal tensor
    # And still writable once the pass is over -- the write that used to raise on a poisoned entry.
    v1.zero_()
    s1.zero_()
    g_tensor_cache.cache.pop(g_tensor_cache.make_key(device, (128,), torch.float, tag), None)
    g_tensor_cache._slice_cache.pop((torch.device(device), 128, torch.float, tag, 100), None)


def test_cache_still_caches_outside_inference_mode(device):
    """The serving path outside a pass keeps caching, unchanged."""
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