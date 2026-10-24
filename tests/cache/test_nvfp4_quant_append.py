# Fenced test for the NVFP4 quant+append fast path (law 12: math change -> unit test +
# latency fence). Bit-exactness against the Python reference on every fast-path-eligible
# shape, and a coarse latency fence: the Triton path must stay under the Python path's
# wall time at decode shape (thresholds generous — the check is direction + order, not
# a performance target; production decode latency is the owner-side referee metric).
import pytest
import torch
from exllamav3.cache.nvfp4 import CacheLayer_nvfp4
from exllamav3.constants import PAGE_SIZE

SHAPES = [(1, 1, 4, 128), (2, 1, 8, 128), (1, 64, 4, 128), (4, 16, 4, 128), (2, 33, 4, 128), (1, 1, 4, 64)]

def _layer(pages, H, D):
    lay = CacheLayer_nvfp4.__new__(CacheLayer_nvfp4)
    lay.shape4 = (pages, PAGE_SIZE, H, D // 2)
    lay.shape_s = (pages, PAGE_SIZE, H, D // 16)
    lay.shape = None
    lay.device = torch.device("cuda")
    lay.k = torch.zeros(lay.shape4, dtype=torch.uint8, device=lay.device)
    lay.v = torch.zeros(lay.shape4, dtype=torch.uint8, device=lay.device)
    lay.ks = torch.zeros(lay.shape_s, dtype=torch.float8_e4m3fn, device=lay.device)
    lay.vs = torch.zeros(lay.shape_s, dtype=torch.float8_e4m3fn, device=lay.device)
    return lay

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(("B", "L", "H", "D"), SHAPES)
def test_quant_append_bit_exact(B, L, H, D):
    assert SHAPES, "positive control: parametrization list must not be empty"
    gen = torch.Generator(device="cpu").manual_seed(11)
    pages = 160
    nblk = (300 + L + PAGE_SIZE - 1) // PAGE_SIZE
    bt = torch.randperm(pages, generator=gen)[: B * nblk].reshape(B, nblk).contiguous().int().cuda()
    seqlen = torch.randint(0, 300, (B,), generator=gen, dtype=torch.int32).cuda()
    k = (torch.randn((B, L, H, D), generator=gen) * 3).half().cuda()
    v = (torch.randn((B, L, H, D), generator=gen) * 3).half().cuda()
    k[0, 0, 0, :16] = 0.0                    # zero block scale
    k[B - 1, -1, -1, :] *= 1500               # scale clamp (>448) path
    k[:, 0, :, 3] = -2.5                      # negative sign codes
    ref = _layer(pages, H, D)
    fast = _layer(pages, H, D)
    fast.update_kv(seqlen.clone(), bt, k, v, L)
    ref._update_kv_python(seqlen.clone(), bt, k, v, L)
    for name in ("k", "v", "ks", "vs"):
        assert torch.equal(getattr(ref, name), getattr(fast, name)), f"{name} diverges at B{B} L{L} H{H} D{D}"
    touched = fast.k.view(torch.uint8).sum().item() + fast.v.view(torch.uint8).sum().item()
    assert touched > 0, "sanity: pages must be written"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_quant_append_decode_latency_fence():
    import time
    lay = _layer(160, 4, 128)
    seqlen = torch.tensor([1000], dtype=torch.int32).cuda()
    bt = torch.arange(160, dtype=torch.int32).reshape(1, 160).cuda()
    k = (torch.randn((1, 1, 4, 128), generator=torch.Generator().manual_seed(3)) * 3).half().cuda()
    v = (torch.randn((1, 1, 4, 128), generator=torch.Generator().manual_seed(4)) * 3).half().cuda()
    def bench(fn, iters=30):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
        return sorted(ts)[len(ts) // 2]
    t_py = bench(lambda: lay._update_kv_python(seqlen.clone(), bt, k, v, 1))
    t_tl = bench(lambda: lay.update_kv(seqlen.clone(), bt, k, v, 1))
    assert t_tl < t_py, f"latency fence: triton {t_tl*1e6:.1f}us must beat python {t_py*1e6:.1f}us"
    assert t_tl < 1.0e-3, f"absolute fence: decode-shape update_kv {t_tl*1e6:.1f}us exceeded 1000us"

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))