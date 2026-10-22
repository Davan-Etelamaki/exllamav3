"""
qsa_sparse_attend_rows over NVFP4 and FP8 pages (CacheLayer_qsa_nvfp4 / CacheLayer_qsa_fp8) against the
fp16 kernel fed the SAME values dequantized in torch: tests the online nibble-decode / fp8-cast page read
and that neither format takes the H32 rotation (both are stored unrotated), not the quantizer.
"""
import pytest
import torch

from exllamav3.constants import PAGE_SIZE
from exllamav3.cache.nvfp4 import nvfp4_quantize, nvfp4_dequantize
from exllamav3.modules.attention_fn.qsa_triton import qsa_sparse_attend_rows
from testlib.compare import assert_rel_close


def _make_case(device, R, H, kvh, hd, topk):
    torch.manual_seed(R * 7 + hd)
    pages = 12
    rows = pages * PAGE_SIZE
    k = torch.randn((rows, kvh, hd), dtype = torch.half, device = device)
    v = torch.randn((rows, kvh, hd), dtype = torch.half, device = device)
    q = torch.randn((R, H, hd), dtype = torch.half, device = device)
    K_pad = -(-topk // 32) * 32
    indices = torch.full((R, K_pad), -1, dtype = torch.int32, device = device)
    n_pos = pages * PAGE_SIZE
    for r in range(R):
        n = min(topk, n_pos - 1)
        indices[r, :n] = torch.randperm(n_pos - 1, device = device)[:n].int()
    bt = torch.stack([torch.randperm(pages, device = device, dtype = torch.int32) for _ in range(R)])
    return rows, k, v, q, indices, bt, hd ** -0.5


@pytest.mark.parametrize("R,H,kvh,hd,topk", [(1, 24, 2, 256, 2051), (5, 8, 2, 128, 300), (3, 16, 4, 64, 96)])
def test_qsa_sparse_nvfp4(device, R, H, kvh, hd, topk):
    rows, k, v, q, indices, bt, scale = _make_case(device, R, H, kvh, hd, topk)

    k4, ks = nvfp4_quantize(k)
    v4, vs = nvfp4_quantize(v)
    k_deq = nvfp4_dequantize(k4.int(), ks.half()).view(rows, kvh, hd)
    v_deq = nvfp4_dequantize(v4.int(), vs.half()).view(rows, kvh, hd)
    assert k_deq.shape == k.shape and v_deq.shape == v.shape

    ref = qsa_sparse_attend_rows(
        q, k_deq, v_deq, indices, scale,
        block_table = bt, page_size = PAGE_SIZE,
    ).clone()
    got = qsa_sparse_attend_rows(
        q, k4.view(-1, kvh, hd // 2), v4.view(-1, kvh, hd // 2), indices, scale,
        block_table = bt, page_size = PAGE_SIZE,
        kv_format = "nvfp4", k_scales = ks, v_scales = vs, n_kv_heads = kvh,
    )
    assert_rel_close(got, ref, 8e-3)


@pytest.mark.parametrize("R,H,kvh,hd,topk", [(1, 24, 2, 256, 2051), (5, 8, 2, 128, 300), (3, 16, 4, 64, 96)])
def test_qsa_sparse_fp8(device, R, H, kvh, hd, topk):
    rows, k, v, q, indices, bt, scale = _make_case(device, R, H, kvh, hd, topk)

    k8 = k.to(torch.float8_e4m3fn)
    v8 = v.to(torch.float8_e4m3fn)
    k_deq = k8.half()
    v_deq = v8.half()

    ref = qsa_sparse_attend_rows(
        q, k_deq, v_deq, indices, scale,
        block_table = bt, page_size = PAGE_SIZE,
    ).clone()
    got = qsa_sparse_attend_rows(
        q, k8.view(-1, kvh, hd), v8.view(-1, kvh, hd), indices, scale,
        block_table = bt, page_size = PAGE_SIZE,
        kv_format = "fp8", n_kv_heads = kvh,
    )
    assert_rel_close(got, ref, 8e-3)