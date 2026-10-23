"""
Fence test: NVFP4 cache generic-path (fallback) correctness.

Drives exllamav3.modules.attention_fn.dispatch.attn_dispatch against a real
CacheLayer_nvfp4 and classifies each path:
  FAST    fp16 q  -> Triton online-dequant + in-kernel append
  BF16    bf16 q  -> dispatch generic branch (get_kv/dequant_full + update_kv post-hook)
  POSCTL  manual torch SDPA over dequantized pages + the new token (harness orientation)
  SABOT   perturbed page must change the fast-path output (prove checks fire)

Expected on a healthy fallback: BF16 arm runs (or declines cleanly) and agrees
with FAST within quantization noise. A crash, a declined-with-no-backend error,
or numerically divergent output is a RED result for the fallback lane.
"""
import sys, traceback
import torch
import torch.nn.functional as F

sys.path.insert(0, "/data/Repos/musicman89/exllamav3")
from exllamav3.constants import PAGE_SIZE
from exllamav3.cache.nvfp4 import CacheLayer_nvfp4, nvfp4_dequantize, nvfp4_quantize
from exllamav3.modules.attention_fn.dispatch import attn_dispatch

DEV = "cuda"
class AttnShim:
    num_kv_heads = 2
    head_dim = 256

def build_layer(tokens: torch.Tensor):
    """tokens [N, h, d] fp16 -> layer holding them at logical rows 0..N-1."""
    n = tokens.shape[0]
    pages = max(4, (n + PAGE_SIZE - 1) // PAGE_SIZE + 2)
    lay = CacheLayer_nvfp4(None, AttnShim(), 0, pages * PAGE_SIZE)
    lay.alloc(torch.device(DEV))
    bt = torch.arange(pages, dtype=torch.int32, device=DEV).view(1, -1)
    cs = torch.tensor([0], dtype=torch.int32, device=DEV)
    lay.update_kv(cs, bt, tokens[:1].unsqueeze(0)[:, :n], tokens[:1].unsqueeze(0)[:, :n], n)
    # NOTE: update_kv quantizes k and v from the SAME tensor here only for fill convenience;
    # distinct real K/V are used for the actual arms below.
    return lay, bt

def main():
    torch.manual_seed(0)
    h, d, qh = AttnShim.num_kv_heads, AttnShim.head_dim, 4
    n0 = 64
    hist_k = torch.randn(1, n0, h, d, dtype=torch.float16, device=DEV)
    hist_v = torch.randn(1, n0, h, d, dtype=torch.float16, device=DEV)
    new_k  = torch.randn(1, 1, h, d, dtype=torch.float16, device=DEV)
    new_v  = torch.randn(1, 1, h, d, dtype=torch.float16, device=DEV)
    q      = torch.randn(1, 1, qh, d, dtype=torch.float16, device=DEV)

    pages = max(4, n0 // PAGE_SIZE + 2)
    def fresh_layer():
        lay = CacheLayer_nvfp4(None, AttnShim(), 0, pages * PAGE_SIZE)
        lay.alloc(torch.device(DEV))
        return lay
    bt = torch.arange(pages, dtype=torch.int32, device=DEV).view(1, -1)
    cs = torch.tensor([n0 - 1], dtype=torch.int32, device=DEV)  # prior tokens: rows 0..n0-2

    def seed(lay):
        lay.update_kv(torch.tensor([0], dtype=torch.int32, device=DEV), bt,
                      hist_k[:, : n0 - 1], hist_v[:, : n0 - 1], n0 - 1)

    results = {}

    # ---------- FAST arm ----------
    lay = fresh_layer(); seed(lay)
    o_fast = attn_dispatch(q, new_k, new_v, cache=lay, block_table=bt,
                           cache_seqlens=cs.clone(), causal=True)
    results["FAST"] = f"ran, out {tuple(o_fast.shape)} finite={bool(torch.isfinite(o_fast).all())}"

    # ---------- post-hook/in-kernel append check: new K/V must be in pages ----------
    pool_k = nvfp4_dequantize(lay.k.int(), lay.ks.half())          # (pages, PS, h, d)
    flat_k = pool_k.reshape(-1, h, d)[:n0]                          # logical rows
    err = (flat_k[n0 - 1] - nvfp4_dequantize(*nvfp4_quantize(new_k)).reshape(h, d)).abs().max()
    results["APPEND"] = f"row{n0-1} vs quantized(new_k) maxerr={err.item():.4f}"

    # ---------- POSCTL: fast output ~= torch SDPA over dequantized cache+new ----------
    kk = flat_k[None]                                               # pool already contains the appended new K/V
    vv = nvfp4_dequantize(lay.v.int(), lay.vs.half()).reshape(-1, h, d)[:n0][None]
    kk = kk.transpose(1, 2); vv = vv.transpose(1, 2)                # [1, h, n0, d]
    qq = q.transpose(1, 2)                                          # [1, qh, 1, d]
    o_ref = F.scaled_dot_product_attention(qq, kk, vv, enable_gqa=True).transpose(1, 2)
    md = (o_fast.float() - o_ref.float()).abs().max().item()
    results["POSCTL"] = f"fast-vs-sdpa(dequant-cache) maxdiff={md:.4f} (noise ceiling ~0.15)"

    # ---------- SABOT ----------
    lay2 = fresh_layer(); seed(lay2)
    o_clean = attn_dispatch(q, new_k, new_v, cache=lay2, block_table=bt,
                            cache_seqlens=cs.clone(), causal=True)
    lay2.k[0, 5, 0, 0] ^= 0xFF                                     # one nibble pair
    o_bad = attn_dispatch(q, new_k, new_v, cache=lay2, block_table=bt,
                          cache_seqlens=cs.clone(), causal=True)
    results["SABOT"] = f"perturbation changed output: {not torch.equal(o_clean, o_bad)}"

    # ---------- BF16 fallback arm ----------
    lay3 = fresh_layer(); seed(lay3)
    try:
        o_fb = attn_dispatch(q.bfloat16(), new_k.bfloat16(), new_v.bfloat16(),
                             cache=lay3, block_table=bt, cache_seqlens=cs.clone(), causal=True)
        d_fb = (o_fb.float() - o_fast.float()).abs().max().item()
        results["BF16"] = f"ran, maxdiff-vs-fast={d_fb:.4f}"
    except Exception as e:
        tb = traceback.extract_tb(sys.exc_info()[2])
        results["BF16"] = f"EXCEPTION {type(e).__name__}: {str(e)[:90]} @ {tb[-1].filename.split('/')[-1]}:{tb[-1].lineno}"

    # ---------- VARLEN fallback arm (fp16 q, cu_seqlens -> generic path, post-hook runs) ----------
    lay4 = fresh_layer(); seed(lay4)
    cu = torch.tensor([0, 1], dtype=torch.int32, device=DEV)
    try:
        o_vl = attn_dispatch(q, new_k, new_v, cache=lay4, block_table=bt,
                             cache_seqlens=cs.clone(), cu_seqlens=cu, max_seqlen=1, causal=True)
        d_vl = (o_vl.float() - o_fast.float()).abs().max().item() if o_vl.shape == o_fast.shape else float('nan')
        results["VARLEN"] = f"ran, maxdiff-vs-fast={d_vl:.4f}"
    except Exception as e:
        tb = traceback.extract_tb(sys.exc_info()[2])
        results["VARLEN"] = f"EXCEPTION {type(e).__name__}: {str(e)[:90]} @ {tb[-1].filename.split('/')[-1]}:{tb[-1].lineno}"

    # ---------- post-state check: fallback writeback must not corrupt pages ----------
    try:
        pk2 = nvfp4_dequantize(lay4.k.int(), lay4.ks.half()).reshape(-1, h, d)[: n0 - 1]
        bad = (pk2 - flat_k[: n0 - 1]).abs().max().item()
        results["FBWRITE4"] = f"varlen-call history rows intact: maxdiff={bad:.4f}"
    except Exception as e:
        results["FBWRITE4"] = f"SKIP ({type(e).__name__})"

    for k in ["FAST", "APPEND", "POSCTL", "SABOT", "BF16", "VARLEN", "FBWRITE4"]:
        print(f"{k:8s}: {results[k]}")

if __name__ == "__main__":
    main()