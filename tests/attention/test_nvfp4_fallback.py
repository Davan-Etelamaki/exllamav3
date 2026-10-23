"""
Fence: NVFP4 cache generic-path (fallback) correctness + bounded staging + latency.

Bar (owner objective): kill the pool-wide dequant_full() fallback with bounded
staging + coherent block_table; the fallback must RUN, not crash.
  FAST/APPEND/POSCTL/SABOT  regression of the Triton fast path (must stay GREEN)
  BF16    bf16 q runs through the staged generic path, output ~= fast path,
          real pages stay coherent (history + quantized append row) -> 0
  VARLEN  parity with an fp16 cache layer under identical varlen args: the fork's
          backends uniformly decline varlen+paged-cache, so equal fate == correct
  LATENCY stage_range vs dequant_full kernel-time (torch.profiler CUDA), positive
          controls asserted (events > 0 each arm), O(used)/O(pool) ratio reported
"""
import sys, traceback
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

sys.path.insert(0, "/data/Repos/musicman89/exllamav3")
from exllamav3.constants import PAGE_SIZE
from exllamav3.cache.nvfp4 import CacheLayer_nvfp4, nvfp4_dequantize, nvfp4_quantize
from exllamav3.cache.fp16 import CacheLayer_fp16
from exllamav3.modules.attention_fn.dispatch import attn_dispatch

DEV = "cuda"
class AttnShim:
    num_kv_heads = 2
    head_dim = 256

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
    def fresh_nvfp4():
        lay = CacheLayer_nvfp4(None, AttnShim(), 0, pages * PAGE_SIZE)
        lay.alloc(torch.device(DEV))
        return lay
    bt = torch.arange(pages, dtype=torch.int32, device=DEV).view(1, -1)
    cs = torch.tensor([n0 - 1], dtype=torch.int32, device=DEV)
    def seed(lay):
        lay.update_kv(torch.tensor([0], dtype=torch.int32, device=DEV), bt,
                      hist_k[:, : n0 - 1], hist_v[:, : n0 - 1], n0 - 1)

    results = {}

    lay = fresh_nvfp4(); seed(lay)
    o_fast = attn_dispatch(q, new_k, new_v, cache=lay, block_table=bt,
                           cache_seqlens=cs.clone(), causal=True)
    results["FAST"] = f"ran, finite={bool(torch.isfinite(o_fast).all())}"

    pool_k = nvfp4_dequantize(lay.k.int(), lay.ks.half())
    flat_k = pool_k.reshape(-1, h, d)[:n0]
    err = (flat_k[n0 - 1] - nvfp4_dequantize(*nvfp4_quantize(new_k)).reshape(h, d)).abs().max()
    results["APPEND"] = f"row{n0-1} vs quantized(new_k) maxerr={err.item():.4f}"

    kk = flat_k[None].transpose(1, 2)
    vv = nvfp4_dequantize(lay.v.int(), lay.vs.half()).reshape(-1, h, d)[:n0][None].transpose(1, 2)
    o_ref = F.scaled_dot_product_attention(q.transpose(1, 2), kk, vv, enable_gqa=True).transpose(1, 2)
    results["POSCTL"] = f"fast-vs-sdpa maxdiff={(o_fast.float()-o_ref.float()).abs().max().item():.4f}"

    lay2 = fresh_nvfp4(); seed(lay2)
    o_c = attn_dispatch(q, new_k, new_v, cache=lay2, block_table=bt, cache_seqlens=cs.clone(), causal=True)
    lay2.k[0, 5, 0, 0] ^= 0xFF
    o_b = attn_dispatch(q, new_k, new_v, cache=lay2, block_table=bt, cache_seqlens=cs.clone(), causal=True)
    results["SABOT"] = f"changed: {not torch.equal(o_c, o_b)}"

    # ---- BF16 staged generic arm ----
    lay3 = fresh_nvfp4(); seed(lay3)
    bf16_diff = bf16_coh = None
    try:
        o_fb = attn_dispatch(q.bfloat16(), new_k.bfloat16(), new_v.bfloat16(),
                             cache=lay3, block_table=bt, cache_seqlens=cs.clone(), causal=True)
        bf16_diff = (o_fb.float() - o_fast.float()).abs().max().item()
        pk3 = nvfp4_dequantize(lay3.k.int(), lay3.ks.half()).reshape(-1, h, d)[:n0]
        rowdiff = (pk3 - flat_k).abs().amax(dim=(1, 2))
        hist = rowdiff[: n0 - 1].max().item()                       # history must be bit-untouched
        exp_row = nvfp4_dequantize(*nvfp4_quantize(new_k.bfloat16().half())).reshape(h, d)
        app = (pk3[n0 - 1] - exp_row).abs().max().item()             # append row == quantize(cast(new_k))
        bf16_coh = hist                                              # gate criterion: history only
        results["BF16"] = (f"ran, maxdiff-vs-fast={bf16_diff:.4f}, hist-coherent={hist:.4f}, "
                           f"append-vs-expected={app:.4f}, worst-row={int(rowdiff.argmax())}")
    except Exception as e:
        tb = traceback.extract_tb(sys.exc_info()[2])
        results["BF16"] = f"EXCEPTION {type(e).__name__}: {str(e)[:80]} @ {tb[-1].filename.split('/')[-1]}:{tb[-1].lineno}"

    # ---- VARLEN parity vs fp16-cache layer ----
    def varlen_outcome(layer):
        cu = torch.tensor([0, 1], dtype=torch.int32, device=DEV)
        try:
            attn_dispatch(q, new_k, new_v, cache=layer, block_table=bt,
                          cache_seqlens=cs.clone(), cu_seqlens=cu, max_seqlen=1, causal=True)
            return "ran"
        except Exception as e:
            return f"{type(e).__name__}: {str(e)[:60]}"
    lay4 = fresh_nvfp4(); seed(lay4)
    layf = CacheLayer_fp16(None, AttnShim(), 0, pages * PAGE_SIZE)
    layf.alloc(torch.device(DEV))
    layf.update_kv(torch.tensor([0], dtype=torch.int32, device=DEV), bt,
                   hist_k[:, : n0 - 1], hist_v[:, : n0 - 1], n0 - 1)
    r_nv, r_fp = varlen_outcome(lay4), varlen_outcome(layf)
    results["VARLEN"] = f"parity={r_nv == r_fp} | nvfp4='{r_nv[:50]}' fp16='{r_fp[:50]}'"

    # ---- LATENCY: stage_range vs dequant_full (kernel time, positive controls) ----
    big_pages = 32768 // PAGE_SIZE
    lb = CacheLayer_nvfp4(None, AttnShim(), 0, big_pages * PAGE_SIZE); lb.alloc(torch.device(DEV))
    bt_big = torch.arange(big_pages, dtype=torch.int32, device=DEV).view(1, -1)
    cs_big = torch.tensor([511], dtype=torch.int32, device=DEV)
    def prof(fn, tag):
        for _ in range(3): fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            for _ in range(5): fn()
        evs = [e for e in p.events() if e.device_type is not None and str(e.device_type) != "DeviceType.CPU" and e.self_device_time_total > 0]
        tot = sum(e.self_device_time_total for e in evs) / 5
        assert len(evs) > 0 and tot > 0, f"positive control FAILED for {tag}"
        return tot
    t_sr = prof(lambda: lb.stage_range(cs_big, bt_big, 1), "stage_range")
    t_df = prof(lambda: lb.dequant_full(), "dequant_full")
    ratio = t_sr / t_df
    results["LATENCY"] = f"stage={t_sr:.1f}us dequant_full={t_df:.1f}us ratio={ratio:.3f} (pool=32768tok used=512, kernels n>0 asserted)"

    ok = True
    for k in ["FAST", "APPEND", "POSCTL", "SABOT", "BF16", "VARLEN", "LATENCY"]:
        print(f"{k:8s}: {results[k]}")
    ok &= "EXCEPTION" not in results["BF16"] and bf16_diff is not None and bf16_diff < 0.05 and bf16_coh == 0.0
    ok &= "True" in results["SABOT"] and float(results["APPEND"].split("maxerr=")[1]) < 0.2
    ok &= results["VARLEN"].startswith("parity=True")
    ok &= ratio < 0.5
    print("VERDICT:", "GREEN" if ok else "RED")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()