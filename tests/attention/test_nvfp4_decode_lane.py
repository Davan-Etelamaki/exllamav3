"""Fence for the NVFP4-dedicated decode lane (fn_nvfp4_paged_decode + NVFP4 branch in
_paged_attn_decode_split_kernel). Law 12: unit oracle check + trace attribution + prod-shape
latency gate. Law 7b: --sabotage must print RED (flips one E4M3 scale byte; the unit arm
catches it). Latency comparator = the general splitdv path the lane replaced (q_len=17 declines
the q_len<=16 gate, so it measures the pre-fix cost on the same cache).

Run:  python tests/attention/test_nvfp4_decode_lane.py [--sabotage]
Regime: shared GPU with live prod traffic; latency = profiler CUDA kernel-time medians
(n=5, interleaved), never wall-clock.
"""
import sys, torch
sys.path.insert(0, ".")
from torch.profiler import profile, ProfilerActivity
from exllamav3.constants import PAGE_SIZE
from exllamav3.cache.nvfp4 import CacheLayer_nvfp4, nvfp4_quantize, nvfp4_dequantize
from exllamav3.modules.attention_fn.dispatch import attn_dispatch
from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton

KH, D, QH = 2, 256, 24                      # prod attn shapes (24q / 2kv / head_dim 256)
SCALE_IDX = ((32768 + PAGE_SIZE) // PAGE_SIZE // 2, PAGE_SIZE // 2, 1, 9)
DEV = "cuda"


class _Spec:
    num_kv_heads = KH
    head_dim = D


def build(used):
    pages = (used + PAGE_SIZE) // PAGE_SIZE
    toks = pages * PAGE_SIZE
    bt = torch.arange(pages, dtype=torch.int32, device=DEV).view(1, -1)
    torch.manual_seed(3)
    hist_k = torch.randn(1, used - 1, KH, D, dtype=torch.float16, device=DEV) * 0.5
    hist_v = torch.randn(1, used - 1, KH, D, dtype=torch.float16, device=DEV) * 0.5
    new_k = torch.randn(1, 1, KH, D, dtype=torch.float16, device=DEV)
    new_v = torch.randn_like(new_k)
    q = torch.randn(1, 1, QH, D, dtype=torch.float16, device=DEV)
    lay = CacheLayer_nvfp4(None, _Spec(), 0, toks)
    lay.alloc(torch.device(DEV))
    z = torch.zeros(1, dtype=torch.int32, device=DEV)
    lay.update_kv(z, bt, hist_k, hist_v, used - 1)
    return lay, bt, q, new_k, new_v


def run_decode(lay, bt, q, nk, nv, used):
    cs = torch.tensor([used - 1], dtype=torch.int32, device=DEV)
    return attn_dispatch(q.clone(), nk, nv, cache=lay, block_table=bt, cache_seqlens=cs, causal=True)


def run_general_q17(lay, bt, q, nk, nv, used):
    """Pre-fix cost comparator: q_len=17 declines the decode-lane gate (q_len<=16) and lands on
    _paged_attn_splitdv_kernel, the path the new lane replaces."""
    q17 = q.expand(1, 17, QH, D).contiguous()
    k17 = nk.expand(1, 17, KH, D).contiguous()
    v17 = nv.expand(1, 17, KH, D).contiguous()
    cs = torch.full((1,), used - 17, dtype=torch.int32, device=DEV)
    return paged_attn_triton(
        q=q17, k=k17, v=v17,
        k_cache=lay.k, v_cache=lay.v,
        k_scales=lay.ks, v_scales=lay.vs,
        block_table=bt, cache_seqlens=cs,
        causal=True, softmax_scale=1.0 / (D ** 0.5),
    )


def kernel_names_and_us(fn):
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn(); torch.cuda.synchronize()
    evs = [e for e in p.events()
           if str(getattr(e, "device_type", "")) != "DeviceType.CPU" and e.self_device_time_total > 0]
    assert evs, "profiler saw zero CUDA kernels"          # law 21 positive control
    return sorted({e.name.split(".")[0] for e in evs}), sum(e.self_device_time_total for e in evs) / 1e3


def main(sabotage: bool):
    ok = True
    # ---- UNIT: decode-lane output vs fp32 oracle built from PRE-RUN cache snapshots, so the
    # oracle encodes the format spec, not whatever bytes the kernel actually saw (--sabotage RED)
    used = 1920
    lay, bt, q, nk, nv = build(used)
    ref_k, ref_ks = lay.k.clone(), lay.ks.clone()
    ref_v, ref_vs = lay.v.clone(), lay.vs.clone()
    if sabotage:
        lay.ks.view(torch.uint8)[:] ^= 0x55       # mangle every K scale: oracle must diverge
    out = run_decode(lay, bt, q, nk, nv, used)
    kk = nvfp4_dequantize(ref_k, ref_ks.float())
    vv = nvfp4_dequantize(ref_v, ref_vs.float())
    kf = kk.reshape(-1, KH, D)[:used].clone()
    vf = vv.reshape(-1, KH, D)[:used].clone()
    kf[used - 1] = nvfp4_dequantize(*nvfp4_quantize(nk.half()))[0][0]
    vf[used - 1] = nvfp4_dequantize(*nvfp4_quantize(nv.half()))[0][0]
    rep = QH // KH
    qf = q.float().permute(0, 2, 1, 3)
    kfr = kf.permute(1, 0, 2).unsqueeze(0).float().repeat_interleave(rep, 1)
    vfr = vf.permute(1, 0, 2).unsqueeze(0).float().repeat_interleave(rep, 1)
    att = torch.softmax((qf @ kfr.transpose(-1, -2)) * (1.0 / (D ** 0.5)), dim=-1)
    ref = (att @ vfr).permute(0, 2, 1, 3)
    md = (out.float() - ref).abs().max().item()
    unit_ok = md < 3e-3
    ok &= unit_ok
    print(f"UNIT    : decode-lane vs fp32-oracle maxdiff={md:.6f} (tol 0.003) -> {'ok' if unit_ok else 'FAIL'}")
    # ---- ATTRIBUTION: decode kernels present, general splitdv kernel absent
    names, _ = kernel_names_and_us(lambda: run_decode(lay, bt, q, nk, nv, used))
    attr_ok = any("decode_split" in n for n in names) and not any("splitdv" in n for n in names)
    ok &= attr_ok
    print(f"ATTRIB  : kernels={names} -> {'ok' if attr_ok else 'FAIL'}")
    # ---- DET + SABOT: determinism control; scale-byte flip bites, restore zeroes
    det = (run_decode(lay, bt, q, nk, nv, used) - out).abs().max().item()
    ok &= det == 0.0
    print(f"DET     : rerun maxdiff={det:.6f}")
    snap = lay.ks.view(torch.uint8).clone()
    lay.ks.view(torch.uint8)[used // PAGE_SIZE // 2, PAGE_SIZE // 2, 1, 9] ^= 0x0F
    sb = (run_decode(lay, bt, q, nk, nv, used) - out).abs().max().item()
    lay.ks.view(torch.uint8).copy_(snap)
    rs = (run_decode(lay, bt, q, nk, nv, used) - out).abs().max().item()
    sab_ok = sb > 0.0 and rs == 0.0
    ok &= sab_ok
    print(f"SABOT   : flip delta={sb:.6f} restored delta={rs:.6f} -> {'ok' if sab_ok else 'FAIL'}")
    # ---- LATENCY: prod-shape decode lane must beat the legacy splitdv path it replaces
    used = 32768
    lay2, bt2, q2, nk2, nv2 = build(used)
    run_decode(lay2, bt2, q2, nk2, nv2, used); run_general_q17(lay2, bt2, q2, nk2, nv2, used)
    meds = {}
    for _ in range(5):
        for tag, call in (("decode", lambda: run_decode(lay2, bt2, q2, nk2, nv2, used)),
                          ("general", lambda: run_general_q17(lay2, bt2, q2, nk2, nv2, used))):
            _, us = kernel_names_and_us(call)
            meds.setdefault(tag, []).append(us)
    dec, gen = sorted(meds["decode"])[2], sorted(meds["general"])[2]
    lat_ok = dec < 0.5 * gen
    ok &= lat_ok
    print(f"LATENCY : median(us) decode={dec*1e3:.1f} general={gen*1e3:.1f} ratio={dec/gen:.3f} (gate <0.5) "
          f"-> {'ok' if lat_ok else 'FAIL'} | shared-GPU regime")
    print("VERDICT:", "GREEN" if ok else "RED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main("--sabotage" in sys.argv))