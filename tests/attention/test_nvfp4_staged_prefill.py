"""Fence for the NVFP4 staged-prefill lane (q_len >= _NVFP4_ONLINE_MAX_Q dispatches to the
bulk page-dequant kernel + fp16-cache backends instead of per-tile online dequant).
Law 12: unit parity (kernel vs torch nvfp4_dequantize, bit-exact) + trace attribution +
prod-shape latency gate (staged <= 1.5x the cache_mode 6,6 comparator). Law 7b: --sabotage
writes a definitely-different E4M3 scale value into a used page; the unit arm must print RED.

Run:  python tests/attention/test_nvfp4_staged_prefill.py [--sabotage]
Regime: shared GPU with live prod traffic; latency = profiler CUDA kernel-time medians
(n=5, interleaved), never wall-clock.
"""
import sys, statistics, torch
sys.path.insert(0, ".")
from torch.profiler import profile, ProfilerActivity
from exllamav3.constants import PAGE_SIZE
from exllamav3.cache import nvfp4 as nvmod
from exllamav3.cache.nvfp4 import CacheLayer_nvfp4, nvfp4_dequantize
from exllamav3.cache.quant import CacheLayer_quant
import exllamav3.modules.attention_fn.dispatch as dsp
from exllamav3.modules.attention_fn.dispatch import attn_dispatch

KH, D, QH = 2, 256, 24                      # prod attn shapes (24q / 2kv / head_dim 256)
DEV = "cuda"
CELLS = [(512, 4096), (512, 16384), (2048, 16384)]
GATE = 1.5
N = 5


class _Spec:
    num_kv_heads = KH
    head_dim = D


def build_cell(q_len, ctx):
    pages = (ctx + q_len + PAGE_SIZE) // PAGE_SIZE
    toks = pages * PAGE_SIZE
    bt = torch.arange(pages, dtype=torch.int32, device=DEV).view(1, -1)
    cs = torch.tensor([ctx], dtype=torch.int32, device=DEV)
    z = torch.zeros(1, dtype=torch.int32, device=DEV)
    hist_k = torch.randn(1, ctx, KH, D, dtype=torch.float16, device=DEV) * 0.5
    hist_v = torch.randn(1, ctx, KH, D, dtype=torch.float16, device=DEV) * 0.5
    q = torch.randn(1, q_len, QH, D, dtype=torch.float16, device=DEV)
    new_k = torch.randn(1, q_len, KH, D, dtype=torch.float16, device=DEV)
    new_v = torch.randn(1, q_len, KH, D, dtype=torch.float16, device=DEV)
    lay = CacheLayer_nvfp4(None, _Spec(), 0, toks); lay.alloc(torch.device(DEV))
    lay_q = CacheLayer_quant(None, _Spec(), 0, toks, 6, 6); lay_q.alloc(torch.device(DEV))
    lay.update_kv(z, bt, hist_k, hist_v, ctx)
    lay_q.update_kv(z, bt, hist_k, hist_v, ctx)
    return lay, lay_q, bt, cs, q, new_k, new_v


def prof_call(fn):
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        o = fn()
    evs = [e for e in p.events() if str(getattr(e, "device_type", "")) != "DeviceType.CPU" and e.self_device_time_total > 0]
    assert len(evs) > 0, "positive control FAILED: zero GPU kernels"
    names = {e.name for e in evs}
    return o, sum(e.self_device_time_total for e in evs), names


def main():
    torch.manual_seed(3)
    sabotage = "--sabotage" in sys.argv
    # ---- UNIT parity (q512_c4096 geometry): bulk kernel vs torch dequantize
    q_len, ctx = CELLS[0]
    lay, lay_q, bt, cs, q, new_k, new_v = build_cell(q_len, ctx)
    z = torch.zeros(1, dtype=torch.int32, device=DEV)
    ks_s, vs_s, _ = lay.stage_range(cs, bt, q_len)
    nb = max(1, (ctx + q_len + PAGE_SIZE - 1) // PAGE_SIZE)
    ks_ref = nvfp4_dequantize(lay.k[:nb].int(), lay.ks[:nb].half())
    vs_ref = nvfp4_dequantize(lay.v[:nb].int(), lay.vs[:nb].half())
    dk = (ks_s[:nb] - ks_ref).abs().max().item()
    dv = (vs_s[:nb] - vs_ref).abs().max().item()
    if sabotage:
        orig = lay.ks[1, 7, 0, 3].clone()
        lay.ks[1, 7, 0, 3] = torch.tensor(1.0 if orig.float().item() != 1.0 else 2.0, dtype=torch.float8_e4m3fn, device=DEV)
        ks_s, _, _ = lay.stage_range(cs, bt, q_len)
        d = (ks_s[:nb] - ks_ref).abs().max().item()
        lay.ks[1, 7, 0, 3] = orig
        if d > 0.0:
            print(f"SABOTAGE RED as required: scale write broke parity, maxdiff {d}")
            sys.exit(0)
        print("SABOTAGE FAILED TO LAND: fence cannot go RED — distrust all greens"); sys.exit(1)
    if dk != 0.0 or dv != 0.0:
        print(f"RED parity: maxdiff k={dk} v={dv}"); sys.exit(1)
    print(f"GREEN unit parity: maxdiff k={dk} v={dv} (bit-exact vs torch nvfp4_dequantize)")

    # ---- batch>1 parity with a shuffled block_table (per-seq pool prefix + phys mapping)
    bsz, q_len2 = 2, 64
    cs2 = torch.tensor([3000, 500], dtype=torch.int32, device=DEV)
    nbp = 12                       # uniform per-seq page capacity: covers full 3064-row writes
    nb2 = torch.div(cs2 + q_len2 + PAGE_SIZE - 1, PAGE_SIZE, rounding_mode='floor')
    bt2 = torch.randperm(bsz * nbp, dtype=torch.int32, device=DEV).view(bsz, nbp)
    lay2 = CacheLayer_nvfp4(None, _Spec(), 0, bsz * nbp * PAGE_SIZE); lay2.alloc(torch.device(DEV))
    hist_k = torch.randn(bsz, nbp * PAGE_SIZE - 8, KH, D, dtype=torch.float16, device=DEV) * 0.5
    hist_v = torch.randn(bsz, nbp * PAGE_SIZE - 8, KH, D, dtype=torch.float16, device=DEV) * 0.5
    lay2.update_kv(torch.zeros(bsz, dtype=torch.int32, device=DEV), bt2, hist_k, hist_v, nbp * PAGE_SIZE - 8)
    ks_p, vs_p, _ = lay2.stage_range(cs2, bt2, q_len2)
    total = int(nb2.sum())
    idx = torch.cat([bt2[0, :nb2[0]].long(), bt2[1, :nb2[1]].long()])
    ks_r = nvfp4_dequantize(lay2.k[idx].int(), lay2.ks[idx].half())
    vs_r = nvfp4_dequantize(lay2.v[idx].int(), lay2.vs[idx].half())
    dkb = (ks_p[:total] - ks_r).abs().max().item(); dvb = (vs_p[:total] - vs_r).abs().max().item()
    if dkb != 0.0 or dvb != 0.0:
        print(f"RED batch2 parity: maxdiff k={dkb} v={dvb}"); sys.exit(1)
    print(f"GREEN batch2+shuffled-bt parity: maxdiff k={dkb} v={dvb} (bsz=2, cs=[3000,500], permuted pages)")

    # ---- attribution + semantics on one staged call vs forced-online call
    def run_staged():  return attn_dispatch(q, new_k, new_v, cache=lay, block_table=bt, cache_seqlens=cs.clone(), causal=True)
    thr = dsp._NVFP4_ONLINE_MAX_Q
    dsp._NVFP4_ONLINE_MAX_Q = 10**9
    def run_online():  return attn_dispatch(q, new_k, new_v, cache=lay, block_table=bt, cache_seqlens=cs.clone(), causal=True)
    run_staged(); run_online(); dsp._NVFP4_ONLINE_MAX_Q = thr
    o_st, t, names = prof_call(run_staged)
    dsp._NVFP4_ONLINE_MAX_Q = 10**9
    o_on, _, names_on = prof_call(run_online)
    dsp._NVFP4_ONLINE_MAX_Q = thr
    assert any("nvfp4_bulk_dequant" in n for n in names), "RED attribution: staged lane lacks bulk kernel"
    assert not any("nvfp4_bulk_dequant" in n for n in names_on), "RED attribution: online lane ran bulk kernel"
    assert torch.isfinite(o_st).all() and torch.isfinite(o_on).all()
    cos = torch.nn.functional.cosine_similarity(o_st.float().flatten(), o_on.float().flatten(), dim=0).item()
    assert cos > 0.995, f"RED semantics: staged vs online cos={cos}"
    print(f"GREEN attribution + semantics: cos(staged,online)={cos:.5f}")

    # ---- prod-shape latency gate vs cache_mode 6,6
    for q_len, ctx in CELLS:
        lay, lay_q, bt, cs, q, new_k, new_v = build_cell(q_len, ctx)
        def rs():   return attn_dispatch(q, new_k, new_v, cache=lay,   block_table=bt, cache_seqlens=cs.clone(), causal=True)
        def rq():   return attn_dispatch(q, new_k, new_v, cache=lay_q, block_table=bt, cache_seqlens=cs.clone(), causal=True)
        rs(); rq(); torch.cuda.synchronize()
        ts, tq = [], []
        for _ in range(N):
            _, t, _ = prof_call(rs); ts.append(t)
            _, t, _ = prof_call(rq); tq.append(t)
        m_s, m_q = statistics.median(ts), statistics.median(tq)
        r = m_s / m_q
        print(f"q_len={q_len:5d} ctx={ctx:6d}: staged={m_s:9.1f}us q66={m_q:9.1f}us ratio={r:.3f} (gate {GATE})")
        assert r <= GATE, f"RED latency gate: {r:.3f} > {GATE}"
    print("FENCE GREEN")


main()