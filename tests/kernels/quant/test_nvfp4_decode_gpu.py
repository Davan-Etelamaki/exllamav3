"""GPU conformance for decode_weights_gpu: bitwise vs CPU oracle + guards + sabotage + bench.

Runs bare: `python test_nvfp4_decode_gpu.py`. Requires CUDA + triton; if absent the runner
reports SKIP for GPU steps (positive control: it must state so, never silently pass zero).
Baseline comparison: fused kernel vs the module's own torch-vectorized GPU decode — the fair
same-semantics baseline, regime-labelled in the printed line.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("nvfp4_fmt", _ROOT / "exllamav3" / "modules" / "quant" / "nvfp4_fmt.py")
fmt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fmt)

import torch

_SHAPES = [(70, 4112), (128, 4096), (256, 12288)]       # ragged / square / prod-ish wide
_PROD = (4096, 12288)

GPU_OK = torch.cuda.is_available() and fmt._HAS_TRITON


def _case(rows, cols, seed):
    torch.manual_seed(seed)
    w = torch.randn(rows, cols) * 10 ** (2 * torch.randn(rows, 1))
    p, s, g = fmt.quantize_weights(w)
    return p, s, g


def t_bitwise_vs_cpu_oracle():
    n = 0
    for i, (r, c) in enumerate(_SHAPES):
        p, s, g = _case(r, c, seed=100 + i)
        ref = fmt.decode_weights(p, s, g, torch.float16)
        got = fmt.decode_weights_gpu(p.cuda(), s.cuda(), g, torch.float16).cpu()
        assert torch.equal(ref, got), (r, c, int((ref != got).sum()))
        n += ref.numel()
    assert n > 1_000_000, n                              # positive control: real coverage
    print(f"  (elements compared bitwise: {n})")


def t_dtypes_and_global_variants():
    p, s, g = _case(64, 256, seed=7)
    for dt in (torch.float16, torch.bfloat16, torch.float32):
        assert torch.equal(fmt.decode_weights(p, s, g, dt),
                           fmt.decode_weights_gpu(p.cuda(), s.cuda(), g, dt).cpu()), dt
    for gg in (1.0, 2.0 ** -30, 2.0 ** 17):
        assert torch.equal(fmt.decode_weights(p, s, gg, torch.float16),
                           fmt.decode_weights_gpu(p.cuda(), s.cuda(), gg, torch.float16).cpu()), gg


def t_gpu_guards_raise():
    p, s, g = _case(8, 32, seed=8)
    bad = s.clone().cuda(); bad[0, 0] = 0x7F
    try:
        fmt.decode_weights_gpu(p.cuda(), bad, g); assert False, "NaN scale accepted"
    except ValueError:
        pass
    bad2 = s.clone().cuda(); bad2[0, 0] |= 0x80
    try:
        fmt.decode_weights_gpu(p.cuda(), bad2, g); assert False, "negative scale accepted"
    except ValueError:
        pass


def t_sabotage_detected():
    p, s, g = _case(16, 256, seed=9)
    d0 = fmt.decode_weights_gpu(p.cuda(), s.cuda(), g, torch.float16)
    p_bad = p.clone().cuda(); p_bad[3, 9] ^= 0x11
    d1 = fmt.decode_weights_gpu(p_bad, s.cuda(), g, torch.float16)
    diff = (d1 != d0)
    assert diff.sum().item() == 2 and diff[3, 18:20].all(), diff.nonzero()
    s_bad = s.clone().cuda(); s_bad[0, 2] -= 1
    d2 = (fmt.decode_weights_gpu(p.cuda(), s_bad, g, torch.float16) != d0)
    assert d2.sum().item() == 16 and d2[0, 32:48].all(), d2.nonzero()


def _bench(fn, iters=20, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(iters):
        fn()
    ev1.record(); torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / iters * 1e3           # µs


def t_prod_shape_latency_fence():
    p, s, g = _case(*_PROD, seed=42)
    pc, sc = p.cuda(), s.cuda()
    t_k = _bench(lambda: fmt.decode_weights_gpu(pc, sc, g, torch.float16))
    t_base = _bench(lambda: fmt.decode_weights(pc, sc, g, torch.float16))
    assert torch.equal(fmt.decode_weights_gpu(pc, sc, g, torch.float16),
                       fmt.decode_weights(pc, sc, g, torch.float16))   # fence on SAME run
    mb = pc.numel() * 5.125 / 1e6                      # bytes/elem: packed 1 + sf 1/8 + f16 out 4 per packed byte
    assert t_k < t_base, (t_k, t_base)
    print(f"  [{_PROD[0]}x{_PROD[1]} device={torch.cuda.get_device_name(0)} sm{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}"
          f" regime=shared-GPU-serving] fused {t_k:.1f}us "
          f"({mb / (t_k * 1e-3):.0f} GB/s incl-out) vs torch-baseline "
          f"{t_base:.1f}us  speedup {t_base / t_k:.2f}x")


def t_arith_full_code_coverage_bitwise():
    # (a) every finite E4M3 scale code 0x00..0x7E as a full row, random payloads
    sf = torch.arange(0x7F, dtype=torch.uint8).unsqueeze(1).repeat(1, 256)      # 127 x 256
    packed = torch.randint(0, 256, (127, 256 * 8), dtype=torch.uint8)
    ref = fmt.decode_weights(packed, sf, 1.0, torch.float32)
    v1 = fmt.decode_weights_gpu(packed.cuda(), sf.cuda(), 1.0, torch.float32).cpu()
    v2 = fmt.decode_weights_gpu_arith(packed.cuda(), sf.cuda(), 1.0, torch.float32).cpu()
    assert torch.equal(ref, v1) and torch.equal(ref, v2)
    assert ref.abs().unique().numel() > 100, "scale sweep collapsed"          # positive control
    # (b) every packed byte 0x00..0xFF as a full row, scale pinned to 1.0 (0x38)
    packed2 = torch.arange(256, dtype=torch.uint8).unsqueeze(1).repeat(1, 32)  # 256 x 32 bytes
    sf2 = torch.full((256, 4), 0x38, dtype=torch.uint8)                    # 32 bytes = 4 blocks
    ref2 = fmt.decode_weights(packed2, sf2, 1.0, torch.float32)
    v1b = fmt.decode_weights_gpu(packed2.cuda(), sf2.cuda(), 1.0, torch.float32).cpu()
    v2b = fmt.decode_weights_gpu_arith(packed2.cuda(), sf2.cuda(), 1.0, torch.float32).cpu()
    assert torch.equal(ref2, v1b) and torch.equal(ref2, v2b)
    assert ref2.unique().numel() == 15, ref2.unique().numel()                  # 14 signed + one zero


def t_arith_v1_paired_bench():
    p, s, g = _case(*_PROD, seed=42)
    pc, sc = p.cuda(), s.cuda()
    assert torch.equal(fmt.decode_weights_gpu(pc, sc, g),
                       fmt.decode_weights_gpu_arith(pc, sc, g))               # fence on SAME run
    import statistics
    t1, t2 = [], []
    for _ in range(7):                                                        # interleaved pairs
        t1.append(_bench(lambda: fmt.decode_weights_gpu(pc, sc, g), iters=10, warmup=0))
        t2.append(_bench(lambda: fmt.decode_weights_gpu_arith(pc, sc, g), iters=10, warmup=0))
    m1, m2 = statistics.median(t1), statistics.median(t2)
    mb = pc.numel() * 5.125 / 1e6
    print(f"  [{_PROD[0]}x{_PROD[1]} device={torch.cuda.get_device_name(0)} sm120 "
          f"regime=shared-GPU-serving] gather-median {m1:.1f}us ({mb / (m1 * 1e-3):.0f} GB/s) "
          f"arith-median {m2:.1f}us ({mb / (m2 * 1e-3):.0f} GB/s)  arith/gather {m2 / m1:.2f}")


def t_prmt_full_code_coverage_bitwise():
    if not hasattr(fmt, "decode_weights_gpu_prmt"):
        raise AssertionError("decode_weights_gpu_prmt missing")       # test-first gate
    torch.manual_seed(7)
    # (a) every finite E4M3 scale code, random payloads — bitwise vs oracle
    sf = torch.arange(0x7F, dtype=torch.uint8).unsqueeze(1).repeat(1, 256)
    packed = torch.randint(0, 256, (127, 256 * 8), dtype=torch.uint8)
    ref = fmt.decode_weights(packed, sf, 1.0, torch.float32)
    v3 = fmt.decode_weights_gpu_prmt(packed.cuda(), sf.cuda(), 1.0, torch.float32).cpu()
    assert torch.equal(ref, v3)
    assert ref.abs().unique().numel() > 100, "scale sweep collapsed"  # positive control
    # (b) localization control: forcing one byte then flipping its lo-sign bit must
    # move EXACTLY ONE column, at the predicted index — proves the comparison bites
    packed3 = packed[:1, :64].clone()
    sf3 = torch.full((1, 8), 0x38, dtype=torch.uint8)                 # scale 1.0
    packed3[0, 5] = 0x01                                              # lo=+0.5, hi=+0.0
    base = fmt.decode_weights_gpu_prmt(packed3.cuda(), sf3.cuda(), 1.0, torch.float32).cpu()
    packed3[0, 5] ^= 0x08                                             # negate lo nibble
    sab = fmt.decode_weights_gpu_prmt(packed3.cuda(), sf3.cuda(), 1.0, torch.float32).cpu()
    assert base[0, 10] == 0.5 and sab[0, 10] == -0.5
    d = (base != sab).nonzero(as_tuple=True)
    assert d[0].tolist() == [0] and d[1].tolist() == [10], (d[0].tolist(), d[1].tolist())
    # (c) every packed byte 0x00..0xFF, scale pinned to 1.0 — bitwise vs oracle
    packed2 = torch.arange(256, dtype=torch.uint8).unsqueeze(1).repeat(1, 32)
    sf2 = torch.full((256, 4), 0x38, dtype=torch.uint8)
    ref2 = fmt.decode_weights(packed2, sf2, 1.0, torch.float32)
    v2 = fmt.decode_weights_gpu_prmt(packed2.cuda(), sf2.cuda(), 1.0, torch.float32).cpu()
    assert torch.equal(ref2, v2)
    assert ref2.unique().numel() == 15, ref2.unique().numel()


def t_prmt_triple_paired_bench():
    p, s, g = _case(*_PROD, seed=42)
    pc, sc = p.cuda(), s.cuda()
    a = fmt.decode_weights_gpu(pc, sc, g)                             # fence on SAME run
    b = fmt.decode_weights_gpu_arith(pc, sc, g)
    c = fmt.decode_weights_gpu_prmt(pc, sc, g)
    assert torch.equal(a, b) and torch.equal(a, c)
    import statistics
    t1, t2, t3 = [], [], []
    for _ in range(5):                                                # interleaved triples
        t1.append(_bench(lambda: fmt.decode_weights_gpu(pc, sc, g), iters=10, warmup=0))
        t2.append(_bench(lambda: fmt.decode_weights_gpu_arith(pc, sc, g), iters=10, warmup=0))
        t3.append(_bench(lambda: fmt.decode_weights_gpu_prmt(pc, sc, g), iters=10, warmup=0))
    m1, m2, m3 = statistics.median(t1), statistics.median(t2), statistics.median(t3)
    mb = pc.numel() * 5.125 / 1e6
    print(f"  [{_PROD[0]}x{_PROD[1]} device={torch.cuda.get_device_name(0)} sm120 "
          f"regime=shared-GPU-serving] gather {m1:.1f}us ({mb / (m1 * 1e-3):.0f} GB/s) "
          f"arith {m2:.1f}us ({mb / (m2 * 1e-3):.0f} GB/s) "
          f"prmt {m3:.1f}us ({mb / (m3 * 1e-3):.0f} GB/s)  prmt/gather {m3 / m1:.2f}  "
          f"gather-spread {max(t1) - min(t1):.1f}us")


def main() -> int:
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("t_") and callable(v)]
    assert tests
    failed = skipped = 0
    for name, fn in tests:
        if not GPU_OK:
            skipped += 1; print(f"SKIP {name} (no CUDA/triton — GPU steps NOT verified)")
            continue
        try:
            fn(); print(f"PASS {name}")
        except Exception as exc:                          # noqa: BLE001
            failed += 1; print(f"FAIL {name}: {exc!r}")
    status = "DONE-PASS" if failed == 0 and not skipped else ("DONE-SKIP-GPU" if not failed else "DONE-FAIL")
    print(f"GPU-CONFORMANCE steps={len(tests)} pass={len(tests) - failed - skipped} "
          f"fail={failed} skip={skipped} {status}")
    return 1 if failed else (2 if skipped else 0)


if __name__ == "__main__":
    sys.exit(main())