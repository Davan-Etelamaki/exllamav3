"""Seam conformance for the packed-resident weight container (nvfp4_tensor.Nvfp4PackedWeight).

Spec-derived goldens only; the container is a trust boundary: illegal E4M3 scale bytes
(NaN 0x7F, sign bit set) and geometry drift must raise at construction, decode must be
BITWISE identical to the CPU oracle on CPU and GPU, and bit flips in either buffer must
change the decoded output (positive control for "decode actually consumes both buffers").

Runs bare: `python test_nvfp4_seam.py`. GPU steps SKIP loudly when CUDA/triton is absent.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fmt = _load("nvfp4_fmt", "exllamav3/modules/quant/nvfp4_fmt.py")
ten = _load("nvfp4_tensor", "exllamav3/modules/quant/nvfp4_tensor.py")

import torch

_SHAPES = [(64, 256), (70, 4112), (128, 4096)]
GPU_OK = torch.cuda.is_available() and fmt._HAS_TRITON


def _case(rows, cols, seed):
    torch.manual_seed(seed)
    w = torch.randn(rows, cols) * 10 ** (2 * torch.randn(rows, 1))
    return ten.Nvfp4PackedWeight.from_float(w)


def t_roundtrip_storage():
    n = 0
    for i, (r, c) in enumerate(_SHAPES):
        tw = _case(r, c, seed=200 + i)
        p, s, g = fmt.quantize_weights(_raw(r, c, seed=200 + i))
        assert torch.equal(tw.packed, p) and torch.equal(tw.scales, s) and tw.global_scale == g
        assert tw.shape == (r, c)
        n += r * c
    assert n > 100_000, n                                   # positive control: real coverage


def _raw(rows, cols, seed):
    torch.manual_seed(seed)
    w = torch.randn(rows, cols) * 10 ** (2 * torch.randn(rows, 1))
    return w


def t_construction_guards():
    p, s, g = fmt.quantize_weights(_raw(8, 32, seed=1))
    cases = 0
    for bad_p, bad_s, bad_g in [
        (p.T.contiguous(), s, g),                            # packed rows/cols swapped
        (p, s.T.contiguous(), g),                            # scales transposed
        (p[1:], s, g),                                        # row mismatch
        (p, s[1:], g),                                        # scale-row mismatch
        (p.to(torch.int8), s, g),                             # wrong dtype packed
        (p, s.to(torch.float32), g),                          # wrong dtype scales
        (p, s, 0.0),                                          # zero global
        (p, s, -1.0),                                         # negative global
        (p, s, float("nan")),                                 # nan global
    ]:
        try:
            ten.Nvfp4PackedWeight(bad_p, bad_s, bad_g)
        except ValueError:
            cases += 1
        else:
            raise AssertionError(f"guard missed bad input {(bad_p.dtype, bad_s.dtype, bad_g)}")
    assert cases == 9, cases


def t_illegal_scale_bytes():
    p, s, g = fmt.quantize_weights(_raw(8, 32, seed=2))
    assert s.numel() >= 2
    for poison in (0x7F, 0xFF, 0x80):
        bad = s.clone(); bad[0, 0] = poison
        try:
            ten.Nvfp4PackedWeight(p, bad, g)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal scale byte 0x{poison:02X} accepted")
    ok = s.clone(); ok[0, 0] = 0x7E                           # largest legal magnitude
    tw = ten.Nvfp4PackedWeight(p, ok, g)                      # positive control: legal passes
    assert tw.scales[0, 0].item() == 0x7E


def t_decode_bitwise_cpu():
    numel = 0
    for i, (r, c) in enumerate(_SHAPES):
        tw = _case(r, c, seed=200 + i)
        for dt in (torch.float16, torch.bfloat16, torch.float32):
            ref = fmt.decode_weights(tw.packed, tw.scales, tw.global_scale, dt)
            assert torch.equal(tw.decode(dt), ref), (r, c, dt)
            numel += ref.numel()
    assert numel > 100_000, numel


def t_decode_bitwise_gpu():
    if not GPU_OK:
        return "SKIP"
    for i, (r, c) in enumerate(_SHAPES):
        tw = _case(r, c, seed=200 + i).to("cuda")
        ref = fmt.decode_weights(tw.packed.cpu(), tw.scales.cpu(), tw.global_scale, torch.float16)
        got = tw.decode(torch.float16).cpu()
        assert torch.equal(ref, got), (r, c, int((ref != got).sum()))
    return None


def t_decode_consumes_both_buffers():
    tw = _case(16, 256, seed=3)
    ref = tw.decode(torch.float32)
    tp = tw.packed.clone(); tp[0, 0] ^= 0x01                  # lowest element, +0.5 step
    assert not torch.equal(ten.Nvfp4PackedWeight(tp, tw.scales, tw.global_scale)
                           .decode(torch.float32), ref), "packed bit flip ignored"
    ts = tw.scales.clone(); ts[0, 0] = ts[0, 0] + 1           # first block scale up one lattice step
    assert ts[0, 0] != tw.scales[0, 0]                        # positive control: bump landed
    assert not torch.equal(ten.Nvfp4PackedWeight(tw.packed, ts, tw.global_scale)
                           .decode(torch.float32), ref), "scale byte change ignored"


def t_move_identity():
    tw = _case(32, 512, seed=4)
    rt = tw.to("cpu")
    assert torch.equal(rt.packed, tw.packed) and torch.equal(rt.scales, tw.scales)
    assert rt.global_scale == tw.global_scale
    if GPU_OK:
        gpu = tw.to("cuda")
        assert torch.equal(gpu.packed.cpu(), tw.packed) and torch.equal(gpu.scales.cpu(), tw.scales)


def t_footprint():
    tw = _case(128, 4096, seed=5)
    elems = 128 * 4096
    assert tw.nbytes == elems // 2 + elems // 16, tw.nbytes   # 4 bits + 1 scale bit per element
    assert tw.nbytes / (elems * 2) < 0.6                       # vs fp16 resident


STEPS = [t_roundtrip_storage, t_construction_guards, t_illegal_scale_bytes,
         t_decode_bitwise_cpu, t_decode_bitwise_gpu, t_decode_consumes_both_buffers,
         t_move_identity, t_footprint]


def main() -> int:
    npass = nfail = nskip = 0
    for step in STEPS:
        try:
            skipped = step()
        except Exception as exc:                              # noqa: BLE001 — report, then fail
            nfail += 1
            print(f"FAIL {step.__name__}: {type(exc).__name__}: {exc}")
            continue
        if skipped == "SKIP":
            nskip += 1
            print(f"SKIP {step.__name__}: no CUDA/triton")
        else:
            npass += 1
            print(f"PASS {step.__name__}")
    total = npass + nfail + nskip
    verdict = "DONE-PASS" if nfail == 0 else "DONE-FAIL"
    print(f"SEAM-CONFORMANCE steps={total} pass={npass} fail={nfail} skip={nskip} {verdict}")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())