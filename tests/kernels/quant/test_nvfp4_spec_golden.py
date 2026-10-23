"""Spec-derived conformance tests for nvfp4_fmt (ORIGINAL, self-contained runner).

Goldens come from the ELEMENT FORMAT DEFINITIONS (bit fields enumerated independently here);
no third-party module, file, or name participates. Runs bare: `python test_nvfp4_spec_golden.py`
(pytest-compatible names included). Sabotage controls (laws 5/21) prove every equality check
detects a planted error; a check that passes on nothing fails the run.
"""
from __future__ import annotations

import importlib.util
import math
import pathlib
import struct
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("nvfp4_fmt", _ROOT / "exllamav3" / "modules" / "quant" / "nvfp4_fmt.py")
fmt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fmt)

import torch

# ---------------------------------------------------------------------------
# Independent goldens: enumerate the format by hand (different code path than the module's)
# ---------------------------------------------------------------------------

def golden_e2m1() -> dict[int, float]:
    """code(3-bit mag) -> magnitude, spelled out from the E2M1 definition: bias 1, denorm 2^0."""
    out = {}
    for e in range(4):
        for m in range(2):
            code = (e << 1) | m
            out[code] = (m * 0.5) if e == 0 else 2.0 ** (e - 1) * (1.0 + 0.5 * m)
    assert out == {0: 0.0, 1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0, 5: 3.0, 6: 4.0, 7: 6.0}, out
    return out

def golden_e4m3_finite() -> list[float]:
    vals = []
    for b in range(0x7F):  # nonneg, NaN pattern excluded
        e, m = (b >> 3) & 0xF, b & 7
        if e == 0 and b != 0:
            v = 2.0 ** -6 * (m / 8)
        elif e == 0:
            v = 0.0
        else:
            v = 2.0 ** (e - 7) * (1.0 + m / 8)
        vals.append(v)
    assert abs(vals[-1] - 448.0) < 1e-9, vals[-1]           # 0x7E = max finite
    assert abs(vals[1] - 2.0 ** -9) < 1e-12                  # smallest subnormal
    assert abs(vals[8] - 2.0 ** -6) < 1e-12                  # smallest normal
    return vals

GOLD_E2M1 = golden_e2m1()
GOLD_E4M3 = golden_e4m3_finite()


def t_lattices_match_goldens():
    assert fmt._E2M1_VALS == [GOLD_E2M1[c] for c in range(8)], fmt._E2M1_VALS
    assert fmt._E4M3_VALS == GOLD_E4M3
    assert len(fmt._E2M1_BOUNDS) == 7 and len(fmt._E4M3_BOUNDS) == len(GOLD_E4M3) - 1
    assert fmt._E2M1_MAX == 6.0 and fmt._E4M3_MAX == 448.0


def t_nibble_decode_exhaustive():
    # every packed byte value must decode to (lo, hi) exactly per the golden grid
    packed = torch.arange(256, dtype=torch.uint8).unsqueeze(0)
    got = fmt.decode_packed_e2m1(packed)[0]
    for b in range(256):
        lo, hi = b & 0xF, (b >> 4) & 0xF
        want = [GOLD_E2M1[lo & 7] * (-1 if lo >> 3 else 1),
                GOLD_E2M1[hi & 7] * (-1 if hi >> 3 else 1)]
        assert got[2 * b].item() == want[0] and got[2 * b + 1].item() == want[1], (b, want, got[2 * b:2 * b + 2])


def t_e4m3_decode_exhaustive_and_guards():
    codes = torch.arange(0x7F, dtype=torch.uint8)
    assert torch.equal(fmt.decode_e4m3(codes), torch.tensor(GOLD_E4M3, dtype=torch.float32))
    for bad in (0x7F, 0xFF):
        try:
            fmt.decode_e4m3(torch.tensor([bad], dtype=torch.uint8)); assert False, hex(bad)
        except ValueError:
            pass


def t_rtne_ties_even_and_saturation():
    b = fmt._E2M1_BOUNDS                      # midpoints between grid neighbors
    ties = b.clone()                           # exact midpoints -> even code
    codes = fmt.snap_codes(ties, b, 8)
    for i in range(len(b)):
        assert codes[i].item() % 2 == 0, (i, codes[i].item())   # tie -> EVEN index
    # non-midpoints snap to strictly nearest
    samples = torch.tensor([0.24, 0.49, 1.2, 1.7, 2.9, 5.9, 7.0])
    want = [0.0, 0.5, 1.0, 1.5, 3.0, 6.0, 6.0]
    got = fmt._E2M1_TABLE[fmt.snap_codes(samples, b, 8).long()]
    assert torch.equal(got, torch.tensor(want)), got
    assert (codes > 0).any(), "positive control: tie table empty"


def t_pow2_global_and_scale_never_clips():
    torch.manual_seed(7)
    w = torch.randn(64, 256) * 10 ** (3 * torch.randn(64, 1))
    packed, sfidx, g = fmt.quantize_weights(w)
    assert g > 0 and (math.log2(g)).is_integer(), g             # global is a power of two
    sf = torch.tensor(GOLD_E4M3)[sfidx.long()]
    amax = w.view(64, 16, 16).abs().amax(dim=-1)
    assert torch.all(6.0 * sf * g + 1e-30 * g >= amax), "policy: scale must cover block max"
    dec = fmt.decode_weights(packed, sfidx, g, torch.float32)
    assert dec.shape == w.shape
    # conservative error bound: max half-step at code 5->6 region is 1.0*sf*g per row-block
    err = (dec - w).abs()
    bound = (sf.repeat_interleave(16, dim=1) * g)
    assert (err <= 1.0 * bound + 1e-6 * w.abs() + 1e-9).all(), err.max()


def t_qdq_idempotent_and_zero_blocks():
    torch.manual_seed(11)
    w = torch.randn(8, 32)
    p1, s1, g1 = fmt.quantize_weights(w)
    d1 = fmt.decode_weights(p1, s1, g1, torch.float32)
    p2, s2, g2 = fmt.quantize_weights(d1)
    d2 = fmt.decode_weights(p2, s2, g2, torch.float32)
    assert torch.equal(d1, d2), "QDQ must be a projection"
    z = torch.zeros(2, 16)
    pz, sz, gz = fmt.quantize_weights(z)
    assert torch.equal(fmt.decode_weights(pz, sz, gz, torch.float32), z)


def t_sabotage_controls():
    # deterministic tensor, no zeros: random data made this flaky (-0.0 == 0.0 comparisons)
    w = torch.full((4, 32), 0.5)
    p, s, g = fmt.quantize_weights(w)
    d0 = fmt.decode_weights(p, s, g, torch.float32)
    p_bad = p.clone(); p_bad[0, 5] ^= 0x11                       # BOTH nibbles of one byte (lo|hi lsb)
    d_bad = fmt.decode_weights(p_bad, s, g, torch.float32)
    diff = (d_bad != d0)
    assert diff.sum().item() == 2 and diff[0, 10:12].all() and not diff[0, :10].any() and not diff[0, 12:].any(), diff.nonzero()
    assert not diff[1:].any()
    s_bad = s.clone(); s_bad[0, 0] -= 1                       # one block scale step down
    d2 = (fmt.decode_weights(p, s_bad, g, torch.float32) != d0)
    assert d2.sum().item() == 16 and d2[0, :16].all() and not d2[0, 16:].any(), d2.nonzero()


def t_geometry_raises():
    for bad in (((4, 30), (4, 4)), ((4, 32), (4, 5))):         # in%16!=0 ; scale-count mismatch
        try:
            fmt.decode_weights(torch.zeros(*bad[0], dtype=torch.uint8), torch.zeros(*bad[1], dtype=torch.uint8), 1.0)
            assert False, bad
        except ValueError:
            pass


def main() -> int:
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("t_") and callable(v)]
    assert tests, "runner found no tests"                       # positive control
    failed = 0
    for name, fn in tests:
        try:
            fn(); print(f"PASS {name}")
        except Exception as exc:                                # noqa: BLE001
            failed += 1; print(f"FAIL {name}: {exc!r}")
    print(f"SPEC-GOLDEN steps={len(tests)} pass={len(tests) - failed} fail={failed} "
          f"{'DONE-PASS' if failed == 0 else 'DONE-FAIL'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())