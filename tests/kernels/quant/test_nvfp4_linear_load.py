# Bare-run conformance suite for the NVFP4 load branch in modules/linear.py:
# detection geometry, bitwise load through Linear.load, orientation/padding/bias,
# and a byte-flip sabotage control. Run:
#   PYTHONPATH=<fork> python tests/kernels/quant/test_nvfp4_linear_load.py
import sys, types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

from exllamav3.modules.linear import Linear  # noqa: E402
from exllamav3.modules.quant import nvfp4_fmt as fmt  # noqa: E402
from exllamav3.modules.quant.nvfp4_tensor import Nvfp4PackedWeight  # noqa: E402

KEY = "model.layers.0.self_attn.q_proj"
OUT, IN_ = 600, 1040          # mis-aligned: exercises pad_to=128 -> (640, 1152)
PAD_OUT, PAD_IN = 640, 1152


class DictSTC:
    def __init__(self, t):
        self.t = t

    def has_tensor(self, n):
        return n in self.t

    def has_tensor_group(self, n, suffixes):
        base = n if isinstance(n, str) else n[0]
        ok = lambda s: ((any(base + "." + a in self.t for a in s)
                         if isinstance(s, (tuple, list)) else base + "." + s in self.t))
        return all(ok(s) for s in suffixes)

    def get_tensor(self, name, dev, optional=False, float2half=False, no_defer=False,
                   arena=True, transpose=False, pad_to=None, fidx=None):
        if name not in self.t:
            if optional:
                return None
            raise KeyError(name)
        x = self.t[name]
        if float2half and x.is_floating_point() and x.dtype != torch.float16:
            x = x.half()
        if transpose:
            x = x.T.contiguous()
        if pad_to is not None and x.dim() >= 1:
            shape = list(x.shape)
            changed = False
            for i, p in enumerate(pad_to if isinstance(pad_to, tuple) else (pad_to,)):
                if i < len(shape) and shape[i] < p:
                    shape[i] = p
                    changed = True
            if changed:
                pad = torch.zeros(*shape, dtype=x.dtype)
                idx = tuple(slice(0, s) for s in x.shape)
                pad[idx] = x
                x = pad
        return x


def raw_shard(seed=3):
    torch.manual_seed(seed)
    w = torch.randn(OUT, IN_, dtype=torch.bfloat16) * 0.05
    packed, scales, g = fmt.quantize_weights(w)
    bias = torch.randn(OUT, dtype=torch.float32) * 0.1
    stc = DictSTC({
        KEY + ".weight": packed,
        KEY + ".weight_scale": scales,
        KEY + ".weight_scale_2": torch.tensor(g, dtype=torch.float32),
        KEY + ".bias": bias,
    })
    return w, packed, scales, float(g), bias, stc


def mk_linear(stc):
    cfg = types.SimpleNamespace(stc=stc)
    lin = Linear(cfg, KEY, IN_, OUT)
    lin.load(torch.device("cpu"))
    return lin


def expected_weight(packed, scales, g):
    dec = Nvfp4PackedWeight(packed, scales, g).decode(torch.float16)  # (OUT, IN_)
    want = torch.zeros(PAD_IN, PAD_OUT, dtype=torch.float16)
    want[:IN_, :OUT] = dec.T
    return want, dec


def t_detect_positive():
    _, packed, scales, g, _, stc = raw_shard()
    triple = Linear.nvfp4_triple(stc, KEY, "cpu")
    assert triple is not None, "detector refused a legal NVFP4 triple"
    p, s, gg = triple
    assert p.dtype == torch.uint8 and p.shape == (OUT, IN_ // 2)
    assert s.dtype == torch.uint8 and s.shape == (OUT, IN_ // 16)
    assert gg == g
    # positive control: detector saw real buffers, not copies of zeros
    assert p.numel() > 100_000 and int(p.view(torch.uint8).sum()) != 0


def t_detect_negatives():
    _, packed, scales, g, bias, _ = raw_shard()
    cases = 0
    # missing global scale
    stc = DictSTC({KEY + ".weight": packed, KEY + ".weight_scale": scales})
    assert Linear.nvfp4_triple(stc, KEY, "cpu") is None; cases += 1
    # E8M0-style coarse grid (not the 16-block NVFP4 grid)
    stc = DictSTC({KEY + ".weight": packed,
                   KEY + ".weight_scale": torch.ones(OUT // 128, IN_ // 2 // 32, dtype=torch.uint8),
                   KEY + ".weight_scale_2": torch.tensor(1.0)})
    assert Linear.nvfp4_triple(stc, KEY, "cpu") is None; cases += 1
    # float weight (plain fp16 checkpoint)
    stc = DictSTC({KEY + ".weight": torch.randn(OUT, IN_),
                   KEY + ".weight_scale": scales,
                   KEY + ".weight_scale_2": torch.tensor(g)})
    assert Linear.nvfp4_triple(stc, KEY, "cpu") is None; cases += 1
    # 1-D scale (per-channel fp8 spelling)
    stc = DictSTC({KEY + ".weight": packed,
                   KEY + ".weight_scale": torch.ones(OUT),
                   KEY + ".weight_scale_2": torch.tensor(g)})
    assert Linear.nvfp4_triple(stc, KEY, "cpu") is None; cases += 1
    assert cases == 4


def t_load_bitwise():
    _, packed, scales, g, bias, stc = raw_shard()
    want, dec = expected_weight(packed, scales, g)
    lin = mk_linear(stc)
    assert lin.quant_type == "fp16"
    assert tuple(lin.inner.weight.shape) == (PAD_IN, PAD_OUT)
    assert torch.equal(lin.inner.weight, want), "loaded weight not bitwise-equal to container decode"
    # orientation control: the un-transposed decode must NOT match
    assert not torch.equal(lin.inner.weight[:OUT, :IN_], dec)


def t_load_bias_and_tail():
    _, packed, scales, g, bias, stc = raw_shard()
    lin = mk_linear(stc)
    assert tuple(lin.inner.weight.shape) == (PAD_IN, PAD_OUT)
    assert lin.inner.bias is not None
    assert torch.equal(lin.inner.bias[:OUT], bias.half())
    assert torch.equal(lin.inner.bias[OUT:], torch.zeros(PAD_OUT - OUT, dtype=torch.float16))
    assert lin.inner.weight[IN_:, :].eq(0).all() and lin.inner.weight[:, OUT:].eq(0).all()


def t_sabotage_byte_flip():
    _, packed, scales, g, bias, stc = raw_shard()
    want, _ = expected_weight(packed, scales, g)
    bad = packed.clone()
    bad.view(torch.uint8).flatten()[3] ^= 0x10
    assert not torch.equal(Nvfp4PackedWeight(bad, scales, g).decode(torch.float16),
                           Nvfp4PackedWeight(packed, scales, g).decode(torch.float16)), \
        "control failed: decode ignores payload bytes"
    stc_bad = DictSTC(dict(stc.t))
    stc_bad.t[KEY + ".weight"] = bad
    lin = mk_linear(stc_bad)
    assert not torch.equal(lin.inner.weight, want), "load silently tolerated a corrupted packed buffer"


SUITE = [t_detect_positive, t_detect_negatives, t_load_bitwise,
         t_load_bias_and_tail, t_sabotage_byte_flip]

if __name__ == "__main__":
    passed = failed = 0
    for fn in SUITE:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"LINEAR-NVFP4 steps={len(SUITE)} pass={passed} fail={failed}",
          "DONE-PASS" if failed == 0 else "DONE-FAIL")
    sys.exit(0 if failed == 0 else 1)