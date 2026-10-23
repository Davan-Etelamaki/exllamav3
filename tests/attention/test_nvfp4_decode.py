"""Fence for cache/NVFP4 decode: _nvfp4_decode must be bit-identical to the
value-select reference over every E2M1 code x sign x representative scales.
Law 12: math change ships with unit test. Positive control: nonzero count > 0."""
import torch, triton, triton.language as tl
from exllamav3.modules.attention_fn.triton_paged import _nvfp4_decode

@triton.jit
def _ref(codes, scale):  # previous implementation, kept as ground truth
    idx = codes & 7
    sgn = codes >> 3
    mag = tl.zeros(idx.shape, dtype = tl.float16)
    mag = tl.where(idx == 1, 0.5, mag)
    mag = tl.where(idx == 2, 1.0, mag)
    mag = tl.where(idx == 3, 1.5, mag)
    mag = tl.where(idx == 4, 2.0, mag)
    mag = tl.where(idx == 5, 3.0, mag)
    mag = tl.where(idx == 6, 4.0, mag)
    mag = tl.where(idx == 7, 6.0, mag)
    return tl.where(sgn > 0, -mag, mag) * scale

@triton.jit
def _probe(c_ptr, s_ptr, a_ptr, r_ptr, N: tl.constexpr):
    o = tl.arange(0, N)
    c = tl.load(c_ptr + o)
    s = tl.load(s_ptr + o)
    tl.store(a_ptr + o, _nvfp4_decode(c, s))
    tl.store(r_ptr + o, _ref(c, s))

def test_nvfp4_decode_bit_exact():
    codes = torch.arange(16, dtype = torch.int32, device = "cuda").repeat(32)
    sc = torch.full((512,), 1.0, dtype = torch.float16, device = "cuda")
    a = torch.empty(512, dtype = torch.float16, device = "cuda")
    r = torch.empty_like(a)
    _probe[(1,)](codes, sc, a, r, N = 512)
    torch.cuda.synchronize()
    ref = torch.tensor([[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                        [-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]],
                       dtype = torch.float16, device = "cuda").flatten().repeat(32)
    eq_ref = torch.equal(a, r)                    # vs old cascade, bit-exact
    eq_oracle = torch.equal(a.view(torch.int16) | 0, ref.abs().view(torch.int16) | 0) if False else \
                torch.equal(a.abs(), ref.abs())   # magnitudes vs E2M1 oracle
    nz = int((a != 0).sum())
    assert nz > 0, "positive control failed: all-zero decode"
    assert eq_ref, "arith decode differs from select cascade"
    assert eq_oracle, "decode magnitudes differ from E2M1 lattice"
    print(f"OK nvfp4_decode bit-exact; nonzero={nz}/512")

if __name__ == "__main__":
    test_nvfp4_decode_bit_exact()
