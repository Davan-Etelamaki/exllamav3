# Fence for triton_paged._fast_dispatch (cut 6): the JIT cache-hit fast path used by the
# NVFP4 paged kernels. Checks it (a) produces identical results to ordinary dispatch,
# (b) actually reaches the CompiledKernel.run path (not the fallback), and (c) keeps
# specialization honest when a runtime scalar changes value (==1 vs >1).
import torch
import triton
import triton.language as tl

from exllamav3.modules.attention_fn import triton_paged as tp


@triton.jit
def _toy(a_ptr, b_ptr, n, scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(b_ptr + offs, tl.load(a_ptr + offs, mask=m) * scale, mask=m)


def _cuda():
    return torch.cuda.is_available()


def test_fast_dispatch_matches_and_hits():
    if not _cuda():
        import pytest
        pytest.skip("needs CUDA")
    # identity: this fence must run against the dev tree, not an installed wheel
    import os
    assert os.path.realpath(tp.__file__).startswith("/tmp/exl3pkg/exllamav3"), \
        f"fence running against {tp.__file__}, not the dev tree"
    a = torch.randn(4096, device="cuda")
    b = torch.zeros_like(a)
    grid = (16,)
    args = (a, b, 4096, 2.0)
    consts = {"BLOCK": 256}
    # first call compiles through the ordinary path and returns the kernel
    k1 = tp._fast_dispatch(_toy, grid, args, consts)
    torch.cuda.synchronize()
    from triton.compiler import CompiledKernel
    assert isinstance(k1, CompiledKernel), "first call must return the compiled kernel"
    ref = a * 2.0
    assert torch.allclose(b, ref), "fast dispatch produced wrong values"

    # fast-path proof: the second call must NOT go through JITFunction.run at all
    # (CompiledKernel.run is a read-only property, so we spy on the slow door instead)
    from triton.runtime.jit import JITFunction
    called = {"n": 0}
    orig_jit_run = JITFunction.run

    def spy(self, *a_, **kw_):
        called["n"] += 1
        return orig_jit_run(self, *a_, **kw_)

    JITFunction.run = spy
    try:
        b.zero_()
        tp._fast_dispatch(_toy, grid, args, consts)
        torch.cuda.synchronize()
        assert called["n"] == 0, "second call fell back to JITFunction.run (fast path dead?)"
        assert torch.allclose(b, ref)
        # positive control for the spy: ordinary dispatch DOES trip it
        _toy[grid](*args, **consts)
        assert called["n"] == 1, "spy never observes ordinary dispatch (test blind)"
    finally:
        JITFunction.run = orig_jit_run

    # positive control for the fallback switch: forcing _FAST_DISPATCH off still works
    saved = tp._FAST_DISPATCH
    try:
        tp._FAST_DISPATCH = False
        b.zero_()
        tp._fast_dispatch(_toy, grid, args, consts)
        torch.cuda.synchronize()
        assert torch.allclose(b, ref), "fallback path broken"
    finally:
        tp._FAST_DISPATCH = saved


def test_fast_dispatch_specialization_change():
    if not _cuda():
        import pytest
        pytest.skip("needs CUDA")
    a = torch.randn(4096, device="cuda")
    b = torch.zeros_like(a)
    grid = (16,)
    # n=4096 (divisible-by-16 specialization), then n=1 (==1 specialization): the binder
    # must key them apart or the second launch reads a stale specialization
    tp._fast_dispatch(_toy, grid, (a, b, 4096, 3.0), {"BLOCK": 256})
    torch.cuda.synchronize()
    assert torch.allclose(b[:4096], a * 3.0)
    b.zero_()
    tp._fast_dispatch(_toy, (1,), (a, b, 1, 5.0), {"BLOCK": 256})
    torch.cuda.synchronize()
    assert b[0].item() == (a[0] * 5.0).item(), "specialization drift on n change"
    assert b[1].item() == 0.0