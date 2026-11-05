"""
Half-integer bitrates (1.5 / 2.5 / 3.5 bpw, mul1) in the fused MoE prefill kernel (ext.exl3_moe), whose instances
are compiled per rate: single 16-row launches, per-range 16 / 32 / 64-row tiles and the mixed-rate runtime-switch
instance, against the dense reconstruct path (testlib.moe.expert_mlp_ref). The CPU expert kernels at these rates
are covered in tests/moe_cpu/test_half_rates_cpu.py.
"""
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib.exl3 import generator, rand_experts, rand_scale, rand_trellis
from testlib.moe import contiguous_layout, expert_mlp_ref, expert_ptr_tables, experts_to, moe_buffers

HALF_RATES = (1.5, 2.5, 3.5)
CAP = 128


@pytest.mark.parametrize("K", HALF_RATES)
@pytest.mark.parametrize("dims", [(256, 128), (512, 256)])   # N = 128 and N = 256 tile shapes
def test_fused_prefill_instances(device, K, dims):
    """The compiled half-rate instances of exl3_moe, 16 / 32 / 64-row tiles, against the reconstruct path"""
    H, I = dims
    counts = [1, 5, 16, 17, 32, 33, 48, 64, 100]
    E = len(counts)
    gen = torch.Generator().manual_seed(11)
    exd = experts_to(rand_experts(E, H, I, K, gen), device)
    tabs = expert_ptr_tables(exd, device)
    A = sum(counts)
    y = (torch.randn((A, H), generator = gen) * 0.05).half().to(device)
    weight_sorted = (torch.rand((A,), generator = gen) * 0.5 + 0.5).half().to(device)
    token_sorted, expert_count, starts = contiguous_layout(counts, device)
    bufs = moe_buffers(H, I, CAP, device)

    def ref_of(ex_, K_down):
        ref = torch.empty((A, H), dtype = torch.float, device = device)
        for e in range(E):
            a = int(starts[e])
            rows = slice(a, a + counts[e])
            ref[rows] = expert_mlp_ref(ex_, e, y[rows], K, K_down) * weight_sorted[rows].float().unsqueeze(1)
        return ref

    def launch(tabs_, K_down, lo, hi, m_tile, out):
        # None, None, None: the per-expert bitrate tables added when the kernel learned mixed-K
        # packs; this test's bundle is uniform per projection
        ext.exl3_moe(y, out, expert_count, token_sorted, weight_sorted, *bufs, 0, K, K, K_down,
                     None, None, None,
                     *tabs_["g"], *tabs_["u"], *tabs_["d"], False, True, False, True, False, True, 0.0, -1,
                     None, None, lo, hi, m_tile)

    ref = ref_of(exd, K)
    scale = ref.abs().max().item()
    single = torch.zeros((A, H), dtype = torch.float, device = device)
    launch(tabs, K, 1, CAP, 16, single)
    tiered = torch.zeros((A, H), dtype = torch.float, device = device)
    launch(tabs, K, 33, CAP, 64, tiered)
    launch(tabs, K, 17, 32, 32, tiered)
    launch(tabs, K, 1, 16, 16, tiered)
    torch.cuda.synchronize()
    assert (single - ref).abs().max().item() <= 3e-3 * scale, "single 16-row launch"
    assert (tiered - ref).abs().max().item() <= 3e-3 * scale, "per-range row tiles"

    # A mixed-rate launch takes the runtime-switch instance: same arithmetic, different kernel
    ex2 = rand_experts(E, H, I, K, gen)
    ex2["d"] = [(rand_trellis(I, H, 4, gen), rand_scale(I, gen), rand_scale(H, gen)) for _ in range(E)]
    exd2 = experts_to(ex2, device)
    mixed = torch.zeros((A, H), dtype = torch.float, device = device)
    launch(expert_ptr_tables(exd2, device), 4, 1, CAP, 16, mixed)
    torch.cuda.synchronize()
    ref2 = ref_of(exd2, 4)
    assert (mixed - ref2).abs().max().item() <= 3e-3 * ref2.abs().max().item(), "mixed-rate launch"


def test_per_expert_mixed_rates_ks_tables(device):
    """Per-expert bitrates through the Ks_gate/up/down tables: the kernel's runtime switch must
    resolve each expert's own rate. Mirrors SAGE-style packs, which grade each expert
    separately (the scenario MultiLinear.mixed_K bundles carry); the uniform-K scalar args
    only describe the codebook once tables are present."""
    H, I = 256, 256
    counts = [1, 3, 8, 16, 17, 33, 48, 64]
    E = len(counts)
    rates_gu = [2, 2.5, 3, 3.5] * (E // 4)
    rates_d = [4, 2.5, 2, 3.5] * (E // 4)
    gen = generator(41)
    ex = {
        "g": [(rand_trellis(H, I, rates_gu[e], gen), rand_scale(H, gen), rand_scale(I, gen)) for e in range(E)],
        "u": [(rand_trellis(H, I, rates_gu[e], gen), rand_scale(H, gen), rand_scale(I, gen)) for e in range(E)],
        "d": [(rand_trellis(I, H, rates_d[e], gen), rand_scale(I, gen), rand_scale(H, gen)) for e in range(E)],
    }
    exd = experts_to(ex, device)
    tabs = expert_ptr_tables(exd, device)
    A = sum(counts)
    y = (torch.randn((A, H), generator = gen) * 0.05).half().to(device)
    weight_sorted = (torch.rand((A,), generator = gen) * 0.5 + 0.5).half().to(device)
    token_sorted, expert_count, _ = contiguous_layout(counts, device)
    bufs = moe_buffers(H, I, CAP, device)
    # Half-bit units, same as k2_from_K in bits_k.cuh: 2 * floor(rate) + half_flag
    k2 = lambda rates: torch.tensor([2 * int(r) + (1 if float(r) - int(r) == 0.5 else 0) for r in rates],
                                    dtype = torch.int32, device = device)

    out = torch.zeros((A, H), dtype = torch.float, device = device)
    ext.exl3_moe(y, out, expert_count, token_sorted, weight_sorted, *bufs, 0, 2, 2, 2,
                 k2(rates_gu), k2(rates_gu), k2(rates_d),
                 *tabs["g"], *tabs["u"], *tabs["d"], False, True, False, True, False, True, 0.0, E,
                 None, None, 1, CAP, 16)
    torch.cuda.synchronize()

    ref = torch.empty((A, H), dtype = torch.float, device = device)
    starts = torch.cat([torch.zeros(1, dtype = torch.long, device = device),
                            expert_count[:-1].cumsum(0)])
    for e in range(E):
        rows = slice(int(starts[e]), int(starts[e]) + counts[e])
        ref[rows] = expert_mlp_ref(exd, e, y[rows], rates_gu[e], rates_d[e]) \
            * weight_sorted[rows].float().unsqueeze(1)
    assert (out - ref).abs().max().item() <= 3e-3 * ref.abs().max().item(), "per-expert Ks tables"
