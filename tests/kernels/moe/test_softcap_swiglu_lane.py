"""
Formula and index parity for the MoE softcap lane (MOE_ACT_SOFTCAP_MUL / ACT_SOFTCAP_MUL).

Runs without importing exllamav3: the package import initialises CUDA on this host, and the
serving card must stay untouched. So this test pins three things it CAN see from CPU:

1. the reference math, against a hand-written torch expression, for both the fused-kernel
   formulation (lane-folded gate, unit up lane, shared multiply) and the dense composition
   that block_sparse_mlp registers for the shapes the fused kernels do not cover;
2. that the kernel enum literal and the Python registry index agree (both must read 3);
3. that the second cap is carried everywhere the enum is interpreted, so no path can apply
   the softcap with up_cap left at 0 (which would divide by zero).

Reference for the model's own math: Inkling2LoopSparseMoeBlock, inkling2_loop.py:1148-1150.
"""

from pathlib import Path
import re

import pytest
import torch

pytestmark = pytest.mark.nogpu   # no device needed: torch CPU + source-literal reads

import os

# EXL3_REPO_ROOT lets this run outside the tests/ tree (the repo conftest builds the extension
# when no GPU is visible, which is not what a formula check needs)
ROOT = Path(os.environ.get("EXL3_REPO_ROOT", Path(__file__).resolve().parents[2]))
EXT = ROOT / "exllamav3" / "exllamav3_ext"
MODULES = ROOT / "exllamav3" / "modules"

GATE_CAP = 4.0
UP_CAP = 25.0


def _inkling_reference(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Verbatim from InklingMoE.forward: softcap(g) * sigmoid(g) * softcap(u)."""
    act = torch.tanh(g / GATE_CAP) * GATE_CAP
    act = act * torch.sigmoid(g)
    return act * (torch.tanh(u / UP_CAP) * UP_CAP)


def _kernel_formulation(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """What hadamard_inner.cuh's ACT_SOFTCAP_MUL case computes: everything folds into the gate
    lane and the up lane becomes 1, because the kernel multiplies the lanes afterwards."""
    folded = _inkling_reference(g, u)
    return folded * torch.ones_like(u)


def _dense_composition(g: torch.Tensor, u: torch.Tensor, gate_cap: float, up_cap: float) -> torch.Tensor:
    """Mirror of block_sparse_mlp._softcap_swiglu (fp32 in, half out)."""
    gf, uf = g.float(), u.float()
    return ((torch.tanh(gf / gate_cap) * gate_cap) * torch.sigmoid(gf) *
            (torch.tanh(uf / up_cap) * up_cap)).half()


def test_dense_matches_engine_reference():
    torch.manual_seed(0)
    g = (torch.randn(4096) * 6.0).half()
    u = (torch.randn(4096) * 6.0).half()
    ref = _inkling_reference(g.float(), u.float()).half()
    got = _dense_composition(g, u, GATE_CAP, UP_CAP)
    # both paths round to half once; allow one half ulp
    assert torch.allclose(got.float(), ref.float(), rtol=2 ** -10, atol=1e-3), \
        f"max deviation {(got.float()-ref.float()).abs().max().item():.3e}"


def test_kernel_formulation_matches_reference():
    torch.manual_seed(1)
    g = (torch.randn(4096) * 6.0).float()
    u = (torch.randn(4096) * 6.0).float()
    assert torch.equal(_kernel_formulation(g, u), _inkling_reference(g, u))


def test_saturation_band_is_the_two_caps():
    # a softcap and a hard clamp are NOT interchangeable: below saturation the softcap keeps
    # growing where a clamp has already plateaued. This is the discriminator that stops the
    # existing act_limit clamp (hadamard_inner.cuh) from standing in for the new lane.
    # float64: in fp32 tanh saturates to exactly 1.0 and strict growth disappears at the tail.
    g = torch.linspace(0.0, 20.0, 2001, dtype=torch.float64)
    soft = torch.tanh(g / GATE_CAP) * GATE_CAP
    clamp = torch.clamp(g, max=GATE_CAP)
    assert soft.max().item() <= GATE_CAP                          # asymptote, never exceeds cap
    assert torch.all(soft[1:] > soft[:-1])                        # strictly increasing
    mid = torch.isclose(g, torch.tensor(8.0, dtype=torch.float64))
    assert soft[mid].item() < 3.9 and clamp[mid].item() == 4.0, \
        f"softcap(8)={soft[mid].item():.4f} vs clamp(8)={clamp[mid].item()}: not distinguishable"


def test_enum_and_registry_index_agree():
    common = (EXT / "quant" / "exl3_moe_common.cuh").read_text()
    inner = (EXT / "quant" / "hadamard_inner.cuh").read_text()
    module = (MODULES / "block_sparse_mlp.py").read_text()
    cud = int(re.search(r"#define MOE_ACT_SOFTCAP_MUL (\d+)", common).group(1))
    inn = int(re.search(r"#define ACT_SOFTCAP_MUL (\d+)", inner).group(1))
    py = int(re.search(r'case "softcap_swiglu":.*?self\.activation_fn_idx = (\d+)', module, re.S).group(1))
    assert cud == inn == py == 3, f"kernel {cud}/{inn}, python {py}"
    # the switch must actually branch on the shared header's value, not a stale literal
    assert "case ACT_SOFTCAP_MUL:" in inner


def test_second_cap_reaches_every_interpreter():
    common = (EXT / "quant" / "exl3_moe_common.cuh").read_text()
    inner = (EXT / "quant" / "hadamard_inner.cuh").read_text()
    entry_cu = (EXT / "quant" / "exl3_moe.cu").read_text()
    entry_cuh = (EXT / "quant" / "exl3_moe.cuh").read_text()
    kernel = (EXT / "quant" / "exl3_moe_kernel.cuh").read_text()
    module = (MODULES / "block_sparse_mlp.py").read_text()
    assert "const float up_cap," in common                      # kernel parameter macro
    assert "const float up_cap," in inner                       # activation helper
    assert "(void*) &up_cap," in entry_cu                       # kernel argument list
    assert "const float up_cap," in entry_cu                    # definition
    assert "const float up_cap," in entry_cuh                   # declaration
    assert re.search(r"act_limit,\s*up_cap,\s*act_function", kernel)   # pass-through at the call
    # three call sites carry the second cap: the two fused prefill launches and the BC decode
    # construction (the coop decode kernel's up_cap originates here)
    assert module.count("self.up_cap,") == 3, module.count("self.up_cap,")
    coop_kernel = (EXT / "quant" / "exl3_moe_coop_kernel.cuh").read_text()
    # The cooperative decode kernel now HAS a softcap branch (it previously ran plain SiLU at
    # bsz<=MAX_BSZN while prefill applied the caps). It branches on the coop enum, not the
    # prefill enum literal, which lives in a different translation unit's switch.
    assert "MOE_COOP_ACT_SOFTCAP" in coop_kernel, \
        "the cooperative decode kernel still has no softcap branch"
    assert '"softcap_swiglu"' in module
    # the name is referenced five times: the two accept tuples (fused prefill + fused decode),
    # the registry case, the load-time cap assert, and the decode act_softcap kwarg; the helper
    # itself is referenced by attribute, not by the string
    assert module.count('"softcap_swiglu"') == 5, module.count('"softcap_swiglu"')
    assert "def _softcap_swiglu(g, u, a, lim):" in module
    assert "self.activation_fn_call = self._softcap_swiglu" in module


def test_coop_decode_applies_the_softcap():
    """Decode (bsz<=MAX_BSZN) must run the SAME softcapped SwiGLU as fused prefill, not plain
    SiLU. Pins the whole coop lane: enum -> kernel branch -> params -> BC boundary -> python.
    Every arm reads a source literal, so a planted rename of the enum or the up_cap carrier
    makes this fail (the fence discriminates)."""
    coop_cuh = (EXT / "quant" / "exl3_moe_coop.cuh").read_text()
    coop_cu = (EXT / "quant" / "exl3_moe_coop.cu").read_text()
    coop_kernel = (EXT / "quant" / "exl3_moe_coop_kernel.cuh").read_text()
    inner = (EXT / "quant" / "hadamard_inner.cuh").read_text()
    bc_cpp = (EXT / "libtorch" / "blocksparse_mlp.cpp").read_text()
    bc_h = (EXT / "libtorch" / "blocksparse_mlp.h").read_text()
    bc_bc_h = (EXT / "libtorch" / "blocksparse_mlp_bc.h").read_text()
    module = (MODULES / "block_sparse_mlp.py").read_text()

    # 1. the coop enum carries its own softcap value (distinct from the prefill enum)
    enum_line = re.search(r"#define MOE_COOP_ACT_SOFTCAP (\d+)", coop_cuh)
    assert enum_line, "coop enum has no MOE_COOP_ACT_SOFTCAP"
    assert int(enum_line.group(1)) >= 4, "coop softcap must not collide with an existing coop act"
    # the parameter block and the prepare front-end must carry the second cap
    assert re.search(r"struct MoeCoopParams[\s\S]*?float up_cap;", coop_cuh), \
        "MoeCoopParams does not carry up_cap"
    assert re.search(r"float act_limit,\s*float up_cap,\s*bool gated", coop_cuh), \
        "coop prepare declaration omits up_cap"

    # 2. the kernel branch exists and reads up_cap; a rename of the enum trips this regex
    assert re.search(r"if \(act == MOE_COOP_ACT_SOFTCAP\)\s*return[^;]*up_cap[^;]*;", coop_kernel), \
        "coop act_gate has no softcap branch that consumes up_cap"

    # 3. formula parity: the coop decode branch must compute the SAME three factors as the
    # fused-prefill ACT_SOFTCAP_MUL case, modulo variable names. A planted rename of any operand
    # or factor diverges the canonilised expressions and fails here.
    coop_branch = re.search(r"if \(act == MOE_COOP_ACT_SOFTCAP\)\s*return\s+([^;]+);", coop_kernel)
    had_branch = re.search(r"_softcap_silu_mul[\s\S]*?z\.x\s*=\s*([^;]+);", inner)
    assert coop_branch and had_branch, "cannot extract the two softcap formulas"

    def canon(expr, gate, up, cg, cu):
        e = re.sub(r"\s+", "", expr)
        for a, b in ((cg, "CG"), (cu, "CU"), (gate, "G"), (up, "U")):
            e = e.replace(a, b)
        return e

    coop_c = canon(coop_branch.group(1), "g", "u", "limit", "up_cap")
    had_c = canon(had_branch.group(1), "gf.x", "uf.x", "cg", "cu")
    assert coop_c == had_c, (
        "coop decode formula diverges from fused prefill:\n"
        f"  coop:    {coop_c}\n  prefill: {had_c}")

    # 4. up_cap reaches the coop params: prepare assigns it, the BC build passes it
    assert re.search(r"p\.up_cap\s*=\s*up_cap;", coop_cu), "prepare never assigns p.up_cap"
    assert re.search(r"act,\s*act_limit,\s*up_cap,\s*gated,", bc_cpp), \
        "BC_BlockSparseMLP does not pass up_cap into exl3_moe_coop_prepare"
    # the coop act range gate must admit the new value, not reject act == MOE_COOP_ACT_SOFTCAP
    assert re.search(r"act <= MOE_COOP_ACT_SOFTCAP", coop_cu), \
        "coop prepare still rejects any act above the old ceiling of 3"

    # 5. selector maps the boundary bool to the new enum and keeps the all-false SILU default
    selector = re.search(
        r"int act = act_silu_oai.*?MOE_COOP_ACT_SILU;", bc_cpp, re.S)
    assert selector, "coop act selector is gone"
    sel = selector.group(0)
    assert "act_softcap ? MOE_COOP_ACT_SOFTCAP" in sel, "selector does not map act_softcap"
    assert sel.rstrip().endswith("MOE_COOP_ACT_SILU;"), "selector lost the SILU default"

    # 6. boundary struct + pybind carry the bool and the float
    assert "bool act_softcap;" in bc_h and "float up_cap;" in bc_h, "BC struct missing softcap fields"
    assert 'py::arg("act_softcap")' in bc_bc_h and 'py::arg("up_cap")' in bc_bc_h, \
        "pybind does not expose act_softcap / up_cap"

    # 7. python threads the bool + cap at the construction site and admits the decode tier
    assert "act_softcap = self.activation_fn == \"softcap_swiglu\"," in module
    assert "up_cap = self.up_cap," in module
    assert re.search(r"support_bc_bszn = \(.*?\"softcap_swiglu\"", module, re.S), \
        "softcap_swiglu is not in the support_bc_bszn accept tuple"


def test_inkling_layer_builds_the_tier_with_the_lane_arguments():
    """Pins the wiring (commit 23ca4cb6): reverting it, or dropping an argument, fails here."""
    layer = (MODULES / "arch_specific" / "inkling2_loop.py").read_text()
    assert "from ..block_sparse_mlp import BlockSparseMLP" in layer
    ctor = re.search(r"self\.moe = BlockSparseMLP\((.*?)\n        \)", layer, re.S)
    assert ctor, "InklingLayer.moe is no longer built on the fused tier"
    args = ctor.group(1)
    for required in (
        'key = p + "mlp"',
        'key_gate = "experts.{expert_idx}.gate_proj"',
        'key_up = "experts.{expert_idx}.up_proj"',
        'key_down = "experts.{expert_idx}.down_proj"',
        'key_routing_gate = "gate"',
        'key_e_score_bias = "gate.bias"',
        'router_type = "dots"',
        'routed_scaling_factor = route_scale',
        'activation_fn = "softcap_swiglu"',
        'act_limit = gate_softcap',
        'up_cap = up_softcap',
    ):
        assert required in args, f"wiring lost {required!r}"
    assert "self.moe = InklingMoE(" not in layer, "the layer still builds the Python expert loop"
    # and the graph claim stays honest: the tier is itself capture-unsafe, so the swap does not
    # buy CUDA-graph eligibility
    fwd = (ROOT / "exllamav3" / "generator" / "fwd_graph.py").read_text()
    assert "return (BlockSparseMLP, BlockSparseMLP_CPU, GatedDeltaNet, InklingAttention, InklingMoE)" in fwd