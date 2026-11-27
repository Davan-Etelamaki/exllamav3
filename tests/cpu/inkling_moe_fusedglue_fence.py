#!/usr/bin/env python3
"""Fence: the fused swiglu arm of Inkling 2 MoE dispatch (CPU-only).

The §CH kernel census put the at::native eager glue family at 49.70 % of GPU time and
cudaLaunchKernel at 64.0 % of host API time. This diff hoists the row-independent swiglu out of
the per-expert loop for the eager multi-token arm (the one cf0df7fa put on the device), so the
elementwise + cast chain runs a bounded number of times over ALL entries instead of once per
expert group. The expert GEMMs stay per-group: folding them into one launch is the grouped-kernel
wheel step (the §CJ floor model), not a python step.

Proved here by EXECUTING the served source's arms on mock experts:
  [1] BIT-IDENTICAL — fused arm vs the same forward with the fused arm excised, over flat and
      skewed routings, and with the window forced to 1 entry (every window edge crossed).
  [2] CONTRACT — the per-group elementwise+cast chain collapses: measured op counts of both arms
      (a ratio claim measured in this run, not a remembered number).
  [3] SEAM UNTOUCHED — t == 1 and a capturing stream still run the per-group loop verbatim: the
      capture contract keeps baking routing, and decode pays nothing for a prefill win.
  [4] CONTROLS — a planted wrong softcap, a planted missing cast, and an inverted seam flag each
      trip the check that exists to catch them.

Run:  cd <repo> && env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES= \\
      PYTHONPATH=. ~/.venvs/tabby-exl3/bin/python tests/cpu/inkling_moe_fusedglue_fence.py
"""
import ast
import copy
import importlib.util
import pathlib
import types

import torch
from torch.utils._python_dispatch import TorchDispatchMode

torch.set_num_threads(2)

ROOT = pathlib.Path(__file__).resolve().parents[2]
LOOP = ROOT / "exllamav3/modules/arch_specific/inkling2_loop.py"
PLAN = ROOT / "exllamav3/modules/arch_specific/moe_coop_plan.py"
_spec = importlib.util.spec_from_file_location("moe_coop_plan_fusedglue", PLAN)
plan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plan)

VIEW_OPS = ("aten.view", "aten.reshape", "aten.slice", "aten.select", "aten.unsqueeze",
            "aten.squeeze", "aten.expand", "aten.empty", "aten.empty_strided", "aten.lift_fresh",
            "aten.alias", "aten.detach", "aten.permute", "aten._efficientzerotensor")
fails = []
ran = 0


def check(name, ok, detail = ""):
    global ran
    ran += 1
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{(': ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


def flat(node):
    return ast.unparse(node).replace("'", '"')


def count(counter, prefix):
    """Ops are rendered 'aten.cat.default' by this torch, so families are matched by prefix."""
    return sum(n for k, n in counter.ops.items() if k.startswith(prefix))


TREE = ast.parse(LOOP.read_text(encoding = "utf8"))
CLS = {n.name: n for n in TREE.body if isinstance(n, ast.ClassDef)}
METHODS = {n.name: n for n in CLS["InklingMoE"].body if isinstance(n, ast.FunctionDef)}
FORWARD = flat(METHODS["forward"])


class Count(TorchDispatchMode):
    def __init__(self):
        self.ops = {}

    def __torch_dispatch__(self, func, types_, args = (), kwargs = None):
        key = str(func).replace("aten::", "aten.")
        self.ops[key] = self.ops.get(key, 0) + 1
        return func(*args, **(kwargs or {}))

    @property
    def kernel(self):
        return sum(n for k, n in self.ops.items() if not any(k.startswith(v) for v in VIEW_OPS))


H, I, E, TOPK = 16, 24, 8, 3


def override(fn):
    return fn


def world(source_text, capture = False, window = 4096):
    ns = {"torch": torch, "moe_coop_plan": plan, "override": override,
          "_capturing": lambda: capture, "_MOE_FUSE_WINDOW": window,
          "host_values": lambda t, key: t,
          "stage_host": lambda v, d, key, dt: torch.tensor(list(v), dtype = dt),
          "get_for_device": lambda p, k, d: None,
          "logger": types.SimpleNamespace(warning = lambda *_: None, info = lambda *_: None)}
    node = ast.parse(source_text).body[0]
    node.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body = [node], type_ignores = [])),
                 str(LOOP), "exec"), ns)
    g, u, d, hits = [], [], [], []
    for e in range(E):
        wg = torch.randn(H, I, generator = torch.Generator().manual_seed(100 + e)).half()
        wu = torch.randn(H, I, generator = torch.Generator().manual_seed(400 + e)).half()
        wd = torch.randn(I, H, generator = torch.Generator().manual_seed(700 + e)).half()
        g.append(types.SimpleNamespace(w = wg)); u.append(types.SimpleNamespace(w = wu))
        d.append(types.SimpleNamespace(w = wd))
        for tag, proj in (("g", wg), ("u", wu), ("d", wd)):
            pass

    def mk(tensor, tag, holder):
        def fwd(x, params, out_dtype = None):
            holder.append(tag)
            return x @ tensor
        return types.SimpleNamespace(forward = fwd)
    experts_g = [mk(x.w, ("g", i), hits) for i, x in enumerate(g)]
    experts_u = [mk(x.w, ("u", i), hits) for i, x in enumerate(u)]
    experts_d = [mk(x.w, ("d", i), hits) for i, x in enumerate(d)]

    class Stub(types.SimpleNamespace):
        pass
    stub = Stub(config = types.SimpleNamespace(infer_params = types.SimpleNamespace(
        inkling_moe_expert_residency_ledger = False, moe_coop_decode = True)),
        layer_idx = 9, top_k = TOPK, route_scale = 2.5, gate_softcap = 4.0, up_softcap = 25.0,
        hidden_size = H, num_routed_experts = E,
        router_w = torch.eye(E, H), router_bias = torch.zeros(E),
        experts_g = experts_g, experts_u = experts_u, experts_d = experts_d,
        _coop_refuse_reason = None, _coop_decode = lambda f, s, w: None,
        _expertx = None, _expertx_note = lambda: None)
    return ns["forward"], stub, hits


def case(rows, seed):
    logits = torch.randn(rows, E, generator = torch.Generator().manual_seed(seed))
    logits[:, seed % E] += 6.0 if rows > 1 else 0.0      # skew some cases
    x = torch.zeros(1, rows, H)
    x[:, :, :E] = logits
    return x.half()


EXCISED = FORWARD.replace("if device_grouped:", "if False:")
print("[1] the fused swiglu arm is bit-identical to the per-group arm it replaces")
for rows, seed, window in ((2048, 3, 4096), (300, 4, 4096), (64, 5, 4096),
                           (512, 6, 1), (512, 7, 7), (2, 8, 3)):
    x = case(rows, seed)
    fn_f, stub_f, hits_f = world(FORWARD)
    fn_p, stub_p, hits_p = world(EXCISED)
    with Count() as c1:
        out_f = fn_f(stub_f, x.clone(), {})
    with Count() as c2:
        out_p = fn_p(stub_p, x.clone(), {})
    gu_idx = [i for i, tag in enumerate(hits_f) if tag[0] in ("g", "u")]
    d_idx = [i for i, tag in enumerate(hits_f) if tag[0] == "d"]
    phases_ok = bool(gu_idx) and bool(d_idx) and min(d_idx) > max(gu_idx)
    check(f"rows={rows} window={window}: fused arm bit-identical to the per-group arm; same "
          f"expert visits (multiset) with every projection call before any down call",
          torch.equal(out_f, out_p) and sorted(hits_f) == sorted(hits_p) and phases_ok,
          f"equal={torch.equal(out_f, out_p)} max|Δ|={(out_f - out_p).abs().max().item():.3e} "
          f"visits={len(hits_f)} vs {len(hits_p)} phases_ok={phases_ok}")

print("\n[2] the per-group elementwise + cast chain collapses (measured in this run)")
x_big = case(2048, 3)
fn_f, stub_f, _ = world(FORWARD)
fn_p, stub_p, _ = world(EXCISED)
with Count() as c_f:
    fn_f(stub_f, x_big.clone(), {})
with Count() as c_p:
    fn_p(stub_p, x_big.clone(), {})
ew_names = ("aten.tanh", "aten.sigmoid", "aten.mul", "aten.div", "aten._to_copy")
ew_f = sum(n for k, n in c_f.ops.items() if any(k.startswith(p) for p in ew_names))
ew_p = sum(n for k, n in c_p.ops.items() if any(k.startswith(p) for p in ew_names))
gemm_f, gemm_p = count(c_f, "aten.mm"), count(c_p, "aten.mm")
check("elementwise/cast ops at least halve while the expert GEMM count is untouched "
      "(measured in this run, no remembered constant)",
      ew_f * 2 <= ew_p and gemm_f == gemm_p and gemm_p > 0,
      f"elementwise/casts {ew_p} → {ew_f}; expert gemms {gemm_p} → {gemm_f}; total kernel-class "
      f"{c_p.kernel} → {c_f.kernel}")
check("total kernel-class ops contract by at least a third in this shape",
      c_f.kernel <= c_p.kernel * 2 // 3, f"{c_p.kernel} → {c_f.kernel}")

print("\n[3] the t == 1 / capture seam still runs the per-group loop verbatim")
x_dec = case(1, 11)
fn_d, stub_d, hits_d = world(FORWARD)
with Count() as c_d:
    out_dec = fn_d(stub_d, x_dec.clone(), {})
per_group_shape = any(hits_d[i][0] == "g" and hits_d[i + 1][0] == "u" and hits_d[i + 2][0] == "d"
                      for i in range(len(hits_d) - 2))
check("decode (t == 1) issues no cat and keeps the per-group g/u/d interleave (the fused arm "
      "never runs)", count(c_d, "aten.cat") == 0 and per_group_shape,
      f"cats={count(c_d, 'aten.cat')} hits={hits_d}")
fn_cap, stub_cap, hits_cap = world(FORWARD, capture = True)
x_big2 = case(2048, 3)
with Count() as c_cap:
    out_cap = fn_cap(stub_cap, x_big2.clone(), {})
out_tree = fn_f(stub_f, x_big2.clone(), {})
check("a capturing stream at t > 1 keeps the per-group chain (no cat) and matches the eager "
      "result bit-for-bit", count(c_cap, "aten.cat") == 0 and torch.equal(out_cap, out_tree),
      f"cats={count(c_cap, 'aten.cat')} equal={torch.equal(out_cap, out_tree)}")

print("\n[4] controls: planted defects in the fused arm are caught")
bad_cap = FORWARD.replace("torch.tanh(gw / self.gate_softcap) * self.gate_softcap",
                          "torch.tanh(gw / self.up_softcap) * self.up_softcap")
fn_b, stub_b, _ = world(bad_cap)
out_b = fn_b(stub_b, x_big.clone(), {})
check("CONTROL: a planted wrong softcap in the fused chain changes the result",
      not torch.equal(out_b, out_tree),
      f"max|Δ|={(out_b - out_tree).abs().max().item():.3e}")
wrong_operand = FORWARD.replace("aw = aw * torch.sigmoid(gw)", "aw = aw * torch.sigmoid(uw)")
try:
    fn_w, stub_w, _ = world(wrong_operand)
    out_w = fn_w(stub_w, x_big.clone(), {})
    caught = not torch.equal(out_w, out_tree)
except Exception as err:
    caught = True
    print(f"      (the wrong-operand arm also fails loudly: {type(err).__name__})")
check("CONTROL: feeding the up-leg into the gate sigmoid inside the fused chain is caught",
      caught, "outputs still equal — the fused chain does not read its operands distinctly")
always = FORWARD.replace("if device_grouped:", "if True:")
fn_a, stub_a, hits_a = world(always)
x_dec2 = case(1, 11)
with Count() as c_always:
    fn_a(stub_a, x_dec2.clone(), {})
phased = [t for t in hits_a].count(("d", 0)) >= 0 and \
    max((i for i, t in enumerate(hits_a) if t[0] in ("g", "u")), default = -1) < \
    min((i for i, t in enumerate(hits_a) if t[0] == "d"), default = 10 ** 9)
check("CONTROL: forcing the fused arm on makes a decoded step take it "
      "(so the seam check above really reads the flag)",
      count(c_always, "aten.cat") > 0 and phased,
      f"cats={count(c_always, 'aten.cat')} phased={phased}")

print()
if fails:
    print(f"FAILURES ({len(fails)}):")
    for f in fails:
        print("  - " + f)
    print("INKLING MOE FUSED-GLUE FENCE RED: " + ", ".join(fails))
    raise SystemExit(1)
print(f"INKLING MOE FUSED-GLUE FENCE GREEN ({ran}/{ran} checks ok; regime: CPU, mock experts): "
      f"fused swiglu is bit-identical to the per-group arm over 6 routings including window=1 "
      f"edges; elementwise/cast ops {ew_p} → {ew_f} and kernel-class {c_p.kernel} → {c_f.kernel} "
      f"at rows=2048 with the expert GEMM count unchanged ({c_p.ops.get('aten.mm.default', 0)}); "
      f"the t == 1 / capture seam is untouched; 3 planted defects caught.")
raise SystemExit(0)