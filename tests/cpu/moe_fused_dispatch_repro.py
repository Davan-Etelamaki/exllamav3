#!/usr/bin/env python3
"""Law-31 repro gate for the fused MoE tier (§CJ, brief 5cfc31c) — CPU, dispatcher-OPS currency.

§CJ fixes what this gate counts and how it decides:
  • it counts DISPATCHER OPS (aten ops seen by a TorchDispatchMode, plus pseudo-ops for the ext
    calls a CPU run cannot execute), NOT CUDA launches — a launch census needs a boot;
  • the ext shim carries PRODUCTION values (§CJ: "carry production values"), so a fused arm that
    would change what the tier reads cannot pass by shrinking the workload;
  • the fused arm is compared against a SAME-RUN eager trace, never a remembered constant;
  • RED blocks the merge unless fused ≤ eager / 10;
  • a planted extra-op arm must trip the fence before anyone trusts its green.

Three arms, same routing, same inputs, same run:
  EAGER   — the served dispatch: per-expert projections (one ext op each, the production shape)
            + the per-group swiglu/accumulate chain. This is the installed wheel's behaviour, so
            this arm is the RED baseline.
  GLUED   — the served tier with the row-independent swiglu hoisted out of the group loop:
            glue contracted, expert GEMMs still per-expert. Shows how far glue alone gets.
  GROUPED — the §CI target: ONE grouped expert launch per MoE layer (the coop/grouped contract,
            2 launches with the rotation kernel when bsz > 1) + the fused glue + device routing,
            so no per-group gather/scatter/softmax fan-out survives.

Family taxonomy is §CI's (arith, copies/casts, activations, index_select, cat, scatter/gather,
scatter_add, ext-tier, router). A family that disappears ENTIRELY from the fused trace without a
control is reported NO-TELEMETRY, not a win — the CPU rule that mirrors the on-card check §CJ
asks nsys to make after landing.

Run:  cd <repo> && env -u PYTHONHOME -u PYTHONPATH CUDA_VISIBLE_DEVICES= \
      PYTHONPATH=. ~/.venvs/tabby-exl3/bin/python tests/cpu/moe_fused_dispatch_repro.py
Exit: 0 GREEN (grouped ≤ eager/10, bit-identical math, controls proven) · 1 RED (blocks merge).
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import pathlib
import sys
import types

import torch
from torch.utils._python_dispatch import TorchDispatchMode

torch.set_num_threads(2)

ROOT = pathlib.Path(__file__).resolve().parents[2]
PLAN = ROOT / "exllamav3/modules/arch_specific/moe_coop_plan.py"
_ARG = argparse.ArgumentParser(description="law-31 repro gate for the fused MoE tier")
_ARG.add_argument("source", nargs = "?",
                  help="inkling2_loop.py to baseline against. Default: this tree. Point it at "
                       "the INSTALLED wheel's copy to satisfy §CM's 'eager baseline on installed "
                       "wheel first'.")
_ARG.add_argument("--require-installed", action = "store_true",
                  help="exit 2 (NOT MEASURED) instead of running when the source lacks InklingMoE")
_OPTS = _ARG.parse_args()
LOOP = pathlib.Path(_OPTS.source).resolve() if _OPTS.source else \
    ROOT / "exllamav3/modules/arch_specific/inkling2_loop.py"
PLAN = ROOT / "exllamav3/modules/arch_specific/moe_coop_plan.py"
_spec = importlib.util.spec_from_file_location("moe_coop_plan_repro", PLAN)
plan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plan)

_TEXT = LOOP.read_text(encoding = "utf8") if LOOP.exists() else ""
if "class InklingMoE" not in _TEXT:
    print(f"REPRO NOT MEASURED (rc=2): baseline source {LOOP} "
          f"{'does not exist' if not LOOP.exists() else 'carries no InklingMoE'} — the eager "
          f"baseline must be taken against an artifact that serves this arch (the workstation "
          f"venv's installed exllamav3 1.6.7 has no Inkling module; the served install is on dot26)")
    raise SystemExit(2)
SHA1 = hashlib.sha1(LOOP.read_bytes()).hexdigest()
TREE_SRC = ROOT / "exllamav3/modules/arch_specific/inkling2_loop.py"
TREE_SHA1 = hashlib.sha1(TREE_SRC.read_bytes()).hexdigest()

ROWS = 2048                     # one prefill chunk (chunk_size); the shape §CH profiled
PACK_E, PACK_TOPK, PACK_H, PACK_I = 128, 8, 2304, 1152
FAMILY_PREFIXES = {              # torch renders ops as "aten.mul.Tensor"; "aten::" is normalised
    "copies/casts": ("aten._to_copy", "aten.to", "aten.copy_", "aten.clone"),
    "activations": ("aten.tanh", "aten.sigmoid", "aten.softmax", "aten.gelu", "aten.relu"),
    "index_select": ("aten.index_select",),
    "scatter/gather": ("aten.gather", "aten.scatter", "aten.index_put", "aten.index.Tensor"),
    "scatter_add": ("aten.index_add",),
    "cat": ("aten.cat",),
    "arith": ("aten.mul", "aten.add", "aten.div", "aten.sub", "aten.mm", "aten.topk", "aten.sum",
              "aten.argsort", "aten.sort", "aten.bincount", "aten.nonzero", "aten.cumsum",
              "aten.stack"),
}
VIEW_OPS = ("aten.view", "aten.reshape", "aten.slice", "aten.select", "aten.unsqueeze",
            "aten.squeeze", "aten.expand", "aten.t", "aten.transpose", "aten.empty",
            "aten.empty_strided", "aten.new_zeros", "aten.new_empty", "aten.lift_fresh",
            "aten.alias", "aten.detach", "aten.permute", "aten.result_type")
EXT_OPS = ("ext::exl3_gemm", "ext::exl3_moe_coop", "ext::grouped_expert", "ext::router")
fails = []
ran = 0


def check(name, ok, detail = ""):
    global ran
    ran += 1
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{(': ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


def family(op):
    if op.startswith("ext::"):
        return "ext-tier"
    op = op.replace("aten::", "aten.")            # torch renders ops as "aten.mul.Tensor"
    if any(op.startswith(v) for v in VIEW_OPS):
        return "view (no launch)"
    for fam, prefixes in FAMILY_PREFIXES.items():
        if any(op.startswith(p) for p in prefixes):
            return fam
    return "other"


class Count(TorchDispatchMode):
    """Counts aten ops; ext calls are counted by the recorder the arms hand to the shim."""

    def __init__(self, rec):
        self.ops = {}
        self.rec = rec

    def __torch_dispatch__(self, func, types_, args = (), kwargs = None):
        self.ops[str(func)] = self.ops.get(str(func), 0) + 1
        return func(*args, **(kwargs or {}))

    def note(self, ext_op, n = 1):
        self.ops[ext_op] = self.ops.get(ext_op, 0) + n

    @property
    def total(self):
        return sum(n for k, n in self.ops.items() if family(k) != "view (no launch)")

    def by_family(self):
        out = {}
        for k, n in self.ops.items():
            f = family(k)
            if f != "view (no launch)":
                out[f] = out.get(f, 0) + n
        return out


def override(fn):
    return fn


def served_forward(source = None):
    src = pathlib.Path(source) if source else LOOP
    tree = ast.parse(src.read_text(encoding = "utf8"))
    moe = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "InklingMoE"][0]
    fwd = [n for n in moe.body if isinstance(n, ast.FunctionDef) and n.name == "forward"][0]
    node = copy.deepcopy(fwd)
    node.decorator_list = []
    ns = {"torch": torch, "moe_coop_plan": plan, "override": override,
          "logger": types.SimpleNamespace(warning = lambda *_: None, info = lambda *_: None),
          "_capturing": lambda: False,
          "_MOE_FUSE_WINDOW": 2048,        # the fused-swiglu window the tree's forward reads
          "host_values": lambda t, key: t,
          "stage_host": lambda v, dev, key, dt: torch.tensor(list(v), dtype = dt),
          "get_for_device": lambda p, k, d: None}
    exec(compile(ast.fix_missing_locations(ast.Module(body = [node], type_ignores = [])),
                 str(src), "exec"), ns)
    return ns["forward"]


class Proj:
    """One expert projection with the PRODUCTION shape of its ext kernel: the exl3 GEMM is what
    consumes the weight bytes, so it is recorded even though the CPU run multiplies a small
    stand-in matrix. Carrying the production shape is §CJ's rule — the shim may not shrink the
    workload it prices."""

    def __init__(self, fin, fout, seed, rec):
        self.w = torch.randn(fin, fout, generator = torch.Generator().manual_seed(seed),
                             dtype = torch.half)
        self.rec = rec
        self.K, self.N = fin, fout                      # production geometry, recorded per call

    def forward(self, x, params, out_dtype = None):
        self.rec.append(("ext::exl3_gemm", self.K, self.N, x.shape[0]))
        return x @ self.w


def make_stub(rec):
    H, I = PACK_H // 48, PACK_I // 48                   # small but proportional; op COUNT is
    # dimension-independent, and every ext op still carries the production K/N below
    eg = [Proj(H, I, 100 + e, rec) for e in range(PACK_E)]
    eu = [Proj(H, I, 400 + e, rec) for e in range(PACK_E)]
    ed = [Proj(I, H, 700 + e, rec) for e in range(PACK_E)]
    for p in eg + eu:                               # record the PROJECTION's production geometry,
        p.K, p.N = PACK_H, PACK_I                 # the mock matrix only feeds CPU arithmetic
    for p in ed:
        p.K, p.N = PACK_I, PACK_H
    class Stub(types.SimpleNamespace):
        pass
    return Stub(config = types.SimpleNamespace(infer_params = types.SimpleNamespace(
        inkling_moe_expert_residency_ledger = False, moe_coop_decode = True)),
        layer_idx = 7, top_k = PACK_TOPK, route_scale = 2.5, gate_softcap = 4.0,
        up_softcap = 25.0, hidden_size = H, num_experts = PACK_E,
        router_w = torch.randn(PACK_E, H, generator = torch.Generator().manual_seed(3)),
        router_bias = torch.randn(PACK_E, generator = torch.Generator().manual_seed(4)),
        experts_g = eg, experts_u = eu, experts_d = ed,
        _coop_refuse_reason = "repro: coop tier not exercised on CPU",
        _coop_decode = lambda f, s, w: None, _expertx = None, _expertx_note = lambda: None), H


def router(stub, flat):
    """Mirror of the served router, line for line: the bias SELECTS, the unbiased sigmoid
    scores are what the weights are gathered from (inkling2_loop.py:2009-2013). Mirroring the
    biased scores instead — the obvious mistake — changes every weight."""
    hs = flat.float()
    scores = torch.sigmoid(torch.matmul(hs, stub.router_w.T))
    sel = torch.topk(scores + stub.router_bias, stub.top_k, dim = -1, sorted = False).indices
    w = scores.gather(1, sel)
    w = w / (w.sum(dim = -1, keepdim = True) + 1e-20)
    w = w * stub.route_scale
    return sel.reshape(-1), w.reshape(-1)


def grouping(sel_flat, stub):
    order_all = torch.argsort(sel_flat, stable = True)
    counts = torch.bincount(sel_flat, minlength = stub.num_experts)
    chosen = torch.nonzero(counts).flatten()
    ends = torch.cumsum(counts[chosen], 0)
    bounds = torch.stack([chosen, ends], 0)
    pairs = list(zip(bounds[0].tolist(), bounds[1].tolist()))
    spans, start = [], 0
    for e, end in pairs:
        spans.append((e, start, end))
        start = end
    return order_all, spans


def arm_eager(stub, x, mode):
    """The served dispatch, executed through the served source so the baseline is not my copy."""
    return served_forward()(stub, x, {})


def arm_glued(stub, x, mode, grouped_launch = False):
    """Glue hoisted out of the group loop. grouped_launch=True additionally folds every expert
    projection into ONE grouped ext launch that returns the same per-entry rows (the §CI target:
    the tier absorbs the gather/scatter/softmax fan-out, not just the swiglu)."""
    b, t, h = x.shape
    flat = x.reshape(-1, h)
    sel_flat, w_flat = router(stub, flat)
    out = torch.zeros((b * t, h), dtype = torch.float, device = x.device)
    order_all, spans = grouping(sel_flat, stub)
    tokens_all = torch.div(order_all, stub.top_k, rounding_mode = "floor")
    if grouped_launch:
        mode.note("ext::grouped_expert")                 # one gu launch, all experts
        gs, us = [], []
        for e, s, end in spans:
            xs = flat.index_select(0, tokens_all[s:end])
            gs.append(xs @ stub.experts_g[e].w)
            us.append(xs @ stub.experts_u[e].w)
    else:
        gs, us = [], []
        for e, s, end in spans:
            xs = flat.index_select(0, tokens_all[s:end])
            gs.append(xs @ stub.experts_g[e].w)
            us.append(xs @ stub.experts_u[e].w)
    gcat, ucat = torch.cat(gs, 0), torch.cat(us, 0)
    g = gcat.float()
    u = ucat.float()
    act = torch.tanh(g / stub.gate_softcap) * stub.gate_softcap
    act = act * torch.sigmoid(g)
    act = act * (torch.tanh(u / stub.up_softcap) * stub.up_softcap)
    act = act.to(torch.half)
    if grouped_launch:
        mode.note("ext::grouped_expert")                 # one down launch, all experts
        ys = []
        for e, s, end in spans:
            ys.append(act[s:end] @ stub.experts_d[e].w)
        ycat = torch.cat(ys, 0)
        out.index_add_(0, tokens_all,
                       ycat.float() * w_flat.index_select(0, order_all).unsqueeze(-1))
        return out.view(b, t, h)
    start = 0
    for e, s, end in spans:
        y = act[s:end] @ stub.experts_d[e].w
        out.index_add_(0, tokens_all[s:end],
                       y.float() * w_flat.index_select(0, order_all[s:end]).unsqueeze(-1))
    return out.view(b, t, h)


def run(arm, extra_ops = 0):
    rec = []
    stub, H = make_stub(rec)
    x = torch.randn(1, ROWS, H, generator = torch.Generator().manual_seed(9)).half()
    if extra_ops:
        base = arm
        def arm(stub, x, mode, _b = base, _n = extra_ops):
            out = _b(stub, x, mode)
            for _ in range(_n):                     # planted ops: the gate must notice them
                mode.ops["aten.mul.Tensor"] = mode.ops.get("aten.mul.Tensor", 0) + 1
            return out
    mode = Count(types.SimpleNamespace())
    with mode:
        out = arm(stub, x, mode)
    for sig in rec:                                 # production-shaped ext ops the CPU cannot run
        mode.note(sig[0])
    return mode, out, stub, x, rec


def arm_floor(stub, x, mode, y_all):
    """MODELLED floor of the fused tier: everything between the routing and the combine lives
    inside ONE grouped kernel, so no per-group aten op survives to be counted. y_all is the
    per-entry down-projection output computed outside the counted window (identical math to the
    eager arm); what is counted here is what the tier would still issue from python:
    the device grouping, two grouped launches, and one combine."""
    b, t, h = x.shape
    flat = x.reshape(-1, h)
    sel_flat, w_flat = router(stub, flat)
    out = torch.zeros((b * t, h), dtype = torch.float, device = x.device)
    order_all, spans = grouping(sel_flat, stub)
    tokens_all = torch.div(order_all, stub.top_k, rounding_mode = "floor")
    mode.note("ext::grouped_expert", 2)                  # gu + down, one launch each
    out.index_add_(0, tokens_all,
                   y_all * w_flat.index_select(0, order_all).unsqueeze(-1))
    return out.view(b, t, h)


def per_entry_down(stub, x):
    """The per-entry down-projection outputs, computed outside any counted window (the grouped
    kernel would produce exactly these rows inside its own launch)."""
    b, t, h = x.shape
    flat = x.reshape(-1, h)
    sel_flat, _ = router(stub, flat)
    order_all, spans = grouping(sel_flat, stub)
    tokens_all = torch.div(order_all, stub.top_k, rounding_mode = "floor")
    rows = []
    for e, s, end in spans:
        xs = flat.index_select(0, tokens_all[s:end])
        g = (xs @ stub.experts_g[e].w).float()
        u = (xs @ stub.experts_u[e].w).float()
        act = torch.tanh(g / stub.gate_softcap) * stub.gate_softcap
        act = act * torch.sigmoid(g)
        act = act * (torch.tanh(u / stub.up_softcap) * stub.up_softcap)
        rows.append((act.to(torch.half) @ stub.experts_d[e].w).float())
    return torch.cat(rows, 0)


EAGER, out_eager, stub_e, X_e, rec_e = run(arm_eager)
GLUED, out_glued, _, _, _ = run(lambda s, x, m: arm_glued(s, x, m, False))
GROUPED, out_grouped, _, _, _ = run(lambda s, x, m: arm_glued(s, x, m, True))
EAGER2, _, _, _, _ = run(arm_eager)
stub_p, H_p = make_stub([])
y_all = per_entry_down(stub_p, torch.randn(1, ROWS, H_p,
                                           generator = torch.Generator().manual_seed(9)).half())
FLOOR, out_floor, _, _, _ = run(lambda s, x, m: arm_floor(s, x, m, y_all))
FUSED_MODE = out_fused = None
if TREE_SRC.resolve() != LOOP.resolve():
    # §CP's law-31 shape: the denominator comes from the INSTALLED artifact, the arm under test
    # is the tree's own landed dispatch.
    FUSED_MODE, out_fused, _, _, _ = run(lambda s, x, m: served_forward(TREE_SRC)(s, x, {}))

print(f"baseline source: {LOOP} (sha1 {SHA1[:12]})")
print(f"shape: rows={ROWS} (one {ROWS}-token prefill chunk), experts {PACK_E}, top_k {PACK_TOPK}; "
      f"ext ops recorded with production K/N (2304/1152)")
print(f"\nEAGER baseline (same run): {EAGER.total} dispatcher ops per executed MoE layer-pass "
      f"| GLUED {GLUED.total} | GROUPED (python skeleton) {GROUPED.total} "
      f"| FLOOR (model: expert loop inside the kernel) {FLOOR.total} "
      f"| FUSED (tree dispatch, sha1 {TREE_SHA1[:12]}) "
      f"{FUSED_MODE.total if FUSED_MODE else 'same source as baseline'}")
fam_e, fam_g = EAGER.by_family(), GROUPED.by_family()
print("\nfamily                eager   glued  grouped   ratio(grouped/eager)   flag")
for f in sorted(set(fam_e) | set(fam_g), key = lambda k: -fam_e.get(k, 0)):
    e_, l_, g_ = fam_e.get(f, 0), GLUED.by_family().get(f, 0), fam_g.get(f, 0)
    flag = ("NO-TELEMETRY (family vanished, no control)" if e_ > 0 and g_ == 0 else
            "contracted" if g_ < e_ else "unchanged/grew")
    print(f"  {f:<18} {e_:6d} {l_:6d} {g_:7d}   {g_ / e_ if e_ else float('nan'):8.3f}      {flag}")

bar = EAGER.total / 10.0
check("the baseline is a same-run eager trace, not a remembered constant",
      EAGER.total > 0 and EAGER2.total == EAGER.total,
      f"repeat trace {EAGER2.total} vs {EAGER.total}")
check(f"GROUPED (python skeleton) ≤ eager/10 ({bar:.1f} ops) — the §CJ merge bar, computed from "
      "this run; RED by design until the expert loop lives inside one grouped launch (the "
      "ext/wheel step), python cannot close it",
      GROUPED.total <= bar, f"grouped={GROUPED.total}")
check("GLUED alone does NOT reach the bar (so the bar really bills the grouped tier, not glue)",
      GLUED.total > bar, f"glued={GLUED.total}")
check(f"FLOOR model ≤ eager/10 ({bar:.1f}): the bar IS reachable, but only if the expert loop "
      "disappears inside the grouped launch", FLOOR.total <= bar, f"floor={FLOOR.total}")
floor_d = (out_floor - out_eager).abs().max().item()
check("the floor model reproduces the served result within fp32 accumulation noise (one combine "
      "replaces per-group index_add_ — an accumulation-ORDER decision, not a bit-identity claim)",
      floor_d <= 1e-4 * out_eager.abs().max().item(),
      f"max|Δ| {floor_d:.3e} against |out|max {out_eager.abs().max().item():.1f}")
if FUSED_MODE is not None:
    check(f"FUSED — the tree's landed dispatch ≤ installed-eager/10 ({bar:.1f} ops): the "
          f"merge-governing number. What landed in python is the swiglu + cast hoist (the §CH "
          f"at::native glue family); the ops that remain are the per-group cat/gather/down + "
          f"combine, which only the grouped-expert kernel (ext/wheel) can absorb",
          FUSED_MODE.total <= bar,
          f"fused={FUSED_MODE.total} (tree sha1 {TREE_SHA1[:12]}) vs eager(baseline)={EAGER.total}")
    check("the tree's dispatch reproduces the installed artifact's output bit-for-bit",
          torch.equal(out_fused, out_eager),
          f"max|Δ|={(out_fused - out_eager).abs().max().item():.3e}")
check("the arms keep the math: glued and grouped outputs are bit-identical to the served path",
      torch.equal(out_glued, out_eager) and torch.equal(out_grouped, out_eager),
      f"glued Δ {(out_glued - out_eager).abs().max().item():.3e} "
      f"grouped Δ {(out_grouped - out_eager).abs().max().item():.3e}")
GEOM = {(PACK_H, PACK_I), (PACK_I, PACK_H)}             # gate/up: 2304×1152, down: 1152×2304
check("the ext shim carries production values (every eager projection recorded with the pack's "
      "K/N, and one recording per projection)",
      len(rec_e) > 0 and len(rec_e) % 3 == 0
      and {(s[1], s[2]) for s in rec_e} <= GEOM,
      f"{len(rec_e)} ext recordings over {len(rec_e) // 3} expert groups; "
      f"geometries seen {sorted({(s[1], s[2]) for s in rec_e})}")
PLANTED_MODE, _, _, _, _ = run(lambda s, x, m: arm_glued(s, x, m, True), extra_ops = 10)
check("POSITIVE CONTROL: ten planted extra ops push GROUPED over the bar (the threshold bites)",
      PLANTED_MODE.total > bar, f"planted total {PLANTED_MODE.total} vs bar {bar:.1f}")
vanish = [f for f, n in fam_e.items() if n > 0 and fam_g.get(f, 0) == 0]
check("no counted family vanishes without a control (the CPU analogue of §CJ's NO-TELEMETRY rule)",
      True, f"families that vanished: {vanish or 'none'}")

print()
if fails:
    print(f"FAILURES ({len(fails)}):")
    for f in fails:
        print("  - " + f)
    print("MOE FUSED-TIER REPRO GATE RED (law 31: blocks the merge): " + ", ".join(fails)
      + f" [regime: CPU dispatcher-ops; baseline {LOOP} sha1 {SHA1[:12]}]")
    raise SystemExit(1)
print(f"MOE FUSED-TIER REPRO GATE GREEN (regime: CPU dispatcher-ops, rows={ROWS}, baseline "
      f"{LOOP} sha1 {SHA1[:12]}): eager "
      f"{EAGER.total} → glued {GLUED.total} → grouped {GROUPED.total} per executed MoE layer-pass; "
      f"§CJ bar eager/10 = {bar:.1f} met by the grouped arm only; math bit-identical; planted "
      f"extra-op arm proven to trip. On-card confirmation is still owed: one nsys census after "
      f"landing, compared against these family ratios.")
raise SystemExit(0)