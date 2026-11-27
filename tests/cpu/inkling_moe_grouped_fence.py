#!/usr/bin/env python3
"""Fence for the grouped multi-token MoE tier (infer_params.inkling_moe_grouped_ext) — CPU,
executed arms on served code extracted from inkling2_loop.py, mock expert interiors. The §CJ
repro gate (moe_fused_dispatch_repro.py) prices the OP COUNT; this fence owns the SEMANTICS:

  [A] equivalence — the grouped tier's output is BIT-IDENTICAL to the tree's own eager
      per-group chain (device-grouping arm, knob off) over executed random routings, at
      several multi-token shapes; the per-token index_add_ sequence is unchanged because
      order_all (stable, keyed on the expert id) IS the concatenation of the per-group
      entry runs, so the combine re-groups the same additions, never re-orders them.
  [B]/[C] REGIME SEAM — t == 1 and CUDA-graph capture keep today's chain verbatim (decline
      reason names the seam; the outputs equal the knob-off chain bit for bit).
  [D] the served default is OFF (config AST) and the ext call carries the served softcaps
      (a shim assertion — planted softcap swap must BREACH it, section [F]).
  [E] the ladder refusal: rows above GROUPED_BUCKETS decline, the eager chain runs.
  [F] planted defects, each must turn an ok-arm RED: wrong token division, gate/up softcap
      swap, and a +1 row shift in the modeled launch's act rows (law 21 — prove the fence
      fires before trusting its green).
  [G] header drift — moe_group_ext.cuh's MOE_GROUPED_MAX_ROWS, entry names and positional
      argument order must equal moe_coop_plan's mirrors (the coop plan fence's discipline).

Run:  cd <repo> && env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES= \
      PYTHONPATH=. ~/.venvs/tabby-exl3/bin/python tests/cpu/inkling_moe_grouped_fence.py
Exit: 0 GREEN · 1 RED (blocks the merge).
"""
import argparse
import ast
import hashlib
import importlib.util
import pathlib
import re
import sys
import textwrap
import types

import torch

torch.set_num_threads(2)

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "exllamav3/modules/arch_specific/inkling2_loop.py"
PLAN = ROOT / "exllamav3/modules/arch_specific/moe_coop_plan.py"
CUH = ROOT / "exllamav3/exllamav3_ext/moe_group/moe_group_ext.cuh"
CFG = ROOT / "exllamav3/model/config.py"
_ARG = argparse.ArgumentParser(description="grouped MoE tier semantics fence")
_ARG.add_argument("--poison", choices = ("division", "softcap", "shift"), default = None,
                  help="plant one defect and re-check: the matching arm MUST go RED")
OPTS = _ARG.parse_args()

_spec = importlib.util.spec_from_file_location("moe_coop_plan_fence_mod", PLAN)
plan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plan)

_TEXT = SRC.read_text(encoding = "utf8")
SHA1 = hashlib.sha1(SRC.read_bytes()).hexdigest()[:12]
E, TOPK, H, I = 128, 8, 48, 24          # op count is dimension-independent; mock stays small
fails = []
ran = 0


def check(name, ok, detail = ""):
    global ran
    ran += 1
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{(': ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


POISON_LANDED = {}                                     # method -> the [F] defect reached it


def extract(name, source_text, shim, capture = False, poisons = ()):
    """Exec one InklingMoE method from the SERVED source with harness globals, taken from
    the ORIGINAL source text (formatting preserved, so the section-[F] poison anchors land
    on served code — never on a normalized copy of it)."""
    tree = ast.parse(source_text)
    moe = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "InklingMoE"][0]
    node = [n for n in moe.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if not node:
        raise LookupError(f"InklingMoE.{name} not found in {SRC}")
    src = textwrap.dedent(ast.get_source_segment(source_text, node[0]))
    landed = False
    for old, new in poisons:
        if old in src:
            src = src.replace(old, new)
            landed = True
    if poisons and not landed:
        # anchors are per-method: one method carries the defect, the others must not raise
        pass
    POISON_LANDED[name] = landed
    ns = {"torch": torch, "moe_coop_plan": plan, "override": lambda fn: fn,
          "logger": types.SimpleNamespace(warning = lambda *_: None, info = lambda *_: None),
          "_capturing": lambda: capture, "_MOE_FUSE_WINDOW": 2048,
          "host_values": lambda t, key: t,
          "stage_host": lambda v, dev, key, dt: torch.tensor(list(v), dtype = dt),
          "get_for_device": lambda p, k, d: None, "ext": shim}
    exec(compile(src, f"<fence:{name}>", "exec"), ns)
    return ns[name]


class Lin:
    """Mock expert projection: .w feeds the CPU arithmetic; .trellis/.suh/.svh/.bias/.K/
    .mcg/.mul1 are the production attribute set the pointer tables and classify_experts read."""

    def __init__(self, fin, fout, seed):
        gen = torch.Generator().manual_seed(seed)
        self.w = torch.randn(fin, fout, generator = gen, dtype = torch.half)
        self.bias = None
        self.K = 4
        self.mcg = False
        self.mul1 = False
        self.trellis = torch.zeros(4, dtype = torch.int64)
        self.suh = torch.zeros(4, dtype = torch.int64)
        self.svh = torch.zeros(4, dtype = torch.int64)

    def forward(self, x, params, out_dtype = None):
        return x @ self.w


def make_stub(shim, knob = True, capture_names = False):
    class Stub(types.SimpleNamespace):
        pass
    stub = Stub(
        config = types.SimpleNamespace(infer_params = types.SimpleNamespace(
            inkling_moe_expert_residency_ledger = False, moe_coop_decode = False,
            inkling_moe_grouped_ext = knob)),
        layer_idx = 7, top_k = TOPK, route_scale = 2.5, gate_softcap = 4.0,
        up_softcap = 25.0, hidden_size = H, intermediate_size = I, num_experts = E,
        router_w = torch.randn(E, H, generator = torch.Generator().manual_seed(3)),
        router_bias = torch.randn(E, generator = torch.Generator().manual_seed(4)),
        experts_g = [Lin(H, I, 100 + e) for e in range(E)],
        experts_u = [Lin(H, I, 400 + e) for e in range(E)],
        experts_d = [Lin(I, H, 700 + e) for e in range(E)],
        _coop_refuse_reason = "fence: coop tier not exercised here",
        _coop_decode = lambda f, s, w: None,
        _grouped_refuse_reason = None, _grouped_emulate_cpu = True,
        _expertx = None, _expertx_note = lambda: None)
    stub._coop_ptr_lists = lambda: {
        tag + "_" + what: [getattr(m, what).data_ptr() for m in mods]
        for tag, mods in (("g", stub.experts_g), ("u", stub.experts_u), ("d", stub.experts_d))
        for what in ("trellis", "suh", "svh")}
    stub._coop_ptrs_probe = lambda: (
        stub.experts_g[0].trellis.data_ptr(), stub.experts_u[0].trellis.data_ptr(),
        stub.experts_d[0].trellis.data_ptr())
    return stub


class Shim:
    """Recording stand-in for the two grouped launches (§CJ: production values, asserted).
    mode=model fills the scratch with the eager chain's own rows so the CPU run completes
    the tier; mode=eager computes the same rows through plain aten (no counting here — the
    gate owns the op count). `shift` poison moves the act rows by +1 inside the launch."""

    def __init__(self, stub, poison = None):
        self.stub = stub
        self.poison = poison
        self.calls = []

    def exl3_moe_group_gu(self, x, order_all, counts, tables, K, mcg, mul1, act, gate_cap, up_cap):
        stub = self.stub
        rows = x.size(0)
        entries = rows * stub.top_k
        assert int(counts.sum()) == order_all.numel() == entries, "gu: order/table mismatch"
        assert tables.numel() == len(plan.TABLE_ORDER) * E, "gu: stacked table size"
        assert act.size(0) >= entries and act.size(1) == I, "gu: act geometry"
        assert 1 <= K <= 8, "gu: bitrate"
        assert gate_cap == stub.gate_softcap and up_cap == stub.up_softcap, \
            f"SOFTCAP DEFECT: launch received gate {gate_cap} up {up_cap}, " \
            f"served module holds {stub.gate_softcap}/{stub.up_softcap}"
        # The gu launch computes act AND the scratch the down launch reads; the down launch
        # consumes THESE rows, so a defect inside gu is visible to the combine (the shift
        # poison plants exactly that).
        model = self._model(order_all)
        if self.poison == "shift":
            model = torch.roll(model, 1, dims = 0)
        self.rows = model
        act.data.numpy()[:entries] = model.numpy()[:entries]
        self.calls.append(("gu", rows, entries))
        return 0

    def exl3_moe_group_down(self, act, entries, order_all, counts, tables, K, mcg, mul1, y):
        stub = self.stub
        assert int(counts.sum()) == order_all.numel() == entries, "down: entries mismatch"
        assert y.size(0) >= entries and y.size(1) == H, "down: y geometry"
        assert getattr(self, "rows", None) is not None, "down before gu"
        ys = torch.empty(self.rows.size(0), H, dtype = torch.half)
        exp = self.stub._fence_sel.index_select(0, order_all)
        for e in torch.unique(exp).tolist():
            m = (exp == e).nonzero().flatten()
            ys[m] = self.rows[m] @ stub.experts_d[e].w
        y.data.numpy()[:entries] = ys.numpy()[:entries]
        self.calls.append(("down", entries))
        return 0

    def _model(self, order_all):
        # per-entry act rows in order_all order: gather → gate/up → softcap swiglu
        sel = self.stub._fence_sel
        flat = self.stub._fence_flat
        tokens = torch.div(order_all, TOPK, rounding_mode = "floor")
        exp = sel.index_select(0, order_all)                      # expert id, sorted entry order
        xs = flat.index_select(0, tokens)
        g = torch.empty(xs.size(0), I, dtype = torch.half)
        u = torch.empty(xs.size(0), I, dtype = torch.half)
        for e in torch.unique(exp).tolist():
            m = (exp == e).nonzero().flatten()
            g[m] = xs[m] @ self.stub.experts_g[e].w
            u[m] = xs[m] @ self.stub.experts_u[e].w
        gf, uf = g.float(), u.float()
        act = torch.tanh(gf / self.stub.gate_softcap) * self.stub.gate_softcap
        act = act * torch.sigmoid(gf)
        act = act * (torch.tanh(uf / self.stub.up_softcap) * self.stub.up_softcap)
        return act.to(torch.half)


def route(stub, x):
    flat = x.reshape(-1, x.size(-1))
    hs = flat.float()
    scores = torch.sigmoid(torch.matmul(hs, stub.router_w.T))
    sel = torch.topk(scores + stub.router_bias, stub.top_k, dim = -1, sorted = False).indices
    w = scores.gather(1, sel)
    w = w / (w.sum(dim = -1, keepdim = True) + 1e-20)
    return sel, w * stub.route_scale


def run_arm(shim, knob, x, capture = False, poisons = ()):
    """Execute the served forward once against a fresh stub; returns (out, stub)."""
    stub = make_stub(shim, knob = knob)
    if shim is not None:
        shim.stub = stub                                  # the shim validates real stub values
    fns = {n: extract(n, _TEXT, shim, capture = capture, poisons = poisons)
           for n in ("forward", "_grouped_moe", "_grouped_ws")}
    if poisons and not any(POISON_LANDED.values()):
        raise AssertionError(f"planted defect landed on no served method: {poisons}")
    for n, fn in fns.items():
        setattr(stub, n, types.MethodType(fn, stub))
    stub._fence_flat = x.reshape(-1, x.size(-1))
    sel, w = route(stub, x)
    stub._fence_sel = sel.reshape(-1)
    out = stub.forward(x, {})
    return out, stub


def make_x(b, t, seed):
    return torch.randn(b, t, H, generator = torch.Generator().manual_seed(seed)).half()


print(f"INKLING MOE GROUPED FENCE — served source sha1 {SHA1}, poison={OPTS.poison or 'none'}")

poisons = ()
if OPTS.poison == "division":
    poisons = (("torch.div(order_all, self.top_k, rounding_mode = \"floor\")",
                "torch.div(order_all, self.top_k - 1, rounding_mode = \"floor\")"),)
elif OPTS.poison == "softcap":
    poisons = (("gate_cap = self.gate_softcap", "gate_cap = self.up_softcap"),)
# "shift" poisons the SHIM's fill, not the served source — the launch semantics themselves

print("\n[A] equivalence vs the tree's own eager chain (bit-identical)")
for (b, t, seed) in ((2, 9, 5), (1, 64, 7), (3, 11, 11)):
    x = make_x(b, t, seed)
    out_eager, _ = run_arm(Shim(None), knob = False, x = x)          # device-grouping chain
    shim = Shim(None)
    out_grouped, stub = run_arm(shim, knob = True, x = x)
    same = torch.equal(out_grouped, out_eager)
    d_ = "0.000e+00" if same else f"{(out_grouped - out_eager).abs().max().item():.3e}"
    check(f"[A] b={b} t={t}: grouped output bit-identical to the eager chain; grouped ran",
          same and len(shim.calls) == 2 and stub._grouped_refuse_reason is None,
          f"Δ={d_} calls={len(shim.calls)} decline={stub._grouped_refuse_reason}")

print("\n[B]/[C] regime seam — t==1 and capture keep today's chain verbatim")
x1 = make_x(1, 1, 21)
out_t1_eager, _ = run_arm(Shim(None), knob = False, x = x1)
shim = Shim(None)
out_t1, stub_t1 = run_arm(shim, knob = True, x = x1)
check("[B] t==1: grouped declines (reason names the seam) and the output is verbatim",
      torch.equal(out_t1, out_t1_eager) and not shim.calls
      and "t == 1" in (stub_t1._grouped_refuse_reason or ""),
      f"decline={stub_t1._grouped_refuse_reason} calls={len(shim.calls)}")
xc = make_x(1, 37, 22)
out_cap_eager, _ = run_arm(Shim(None), knob = False, x = xc, capture = True)
shim = Shim(None)
out_cap, stub_cap = run_arm(shim, knob = True, x = xc, capture = True)
check("[C] capturing: grouped declines (capture seam) and the output is verbatim",
      torch.equal(out_cap, out_cap_eager) and not shim.calls
      and "capture" in (stub_cap._grouped_refuse_reason or ""),
      f"decline={stub_cap._grouped_refuse_reason} calls={len(shim.calls)}")

print("\n[D] served default OFF; launch carries the served softcaps")
cfg_src = CFG.read_text(encoding = "utf8")
m = re.search(r"^(\s+)inkling_moe_grouped_ext:\s*bool\s*=\s*(\w+)", cfg_src, re.M)
check("[D] config.py declares inkling_moe_grouped_ext default False (served path unchanged)",
      m is not None and m.group(2) == "False", f"default={m.group(2) if m else 'ABSENT'}")
x = make_x(1, 33, 31)
try:
    run_arm(Shim(None), knob = True, x = x,
            poisons = (("gate_cap = self.gate_softcap", "gate_cap = self.up_softcap"),))
    softcap_caught = False
except AssertionError as e:
    softcap_caught = "SOFTCAP DEFECT" in str(e)
check("[D] a gate/up softcap swap at the launch boundary is CAUGHT by the shim assertion",
      softcap_caught, "production values are validated, not assumed")

print("\n[E] ladder refusal above GROUPED_BUCKETS keeps the eager chain")
xr = make_x(1, plan.GROUPED_BUCKETS[-1] + 1, 41)
out_hi_eager, _ = run_arm(Shim(None), knob = False, x = xr)
shim = Shim(None)
out_hi, stub_hi = run_arm(shim, knob = True, x = xr)
check("[E] rows above the ladder: grouped declines with the ladder reason, output verbatim",
      torch.equal(out_hi, out_hi_eager) and not shim.calls
      and "above the grouped ladder" in (stub_hi._grouped_refuse_reason or ""),
      f"decline={stub_hi._grouped_refuse_reason}")

print("\n[F] planted defects must RED the equivalence")
x = make_x(1, 25, 51)
out_eager, _ = run_arm(Shim(None), knob = False, x = x)
if OPTS.poison == "division":
    try:
        out_bad, _ = run_arm(Shim(None), knob = True, x = x, poisons = poisons)
        caught = not torch.equal(out_bad, out_eager)
        check("[F-division] wrong token division changes the output (equivalence is load-bearing)",
              caught, f"differed={caught}")
    except (AssertionError, RuntimeError) as e:
        check("[F-division] wrong token division caught", True, str(e)[:80])
elif OPTS.poison == "shift":
    out_bad, _ = run_arm(Shim(None, poison = "shift"), knob = True, x = x)
    caught = not torch.equal(out_bad, out_eager)
    check("[F-shift] a +1 row shift inside the modeled launch changes the output",
          caught, f"differed={caught}")
elif OPTS.poison == "softcap":
    try:
        run_arm(Shim(None), knob = True, x = x, poisons = poisons)
        caught = False
    except AssertionError as e:
        caught = "SOFTCAP DEFECT" in str(e)
    check("[F-softcap] planted softcap swap breaches the shim (must raise)", caught)
else:
    # green-state sanity: all three defects caught when planted (run them INLINE so one
    # process proves the fence is not vacuous — the --poison arms re-prove each in isolation)
    bad_div = None
    try:
        out_bad, _ = run_arm(Shim(None), knob = True, x = x,
                             poisons = (("torch.div(order_all, self.top_k, "
                                         "rounding_mode = \"floor\")",
                                         "torch.div(order_all, self.top_k - 1, "
                                         "rounding_mode = \"floor\")"),))
        bad_div = not torch.equal(out_bad, out_eager)
    except (AssertionError, RuntimeError):
        bad_div = True
    check("[F] planted division defect caught", bad_div is True)
    out_bad = run_arm(Shim(None, poison = "shift"), knob = True, x = x)[0]
    check("[F] planted launch-row shift defect caught", not torch.equal(out_bad, out_eager))

print("\n[G] header drift — the .cuh mirrors moe_coop_plan (the coop plan fence's discipline)")
if CUH.exists():
    cuh = CUH.read_text(encoding = "utf8")
    mr = re.search(r"constexpr int MOE_GROUPED_MAX_ROWS\s*=\s*(\d+)", cuh)
    gu_sig = re.search(r"int\s+exl3_moe_group_gu\s*\(([^)]*)\)", cuh, re.S)
    dn_sig = re.search(r"int\s+exl3_moe_group_down\s*\(([^)]*)\)", cuh, re.S)
    names = lambda sig: [a.strip().split()[-1] for a in sig.group(1).split(",")] if sig else []
    check("[G] MOE_GROUPED_MAX_ROWS equals the python mirror",
          mr is not None and int(mr.group(1)) == plan.MOE_GROUPED_MAX_ROWS,
          f"cuh={mr.group(1) if mr else 'ABSENT'} py={plan.MOE_GROUPED_MAX_ROWS}")
    check("[G] gu entry signature argument order equals GU_CALL_ARGS",
          names(gu_sig) == list(plan.GU_CALL_ARGS), f"cuh={names(gu_sig)}")
    check("[G] down entry signature argument order equals DOWN_CALL_ARGS",
          names(dn_sig) == list(plan.DOWN_CALL_ARGS), f"cuh={names(dn_sig)}")
else:
    check("[G] moe_group_ext.cuh present (the fence lands WITH the CUDA sources)", False,
          f"absent: {CUH}")

print()
if fails:
    print(f"FAILURES ({len(fails)}):")
    for f in fails:
        print("  - " + f)
    print(f"INKLING MOE GROUPED FENCE RED (blocks the merge) [regime: CPU, mock experts, "
          f"served sha1 {SHA1}]")
    raise SystemExit(1)
print(f"INKLING MOE GROUPED FENCE GREEN ({ran} checks ok; regime: CPU, mock experts, served "
      f"sha1 {SHA1}): equivalence bit-identical, regime seams verbatim, served default OFF, "
      f"production values asserted, planted defects caught.")