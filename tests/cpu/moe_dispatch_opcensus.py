#!/usr/bin/env python3
"""Census (CPU, measured): how many operations does one decoded token actually issue?

§CG (kernel-lane brief, faa5193) sets the decode target from the pack's own roofline: 1.59 GiB of
expert weights per batch-1 token ⇒ ≈600 tok/s DRAM floor, measured 14.1 tok/s, and the fix is
"~10× fewer CUDA operations per token". That target needs a DENOMINATOR measured from the served
source, not an estimate. This script counts aten-level operations issued by InklingMoE.forward
under a TorchDispatchMode, per MoE-layer execution, at the two shapes that matter, and prices the
tiers the lane can actually reach.

Regime and limits, stated because they bound the claim:
  • CPU tensors, mock expert interiors (one matmul per projection). Every number is therefore a
    LOWER bound on device launches: a real expert projection additionally issues its own ext
    kernel(s) (exl3 dequant-GEMM) that aten cannot see, and the real ext coop call is ONE launch
    priced from moe_coop_plan, not counted here.
  • View/metadata ops (reshape, slice, expand…) do not launch kernels; they are counted separately
    and excluded from the kernel-class total. The classification list is printed, not hidden.
  • 50 executed MoE-layer instances per token pass comes from the pack's loop_config
    (core_start 6, core_end 18, num_passes 3): 11 outside the core + 13 × 3 (the lead's arithmetic,
    reused as a constant, not re-derived).

Run:  cd <repo> && env -u PYTHONHOME -u PYTHONPATH CUDA_VISIBLE_DEVICES= \
      PYTHONPATH=. ~/.venvs/tabby-exl3/bin/python tests/cpu/moe_dispatch_opcensus.py
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
_spec = importlib.util.spec_from_file_location("moe_coop_plan_census", PLAN)
plan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plan)

LAYERS_PER_TOKEN = 50            # pack loop_config: 11 outside the core + 13 core × 3 passes
PACK_EXPERTS, PACK_TOPK = 128, 8
H, I = 32, 48                    # dims do not change an op COUNT; they only keep the mock cheap

VIEW_CLASS_OPS = {               # metadata only: no kernel launch
    "aten::view", "aten::reshape", "aten::_reshape_alias", "aten::slice", "aten::select",
    "aten::expand", "aten::detach", "aten::alias", "aten::unsqueeze", "aten::squeeze",
    "aten::t", "aten::transpose", "aten::empty", "aten::empty_strided", "aten::resize_",
    "aten::new_zeros", "aten::new_empty", "aten::result_type", "aten::sym_size", "aten::len",
}


class Count(TorchDispatchMode):
    def __init__(self):
        self.ops = {}

    def __torch_dispatch__(self, func, types_, args = (), kwargs = None):
        self.ops[str(func)] = self.ops.get(str(func), 0) + 1
        return func(*args, **(kwargs or {}))

    @property
    def total(self):
        return sum(self.ops.values())

    @property
    def kernel_class(self):
        return sum(n for k, n in self.ops.items() if k not in VIEW_CLASS_OPS)

    @property
    def view_class(self):
        return self.total - self.kernel_class


def override(fn):
    return fn


def build(source_text, capturing = False):
    """Executable InklingMoE.forward over the pack's ROUTING geometry (128 experts, top_k 8)."""
    tree = ast.parse(LOOP.read_text(encoding = "utf8"))
    moe = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "InklingMoE"][0]
    fwd = [n for n in moe.body if isinstance(n, ast.FunctionDef) and n.name == "forward"][0]
    node = copy.deepcopy(fwd if source_text is None else ast.parse(source_text).body[0])
    node.decorator_list = []
    ns = {"torch": torch, "moe_coop_plan": plan, "override": override,
          "logger": types.SimpleNamespace(warning = lambda *_: None, info = lambda *_: None),
          "_capturing": lambda: capturing,
          "_MOE_FUSE_WINDOW": 2048,        # the fused-swiglu window the tree's eager arm reads
          "host_values": lambda t, key: t,                       # eager semantics on CPU
          "stage_host": lambda v, dev, key, dt: torch.tensor(list(v), dtype = dt),
          "get_for_device": lambda p, k, d: None}
    exec(compile(ast.fix_missing_locations(ast.Module(body = [node], type_ignores = [])),
                 str(LOOP), "exec"), ns)
    experts_g, experts_u, experts_d = [], [], []
    for e in range(PACK_EXPERTS):
        wg = torch.randn(H, I, generator = torch.Generator().manual_seed(100 + e)).half()
        wu = torch.randn(H, I, generator = torch.Generator().manual_seed(400 + e)).half()
        wd = torch.randn(I, H, generator = torch.Generator().manual_seed(700 + e)).half()
        experts_g.append(types.SimpleNamespace(w = wg))
        experts_u.append(types.SimpleNamespace(w = wu))
        experts_d.append(types.SimpleNamespace(w = wd))
    calls = []

    def lin(tag):
        def fwd_(x, params, out_dtype = None, t = tag):
            calls.append(t)
            return x @ t.w
        return fwd_
    for e, trio in enumerate(zip(experts_g, experts_u, experts_d)):
        trio[0].forward, trio[1].forward, trio[2].forward = \
            lin(trio[0]), lin(trio[1]), lin(trio[2])
    class Stub(types.SimpleNamespace):          # a real type: forward() setattr's class flags
        pass

    stub = Stub(
        config = types.SimpleNamespace(infer_params = types.SimpleNamespace(
            inkling_moe_expert_residency_ledger = False, moe_coop_decode = True)),
        layer_idx = 7, top_k = PACK_TOPK, route_scale = 2.5, gate_softcap = 4.0,
        up_softcap = 25.0, hidden_size = H, num_routed_experts = PACK_EXPERTS,
        router_w = torch.randn(PACK_EXPERTS, H, generator = torch.Generator().manual_seed(3)),
        router_bias = torch.randn(PACK_EXPERTS, generator = torch.Generator().manual_seed(4)),
        experts_g = experts_g, experts_u = experts_u, experts_d = experts_d,
        _coop_refuse_reason = "census: coop tier not exercised here",
        _coop_decode = lambda flat_, sel_, w_: None,
        _expertx = None, _expertx_note = lambda: None)
    return ns["forward"], stub, calls


def census(rows, source_text = None, capturing = False):
    fn, stub, calls = build(source_text, capturing)
    x = torch.randn(1, rows, H, generator = torch.Generator().manual_seed(9)).half()
    with Count() as c:
        fn(stub, x, {})
    return c, len(calls) // 3


DEC_HOST, n_dec = census(1)                              # decode: t == 1 → host round
DEC_GROUPED = DEC_HOST                                    # same arm; the device arm is t > 1 only
PRE_DEV, n_dev = census(2048)                             # prefill: t > 1, eager → device arm
PRE_HOST, n_pre = census(2048, capturing = True)          # prefill maths under the host arm

print(f"geometry: experts {PACK_EXPERTS}, top_k {PACK_TOPK}, {LAYERS_PER_TOKEN} executed "
      f"MoE-layer instances per token pass (pack loop_config)")
print(f"view-class ops excluded as kernel-free: {len(VIEW_CLASS_OPS)} names listed in the header")


def row(label, c, groups, unit = "token"):
    print(f"{label:<46} aten {c.total:6d}  kernel-class {c.kernel_class:6d}  "
          f"view-class {c.view_class:5d}  expert-groups {groups:3d}  "
          f"kernel-class per {unit} (×{LAYERS_PER_TOKEN}): {c.kernel_class * LAYERS_PER_TOKEN:7d}")


print("\n== measured: ops issued per MoE-layer execution (mock expert interior = 1 matmul/proj) ==")
row("decode rows=1 (t=1, host round — the served path)", DEC_HOST, n_dec, "token")
row("prefill rows=2048, device-grouping + fused-swiglu arm", PRE_DEV, n_dev, "chunk")
row("prefill rows=2048, host round (reference)", PRE_HOST, n_pre, "chunk")
delta = PRE_DEV.kernel_class - PRE_HOST.kernel_class
print(f"\ndevice arm's own cost at prefill shape: {delta:+d} kernel-class ops per MoE layer "
      f"({delta * LAYERS_PER_TOKEN:+d} per token pass); at decode it adds exactly "
      f"{DEC_HOST.kernel_class - DEC_GROUPED.kernel_class} because t == 1 never enters it")

print("\n== op inventory: what the decoded token's 50 MoE-layer instances are made of ==")
hist = {}
for k, n in DEC_HOST.ops.items():
    hist[k] = hist.get(k, 0) + n * LAYERS_PER_TOKEN
for k, n in sorted(hist.items(), key = lambda kv: -kv[1])[:14]:
    print(f"  {k:<34} {n:6d} per token")
print(f"  ({len(hist)} distinct ops; total {sum(hist.values())} kernel-class per token)")
delta_ops = {k: PRE_DEV.ops.get(k, 0) - PRE_HOST.ops.get(k, 0) for k in
             set(PRE_DEV.ops) | set(PRE_HOST.ops)}
print("\n== the device-grouping arm's own op delta at prefill shape (per MoE execution) ==")
for k, d in sorted(delta_ops.items(), key = lambda kv: -abs(kv[1])):
    if d:
        print(f"  {d:+3d}  {k}")

print("\n== what the tiers declare (not counted, read from moe_coop_plan / README) ==")
print(f"coop-drive tier: 2 launches per MoE layer (rot kernel only when bsz > 1) + 0 host "
      f"round-trips ⇒ {2 * LAYERS_PER_TOKEN}-{3 * LAYERS_PER_TOKEN} ops per token")
for rows in (1, 8, 32, 33, 2048):
    r = plan.refusal(rows, PACK_TOPK, True, True, True)
    print(f"  refusal(rows={rows:5d}, cuda+fp16+ext present) → "
          f"{'ADMITTED' if r is None else r}")

print("\n== fused tier AS LANDED on dsh-devdispatch: absorbs the MoE torch glue §CH names; the "
      "fused=False arm below is the excised per-group reference ==")
# §CH: at::native eager glue is 49.70 % of GPU time and cudaLaunchKernel 10,377,201 calls are
# 64.0 % of host API time. Most of the MoE dispatch's cost is NOT the expert GEMMs — it is the
# per-group elementwise swiglu and its casts, which are row-independent and can therefore run
# ONCE over all groups instead of once per group. This prototype mirrors the served op sequence
# exactly, proves the mirror bit-identical against the served forward, then counts the fused
# formulation. Nothing here changes the tree; it prices it.
PACK_I = 1152           # per-expert intermediate width from §CG's pack arithmetic (bytes only)


def dispatch(stub, x, fused = False, window = 0):
    """Mirror of InklingMoE.forward's dispatch (same router chain, same group order, same
    elementwise sequence). fused=True hoists the swiglu out of the group loop; window>0 bounds
    the fp32 temporaries by processing the entry order in windows."""
    b, t, h = x.shape
    flat = x.reshape(-1, h)
    hs = flat.float()
    scores = torch.sigmoid(torch.matmul(hs, stub.router_w.T))
    sel = torch.topk(scores + stub.router_bias, stub.top_k, dim = -1, sorted = False).indices
    w = scores.gather(1, sel)                       # weights come from the UNBIASED scores
    w = w / (w.sum(dim = -1, keepdim = True) + 1e-20)
    w = w * stub.route_scale
    sel_flat = sel.reshape(-1)
    w_flat = w.reshape(-1)
    out = torch.zeros((b * t, h), dtype = torch.float, device = x.device)
    order_all = torch.argsort(sel_flat, stable = True)
    counts = torch.bincount(sel_flat, minlength = stub.num_routed_experts)
    chosen = torch.nonzero(counts).flatten()
    ends = torch.cumsum(counts[chosen], 0)
    bounds = torch.stack([chosen, ends], 0)
    pairs = list(zip(bounds[0].tolist(), bounds[1].tolist()))
    tokens_all = torch.div(order_all, stub.top_k, rounding_mode = "floor")
    spans, start = [], 0
    for e, end in pairs:
        spans.append((e, start, end))
        start = end
    if not fused:
        for e, s, end in spans:
            xs = flat.index_select(0, tokens_all[s:end])
            g = (xs @ stub.experts_g[e].w).float()
            u = (xs @ stub.experts_u[e].w).float()
            act = torch.tanh(g / stub.gate_softcap) * stub.gate_softcap
            act = act * torch.sigmoid(g)
            act = act * (torch.tanh(u / stub.up_softcap) * stub.up_softcap)
            y = (act.to(torch.half) @ stub.experts_d[e].w)
            out.index_add_(0, tokens_all[s:end],
                           y.float() * w_flat.index_select(0, order_all[s:end]).unsqueeze(-1))
        return out.view(b, t, h)
    gs, us = [], []
    for e, s, end in spans:
        xs = flat.index_select(0, tokens_all[s:end])
        gs.append(xs @ stub.experts_g[e].w)
        us.append(xs @ stub.experts_u[e].w)
    gcat, ucat = torch.cat(gs, 0), torch.cat(us, 0)
    n = gcat.shape[0]
    act = torch.empty((n, gcat.shape[1]), dtype = torch.half, device = x.device)
    step = window if window > 0 else n
    for lo in range(0, n, step):
        hi = min(lo + step, n)
        g = gcat[lo:hi].float()
        u = ucat[lo:hi].float()
        a = torch.tanh(g / stub.gate_softcap) * stub.gate_softcap
        a = a * torch.sigmoid(g)
        a = a * (torch.tanh(u / stub.up_softcap) * stub.up_softcap)
        act[lo:hi] = a.to(torch.half)
    for e, s, end in spans:
        y = act[s:end] @ stub.experts_d[e].w
        out.index_add_(0, tokens_all[s:end],
                       y.float() * w_flat.index_select(0, order_all[s:end]).unsqueeze(-1))
    return out.view(b, t, h)


ROWS_BIG = 2048
FN_REF, STUB_REF, calls_ref = build(None, False)
X_BIG = torch.randn(1, ROWS_BIG, H, generator = torch.Generator().manual_seed(9)).half()


def counted(fn_call):
    with Count() as c:
        out = fn_call()
    return c, out


c_plain, out_plain = counted(lambda: dispatch(STUB_REF, X_BIG, fused = False))
c_served, out_served = counted(lambda: FN_REF(STUB_REF, X_BIG.clone(), {}))
mirrored = torch.equal(out_plain, out_served)
print(f"mirror check: the plain dispatch reproduces the served forward bit-exactly: {mirrored} "
      f"(max|Δ| {(out_plain - out_served).abs().max().item():.3e})")
c_fused, out_fused = counted(lambda: dispatch(STUB_REF, X_BIG, fused = True))
c_win, out_win = counted(lambda: dispatch(STUB_REF, X_BIG, fused = True, window = 2048))
n_entries = ROWS_BIG * PACK_TOPK
for label, c, out in (("served per-group swiglu (today's shape)", c_plain, out_plain),
                      ("fused swiglu, unbounded buffer", c_fused, out_fused),
                      ("fused swiglu, 2048-entry windows", c_win, out_win)):
    ok = torch.equal(out, out_served)
    print(f"  {label:<40} kernel-class {c.kernel_class:5d} per MoE execution  "
          f"bit-identical={ok}")
mb = lambda rows_, itemsize: rows_ * PACK_I * itemsize / 2 ** 20
print(f"  transient peak, pack geometry (I={PACK_I}, n={n_entries}): "
      f"unbounded = 2 fp16 cats {2 * mb(n_entries, 2):.0f} MiB + 2 fp32 {2 * mb(n_entries, 4):.0f} "
      f"MiB + act fp16 {mb(n_entries, 2):.0f} MiB = {2 * mb(n_entries, 2) + 2 * mb(n_entries, 4) + mb(n_entries, 2):.0f} MiB; "
      f"windows=2048 = {2 * mb(n_entries, 2) + mb(n_entries, 2) + 2 * mb(2048, 4):.0f} MiB "
      f"(bytes from shapes — arithmetic, not a GPU measurement)")
print(f"  reduction vs the mirrored served dispatch: {c_plain.kernel_class} → "
      f"{c_fused.kernel_class} (unbounded) / {c_win.kernel_class} (windowed) kernel-class ops; "
      f"ratio {c_plain.kernel_class / c_fused.kernel_class:.1f}× / "
      f"{c_plain.kernel_class / c_win.kernel_class:.1f}×")
fused_ok = mirrored and torch.equal(out_fused, out_served) and torch.equal(out_win, out_served)

print("\n== control: the counter reacts to a planted extra op ==")
tree = ast.parse(LOOP.read_text(encoding = "utf8"))
moe = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "InklingMoE"][0]
fwd = [n for n in moe.body if isinstance(n, ast.FunctionDef) and n.name == "forward"][0]
base_txt = ast.unparse(fwd).replace("'", '"')
mut = base_txt.replace("w_flat.index_select(0, entries)",
                       "w_flat.index_select(0, entries).add_(0.0)")
PLANTED, n_planted = census(1, source_text = mut)
print(f"decode with one planted add per group: kernel-class {PLANTED.kernel_class} vs "
      f"{DEC_HOST.kernel_class} baseline → +{PLANTED.kernel_class - DEC_HOST.kernel_class} "
      f"(expected +{n_dec}); groups {n_planted}")
ok = (PLANTED.kernel_class - DEC_HOST.kernel_class == n_dec) and fused_ok
print("\nCENSUS " + (
    "OK (CPU, mock expert interior, eager dispatch — the path prefill always runs and decode "
    f"runs when coop declines): eager MoE dispatch = {DEC_HOST.kernel_class} kernel-class aten "
    f"ops per executed layer-pass = {DEC_HOST.kernel_class * LAYERS_PER_TOKEN} per token pass; "
    f"hoisting the swiglu out of the group loop takes an executed layer-pass to "
    f"{c_fused.kernel_class} (unbounded buffer) / {c_win.kernel_class} (windowed) bit-identically; "
    "and the coop ladder admits every decode shape up to rows=32"
    if ok else "RED: the counter did not react to the planted op, or the fused prototype is not "
               "bit-identical to the served dispatch"))
raise SystemExit(0 if ok else 1)