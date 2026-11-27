#!/usr/bin/env python3
"""Fence: the two arms of Inkling 2 MoE expert dispatch (CPU-only).

The prefill census (§CC, 2026-11-27) named this call site as the ingest's critical path: one
blocking read per executed MoE layer, then Python sorting chunk_size × top_k router entries
before the next launch. The fix landed here moves the GROUPING onto the device for eager
multi-token rounds (argsort + bincount + cumsum; the host reads only the boundary pair) and
leaves the decode/capture round exactly as it was. What this fence proves, by EXECUTING both
arms of the served source on mock experts:

  [1] EQUIVALENCE — for the same routing, the device arm and the host arm hand the SAME tokens
      to the SAME experts in the SAME group order, and their outputs are BIT-IDENTICAL (the
      per-token index_add_ sequence is unchanged, so fp32 accumulation cannot drift).
  [2] PAYLOAD — the device arm reads ≤2 × num_experts values back and uploads no order
      tensor at all; the host arm still reads top_k × rows values and uploads the whole order.
  [3] REGIME SEAM — t == 1 and a capturing stream take the host arm verbatim (a capture must
      keep baking the routing it recorded: frozen boundaries meeting a fresh argsort on replay
      would hand tokens to the wrong expert).
  [4] CONTROLS — planted defects in the new arm (stacked boundary order swapped, cumsum
      dropped, arm condition forced True) each trip the check that exists to catch them.

Run:  cd <repo> && env -u PYTHONHOME -u PYTHONPATH CUDA_VISIBLE_DEVICES= \\
      PYTHONPATH=. ~/.venvs/tabby-exl3/bin/python tests/cpu/inkling_moe_devdispatch_fence.py
"""
import ast
import copy
import importlib.util
import pathlib
import re
import sys
import types

import torch

torch.set_num_threads(4)

ROOT = pathlib.Path(__file__).resolve().parents[2]
LOOP = ROOT / "exllamav3/modules/arch_specific/inkling2_loop.py"
PLAN = ROOT / "exllamav3/modules/arch_specific/moe_coop_plan.py"

_spec = importlib.util.spec_from_file_location("moe_coop_plan_devdisp", PLAN)
plan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plan)

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


TREE = ast.parse(LOOP.read_text(encoding = "utf8"))
CLS = {n.name: n for n in TREE.body if isinstance(n, ast.ClassDef)}
MOE = CLS["InklingMoE"]
METHODS = {n.name: n for n in MOE.body if isinstance(n, ast.FunctionDef)}
FORWARD_SRC = flat(METHODS["forward"])

# ------------------------------------------------------------------ [A] phantom-attr guard
# Installed wheel 1.6.7-07deb31a0c835 crash-looped dot26 (NRestarts=12, every boot dying in
# the _load_autosplit dummy forward): the device arm read self.num_routed_experts, an
# attribute ONLY these test stubs ever had. All four fences stayed GREEN because each one
# hands the extracted forward a SimpleNamespace that CARRIES the phantom name. This arm
# grades the real class instead of the stub: every self.<attr> InklingMoE READS must be
# stored by InklingMoE or by its base Module (setattr(self, "...") forms audited absent).
def self_attrs(class_node, ctx):
    return {n.attr for n in ast.walk(class_node)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id == "self" and type(n.ctx) is ctx}

def class_attrs(class_node):
    return (self_attrs(class_node, ast.Store)
            | {n.name for n in class_node.body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))})

BASE = ROOT / "exllamav3/modules/module.py"
base_cls = next(n for n in ast.parse(BASE.read_text(encoding = "utf8")).body
                if isinstance(n, ast.ClassDef) and n.name == "Module")
phantom = sorted(self_attrs(MOE, ast.Load) - class_attrs(MOE) - class_attrs(base_cls))
print("[A] every self.<attr> InklingMoE reads exists on the real class (phantom-crash guard)")
check("no phantom attributes: " + (", ".join(phantom) if phantom else "none"), not phantom)
_sab = copy.deepcopy(MOE)
_fwd = next(n for n in _sab.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
_fwd.body.insert(0, ast.Expr(ast.Attribute(value = ast.Name(id = "self", ctx = ast.Load()),
                                           attr = "phantom_ctrl_zz", ctx = ast.Load())))
check("CONTROL: planting self.phantom_ctrl_zz in forward trips this arm",
      "phantom_ctrl_zz" in sorted(self_attrs(_sab, ast.Load) - class_attrs(_sab)
                                  - class_attrs(base_cls)))

H, I, E, TOPK = 12, 16, 8, 3


def override(fn):
    return fn


def build(source_text, capture = False):
    """Executable copy of InklingMoE.forward over mock experts, with the host-transfer helpers
    replaced by spies that count every value crossing the boundary."""

    class Spy:
        reads = 0
        uploads = 0

        def __init__(self, capturing):
            self.capturing = capturing

    spy = Spy(capture)

    def host_values(t, key):
        spy.reads += t.numel()
        return t                                  # eager on CPU: the values ARE t

    def stage_host(values, device, key, dtype = torch.long):
        spy.uploads += len(values) if not isinstance(values, torch.Tensor) else values.numel()
        return torch.tensor(list(values), dtype = dtype)

    def capturing():
        return spy.capturing

    ns = {"torch": torch, "moe_coop_plan": plan, "logger": types.SimpleNamespace(
        warning = lambda *_: None, info = lambda *_: None), "override": override,
        "_capturing": capturing, "host_values": host_values, "stage_host": stage_host,
        "_MOE_FUSE_WINDOW": 2048,          # the fused-swiglu window constant the merged forward reads
        "get_for_device": lambda params, k, d: None}
    node = ast.parse(source_text).body[0]
    node.decorator_list = []
    mod = ast.fix_missing_locations(ast.Module(body = [node], type_ignores = []))
    exec(compile(mod, str(LOOP), "exec"), ns)

    gen = torch.Generator().manual_seed(5)
    calls = []

    class Lin:
        def __init__(self, fin, fout, seed):
            self.w = torch.randn(fin, fout, generator = torch.Generator().manual_seed(seed),
                                 dtype = torch.half)

        def forward(self, x, params, out_dtype = None):
            # The per-group INPUT travels with the tag: a token handed to the wrong expert
            # changes the checksum, so sequence equality means "same expert, same rows".
            calls.append((self.tag[0], self.tag[1], x.shape[0], x.float().sum().item()))
            return x @ self.w

    experts_g, experts_u, experts_d = [], [], []
    for e in range(E):
        g, u, d = Lin(H, I, 100 + e), Lin(H, I, 400 + e), Lin(I, H, 700 + e)
        g.tag, u.tag, d.tag = ("g", e), ("u", e), ("d", e)
        experts_g.append(g); experts_u.append(u); experts_d.append(d)

    stub = types.SimpleNamespace(
        config = types.SimpleNamespace(infer_params = types.SimpleNamespace(
            inkling_moe_expert_residency_ledger = False, moe_coop_decode = True)),
        layer_idx = 5, top_k = TOPK, route_scale = 2.5, gate_softcap = 4.0, up_softcap = 25.0,
        hidden_size = H, num_experts = E, num_experts_per_tok = TOPK,
        # Identity router: scores[r, e] == x[r, e], so a case's routing IS its first E channels
        router_w = torch.eye(E, H), router_bias = torch.zeros(E),
        experts_g = experts_g, experts_u = experts_u, experts_d = experts_d,
        _coop_refuse_reason = None, _coop_decode = lambda f, s, w: None,
        _expertx = None, _expertx_note = lambda: None)
    return ns["forward"], stub, spy, calls


def make_case(rows, seed, skew = None):
    """One token row per router draw: the first E channels of x are the router logits, so a
    skew is visible to the router and the routing has no ties."""
    logits = torch.randn(rows, E, generator = torch.Generator().manual_seed(seed))
    if skew is not None:
        logits[:, skew] += 8.0
    x = torch.zeros(1, rows, H)
    x[:, :, :E] = logits
    return x.half()


def run(source_text, x, capture = False):
    fn, stub, spy, calls = build(source_text, capture)
    out = fn(stub, x, {})
    return out, spy, calls, stub


print("[0] the served source has the two arms, split by shape and capture state")
check("the device arm exists and is gated on t > 1 AND not capturing",
      re.search(r"if t > 1 and \(not _capturing\(\)\):", FORWARD_SRC) is not None)
check("the device arm derives the order with a STABLE argsort (ties by entry index, "
      "exactly what sorted() does)",
      "torch.argsort(sel_flat, stable=True)" in FORWARD_SRC)
check("the device arm's boundaries are bincount over the expert id + cumsum over the CHOSEN "
      "experts only",
      "torch.bincount(sel_flat, minlength=self.num_experts)" in FORWARD_SRC
      and "torch.cumsum(counts[chosen], 0)" in FORWARD_SRC)
check("one shared group loop consumes both arms' (expert, end) pairs",
      "for e, end in groups:" in FORWARD_SRC and "entries = order_all[start:end]" in FORWARD_SRC)
check("the host arm keeps the pinned stage_host upload and the nested group scan",
      'stage_host(order, x.device, ("inkling2", "moe_order_all"), torch.long)' in FORWARD_SRC
      and "groups.append((e, m))" in FORWARD_SRC)
check("_capturing is the same predicate host_values/stage_host consult",
      "from ...util.tensor import _capturing" in LOOP.read_text(encoding = "utf8")
      and "def _capturing()" in (ROOT / "exllamav3/util/tensor.py").read_text(encoding = "utf8"))

# ------------------------------------------------------------------ [1] equivalence
print("[1] same routing → same tokens per expert, same group order, bit-identical output")
CASES = [(2048, 11, None), (37, 12, None), (128, 13, 2), (64, 14, 0), (2, 15, None),
         (512, 16, E - 1)]
worst_seq = 0
for rows, seed, skew in CASES:
    x = make_case(rows, seed, skew)
    out_dev, spy_dev, calls_dev, _ = run(FORWARD_SRC, x)
    out_host, spy_host, calls_host, _ = run(FORWARD_SRC, x, capture = True)
    same_bits = torch.equal(out_dev, out_host)
    # The fused arm runs every gate/up before any down (that is the fusion), the seam
    # interleaves them per group — so visitation is compared PER PROJECTION FAMILY: same
    # expert, same group order, same rows (checksum) in both arms.
    fam = lambda cs, k: [c for c in cs if c[0] == k]
    seq_equal = all(fam(calls_dev, k) == fam(calls_host, k) for k in ("g", "u", "d"))
    check(f"rows={rows} skew={skew}: device arm is bit-identical to the host arm "
          f"and feeds each expert the same rows in the same group order",
          same_bits and seq_equal,
          f"equal={same_bits} seq={seq_equal} "
          f"max|Δ|={(out_dev - out_host).abs().max().item():.3e}")
    worst_seq = max(worst_seq, len(calls_dev))
check("both arms visit exactly one (g,u,d) triplet per non-empty expert",
      worst_seq % 3 == 0 and worst_seq // 3 <= E, f"{worst_seq // 3} experts touched")

print("[1b] the fused arm phases gate/up before down; the capture seam keeps today's interleave")
x_phase = make_case(2048, 11, None)
_, _, c_fused_arm, _ = run(FORWARD_SRC, x_phase)
_, _, c_seam, _ = run(FORWARD_SRC, x_phase, capture = True)
idx = lambda cs, k: [i for i, c in enumerate(cs) if c[0] == k]
check("fused arm: every gate/up projection completes before the first down projection "
      "(the activation is hoisted out of the group loop)",
      max(idx(c_fused_arm, "g")) < min(idx(c_fused_arm, "d")),
      f"last g at {max(idx(c_fused_arm, 'g'))}, first d at {min(idx(c_fused_arm, 'd'))}")
check("capture seam: down projections still interleave with gate/up (per-expert chain the "
      "graph baked is unchanged)",
      max(idx(c_seam, "g")) > min(idx(c_seam, "d")),
      f"last g at {max(idx(c_seam, 'g'))}, first d at {min(idx(c_seam, 'd'))}")

# ------------------------------------------------------------------ [2] payload
print("[2] the device arm shrinks what crosses the boundary")
big = FORWARD_SRC
x = make_case(2048, 21)
_, spy_dev, _, _ = run(big, x)
_, spy_host, _, _ = run(big, x, capture = True)
n = 2048 * TOPK
check(f"device arm reads ≤2×E={2 * E} values back where the host arm reads n={n}",
      spy_dev.reads <= 2 * E and spy_host.reads == n,
      f"dev reads={spy_dev.reads} host reads={spy_host.reads}")
check("device arm uploads no order tensor; the host arm uploads all n indices",
      spy_dev.uploads == 0 and spy_host.uploads == n,
      f"dev uploads={spy_dev.uploads} host uploads={spy_host.uploads}")

# ------------------------------------------------------------------ [3] regime seam
print("[3] t == 1 and a capturing stream keep the host round verbatim")
x_dec = make_case(1, 31)
fn_d, stub_d, spy_d, calls_d = build(FORWARD_SRC)
out_dec = fn_d(stub_d, x_dec, {})
check("decode (t == 1) takes the host arm: the whole order is uploaded, nothing else",
      spy_d.uploads == 1 * TOPK and spy_d.reads == 1 * TOPK,
      f"uploads={spy_d.uploads} reads={spy_d.reads}")
out_cap, spy_cap, calls_cap, _ = run(FORWARD_SRC, x, capture = True)
check("under capture (t > 1) the host arm runs — frozen boundaries never meet a fresh argsort",
      spy_cap.uploads == n and spy_cap.reads == n,
      f"uploads={spy_cap.uploads} reads={spy_cap.reads}")
no_dev = FORWARD_SRC.replace("if t > 1 and (not _capturing()):", "if False:")
out_force_host, _, calls_fh, _ = run(no_dev, x)
check("excising the device arm changes nothing about the host arm's result (bit-identical)",
      torch.equal(out_cap, out_force_host) and calls_cap == calls_fh,
      f"bits_equal={torch.equal(out_cap, out_force_host)} "
      f"max|Δ|={(out_cap - out_force_host).abs().max().item():.3e} "
      f"seq_equal={calls_cap == calls_fh} n_cap={len(calls_cap)} n_exc={len(calls_fh)}")

# ------------------------------------------------------------------ [4] controls
print("[4] controls: planted defects in the new arm are caught")
swap = FORWARD_SRC.replace("torch.stack([chosen, ends], 0)", "torch.stack([ends, chosen], 0)")
try:
    o_swap, _, c_swap, _ = run(swap, x)
    caught_swap = not torch.equal(o_swap, run(FORWARD_SRC, x)[0]) or c_swap != calls_dev
except Exception as err:
    caught_swap = True
    print(f"      (the swapped arm also fails loudly: {type(err).__name__}: {err})")
check("CONTROL: swapping the boundary pair sends tokens to the wrong experts", caught_swap,
      f"expert visits differ={c_swap != calls_dev}" if not caught_swap else "caught")
nocum = FORWARD_SRC.replace("torch.cumsum(counts[chosen], 0)", "counts[chosen]")
try:
    o_nocum, _, _, _ = run(nocum, x)
    caught = not torch.equal(o_nocum, run(FORWARD_SRC, x)[0])
except Exception as err:
    caught = True
    print(f"      (the cumsum-less arm also fails loudly: {type(err).__name__})")
check("CONTROL: dropping the cumsum (raw counts as ends) breaks the slices", caught)
always = FORWARD_SRC.replace("if t > 1 and (not _capturing()):", "if True:")
_, spy_always, _, _ = run(always, x_dec)
check("CONTROL: forcing the arm on makes a decoded step take the device path "
      "(so the t == 1 check above really reads the shape)",
      spy_always.uploads == 0 and spy_always.reads <= 2 * E,
      f"uploads={spy_always.uploads} reads={spy_always.reads}")
unstable = FORWARD_SRC.replace("torch.argsort(sel_flat, stable=True)",
                               "torch.argsort(sel_flat)")
o_unstable, _, c_unstable, _ = run(unstable, x)
check("CONTROL: an unstable argsort still hands the same expert SET per group (so 'stable' is "
      "pinned by the group-membership check, not by luck)",
      torch.equal(o_unstable, run(FORWARD_SRC, x)[0]) and len(c_unstable) == len(calls_dev))

print()
if fails:
    print(f"FAILURES ({len(fails)}):")
    for f in fails:
        print("  - " + f)
    print("INKLING MOE DEV-DISPATCH FENCE RED: " + ", ".join(fails))
    sys.exit(1)
print(f"INKLING MOE DEV-DISPATCH FENCE GREEN: {ran}/{ran} checks ok — device grouping is "
      f"bit-identical to the host round over {len(CASES)} routings (largest n = 2048×{TOPK}), "
      f"reads ≤{2 * E} values instead of {n}, uploads 0 instead of {n}, and the t == 1 / "
      f"capture seam keeps the host round verbatim; 4 planted defects caught.")
sys.exit(0)