#!/usr/bin/env python3
"""Fence: the Inkling 2 routed-expert WEIGHT-RESIDENCE ledger (CPU-only).

The prefetch-overlap question this lane started with — hide the next MoE layer's routed-expert
weight movement behind the current layer's compute — has a precondition that must be measured,
not assumed: something must actually move per selection. This fence executes the ledger that
measures it, on the served source (never the installed wheel; exllamav3 is never imported — the
module is parsed by AST and its functions are compiled into a mock world):

  [1] CENSUS — every expert's trellis/suh/svh are classified by device. A layer whose routed
      weights all sit on the compute device reports device=N/host=0; a layer with one
      host-resident expert reports it and names its index.
  [2] MOVEMENT — an address that changes with no unload() between is counted as a move and the
      verdict names it: that is the regime in which a prefetch lane would have work to hide.
  [3] CADENCE — the first forward censuses and logs (a boot reports residence instead of
      promising it), then censuses every _MOE_EXPERTX_RECENSUS forwards and logs every
      _MOE_EXPERTX_LOG_EVERY.
  [4] ARMED PATH IS PURE — no host sync, no allocation, no stream/event in the armed path: it
      cannot reorder work, and cannot break the coop arena's row-padding contract or the
      capture path's recorded ops. Sabotage arm: a planted torch.cuda.synchronize() trips it.
  [5] GATE OFF = UNCHANGED — InklingMoE.forward gains exactly one guarded statement, and with
      infer_params.inkling_moe_expert_residency_ledger False the ledger is never entered and the
      output is BIT-IDENTICAL to the same forward with that statement excised. Executed, not
      eyeballed; the polarity of the guard is itself under a control.
  [6] SURFACE — the wheel-gate markers exist in the served source, the switch is declared
      default OFF, and InklingMoE gained no CUDA stream/event (the negative result landed as a
      measurement; the side-stream idiom stays only where it was, in the capture path).

POSITIVE CONTROLS in every section: a planted defect in the new code must trip the check that
exists to catch it, and the fence proves that it tripped.

Run:  cd <repo> && env -u PYTHONHOME -u PYTHONPATH CUDA_VISIBLE_DEVICES= \\
      PYTHONPATH=. ~/.venvs/tabby-exl3/bin/python tests/cpu/inkling_moe_expertx_fence.py
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
PARAMS = ROOT / "exllamav3/model/config.py"
PLAN = ROOT / "exllamav3/modules/arch_specific/moe_coop_plan.py"

_spec = importlib.util.spec_from_file_location("moe_coop_plan_expertx", PLAN)
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


def src(path):
    return path.read_text(encoding = "utf8")


def summ(led):
    """Compact ledger view for check details (a full fingerprint is 72 addresses)."""
    return (f"fwd={led['forwards']} censuses={led['censuses']} moves={led['moves']} "
            f"residency={led['census']}")


def flat(node):
    """ast.unparse with double quotes, so a mutation pattern can name a string literal
    (the same normalization tests/cpu/inkling_attn_graph_fence.py uses)."""
    return ast.unparse(node).replace("'", '"')


TREE = ast.parse(src(LOOP))
CLS = {n.name: n for n in TREE.body if isinstance(n, ast.ClassDef)}
MOE = CLS["InklingMoE"]
METHODS = {n.name: n for n in MOE.body if isinstance(n, ast.FunctionDef)}
TOP_FN = {n.name: n for n in TREE.body if isinstance(n, ast.FunctionDef)}
CONSTS = [n for n in TREE.body if isinstance(n, ast.Assign)
          and getattr(n.targets[0], "id", "").startswith("_MOE_EXPERTX")]
LEDGER_NAMES = ("_expertx_census", "_expertx_verdict", "_expertx_summary_line", "_expertx_note")


class Book:
    """Logger stand-in: the fence reads the summary lines straight out of it."""

    def __init__(self):
        self.lines = []

    def warning(self, line):
        self.lines.append(line)

    def info(self, line):
        self.lines.append(line)


def override(fn):
    return fn


BASE_NS = {"torch": torch, "moe_coop_plan": plan, "logger": Book(), "override": override,
           # InklingMoE.forward consults the capture predicate (the device-grouping arm landed by
           # cf0df7fa) and the fused-swiglu window constant (the fused arm), so the namespace carries both.
           "_capturing": lambda: False, "_MOE_FUSE_WINDOW": 2048}


def run_nodes(nodes, ns):
    mod = ast.fix_missing_locations(ast.Module(body = nodes, type_ignores = []))
    exec(compile(mod, str(LOOP), "exec"), ns)
    return ns


def ledger_ns(mut = None, book = None):
    """The ledger's functions, compiled out of the served source and optionally rewritten by
    mut (a planted defect), with a logger the fence can read."""
    nodes = []
    for node in CONSTS + [TOP_FN["_fresh_expertx_ledger"]] + [METHODS[n] for n in LEDGER_NAMES]:
        node = copy.deepcopy(node)
        if mut is not None and isinstance(node, ast.FunctionDef):
            parsed = ast.parse(mut(flat(node))).body
            node = parsed[0]
        nodes.append(node)
    ns = dict(BASE_NS)
    ns["logger"] = book or Book()
    return run_nodes(nodes, ns)


def tensor_mock(device, addr):
    """Weight-tensor stand-in: the census reads .device and .data_ptr() and nothing else."""
    return types.SimpleNamespace(device = device, data_ptr = lambda: addr)


def expert_mock(base, off_device = None, missing = ()):
    """One routed expert: a Linear WRAPPER carrying .inner, exactly the shape
    moe_coop_plan.unwrap_linear expects."""
    inner = {}
    for proj in ("trellis", "suh", "svh"):
        inner[proj] = None if proj in missing else tensor_mock(off_device or "cuda:0",
                                                              base + 10 * ord(proj[0]))
    return types.SimpleNamespace(inner = types.SimpleNamespace(**inner))


def layer(experts, ns, layer_idx = 7):
    """Stub InklingMoE self: the roster plus the ledger and its methods, as forward sees them."""
    stub = types.SimpleNamespace(layer_idx = layer_idx, experts_g = experts, experts_u = experts,
                                 experts_d = experts, _expertx = ns["_fresh_expertx_ledger"]())
    for name in LEDGER_NAMES:
        setattr(stub, name, (lambda n: (lambda: ns[n](stub)))(name))
    return stub


def roster(count = 8, off_at = None, missing_at = None):
    experts = [expert_mock(1000 + e * 100) for e in range(count)]
    if off_at is not None:
        experts[off_at] = expert_mock(9000 + off_at, off_device = "cpu:0")
    if missing_at is not None:
        experts[missing_at] = expert_mock(7000 + missing_at, missing = ("svh",))
    return experts


FULL = 8

# ------------------------------------------------------------------ [1] census
print("[1] census classifies every routed expert by the device its weights live on")
NS = ledger_ns()
cen, fp = NS["_expertx_census"](layer(roster(), NS))
check("a fully device-resident layer reports every expert resident and none off-device",
      cen["experts"] == FULL and cen["device"] == FULL and cen["host"] == 0
      and cen["missing"] == 0 and cen["first_offdevice"] == -1, str(cen))
check("the fingerprint carries one address per weight tensor (nothing is skipped)",
      len(fp) == FULL * 9, f"{len(fp)} != {FULL * 9}")
cen_m, _ = NS["_expertx_census"](layer(roster(off_at = 3), NS))
check("one host-resident expert is counted and named by index (a prefetch candidate)",
      cen_m["host"] == 1 and cen_m["device"] == FULL - 1 and cen_m["first_offdevice"] == 3,
      str(cen_m))
cen_g, fp_g = NS["_expertx_census"](layer(roster(missing_at = 5), NS))
check("an absent tensor counts as missing AND off-device (an unloaded module is not resident)",
      cen_g["missing"] == 3 and cen_g["host"] == 1 and cen_g["first_offdevice"] == 5
      and fp_g.count(0) == 3, str(cen_g))
check("the census reaches the quant tensors through unwrap_linear, like the pointer tables",
      "moe_coop_plan.unwrap_linear" in flat(METHODS["_expertx_census"]))
bad_ns = ledger_ns(lambda s: s.replace('torch.device(t.device).type != "cuda"',
                                       'torch.device(t.device).type == "cuda"'))
cen_bad, _ = bad_ns["_expertx_census"](layer(roster(), bad_ns))
check("POSITIVE CONTROL: an inverted residence test calls a resident layer all-off-device",
      cen_bad["device"] == 0 and cen_bad["host"] == FULL, str(cen_bad))

# ------------------------------------------------------------------ [2] movement
print("[2] an address that moves without unload is counted, and the verdict names it")
book = Book()
NS2 = ledger_ns(book = book)
live = roster()
stub = layer(live, NS2)
NS2["_expertx_note"](stub)
check("the first forward censuses once (a boot reports residence, not a promise)",
      stub._expertx["censuses"] == 1 and stub._expertx["moves"] == 0 and len(book.lines) == 1,
      summ(stub._expertx))
check("that line carries the residence counts and the negative verdict word",
      book.lines[0].startswith("INKLING-MOE-EXPERTX L07 ")
      and "device_resident=8/8" in book.lines[0]
      and "verdict=all_device_resident_at_load" in book.lines[0], book.lines[0])
live[2] = expert_mock(555_000)                    # promote/offload/reload: fresh addresses
stub._expertx["censuses"] = 0                     # the next note re-censuses
NS2["_expertx_note"](stub)
check("moved weight addresses raise the move counter and change the verdict",
      stub._expertx["moves"] == 1
      and NS2["_expertx_verdict"](stub) == "weights_moved_without_unload", summ(stub._expertx))
host_stub = layer(roster(off_at = FULL - 1), NS2)
NS2["_expertx_note"](host_stub)
check("with no movement an off-device expert gives the offdevice verdict, not the resident one",
      NS2["_expertx_verdict"](host_stub) == "offdevice_experts:1:first_e7",
      NS2["_expertx_verdict"](host_stub))
mv_ns = ledger_ns(lambda s: s.replace('led["moves"] += 1', "pass"))
mv_stub = layer(roster(), mv_ns)
mv_ns["_expertx_note"](mv_stub)
mv_live = mv_stub.experts_g
mv_live[4] = expert_mock(777_000)
mv_stub._expertx["censuses"] = 0
mv_ns["_expertx_note"](mv_stub)
check("POSITIVE CONTROL: silencing the move counter loses the movement (moves stayed 0)",
      mv_stub._expertx["moves"] == 0, summ(mv_stub._expertx))

# ------------------------------------------------------------------ [3] cadence
print("[3] cadence: census per N forwards, log per M, nothing in between")
book = Book()
NS3 = ledger_ns(book = book)
stub = layer(roster(), NS3)
REC, LOG = NS3["_MOE_EXPERTX_RECENSUS"], NS3["_MOE_EXPERTX_LOG_EVERY"]
for _ in range(LOG):
    NS3["_expertx_note"](stub)
check(f"over {LOG} forwards the census runs {1 + LOG // REC} times (first + every {REC})",
      stub._expertx["censuses"] == 1 + LOG // REC, str(stub._expertx["censuses"]))
check(f"over {LOG} forwards exactly 2 summary lines are written (boot + every {LOG})",
      len(book.lines) == 2, f"{len(book.lines)} lines")
check("neither dial taxes decode (census >= 8 forwards apart, log >= 512)",
      REC >= 8 and LOG >= 512, f"recensus={REC} log_every={LOG}")
cad_book = Book()
cad_ns = ledger_ns(lambda s: s.replace('led["forwards"] == 1', "False"), book = cad_book)
cad_ns["_expertx_note"](layer(roster(), cad_ns))
check("POSITIVE CONTROL: suppressing the boot line leaves a resident pack silent",
      cad_book.lines == [], str(cad_book.lines))

# ------------------------------------------------------------------ [4] purity
print("[4] the armed path is sync-free, allocation-free and stream-free")
BODY = "\n".join(flat(METHODS[n]) for n in LEDGER_NAMES)
FORBID = (".item()", ".cpu()", ".tolist()", "torch.cuda.synchronize",
          "torch.cuda.current_stream", "torch.empty", "torch.zeros", "torch.ones",
          "torch.tensor(", "torch.cuda.Stream", "torch.cuda.Event", "record_stream")
found = [t for t in FORBID if t in BODY]
check(f"the armed path reads no device value back and allocates nothing (checked: {len(FORBID)} "
      "constructs)", not found, f"found {found}")
coop = flat(METHODS["_coop_decode"])
check("the coop arena's bucket keying and pinned-staging refill are untouched",
      "bucket = moe_coop_plan.padded_bucket_rows(rows)" in coop
      and "st = self._coop_by_bucket.get(bucket) or self._coop_state(flat.device, bucket)" in coop
      and 'args[k].copy_(st["stage"][lo:hi], non_blocking=True)' in coop)
planted = BODY.replace('led["forwards"] += 1', 'led["forwards"] += 1\ntorch.cuda.synchronize()')
check("SABOTAGE ARM: a planted synchronize in the armed path is caught by this same detector",
      "torch.cuda.synchronize" in planted and not found, str(found))

# ------------------------------------------------------------------ [5] gate off
print("[5] switch OFF: the served path is untouched and bit-identical")
param_src = src(PARAMS)
check("InferParams declares the switch with default False",
      re.search(r"inkling_moe_expert_residency_ledger:\s*bool\s*=\s*False", param_src) is not None)
FWD = copy.deepcopy(METHODS["forward"])
guarded = [i for i, st in enumerate(FWD.body)
           if isinstance(st, ast.If) and "inkling_moe_expert_residency_ledger" in flat(st)]
check("forward gained exactly ONE guarded statement and it is its first statement",
      len(guarded) == 1 and guarded[0] == 0
      and "self._expertx_note()" in flat(FWD.body[guarded[0]]), str(guarded))
STRIPPED = copy.deepcopy(METHODS["forward"])
STRIPPED.body[guarded[0]] = ast.Pass()

H, I, NE, TOPK = 16, 24, 6, 2


class Lin:
    """Mock expert projection: one fixed matrix, so a dispatch is deterministic on CPU."""

    def __init__(self, fin, fout, seed):
        self.w = torch.randn(fin, fout, generator = torch.Generator().manual_seed(seed),
                             dtype = torch.half)

    def forward(self, x, params, out_dtype = None):
        return x @ self.w


def strip(fn):
    fn.decorator_list = []
    return fn


def world(flag, source = None):
    """Executable InklingMoE.forward over mock experts: coop declines (x is not CUDA), so the
    python dispatch runs end to end on CPU."""
    node = strip(copy.deepcopy(source or METHODS["forward"]))
    ns = dict(BASE_NS)
    ns["host_values"] = lambda t, tag: t
    ns["stage_host"] = lambda vals, device, tag, dtype: torch.tensor(vals, dtype = dtype)
    ns["ext"] = types.SimpleNamespace(exl3_moe_coop = None)
    run_nodes([node], ns)
    calls = []
    stub = types.SimpleNamespace(
        config = types.SimpleNamespace(infer_params = types.SimpleNamespace(
            inkling_moe_expert_residency_ledger = flag, moe_coop_decode = True)),
        layer_idx = 3, top_k = TOPK, route_scale = 2.5, gate_softcap = 4.0, up_softcap = 25.0,
        hidden_size = H, num_routed_experts = NE,
        router_w = torch.randn(NE, H, generator = torch.Generator().manual_seed(11)),
        router_bias = torch.randn(NE, generator = torch.Generator().manual_seed(12)),
        experts_g = [Lin(H, I, 100 + e) for e in range(NE)],
        experts_u = [Lin(H, I, 400 + e) for e in range(NE)],
        experts_d = [Lin(I, H, 700 + e) for e in range(NE)],
        _coop_refuse_reason = None,
        _coop_decode = lambda flat_, sel_, w_: calls.append("coop") or None,
        _expertx = {"forwards": 3, "censuses": 1, "moves": 0, "fingerprint": (), "census": {}},
        _expertx_note = lambda: calls.append("expertx"))
    return stub, ns["forward"], calls


X = torch.randn(2, 5, H, generator = torch.Generator().manual_seed(77)).half()
off_stub, off_fn, off_calls = world(False)
base_stub, base_fn, base_calls = world(False, STRIPPED)
out_off = off_fn(off_stub, X.clone(), {})
out_base = base_fn(base_stub, X.clone(), {})
check("switch OFF: output is BIT-IDENTICAL to the same forward with the guarded line excised",
      torch.equal(out_off, out_base) and out_off.dtype == torch.float
      and tuple(out_off.shape) == (2, 5, H),
      f"max|Δ| {(out_off.float() - out_base.float()).abs().max().item():.3e}")
check("switch OFF: the ledger is never entered",
      off_calls == ["coop"] and off_stub._expertx["forwards"] == 3, str(off_calls))
on_stub, on_fn, on_calls = world(True)
out_on = on_fn(on_stub, X.clone(), {})
check("switch ON: the note runs once per forward and the maths is unchanged",
      on_calls == ["expertx", "coop"] and torch.equal(out_on, out_off), str(on_calls))
inverted = flat(METHODS["forward"]).replace(
    "if self.config.infer_params.inkling_moe_expert_residency_ledger:",
    "if not self.config.infer_params.inkling_moe_expert_residency_ledger:")
inv_node = ast.parse(inverted).body[0]
inv_stub, inv_fn, inv_calls = world(False, inv_node)
inv_fn(inv_stub, X.clone(), {})
check("POSITIVE CONTROL: with the guard's polarity inverted the same OFF run arms the ledger "
      "(so the gate-OFF check above really reads polarity)",
      inv_calls == ["expertx", "coop"], str(inv_calls))

# ------------------------------------------------------------------ [6] surface
print("[6] markers, default OFF, and no side stream in the MoE block")
served = src(LOOP)
for marker in ("INKLING-MOE-EXPERTX L", '"all_device_resident_at_load"',
               '"weights_moved_without_unload"', "inkling_moe_expert_residency_ledger",
               "_expertx_census", "_MOE_EXPERTX_RECENSUS", 'f"verdict={self._expertx_verdict()}'):
    check(f"marker present in the served source: {marker}", marker in served)
moe_text = flat(MOE)
check("the MoE block gained no CUDA stream/event: the negative landed as a measurement, not a "
      "speculative side stream",
      "torch.cuda.Stream" not in moe_text and "torch.cuda.Event" not in moe_text
      and "wait_event" not in moe_text)
attn_text = flat(CLS["InklingAttention"])
check("the side-stream idiom stays where it already was — the sliding-graph capture path",
      "stream = torch.cuda.Stream()" in attn_text
      and "stream.wait_stream(torch.cuda.current_stream())" in attn_text)
check("unload resets the ledger with the load epoch (dead addresses cannot be compared)",
      "_fresh_expertx_ledger()" in flat(METHODS["unload"]))
check("the ledger comes from a factory, not a shared mutable default",
      served.count("def _fresh_expertx_ledger") == 1 and served.count("_fresh_expertx_ledger()") >= 2)

print()
if fails:
    print(f"FAILURES ({len(fails)}):")
    for f in fails:
        print("  - " + f)
    print("INKLING MOE EXPERTX FENCE RED: " + ", ".join(fails))
    sys.exit(1)
print(f"INKLING MOE EXPERTX FENCE GREEN: {ran}/{ran} checks ok — census, movement, cadence and "
      f"purity executed on the served source; gate OFF bit-identical on a {FULL}-expert mock "
      f"layer; 6 planted defects caught (census, moves, boot line, sync, polarity, marker)")
sys.exit(0)