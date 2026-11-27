#!/usr/bin/env python3
"""Piecewise CUDA-graph capture pilot fence for Inkling 2 sliding decode (CPU-only).

Pins the four contracts the pilot claims without a GPU, on the served source (never the
installed wheel):
  [1] INDEX IDENTITY — the device-tensor ring geometry reproduces the base scalar path
      (f02a7468 ring_read, embedded verbatim) for a swept (positions, seqlen, window, ring,
      batch) grid: same gathered K/V, same kv_pos, no stale column left unmasked.
  [2] NUMERICS — the attention output is BIT-IDENTICAL when the row runs at its oracle
      column count (the control arm's exact slice), and within atol 1e-4 when the extent is
      padded to the capture bucket (softmax reduces over extra masked columns).
      Rule inherited from tests/graph/test_inkling_attend_peak.py:303-310.
  [3] CAPTURE KEY — the rows leg of the key is the shared arena rule
      moe_coop_plan.padded_bucket_rows; the ctx leg is attn_extent_bucket (monotone, power of
      two, never below need, capped at window+chunk); above the ladder there is no key.
  [4] ONE BUCKET LOOP — no second bucket-ladder implementation appeared in the lane.
  [5] SWITCH OFF — with infer_params.inkling_attn_graph False (and with it True at t > 1) the
      capture helper is never called: the control arm runs at live shapes with exact per-row
      extents. Executed, not eyeballed.
  [6] NO HOST SIDE IN THE CAPTURED BODY — the captured functions allocate no pageable host
      source (torch.tensor(..., device=)) and read no device value back (`.cpu()`/`.item()`),
      which is the failure that killed every capture of this model (900 STREAM_CAPTURE_UNSUPPORTED).
  [7] GLOBAL LAYER UNTOUCHED — the non-sliding branch and _gather_global keep the base source;
      InferParams declares the switch, default OFF; unload() drops the captured graphs.
Planted controls in every section: a mutation that would silence the arm must trip it.

Run:  CUDA_VISIBLE_DEVICES= env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH \\
      PYTHONPATH=. ~/.venvs/tabby-exl3/bin/python tests/cpu/inkling_attn_graph_fence.py
"""
import ast
import importlib.util
import os
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
# the lane this pilot touches; scanned by NAME (workstation law: no whole-tree grep sweeps)
LANE = [PLAN, LOOP, ROOT / "exllamav3/generator/fwd_graph.py",
        ROOT / "exllamav3/generator/generator.py", ROOT / "exllamav3/model/model.py",
        PARAMS, ROOT / "exllamav3/modules/arch_specific/inkling2_mtp.py",
        ROOT / "exllamav3/modules/short_conv.py", ROOT / "exllamav3/modules/sliding_attn.py",
        ROOT / "exllamav3/modules/gated_delta_net.py",
        ROOT / "exllamav3/modules/block_sparse_mlp.py",
        ROOT / "exllamav3/modules/block_sparse_mlp_cpu.py"]

fails = []


def check(name, ok, detail = ""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{(': ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


def src(path):
    return path.read_text(encoding = "utf8")


def flat(node):
    """ast.unparse with double quotes: string literals come back single-quoted, which would
    never match the source text the arms compare against."""
    return ast.unparse(node).replace("'", '"')


TREE = ast.parse(src(LOOP))
TOP_FN = {n.name: n for n in TREE.body if isinstance(n, ast.FunctionDef)}
CLS = {n.name: n for n in TREE.body if isinstance(n, ast.ClassDef)}
METHODS = {c: {n.name: n for n in CLS[c].body if isinstance(n, ast.FunctionDef)}
           for c in CLS}


def run_src(nodes, ns):
    """Compile the named AST nodes (extracted from the served source) into a namespace."""
    mod = ast.fix_missing_locations(ast.Module(body = nodes, type_ignores = []))
    exec(compile(mod, str(LOOP), "exec"), ns)
    return ns


_spec = importlib.util.spec_from_file_location("moe_coop_plan_pilot", PLAN)
plan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plan)

BASE_NS = {"torch": torch, "torch.nn.functional": torch.nn.functional,
           "F": torch.nn.functional, "_os": os, "moe_coop_plan": plan}
CONSTS = [n for n in TREE.body if isinstance(n, ast.Assign)
          and getattr(n.targets[0], "id", "").startswith("_ATTEND")]
NS_GEOM = run_src(CONSTS + [TOP_FN["ring_geometry"], TOP_FN["attn_extent_bucket"],
                            TOP_FN["attn_capture_key"], METHODS["InklingConvs"]["ring_read"]],
                  dict(BASE_NS))

# ---------------------------------------------------------------------------
# [1] the base scalar path, verbatim from f02a7468 (git show), as the oracle
# ---------------------------------------------------------------------------
BASE_RING_READ = '''
def base_ring_read(self, params, seqlen):
    st = self._state_slot(params)[0]
    s = st.k_state
    n = s.shape[1]
    positions = self._positions(params)
    slots = self._slots(params)
    ks, vs, los, lens = [], [], [], []
    for b, slot in enumerate(slots):
        pos = int(positions[b])
        lo = max(0, pos + seqlen - self.sliding_window)
        cnt = max(0, pos - lo)
        if cnt > 0:
            idx = (lo + torch.arange(cnt, device = s.device)) % n
            ks.append(s[slot, idx])
            vs.append(st.v_state[slot, idx])
        else:
            ks.append(s[slot, :0])
            vs.append(st.v_state[slot, :0])
        los.append(lo)
        lens.append(cnt)
    mlen = max(lens)
    def pad(rows):
        if all(r.shape[0] == mlen for r in rows):
            return torch.stack(rows)
        out = torch.zeros((len(rows), mlen) + tuple(rows[0].shape[1:]), dtype = s.dtype, device = s.device)
        for i, r in enumerate(rows):
            out[i, :r.shape[0]] = r
        return out
    return pad(ks), pad(vs), los, lens
'''
BASE_SCOPE = {"torch": torch}
exec(BASE_RING_READ, BASE_SCOPE)
base_ring_read = BASE_SCOPE["base_ring_read"]

print("[1] device index math == the base scalar path (f02a7468), swept")
CASES = [  # (positions, seqlen, window, ring_rows, kv_heads, head_dim)
    ([5], 1, 8, 8, 2, 4),                      # single row, mid-window
    ([0], 1, 8, 8, 2, 4),                      # nothing in the ring yet
    ([8], 1, 8, 8, 2, 4),                      # window saturated
    ([40, 12], 1, 8, 16, 2, 4),                # ragged pair, ring wrap-around
    ([7, 7, 3], 1, 8, 8, 3, 2),                # uneven third row
    ([6, 20], 2, 16, 32, 2, 4),                # two-token chunk (MTP verify shape)
    ([0, 0], 1, 8, 8, 2, 4),                   # all-empty batch
    ([255, 1], 1, 256, 256, 4, 8),             # page-rounded ring, one long one short
]
stale_nonzero = 0
for pos_list, seq, window, n_ring, kvh, hd in CASES:
    b = len(pos_list)
    torch.manual_seed(11)
    n_slots = b + 1                                        # pool is per-cache, never per-batch
    pool_k = torch.randn(n_slots * n_ring, kvh, hd).half()
    pool_v = torch.randn(n_slots * n_ring, kvh, hd).half()
    state = types.SimpleNamespace(
        k_state = pool_k.view(n_slots, n_ring, kvh, hd),
        v_state = pool_v.view(n_slots, n_ring, kvh, hd))
    slots = list(range(b))
    stub = types.SimpleNamespace(
        sliding_window = window,
        _state_slot = lambda params: (state, None),
        _positions = lambda params: list(pos_list),
        _slots = lambda params: list(slots))
    old_k, old_v, old_los, old_lens = base_ring_read(stub, {}, seq)
    mlen = max(old_lens)
    new_k, new_v, kv_pos, kv_valid, cnt = NS_GEOM["ring_read"](
        stub, {}, seq, torch.tensor(slots, dtype = torch.int64),
        torch.tensor(pos_list, dtype = torch.int64), mlen)
    tag = f"pos={pos_list} seq={seq} W={window} n={n_ring}"
    ok_kv = bool(torch.equal(old_k, new_k) and torch.equal(old_v, new_v))
    ok_cnt = bool([int(c) for c in cnt] == old_lens)
    # the per-row index the old code fed to _attend_one: arange(lo, lo+cnt+seq)
    ok_pos = all(bool(torch.equal(kv_pos[bi, :old_lens[bi] + seq],
                                  torch.arange(old_los[bi], old_los[bi] + old_lens[bi] + seq)))
                 for bi in range(b))
    # every column the old slice kept must be valid; the padded ring tail must be masked
    ok_valid = all(bool(torch.equal(
        kv_valid[bi],
        torch.cat([torch.arange(mlen) < old_lens[bi],
                   torch.ones(seq, dtype = torch.bool)]))) for bi in range(b))
    # a stale column that happens to hold zeros would make the zero-mask arm vacuous
    for bi, slot in enumerate(slots):
        raw = state.k_state[slot][old_los[bi] + torch.arange(old_lens[bi], mlen)]
        if raw.numel() and float(raw.float().abs().max()) > 0:
            stale_nonzero += 1
    check(f"gather+positions match: {tag}", ok_kv and ok_cnt and ok_pos and ok_valid,
          f"kv={ok_kv} cnt={ok_cnt} pos={ok_pos} valid={ok_valid}")
check("CONTROL: the sweep exercised stale ring columns with non-zero raw values "
      "(otherwise the zero-mask arm would prove nothing)", stale_nonzero > 0,
      f"{stale_nonzero} rows")

print("[2] attention numerics: exact at the oracle extent, <=1e-4 padded")
A_NS = dict(BASE_NS)
run_src(CONSTS + [METHODS["InklingAttention"]["_attend_one"],
                  METHODS["InklingAttention"]["_attend_one_tile"]], A_NS)


class StubAttn:
    _attend_one = A_NS["_attend_one"]
    _attend_one_tile = A_NS["_attend_one_tile"]

    def __init__(self, h, kvh, ext):
        self.num_kv_heads = kvh
        self.num_kv_groups = h // kvh
        self.rel_extent = ext
        self.scaling = 8 ** -0.5


h, kvh, d, ext = 4, 2, 8, 16
torch.manual_seed(3)
maxabs_worst = 0.0
nomask_worst = 0.0
for cnt, pad_extra in [(20, 0), (20, 4), (20, 32), (256, 7), (1, 1)]:
    # ONE decode row as the serving path lays it out: `cnt` ring columns + this token, and the
    # capture bucket pads the ring by pad_extra. lo = 0 (window not reached), so pos == cnt.
    attn = StubAttn(h, kvh, ext)
    q = torch.randn(h, 1, d).half()
    rel = torch.randn(h, 1, ext).float()
    sink = torch.randn(1, h).float()
    k_new = torch.randn(1, kvh, d).half()
    v_new = torch.randn(1, kvh, d).half()
    k_ring = torch.randn(cnt, kvh, d).half()
    v_ring = torch.randn(cnt, kvh, d).half()
    pos = cnt
    q_pos = torch.tensor([pos], dtype = torch.int64)
    k_oracle = torch.cat([k_ring, k_new])
    v_oracle = torch.cat([v_ring, v_new])
    pos_oracle = torch.arange(cnt + 1, dtype = torch.int64)
    oracle = attn._attend_one(q, k_oracle, v_oracle, rel, sink, q_pos, pos_oracle)
    fit = attn._attend_one(q, k_oracle, v_oracle, rel, sink, q_pos, pos_oracle,
                           torch.ones(cnt + 1, dtype = torch.bool))
    _, kv_pos_p, kv_valid_p, _ = NS_GEOM["ring_geometry"](
        torch.tensor([pos]), 1, 4096, 4096, cnt + pad_extra)
    k_pad = torch.cat([k_ring, torch.randn(pad_extra, kvh, d).half(), k_new])
    v_pad = torch.cat([v_ring, torch.randn(pad_extra, kvh, d).half(), v_new])
    Sp = cnt + pad_extra + 1
    padded = attn._attend_one(q, k_pad, v_pad, rel, sink, q_pos, kv_pos_p[0], kv_valid_p[0])
    nomask = attn._attend_one(q, k_pad, v_pad, rel, sink, q_pos, kv_pos_p[0],
                              torch.ones(Sp, dtype = torch.bool))
    nomask_worst = max(nomask_worst, float((nomask.float() - oracle.float()).abs().max()))
    exact = bool(torch.equal(oracle.view(torch.uint16), fit.view(torch.uint16)))
    maxabs = float((padded.float() - oracle.float()).abs().max())
    maxabs_worst = max(maxabs_worst, maxabs)
    check(f"bit-identical at the oracle column count (cnt={cnt}); padded extent {Sp} within 1e-4",
          exact and maxabs <= 1.0e-4 and padded.shape == oracle.shape,
          f"bit={exact} maxabs={maxabs:.3e} (softmax/gemm over extra masked columns)")
check("CONTROL: padding WITHOUT the validity mask moves the output (the stale ring column at "
      "dist == 0 would be attended; the mask, not the pad, keeps a replay faithful)",
      nomask_worst > 1.0e-3, f"maxabs={nomask_worst:.3e}")

print("[3] capture key: rows leg is the shared arena rule, ctx leg is one bucket fn")
for rows, want in [(1, 1), (3, 4), (5, 8), (17, 32), (32, 32), (33, None), (4096, None)]:
    key = NS_GEOM["attn_capture_key"](rows, 64, 4242)
    expect = None if want is None else (want, 64, 4242)
    check(f"attn_capture_key(rows={rows}) -> {expect}", key == expect, str(key))
check("rows leg == moe_coop_plan.padded_bucket_rows over a sweep",
      all(NS_GEOM["attn_capture_key"](r, 8, 0) is None
          if plan.padded_bucket_rows(r) is None
          else NS_GEOM["attn_capture_key"](r, 8, 0)[0] == plan.padded_bucket_rows(r)
          for r in range(1, 70)))
check("pool identity joins the key (a moved pool cannot replay the old graph)",
      NS_GEOM["attn_capture_key"](1, 8, 1) != NS_GEOM["attn_capture_key"](1, 8, 2))
buckets_ok = True
for need in list(range(1, 40)) + [255, 256, 257, 4095, 4096, 4097]:
    ceil = 4096 + 1
    got = NS_GEOM["attn_extent_bucket"](need, ceil)
    if not (got >= need and got <= ceil and (got & (got - 1)) == 0 or got == ceil):
        buckets_ok = False
check("ctx bucket: never below need, never above the ceiling, power of two (or the ceiling)",
      buckets_ok)
check("ctx bucket is monotone and collapses ragged needs into one bucket",
      NS_GEOM["attn_extent_bucket"](3000, 4097) == NS_GEOM["attn_extent_bucket"](4096, 4097) == 4096
      and NS_GEOM["attn_extent_bucket"](1, 4097) == 1)
cap_src = ast.unparse(TOP_FN["attn_capture_key"])
check("key builder calls the shared rule, not its own ladder",
      "moe_coop_plan.padded_bucket_rows(rows)" in cap_src,
      "")
planted = cap_src.replace("moe_coop_plan.padded_bucket_rows(rows)", "min(rows, 32)")
check("CONTROL: a re-derived bucket in the key builder trips the arm",
      planted != cap_src and "moe_coop_plan.padded_bucket_rows(rows)" not in planted)

print("[4] one bucket-loop implementation in the lane")
lane_hits = {p.name: src(p) for p in LANE if p.exists()}
defs = [name for name, text in lane_hits.items() if "def padded_bucket_rows" in text]
check("padded_bucket_rows is defined exactly once in the lane", defs == ["moe_coop_plan.py"],
      str(defs))
loops = [name for name, text in lane_hits.items()
         if "for bucket in" in text and name != "moe_coop_plan.py"]
check("no second bucket-walk appears outside moe_coop_plan.py", loops == [], str(loops))
check("inkling2_loop.py does not declare its own decode ladder",
      "DECODE_BUCKETS =" not in lane_hits["inkling2_loop.py"])
planted_loop = src(LOOP).replace(src(LOOP), "for bucket in (1, 2, 4, 8):\n        pass\n"
                                             + src(LOOP))
check("CONTROL: planting a second bucket walk trips the arm",
      "for bucket in" in planted_loop.split("\n", 1)[0]
      and "for bucket in" not in src(LOOP))

print("[5] switch OFF: the capture helper is never reached (executed)")
SLOTS_STUB = torch.tensor([0, 1])
SL_NS = run_src([METHODS["InklingAttention"]["_attend_sliding"]],
                dict(BASE_NS, get_for_device = lambda p, k, dev, default = None: SLOTS_STUB))


def sliding_stub(flag, extents_seen, capture_calls):
    def _capture(inp, params, b, t, cnts):
        capture_calls.append((b, t))
        return None

    def _attend(inp, kr, vr, kv_pos, kv_valid, rb, t, extents):
        extents_seen.extend(extents if extents is not None else [None])
        return torch.zeros(rb, t, 9)

    convs = types.SimpleNamespace(
        sliding_window = 8,
        _positions = lambda params: [7, 2],
        ring_read = lambda params, t, slots, pos_first, ring_cols: (
            torch.zeros(len(slots), ring_cols, 2, 3), torch.zeros(len(slots), ring_cols, 2, 3),
            torch.zeros(len(slots), ring_cols + t, dtype = torch.int64),
            torch.ones(len(slots), ring_cols + t, dtype = torch.bool),
            torch.tensor([7, 2])),
        ring_write = lambda k, v, params: None)
    return types.SimpleNamespace(
        convs = convs, device = "cpu",
        config = types.SimpleNamespace(infer_params = types.SimpleNamespace(
            inkling_attn_graph = flag)),
        _sliding_capture = _capture, _sliding_attend = _attend)


for label, flag, t, want_calls, want_extents in [
        ("switch OFF, t=1", False, 1, [], [8, 3]),
        ("switch ON, t=1, capture declined", True, 1, [(2, 1)], [8, 3]),
        ("switch ON but t=2 (verify window: pilot declines capture)", True, 2, [], [9, 4])]:
    calls, extents = [], []
    inp = {"q": torch.zeros(2, 4, t, 8), "kh": torch.zeros(2, t, 2, 8),
           "vh": torch.zeros(2, t, 2, 8), "rel_small": torch.zeros(2, 4, t, 16),
           "sink": torch.zeros(2, t, 4), "pos_row": torch.full((2, t), 7, dtype = torch.int64)}
    out = SL_NS["_attend_sliding"](sliding_stub(flag, extents, calls), inp, {}, 2, t)
    check(f"{label}: capture {'attempted' if want_calls else 'skipped'}, "
          f"control arm ran with exact extents {want_extents}",
          calls == want_calls and extents == want_extents and out.shape[0] == 2,
          f"calls={calls} extents={extents}")
off_src = ast.unparse(METHODS["InklingAttention"]["_attend_sliding"])
check("the gate reads the declared infer_params field",
      "self.config.infer_params.inkling_attn_graph and t == 1" in off_src)
planted_off = off_src.replace("self.config.infer_params.inkling_attn_graph and t == 1", "True")
check("CONTROL: a gate that always captures trips the arm",
      planted_off != off_src and "infer_params.inkling_attn_graph" not in planted_off)

print("[6] no pageable H2D, no host read-back inside the captured body")
GRAPH_BODY = [TOP_FN["ring_geometry"], METHODS["InklingConvs"]["ring_read"],
              METHODS["InklingAttention"]["_sliding_body"],
              METHODS["InklingAttention"]["_sliding_attend"]]
CAPTURED = GRAPH_BODY + [METHODS["InklingAttention"]["_sliding_capture"]]
captured_src = "\n".join(flat(n) for n in CAPTURED)
body_src = "\n".join(flat(n) for n in GRAPH_BODY)
check("captured region builds no pageable host tensor",
      "torch.tensor(" not in captured_src)
check("captured region never synchronises the stream it records (no .cpu()/.item())",
      ".cpu()" not in captured_src and ".item()" not in captured_src)
check("captured region takes slots/positions as tensors (no python-scalar arange index)",
      "torch.arange(cnt" not in captured_src and "ring_geometry(" in captured_src)
check("the graph body mutates nothing: ring_write runs in the caller, after the read",
      "ring_write" not in body_src)
planted_sync = captured_src.replace("slots.long()", "slots.cpu().long()")
check("CONTROL: a host read-back inside the body trips the arm",
      planted_sync != captured_src and ".cpu()" in planted_sync)
pos_src = flat(METHODS["InklingConvs"]["_positions"])
check("host mirror goes through pinned staging (host_values), not a blocking .cpu()",
      'host_values(positions, ("inkling2", "positions"))' in pos_src)
planted_mirror = pos_src.replace('host_values(positions, ("inkling2", "positions"))',
                                 "positions.cpu()")
check("CONTROL: reverting the host mirror to .cpu() trips the arm",
      planted_mirror != pos_src and "host_values" not in planted_mirror)

print("[7] global layer untouched + the declared seam")
fas = ast.unparse(METHODS["InklingAttention"]["forward_attn"])
global_branch = fas[fas.index("else:"):]
for needle in ('cache.update_layer_direct(self.layer_idx, cache_seqlens, block_table, kh, vh, t, inst)',
               'self._gather_global(\ncache, cache_seqlens, block_table, inst, kh, vh, t, params)'
               .replace("\n", ""),
               'kv_pos = torch.arange(S, device = q.device)',
               'q[bi], kv_cat_k[bi][:S], kv_cat_v[bi][:S], rel_small[bi], sink[bi], pos_row[bi], kv_pos'):
    check(f"global branch still contains: {needle[:58]}…",
          needle.replace(" ", "") in global_branch.replace(" ", ""))
for banned in ("_attn_graphs", "kv_valid", "attn_capture_key", "ring_geometry"):
    check(f"global branch names no graph machinery ({banned})",
          banned not in global_branch)
gather = ast.unparse(METHODS["InklingAttention"]["_gather_global"])
check("_gather_global unchanged (host lengths via _host_lens, no graph state)",
      "_host_lens(params, cache_seqlens)" in gather and "_attn_graph" not in gather)
check("ring_write runs in the caller of both arms, after the read (read-before-write)",
      'self.convs.ring_write(inp["kh"], inp["vh"], params)'
      in flat(METHODS["InklingAttention"]["_attend_sliding"])
      and flat(METHODS["InklingAttention"]["_attend_sliding"]).count("ring_write(inp[") == 2)
param_src = src(PARAMS)
check("InferParams declares inkling_attn_graph, default OFF",
      re.search(r"inkling_attn_graph:\s*bool\s*=\s*False", param_src) is not None)
check("the switch is read per call (both arms live in one process)",
      "self.config.infer_params.inkling_attn_graph" in off_src)
unload_src = ast.unparse(METHODS["InklingAttention"]["unload"])
check("unload drops the captured graphs (pool addresses die with the cache layers)",
      "self._attn_graphs = {}" in unload_src and "cl.free()" in unload_src)
planted_unload = unload_src.replace("self._attn_graphs = {}", "pass")
check("CONTROL: an unload that keeps the graphs trips the arm",
      planted_unload != unload_src and "_attn_graphs" not in planted_unload)
check("stats are exposed for the A/B (a zero delta must be explainable)",
      "runner" in ast.unparse(METHODS["InklingAttention"]["attn_graph_stats"]))

print()
if fails:
    print(f"FAILURES ({len(fails)}):")
    for f in fails:
        print("  - " + f)
    sys.exit(1)
print("VERDICT: inkling sliding-graph pilot fence GREEN "
      f"(index identity + numerics worst maxabs {maxabs_worst:.3e}; key + seams intact)")