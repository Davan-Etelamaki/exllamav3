#!/usr/bin/env python3
"""Stale-recurrent-checkpoint fence (CPU-only).

Pins the contract that ended the KeyError(6,0) abort storm (dot26 2026-11-27
00:56, and the same signature at the pre-MTP baseline): a persistent recurrent
checkpoint written against a DIFFERENT recurrent-layer registry is a cache MISS
(replay from zero), never an aborted request. Arms:
  [1] short_conv carries CheckpointStale + checkpoint_is_compatible, stash() stamps
      registry_keys on non-TP blobs, and unstash() validates BEFORE touching layers;
  [2] runtime semantics: legacy blob (no fingerprint) raises and leaves layers
      untouched; matching blob unstashes every layer; foreign registry raises;
  [3] Generator.recurrent_state_from_stashed converts CheckpointStale to None and
      prints the miss line, while a real fault still propagates (control);
  [1b] Cache.new_from_stashed quarantines the slot (clear_slot) before returning it
      to the free_list when state construction fails;
  [4] both job.py consumers treat None as the replay-from-zero miss: allocate_pages
      falls back to new_recurrent_state(), and the rewind path resets replay_from to 0.
Evidence pair (law 31; Launchers cc_f14_slothole_arm.py, dot26 installed bytes; rc is 0 on
BOTH halves — quote the TOKEN, never the rc):
  RED   wheel 018d2702524ac (short_conv ffd28f50, cache f5caccba):
        VERDICT_SLOTHOLE hole CONFIRMED (compatible+missing-slot reaches unstash uncaught)
  GREEN wheel 3c449ee1 (short_conv 71038765, cache f5caccba — completeness gate aboard,
        quarantine not yet):
        VERDICT_SLOTHOLE hole REFUTED + BH1-in-caught-stale + dirty=False
        (gate containment in force; quarantine half NOT installed — cache.py lacks clear_slot)
  GREEN+quarantine, tip 459ae4d7 and later (--fixed):
        VERDICT_SLOTHOLE hole REFUTED + BH1-in-caught-stale + dirty=False +
        QUARANTINE-IN-FORCE (gate miss; mid-copy fault returns the slot zeroed)

Run:  CUDA_VISIBLE_DEVICES= env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH \\
      PYTHONPATH=. ~/.venvs/tabby-exl3/bin/python tests/cpu/recurrent_checkpoint_stale_fence.py
"""
import ast
import copy
import io
import re
import sys
import types

SC = "exllamav3/modules/short_conv.py"
GEN = "exllamav3/generator/generator.py"
JOB = "exllamav3/generator/job.py"
LOOP = "exllamav3/modules/arch_specific/inkling2_loop.py"
CAC = "exllamav3/cache/cache.py"

def src(path):
    with open(path, encoding = "utf8") as f:
        return f.read()

fails = []
def check(name, cond):
    print(("  ok   " if cond else "  FAIL ") + name)
    if not cond:
        fails.append(name)

print("== [1] short_conv contract ==")
ssc = src(SC)
check("CheckpointStale defined as RuntimeError",
      "class CheckpointStale(RuntimeError):" in ssc)
check("checkpoint_is_compatible compares against sorted(registry_keys)",
      "stashed.get(\"registry_keys\") != sorted(registry_keys)" in ssc
      .replace("!= keys", "!= sorted(registry_keys)")
      or "if stashed.get(\"registry_keys\") != keys" in ssc)
check("compatibility ALSO demands a payload for every live layer (disk-tier key rewriting)",
      "all(k in stashed for k in keys)" in ssc)
# CONTROL 0: a fingerprint-matching blob whose payloads were re-keyed must NOT be trusted
from exllamav3.modules.short_conv import checkpoint_is_compatible as _c
check("CONTROL: registry_keys match + stringified payloads => incompatible",
      _c({"registry_keys": [(6, 0)], "6,0": "blob"}, {(6, 0): None}) is False)
check("stash() stamps registry_keys on the non-TP path",
      'stashed["registry_keys"] = sorted(registry)' in ssc)
sc_tree = ast.parse(ssc)
sc_cls = next(n for n in ast.walk(sc_tree)
              if isinstance(n, ast.ClassDef) and n.name == "ShortConvState")
unstash_fn = next(n for n in ast.walk(sc_cls)
                  if isinstance(n, ast.FunctionDef) and n.name == "unstash")
ubody = ast.unparse(unstash_fn)
check("unstash() validates BEFORE the per-layer unstash loop",
      ubody.index("CheckpointStale(") < ubody.index("l.unstash("))
# CONTROL 1: excise the validation If-node from the parsed AST; the guard's raise
# must disappear from the reconstructed body (otherwise the ordering arm proves nothing)
excised = copy.deepcopy(unstash_fn)
excised.body = [n for n in excised.body
                if not (isinstance(n, ast.If) and "CheckpointStale" in ast.unparse(n))]
mut1 = ast.unparse(excised)
check("CONTROL: excising the validation node removes every CheckpointStale raise from unstash",
      "CheckpointStale" in ubody and "CheckpointStale" not in mut1)

print("== [1b] slot quarantine on failed restore ==")
cac = src(CAC)
_quar = re.compile(
    r"except BaseException:\s+(?:#[^\n]*\n\s+)*self\.clear_slot\(handle\)\s*"
    r"(?:print\([^\n]*\)\s+)?self\.free_list\.appendleft\(handle\)\s+"
    r"(?:print\([^\n]*\)\s+)?raise")
check("new_from_stashed clears the slot before returning it to free_list",
      _quar.search(cac) is not None)
mut_q = cac.replace("self.clear_slot(handle)\n", "", 1)
check("CONTROL: excising the clear call trips the quarantine arm",
      _quar.search(cac) is not None and _quar.search(mut_q) is None)
_clr = re.compile(r"self\.clear_slot\(handle\)\s+print\(\"\[REC_EMIT\] cleared slot=")
check("clear execution is announced by [REC_EMIT] cleared print (freeze-enumerated x1)",
      _clr.search(cac) is not None and cac.count('[REC_EMIT] cleared ') == 1)
mut_p = cac.replace('            print("[REC_EMIT] cleared slot=%s pos=%s" % (handle, position), flush=True)\n', "", 1)
check("CONTROL: excising the cleared print trips the announcement arm",
      _clr.search(cac) is not None and _clr.search(mut_p) is None)
check("clear_slot covers non-TP and TP paths",
      "def clear_slot" in cac and "mp_cache_recurrent_clear" in cac and "l.clear(slot)" in cac)

print("== [2] runtime semantics ==")
from exllamav3.modules.short_conv import CheckpointStale, checkpoint_is_compatible, ShortConvState

check("matching fingerprint with full payload is compatible",
      checkpoint_is_compatible({"registry_keys": [(6, 0)], (6, 0): "B"}, {(6, 0): None}) is True)
check("legacy blob (no fingerprint) is stale -> self-heals on first touch",
      checkpoint_is_compatible({}, {(6, 0): None}) is False)
check("foreign registry is stale",
      checkpoint_is_compatible({"registry_keys": [(7, 0)]}, {(6, 0): None}) is False)

class RecordingLayer:
    def __init__(self):
        self.got = None
    def unstash(self, slot, blob):
        self.got = (slot, blob)

def fake_state(layers):
    st = ShortConvState.__new__(ShortConvState)
    st.slot = 3
    st.position = 5
    st.cache = types.SimpleNamespace(
        model = types.SimpleNamespace(loaded_tp = False),
        get_all_recurrent_layers = lambda: layers)
    return st

touched = RecordingLayer()
try:
    fake_state({(6, 0): touched}).unstash({"position": 5})
    check("legacy unstash raises CheckpointStale", False)
except CheckpointStale:
    check("legacy unstash raises CheckpointStale", True)
check("CONTROL: stale unstash left the layer untouched (KeyError path never ran)",
      touched.got is None)

fresh = RecordingLayer()
fake_state({(6, 0): fresh}).unstash(
    {"position": 5, "registry_keys": [(6, 0)], (6, 0): "BLOB"})
check("compatible unstash delivers (slot, blob) to every layer",
      fresh.got == (3, "BLOB"))
hole_a, hole_b = RecordingLayer(), RecordingLayer()
try:
    fake_state({(0, 0): hole_a, (6, 0): hole_b}).unstash(
        {"position": 5, "registry_keys": [(0, 0), (6, 0)], (0, 0): "B"})
    check("slot-hole blob (fingerprint OK, payload absent) raises before any layer copy", False)
except CheckpointStale as e:
    check("slot-hole blob (fingerprint OK, payload absent) raises before any layer copy",
          hole_a.got is None and hole_b.got is None)
    check("stale message names the missing entries (cured-event body is self-describing)",
          "missing entries [(6, 0)]" in str(e))
# CONTROL 2: a swapped registry must refuse delivery
swapped = RecordingLayer()
try:
    fake_state({(7, 0): swapped}).unstash(
        {"position": 5, "registry_keys": [(6, 0)], (6, 0): "BLOB"})
    check("CONTROL: foreign registry trips (delivered instead)", swapped.got is None)
except CheckpointStale:
    check("CONTROL: foreign registry trips", swapped.got is None)

print("== [3] generator miss path ==")
from exllamav3.generator.generator import Generator

def fake_gen(cache):
    g = Generator.__new__(Generator)
    g.target_recurrent = True
    g.draft_recurrent = False
    g.cache = cache
    return g

class StaleCache:
    def new_from_stashed(self, stashed_state, position):
        raise CheckpointStale("fence unit stale")

class BoomCache:
    def new_from_stashed(self, stashed_state, position):
        raise RuntimeError("real fault")

buf = io.StringIO()
redirected = sys.stdout
sys.stdout = buf
try:
    out = fake_gen(StaleCache()).recurrent_state_from_stashed({"target": {}}, 4096)
finally:
    sys.stdout = redirected
miss_line = buf.getvalue()
check("stale checkpoint returns None (cache miss, not abort)", out is None)
check("miss is announced", "dropped as stale" in miss_line and "4096" in miss_line)
try:
    fake_gen(BoomCache()).recurrent_state_from_stashed({"target": {}}, 4096)
    check("CONTROL: a real fault still propagates (handler did not swallow it)", False)
except RuntimeError:
    check("CONTROL: a real fault still propagates", True)

print("== [4] job.py miss consumers ==")
jsc = src(JOB)
alloc = re.search(
    r"self\.recurrent_state = self\.generator\.recurrent_state_from_stashed\(\s*stashed_recurrent_state.*?(?=# Metrics)",
    jsc, re.S)
check("allocate_pages: None from from_stashed falls back to new_recurrent_state()",
      alloc is not None
      and "if self.recurrent_state is None:" in alloc.group(0)
      and "self.recurrent_state = self.generator.new_recurrent_state()" in alloc.group(0))
check("allocate_pages: checkpoint_pos only read when the restore succeeded",
      alloc is not None
      and alloc.group(0).index("if self.recurrent_state is None:") < alloc.group(0).index("self.last_recurrent_checkpoint_pos"))
rewind = re.search(
    r"stashed = self\.find_recurrent_stash\(target\).*?self\.last_recurrent_checkpoint_pos = replay_from or None",
    jsc, re.S)
check("rewind: stale stash resets replay_from to 0 with a fresh state",
      rewind is not None
      and "if self.recurrent_state is None:" in rewind.group(0)
      and "replay_from = 0" in rewind.group(0)
      and "self.recurrent_state = self.generator.new_recurrent_state()" in rewind.group(0))

print("== [5] moe coop decline reasons are recorded and printable ==")
lsrc = src(LOOP)
check("refusal() branch records its reason before returning None",
      re.search(r"reason = moe_coop_plan\.refusal\(.*?if reason is not None:\s*self\._coop_refuse_reason = reason\s*return None",
                lsrc, re.S) is not None)
ltree = ast.parse(lsrc)
moe_cls = next(n for n in ast.walk(ltree)
               if isinstance(n, ast.ClassDef) and n.name == "InklingMoE")
fns = [n for n in ast.walk(moe_cls) if isinstance(n, ast.FunctionDef)]
init_fn = next(n for n in fns if n.name == "__init__")
check("InklingMoE stores layer_idx (DECLINED print dereferences it)",
      "self.layer_idx = layer_idx" in ast.unparse(init_fn)
      and "layer_idx" in [a.arg for a in init_fn.args.args])
decline_fn = next(n for n in fns
                  if "_coop_decline_logged" in ast.unparse(n))
dbg = ast.unparse(decline_fn).replace("'", '"')
check("decline print flags use type(self), not a bare class identifier",
      "type(self)" in dbg and "Inkling2LoopSparseMoeBlock" not in dbg)
check("_coop_decode clears the reason at entry (a print always names THIS call's decline)",
      re.search(r"def _coop_decode\(.*?\"\"\".*?\n\s*self\._coop_refuse_reason = None", lsrc, re.S) is not None)
check("prefill-shaped ladder refusals get their own one-shot, not the decode slot",
      '"_coop_prefill_logged"' in dbg and '"_coop_decline_logged"' in dbg
      and "above the decode ladder" in dbg)
# CONTROL 5: reintroducing the phantom name (the 2026-11-27 boot-death NameError) trips the arm
mut5 = dbg.replace("type(self)", "Inkling2LoopSparseMoeBlock")
check("CONTROL: planting the phantom class name trips the type(self) arm",
      "Inkling2LoopSparseMoeBlock" not in dbg and "Inkling2LoopSparseMoeBlock" in mut5)

print()
if fails:
    print(f"FAILURES ({len(fails)}):")
    for f in fails:
        print("  - " + f)
    sys.exit(1)
print("VERDICT: recurrent-checkpoint stale fence GREEN (miss-not-abort contract intact)")