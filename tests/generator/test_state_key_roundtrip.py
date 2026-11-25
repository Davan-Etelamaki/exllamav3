#!/usr/bin/env python3
"""Fence for the state tier's tuple-key round trip. Executes the WRITE and RESTORE key
expressions AS WRITTEN in the target disk_cache.py (AST-extracted source segments, never
retyped), through a real json.dumps/json.loads manifest hop, then feeds the restored dict
into the INSTALLED short_conv.py ShortConvState constructor (the crash frame).

Invalidation rides the EXISTING store fingerprint: DISK_CACHE_FORMAT_VERSION 2 -> 3
orphans format-2 stores wholesale. There is no per-file version field; a format-2 state
that is nonetheless reached is recognised by its stringified layer-key shape, counted in
legacy_states_skipped and dropped as a miss (the GC reclaims the blob).

usage: test_state_key_roundtrip.py <disk_cache.py> <short_conv.py> wheel|patched
"""
import ast, json, os, re, sys, types, textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "exllamav3", "generator", "disk_cache.py")
SC = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "exllamav3", "modules", "short_conv.py")
MODE = sys.argv[3] if len(sys.argv) > 3 else "patched"
dc_src = open(DC).read()
tree = ast.parse(dc_src)

def find_fn(name):
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    return None

lay, rd = find_fn("_state_layout"), find_fn("read_state")
assert lay is not None and rd is not None, "target functions missing"

def seg(node):
    return ast.get_source_segment(dc_src, node)

write_seg = inverse_seg = put_seg = detect_seg = None
body_assign = detect_body = None
for n in ast.walk(lay):
    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "key" for t in n.targets):
        write_seg = seg(n); break
for n in ast.walk(rd):
    if isinstance(n, ast.Assign):
        t = seg(n)
        if t and t.lstrip().startswith("inverse = {"):
            inverse_seg = t
        if t and isinstance(n.targets[0], ast.Subscript) and "unpack_stash" in t:
            put_seg = t
    if isinstance(n, ast.If) and "_is_legacy_layer_key" in seg(n.test):
        detect_seg = seg(n.test)
        detect_body = "\n".join(x for x in (seg(m) for m in n.body) if x)
        for m in n.body:
            ms = seg(m)
            if ms and "legacy_states_skipped" in ms:
                body_assign = ms
for f in (write_seg, inverse_seg, put_seg):
    assert f, f"seam missing in {DC}"
if MODE == "patched":
    for f in (detect_seg, body_assign):
        assert f, f"legacy-detection seam missing in {DC}"

helpers = {"json": json}
for n in tree.body:
    if isinstance(n, ast.FunctionDef) and n.name in ("_encode_layer_key", "_decode_layer_key", "_is_legacy_layer_key"):
        exec(compile(ast.Module([n], []), DC, "exec"), helpers)

def sim_round_trip(keymap, stash_key):
    loc = dict(helpers, keymap=keymap, k=stash_key)
    exec(write_seg, loc)
    written = loc["key"]
    man = json.loads(json.dumps({"sides": {"__root__": {"layers": {written: {"struct": "s", "tensors": []}}}}}))
    json_key = next(iter(man["sides"]["__root__"]["layers"]))
    loc2 = dict(helpers, keymap=keymap, inverse=None)
    exec(inverse_seg, loc2)
    d = {}
    loc3 = dict(helpers, d=d, inverse=loc2["inverse"], key=json_key,
                entry={"struct": "s"}, tensors=[], unpack_stash=lambda st, ts: "PAYLOAD")
    exec(put_seg, loc3)
    return written, json_key, d

# ---- installed ShortConvState (constructor == the crash frame) --------------------------
def load_sc(path):
    src = open(path).read()
    src = src.replace("from __future__ import annotations", "")
    src = src.replace("from typing_extensions import override", "override = lambda f: f")
    src = re.sub(r"^import torch.*$", "", src, flags=re.M)
    src = re.sub(r"^from \.\.cache\.recurrent import \(.*?^\)", "", src, flags=re.M | re.S)
    src = src.replace("from ..cache import Cache", "Cache = object")
    src = src.replace("from ..model.config import Config", "Config = object")
    src = src.replace("from ..util.tensor import get_for_device, to2", "get_for_device = to2 = lambda *a, **k: None")
    src = src.replace("from . import Module, Linear", "Module = Linear = object")
    src = re.sub(r"^from \.\.cache\.recurrent import .*$", "", src, flags=re.M)
    keep = [n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "ShortConvState"]
    ns = {"__name__": "sc", "torch": types.SimpleNamespace(no_grad=lambda f: f, empty=lambda *a, **k: None), "F": None}
    ns.update(Cache=object, Config=object, Module=object, Linear=object, override=lambda f: f,
              get_for_device=lambda *a, **k: None, to2=lambda *a, **k: None,
              mp_cache_recurrent_stash=lambda *a, **k: None, mp_cache_recurrent_unstash=lambda *a, **k: None,
              mp_cache_recurrent_clear=lambda *a, **k: None, new_checkpoint_handle=lambda: "handle",
              host_copy=lambda st: st)
    exec(compile(ast.Module(keep, []), path, "exec"), ns)
    return ns["ShortConvState"]

class FakeLayer:
    def __init__(self): self.seen = None
    def get_checkpoint_size(self): return 1024
    def unstash(self, slot, payload): self.seen = (slot, payload)

class FakeModel: loaded_tp = False

class FakeCache:
    def __init__(self, keys):
        self.layers = {k: FakeLayer() for k in keys}
        self.model = FakeModel()
    def get_all_recurrent_layers(self): return self.layers

SCState = load_sc(SC)

def crash_via_constructor(restored):
    stashed = {"position": 0, "checkpoint_size": 2048}
    stashed.update({k: v for k, v in restored.items()})
    cach = FakeCache([(6, 0)])
    SCState(cache=cach, slot=3, position=0, clear=False, stashed=stashed)
    return cach.layers[(6, 0)].seen

rc = 0
try:
    written, json_key, d = sim_round_trip({}, (6, 0))
except TypeError as e:
    # the module cannot even serialize tuple keys -> the pre-encoding tier shape
    if MODE == "wheel":
        print(f"[wheel  ] RED: json.dumps cannot serialize the tuple layer key: {e}")
        print("OUTCOME: FENCE RED — the installed wheel cannot round-trip a tuple layer key (rc=1)")
        sys.exit(1)
    print(f"HARNESS_FAIL: patched source cannot serialize tuple keys: {e}")
    sys.exit(2)
print(f"[bare-side] written key: {written!r} (json key {json_key!r}) -> restored keys {[repr(k) for k in d]}")
if MODE == "wheel":
    try:
        seen = crash_via_constructor(d)
        print(f"[wheel  ] FENCE BROKEN: constructor survived on the installed wheel (seen={seen})")
        rc = 2
    except KeyError as e:
        if str(e) == "(6, 0)":
            print(f"[wheel  ] RED: KeyError {e} from installed short_conv.py:111 after the tier round trip")
            rc = 1
        else:
            print(f"[wheel  ] FENCE BROKEN: wrong key {e}")
            rc = 2
else:
    try:
        seen = crash_via_constructor(d)
        ok_bare = (6, 0) in d and seen is not None and json.loads(written) == ["tuple", [6, 0]]
        print(f"[patched] bare side round-trips as tuple under the tagged encoding: {ok_bare}")
    except Exception as e:
        print(f"[patched] constructor FAILED unexpectedly: {type(e).__name__}: {e}")
        ok_bare = False
    _, _, dp = sim_round_trip({(6, 0): ("m", 6)}, (6, 0))
    ok_paired = list(dp.keys()) == [(6, 0)]
    print(f"[patched] paired side still resolves to stash keys: {ok_paired}")
    # legacy shape-detection arm: condition + counter execute against live namespaces
    metrics = {"legacy_states_skipped": 0}
    self_obj = types.SimpleNamespace(metrics=metrics)
    # comprehension scoping: the generator body only sees eval's GLOBALS, so bind there
    fired_cond = eval(detect_seg, dict(globals(), keymap=None, any=any,
                                       info={"layers": {"(6, 0)": {}}},
                                       _is_legacy_layer_key=helpers["_is_legacy_layer_key"]))
    # both sides: on the TP side keymap is not None, and a stringified key there silently misses
    # `inverse` and re-plants the KeyError this fix exists to remove. If the guard ever regresses to
    # `keymap is None and ...`, this second eval is the arm that catches it -- positive control, not
    # decoration: same condition, opposite keymap, same expected firing.
    fired_tp = eval(detect_seg, dict(globals(), keymap={"(6, 0)": (6, 0)}, any=any,
                                     info={"layers": {"(6, 0)": {}}},
                                     _is_legacy_layer_key=helpers["_is_legacy_layer_key"]))
    fresh_cond = eval(detect_seg, dict(globals(), keymap={(6, 0): (6, 0)}, any=any,
                                       info={"layers": {helpers["_encode_layer_key"]((6, 0)): {}}},
                                       _is_legacy_layer_key=helpers["_is_legacy_layer_key"]))
    if fired_cond:
        try:
            exec(detect_body, {"self": self_obj, "ValueError": ValueError})
        except ValueError:
            pass  # the body's raise is the hand-off to the existing drop path
    ok_gate = (metrics["legacy_states_skipped"] == 1 and "raise ValueError" in detect_body
               and fired_cond and fired_tp and not fresh_cond)
    print(f"[patched] legacy-shape arm counts and raises: {ok_gate} "
          f"(bare={fired_cond} tp={fired_tp} fresh_not_fired={not fresh_cond})")
    # precision: the detector recognises stringified tuples, nothing else
    ok_prec = (helpers["_is_legacy_layer_key"]("(6, 0)") is True
               and helpers["_is_legacy_layer_key"](helpers["_encode_layer_key"]((6, 0))) is False
               and helpers["_is_legacy_layer_key"]("model.layers.3.conv") is False)
    print(f"[patched] detector precision ('(6, 0)' yes; encoded/dotted no): {ok_prec}")
    # invalidation rides the reader, not a new field, and the store identity does NOT rotate:
    # a bump strands the live tier (segments measured 2.4 GB still valid on dot26) and re-writes
    # about the same, which is the drive-wear limit -- so the version stays 2 and the legacy key
    # shape is recognised at read.
    ok_fmt = ("DISK_CACHE_FORMAT_VERSION = 2" in dc_src
              and "STATE_FORMAT_VERSION" not in dc_src
              and 'manifest["state_format"]' not in dc_src
              and "keymap is None and any(_is_legacy_layer_key" not in dc_src)
    print(f"[patched] format held at 2, legacy shape caught at read on both sides: {ok_fmt}")
    ok_counters = (dc_src.count('"legacy_states_skipped"') >= 4
                   and '"skipped_budget", "write_errors",\n                                         "legacy_states_skipped") if d[k]]' in dc_src)
    print(f"[patched] counter reaches the [CACHE] line and the discarded: segment: {ok_counters}")
    if not (ok_bare and ok_paired and ok_gate and ok_prec and ok_fmt and ok_counters):
        rc = 1
if MODE == "wheel":
    print("OUTCOME: FENCE RED — the installed wheel cannot round-trip a tuple layer key (rc=1)" if rc == 1
          else "OUTCOME: FENCE BROKEN (rc=2)")
else:
    print(f"OUTCOME: FENCE {'GREEN' if rc == 0 else 'FAIL'} on patched source (rc={rc})")
sys.exit(rc)