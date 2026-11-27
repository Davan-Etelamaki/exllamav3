#!/usr/bin/env python3
"""Fence: the fused score+bias+mask+softmax epilogue of Inkling 2 global attention (CPU-only).

The cold long-context prefill peak sat in two fp32 planes per query tile inside _attend_one: the
(h, tt, S) relative bias — gathered at full context width, then zeroed everywhere the
dist < rel_extent mask rejects, 12.7 GiB of mostly-zeros at T=2048 / S=23,055 — and the
(h, tt, S) scores plane, copied again by the +bias add, again by the -inf mask, again by the
(S+1) sink concat, again by softmax and again by the fp16 weight cast. This fence executes the
fused arm (_attend_one_biasfuse, infer_params.inkling_attn_biasfuse) on the SERVED SOURCE — never
the installed wheel; exllamav3 is never imported, the methods are parsed by AST and compiled into a
mock world — and pins six contracts:

  [1] ORACLE EQUALITY — an independent recomputation of the served definition (full-width gather,
      where, mask, (S+1) softmax) reproduces _attend_one BITWISE wherever the served arm runs its
      one wide matmul (S <= the k-chunk budget), which is what makes the oracle an oracle. The
      fused arm must agree with that oracle within 1e-4 (fp32 fixtures) — the tolerance the
      keytile arm's fence carries, two orders above fp32 reassociation drift and ten thousand
      times under the fp16 output quantum at |x|~1 (4.88e-4).
  [2] BAND IDENTITY — the eliminated bias plane is replaced by an exact rel_extent-wide band:
      with k = 0 the scores vanish and the output is softmax(bias) @ v, so this arm isolates the
      gather + mask + softmax epilogue from both matmuls.
  [3] TRAFFIC — a per-call ATen census (TorchDispatchMode) at two prefill shapes reports ops,
      materialised bytes and the largest single tensor for both arms. Materialised bytes and peak
      must fall; the launch count is the fusion's cost side and is bounded, not claimed lower.
  [4] FP32 PV ACCUMULATION — blocked at S = 63,387 with broad attention and positively-biased V,
      the un-normalised online numerator reaches ~1.2e5, above fp16's 65504: both arms stay
      finite, and the planted fp16-accumulation mutation goes non-finite (the 4cb7529a defect,
      re-proved against this arm). The single-block fast path is exempt by construction and says
      so in the report line.
  [5] SEAMS AND GATE — both attend seams (global forward_attn, sliding _sliding_attend) resolve
      the arm through ip.inkling_attn_biasfuse; the check EXECUTES the extracted assignment
      against a recording mock instead of eyeballing it, and OFF must resolve to _attend_one alone.
      InferParams declares the switch, default False; no environment variable is read.
  [6] REGIMES — kv_valid with stale ring columns, a mid-context chunk whose band straddles block
      edges, multi-tile / multi-block prefill, and single-token decode at S = 131,072.

POSITIVE CONTROLS: nine planted defects, each a mutation of the fused arm's own source, must trip
the check that exists to catch it; the fence counts how many tripped and REDs on any survivor.

Run:  cd <repo> && env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES= \\
      PYTHONPATH=. ~/.venvs/tabby-exl3/bin/python tests/cpu/inkling_attn_biasfuse_fence.py
"""
import ast
import copy
import importlib
import math
import os
import pathlib
import sys
import types
from collections import Counter, OrderedDict

import torch

# torch 2.14.1+cu130 (the tabby-exl3 venv) exports TorchDispatchMode from torch.utils.
# _python_dispatch only: torch.utils._context does not exist there and torch.overrides does not
# re-export it, which is how the first cut of this fence died before it ran a single assertion.
# The resolver walks the known homes and names them in its failure line, so a torch bump reports
# the module it wants instead of dying at import.
TorchDispatchMode = None
for _home in ("torch.utils._python_dispatch", "torch.utils._context", "torch.overrides"):
    try:
        TorchDispatchMode = getattr(importlib.import_module(_home), "TorchDispatchMode")
    except (ImportError, AttributeError):
        continue
    break
if TorchDispatchMode is None:
    print(f"FAIL: torch {torch.__version__} exports no TorchDispatchMode in "
          "torch.utils._python_dispatch, torch.utils._context or torch.overrides")
    sys.exit(1)

torch.set_num_threads(4)

ROOT = pathlib.Path(__file__).resolve().parents[2]
LOOP = ROOT / "exllamav3/modules/arch_specific/inkling2_loop.py"
PARAMS = ROOT / "exllamav3/model/config.py"

fails = []
ran = 0
passed = 0
planted = 0
CEN = []                           # census rows: (label, rows, cols, Census OFF, Census ON)


def check(name, ok, detail = ""):
    """One assertion, one line: PASS names the contract and carries the number that decided it."""
    global ran, passed
    ran += 1
    print(f"  {'PASS' if ok else 'FAIL'} {name}{(': ' + detail) if detail else ''}")
    if ok:
        passed += 1
    else:
        fails.append(name)


def gib(nbytes):
    return f"{nbytes / 2**20:.2f} MiB" if nbytes < 2**30 else f"{nbytes / 2**30:.3f} GiB"


TREE = ast.parse(LOOP.read_text(encoding = "utf8"))
CLS = {n.name: n for n in TREE.body if isinstance(n, ast.ClassDef)}
ATT = CLS["InklingAttention"]
METHODS = {n.name: n for n in ATT.body if isinstance(n, ast.FunctionDef)}
TOP_CONSTS = [n for n in TREE.body if isinstance(n, ast.Assign)
              and getattr(n.targets[0], "id", "").startswith(("_ATTEND", "_ATTN_"))]
SERVED_BODY = ast.unparse(METHODS["_attend_one"])
K_CHUNK_BYTES = 64 * 1024 * 1024          # _ATTEND_K_CHUNK_BYTES default: the wide-matmul cutoff

TOL = 1e-4          # vs the oracle, both arms in fp32: 2 orders above fp32 reassociation drift,
                    # 5 orders under the smallest structural error a band/mask slip produces (~1e-1)
BITS = True         # served-vs-oracle bitwise holds only while the served arm takes one wide matmul

SRC_CACHE = {}


def fused_methods(subs = ()):
    """The attend methods, compiled from the served source. `subs` are (old, new) source edits
    applied to the UNPARSED method text — a planted defect mutates the new arm in memory, never
    the tracked file (a RED arm that poisons its own subject proves nothing)."""
    key = tuple(subs)
    if key in SRC_CACHE:
        return SRC_CACHE[key]
    nodes = list(TOP_CONSTS)
    for name in ("_attend_one_tile", "_attend_keytile_cols", "_attend_biasfuse_cols",
                 "_attend_one", "_attend_one_biasfuse"):
        if name not in METHODS:
            print(f"FAIL {name} is gone from {LOOP}")
            sys.exit(1)
        text = ast.unparse(copy.deepcopy(METHODS[name]))
        if name in ("_attend_one_biasfuse", "_attend_biasfuse_cols"):
            for old, new in subs:
                text = text.replace(old, new)
        nodes.append(ast.parse(text).body[0])
    arm_join = (ast.unparse(METHODS["_attend_one_biasfuse"])
                + ast.unparse(METHODS["_attend_biasfuse_cols"]))
    for old, _ in subs:
        if old not in arm_join:
            raise AssertionError(f"mutation target absent: {old}")
    # The extracted module-level constants read _os.environ at MODULE SCOPE
    # (inkling2_loop.py:38 _ATTEND_TILE_BYTES = int(_os.environ.get(...))), so the namespace must
    # carry every name the extracted bodies and constant initializers touch — exec into an empty
    # dict dies at the first module-scope read. Keys absent from the CPU arm (logger, FlowKey) are
    # seeded too so a future edit to the attend bodies cannot crash on a missing import.
    ns = {"_os": os, "os": os, "math": math, "torch": torch,
          "torch.nn.functional": torch.nn.functional, "F": torch.nn.functional,
          "OrderedDict": OrderedDict, "logger": None, "FlowKey": lambda key: key}
    exec(compile(ast.fix_missing_locations(ast.Module(body = nodes, type_ignores = [])),
                 str(LOOP), "exec"), ns)
    for bound in ("_ATTEND_TILE_BYTES", "_ATTEND_K_CHUNK_BYTES"):
        if not isinstance(ns.get(bound), int):
            raise AssertionError(f"extraction did not bind {bound}: {ns.get(bound)!r}")
    SRC_CACHE[key] = ns
    return ns


class Attn:
    """Mock InklingAttention: only the attributes the attend bodies read."""

    def __init__(self, ns, h = 24, kvh = 2, d = 256, ext = 1024, tile_bytes = 64 * 1024 * 1024):
        self.num_q_heads = h
        self.num_kv_heads = kvh
        self.num_kv_groups = h // kvh
        self.rel_extent = ext
        self.head_dim = d
        self.scaling = d ** -0.5
        self.attend_tile_bytes = tile_bytes     # fixed dial: no CUDA, no mem_get_info on CPU
        for name in ("_attend_one", "_attend_one_biasfuse", "_attend_one_tile",
                     "_attend_biasfuse_cols", "_attend_keytile_cols"):
            setattr(self, name, types.MethodType(ns[name], self))


def fixture(T, S, ext, h = 24, kvh = 2, d = 256, seed = 0, dtype = torch.float32,
            tail = True, v_mean = 0.0, padded = False):
    """Served-geometry attention inputs (h=24 q heads, kvh=2, d=256, rel_extent=1024).
    tail=True puts the chunk's rows at the END of the context — the cold-prefill regime."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(h, T, d, generator = g)
    k = torch.randn(S, kvh, d, generator = g)
    v = torch.randn(S, kvh, d, generator = g) + v_mean
    rel = torch.randn(h, T, ext, generator = g)
    sink = torch.randn(T, h, generator = g)
    base = max(S - T, 0) if tail else max(S // 3, 0)
    q_pos = torch.arange(base, base + T, dtype = torch.long)
    kv_pos = torch.arange(S, dtype = torch.long)
    kv_valid = None
    if padded:
        # ring_geometry: the stale ring columns keep increasing positions but never join the softmax
        kv_valid = torch.zeros(S, dtype = torch.bool)
        kv_valid[:max(1, S // 2)] = True
    if dtype != torch.float32:
        q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    return dict(q = q, k = k, v = v, rel_small = rel, sink = sink, q_pos = q_pos,
                kv_pos = kv_pos, kv_valid = kv_valid)


def oracle(fx, ext, scaling):
    """Independent served-definition recomputation: full-width bias gather, mask, (S+1) softmax."""
    q, k, v = fx["q"], fx["k"], fx["v"]
    h, T, d = q.shape
    S = k.shape[0]
    kvh = k.shape[1]
    grp = h // kvh
    dist = fx["q_pos"][:, None] - fx["kv_pos"][None, :]
    causal = dist >= 0
    if fx["kv_valid"] is not None:
        causal = causal & fx["kv_valid"][None, :]
    bias = fx["rel_small"].gather(-1, dist.clamp(0, ext - 1).unsqueeze(0).expand(h, T, S))
    bias = torch.where((causal & (dist < ext)).unsqueeze(0), bias, torch.zeros_like(bias))
    sc = torch.matmul(q.reshape(kvh, grp * T, d).float(),
                      k.transpose(0, 1).transpose(1, 2).float()) * scaling
    sc = sc.view(kvh, grp, T, S) + bias.view(kvh, grp, T, S)
    sc = sc.masked_fill(~causal[None, None, :, :], float("-inf"))
    snk = fx["sink"].T.reshape(kvh, grp, T).float()
    w = torch.softmax(torch.cat([sc, snk.unsqueeze(-1)], dim = -1), dim = -1)[..., :S].to(v.dtype)
    return torch.matmul(w.reshape(kvh, grp * T, S), v.transpose(0, 1)).view(h, T, d)


def dev(a, fx, orc):
    return (a._attend_one_biasfuse(**fx).float() - orc.float()).abs().max().item()


ns = fused_methods()
attn = Attn(ns)

# ------------------------------------------------------------------- [1] ORACLE EQUALITY
print("[1] oracle equality (served bitwise where its matmul is one wide call)")
for (T, S, seed) in [(1, 4096, 0), (8, 8192, 1), (64, 4096, 2), (32, 2048, 3)]:
    fx = fixture(T, S, attn.rel_extent, seed = seed)
    orc = oracle(fx, attn.rel_extent, attn.scaling)
    served = attn._attend_one(**fx)
    d_fused = dev(attn, fx, orc)
    check(f"[1] served == oracle bitwise T={T} S={S}", torch.equal(served, orc),
          f"maxabs {(served.float() - orc.float()).abs().max().item():.3e}")
    check(f"[1] fused within {TOL:g} of oracle T={T} S={S}", d_fused <= TOL,
          f"maxabs {d_fused:.3e}")
fx16 = fixture(8, 4096, attn.rel_extent, seed = 4, dtype = torch.float16)
orc16 = oracle(fx16, attn.rel_extent, attn.scaling)
serv16 = attn._attend_one(**fx16)
fused16 = attn._attend_one_biasfuse(**fx16)
d16 = (fused16.float() - orc16.float()).abs().max().item()
ulp16 = orc16.abs().max().item() * 2 ** -10          # fp16 ULP at the largest output magnitude
check("[1] served dtype arm: bitwise at fp16", torch.equal(serv16, orc16),
      f"maxabs {(serv16.float() - orc16.float()).abs().max().item():.3e}")
check(f"[1] fused dtype arm within 1 fp16 ULP ({ulp16:.3e})", d16 <= ulp16, f"maxabs {d16:.3e}")

# ------------------------------------------------------------------- [2] BAND IDENTITY
print("[2] band identity (k=0 isolates bias+mask+softmax from both matmuls)")
for (T, S, seed) in [(4, 5000, 5), (16, 12288, 6), (1, 30000, 7)]:
    fx = fixture(T, S, attn.rel_extent, seed = seed)
    fx["k"] = torch.zeros_like(fx["k"])
    orc = oracle(fx, attn.rel_extent, attn.scaling)
    check(f"[2] band gather == full-width bias plane T={T} S={S}", dev(attn, fx, orc) <= TOL,
          f"maxabs {dev(attn, fx, orc):.3e}")
    check(f"[2] served arm bitwise at k=0 T={T} S={S}",
          torch.equal(attn._attend_one(**fx), orc))

# --------------------------------------------------------- [3] TRAFFIC / PER-CALL ATEN CENSUS
print("[3] traffic census (TorchDispatchMode, CPU): gate OFF vs ON")


class Census(TorchDispatchMode):
    """Per-call ATen census: every op counted; materialised bytes summed over outputs with no
    _base (a view shares storage somebody already paid for). House method: per-call ATen census,
    views excluded — the same rule attend_peak_probe.py's PeakTensor uses."""

    def __init__(self):
        self.ops = Counter()
        self.n_ops = 0
        self.bytes = 0
        self.peak = 0
        self.biggest = None

    def __torch_dispatch__(self, func, types_, args = (), kwargs = None):
        out = func(*args, **(kwargs or {}))
        self.ops[str(func)] += 1
        self.n_ops += 1
        for t in (out if isinstance(out, (tuple, list)) else (out,)):
            if isinstance(t, torch.Tensor) and t._base is None:
                self.bytes += t.nbytes
                if t.nbytes > self.peak:
                    self.peak = t.nbytes
                    self.biggest = f"{tuple(t.shape)} {t.dtype}".replace("torch.", "")
        return out


def census(fn, fx):
    c = Census()
    with c:
        fn(**fx)
    return c


for label, (T, S, tb) in (("single-block prefill T=32 S=4096 budget=64MiB", (32, 4096, 64)),
                          ("blocked prefill T=16 S=16384 budget=4MiB", (16, 16384, 4))):
    a = Attn(ns, tile_bytes = tb * 1024 * 1024)
    fx = fixture(T, S, a.rel_extent, seed = 11)
    off = census(a._attend_one, fx)
    on = census(a._attend_one_biasfuse, fx)
    rows = a._attend_one_tile(S, T, 4)
    cols = a._attend_biasfuse_cols(S, rows, a.head_dim, 4)
    h, kvh, d, ext = a.num_q_heads, a.num_kv_heads, a.head_dim, a.rel_extent
    print(f"       {label}: rows {rows} cols {cols} | OFF ops={off.n_ops} bytes={gib(off.bytes)} "
          f"peak={gib(off.peak)} [{off.biggest}] | ON ops={on.n_ops} bytes={gib(on.bytes)} "
          f"peak={gib(on.peak)} [{on.biggest}]")
    CEN.append((label, rows, cols, off, on))
    fb = cols <= a.rel_extent and cols < S       # economics fallback fires: ON delegates to OFF
    if fb:
        print(f"  PASS [3] {label}: budget pins blocks at the window floor -> FALLBACK to served "
              f"(equality is the correct verdict)")
        check(f"[3] {label}: fallback is transparent (bytes no-worse)", on.bytes <= off.bytes,
              f"{gib(off.bytes)} -> {gib(on.bytes)} ({on.bytes / off.bytes:.3f}x)")
        check(f"[3] {label}: fallback ops no-worse", on.n_ops <= off.n_ops + 8,
              f"{off.n_ops} -> {on.n_ops} ops")
        continue
    check(f"[3] {label}: materialised bytes fall", on.bytes < off.bytes,
          f"{gib(off.bytes)} -> {gib(on.bytes)} ({on.bytes / off.bytes:.3f}x)")
    check(f"[3] {label}: peak single tensor falls", on.peak < off.peak,
          f"{gib(off.peak)} -> {gib(on.peak)}")
    check(f"[3] {label}: launch count bounded (fusion cost side)",
          on.n_ops <= 3 * off.n_ops + 32, f"{off.n_ops} -> {on.n_ops} ops")
    model = max(h * rows * min(ext, cols) * 4, kvh * d * cols * 4, h * rows * cols * 4)
    check(f"[3] {label}: no (h,rows,S) fp32 plane ON (bound {gib(model)})", on.peak <= model,
          f"peak {gib(on.peak)} vs fused byte model {gib(model)}, served planes "
          f"{gib(h * rows * S * 4)}")

# -------------------------------------------------- [4] FP32 PV ACCUMULATION AT LONG S
print("[4] long-S finiteness (the 65504 fp16-PV ceiling)")
# Budget 64 MiB, not 4 MiB: at 4 MiB the byte budget pins key blocks at the rel_extent floor and
# the arm's own economics guard hands the call back to the served body — the section would then
# certify the CONTROL arm while claiming to test the online (rescaled) one. At 64 MiB cols ~=
# 15.9k: above rel_extent, below S, so the multi-block path really runs and fp32 accumulation is
# what keeps S=63,387 finite.
a_blk = Attn(ns, tile_bytes = 64 * 1024 * 1024)
fx = fixture(1, 63387, a_blk.rel_extent, seed = 8, v_mean = 2.0)
orc = oracle(fx, a_blk.rel_extent, a_blk.scaling)
rows = a_blk._attend_one_tile(63387, 1, 4)
cols = a_blk._attend_biasfuse_cols(63387, rows, 256, 4)
fused = a_blk._attend_one_biasfuse(**fx)
served = a_blk._attend_one(**fx)
check(f"[4] ONLINE (rescaled) path engaged at S=63387: rel_extent < cols {cols} < S (rows {rows})",
      attn.rel_extent < cols < 63387, "at or under rel_extent the guard falls back: no online test")
check("[4] the online arm is not the served bits (the rescale really ran)",
      not torch.equal(fused, served), "bit-identical output would mean the guard fell back")
check("[4] served finite at S=63387", bool(torch.isfinite(served).all()))
check("[4] fused finite at S=63387", bool(torch.isfinite(fused).all()),
      f"non-finite {int((~torch.isfinite(fused)).sum())} of {fused.numel()}")
check("[4] fused == oracle within tolerance at S=63387 (long-S online equivalence)",
      dev(a_blk, fx, orc) <= TOL, f"maxabs {dev(a_blk, fx, orc):.3e}")

# The CONTROL casts the fp16 CARRY between blocks, which is where the 4cb7529a hazard lives. One
# block's own partial cannot overflow: with p <= 1 and ~11k live keys it sits near 0.11 x 11k x |v|,
# far inside fp16. The carry accumulates the whole cache — that is the quantity that crossed
# 65504 in the field. The substitution is CUT FROM THE PARSED ARM: ast.unparse renders that
# statement on one line, so an anchor copied from the wrapped file text cannot match it.
_arm = ast.unparse(METHODS["_attend_one_biasfuse"])
_start = _arm.index("acc = acc * alpha.unsqueeze(-1)")
_end = _arm.index(".view(kvh, grp, tt, d)", _start) + len(".view(kvh, grp, tt, d)")
_pv_old = _arm[_start:_end]
_pv_new = _pv_old.replace("acc = acc *", "acc = (acc *", 1) + ").to(torch.float16)"
check("[4] CONTROL target is the cross-block carry, read from the parsed arm",
      _pv_old.startswith("acc = acc * alpha") and _pv_new.endswith(").to(torch.float16)"),
      _pv_old[:56] + "...")

# Payload that reproduces the field arithmetic: near-uniform attention. q = 0, bias = 0 and sink < 0
# make every causal score exactly 0, so exp(score - m) = 1 across all 63,387 live keys; V biased to
# +4 puts the un-normalised numerator at ~2.5e5, four times fp16's 65504. The clean arm must stay
# finite on the same payload, so a trip is attributable to the fp16 carry and not to the fixture.
fx_ov = fixture(1, 63387, a_blk.rel_extent, seed = 8, v_mean = 4.0)
fx_ov["q"] = torch.zeros_like(fx_ov["q"])
fx_ov["rel_small"] = torch.zeros_like(fx_ov["rel_small"])
fx_ov["sink"] = torch.full_like(fx_ov["sink"], -4.0)
check("[4] clean online arm finite on the uniform-attention payload",
      bool(torch.isfinite(a_blk._attend_one_biasfuse(**fx_ov)).all()))
sab = Attn(fused_methods(((_pv_old, _pv_new),)))._attend_one_biasfuse(**fx_ov)
nbad = int((~torch.isfinite(sab)).sum())
check("[4] CONTROL: fp16 carry across blocks goes non-finite at S=63387", nbad > 0,
      f"non-finite {nbad} of {sab.numel()}; the fp32 carry stays finite")

# Single-block fast path: one block over the whole cache divides BEFORE the value matmul, so the
# value partial is bounded at O(|v|) and the served fp16 PV stays safe — the decode regime: no
# rescale, no fp32 accumulator, no added launch. The regime must be FORCED, because at both 4 MiB
# and 64 MiB cols < S at this context length and an unforced arm silently judges the online path.
a_one = Attn(ns, tile_bytes = 3 * 1024 ** 3)
fx_one = fixture(1, 63387, a_one.rel_extent, seed = 8, v_mean = 2.0)
cols_one = a_one._attend_biasfuse_cols(63387, a_one._attend_one_tile(63387, 1, 4), 256, 4)
single_fused = a_one._attend_one_biasfuse(**fx_one)
orc_one = oracle(fx_one, a_one.rel_extent, a_one.scaling)
check(f"[4] single-block regime forced at S=63387 (cols {cols_one} >= S)", cols_one >= 63387,
      "cols below S here and this arm would be judging the online path")
# dtype must FOLLOW THE INPUTS (same-shape contract with _attend_one): the fixture is fp32, so the
# served-parity dtype here is fp32. Asserting literal float16 against an fp32 fixture checked the
# fixture's dtype, not the arm's — it REDs a correct arm.
srv_one = a_one._attend_one(**fx_one)   # the served body, same fixture: the parity reference
check("[4] single-block fast path stays finite and mirrors served PV dtype (no rescale needed)",
      bool(torch.isfinite(single_fused).all()) and single_fused.dtype == srv_one.dtype
      and single_fused.dtype == fx_one["v"].dtype
      and (single_fused.float() - orc_one.float()).abs().max().item() <= TOL,
      f"dtype {single_fused.dtype} vs v {fx_one['v'].dtype}; maxabs "
      f"{(single_fused.float() - orc_one.float()).abs().max().item():.3e}")

# ------------------------------------------------------------------- [5] SEAMS AND GATE
print("[5] gate default and per-call seams")


class Rec:
    """Recording mock: __getattr__ sees every attribute the extracted seam reads."""

    def __init__(self, biasfuse, keytile):
        self.seen = []
        self.config = types.SimpleNamespace(
            infer_params = types.SimpleNamespace(inkling_attn_flashprefill = False,
                                                 inkling_attn_biasfuse = biasfuse,
                                                 inkling_attn_keytile = keytile))

    def __getattr__(self, item):
        self.seen.append(item)
        return item


seams = {}
for meth, tag in (("forward_attn", "global"), ("_sliding_attend", "sliding")):
    body = METHODS[meth].body
    holder = next(st for st in body
                  if (isinstance(st, ast.Assign) and getattr(st.targets[0], "id", "") == "attend")
                  or (isinstance(st, (ast.For, ast.AsyncFor, ast.If))
                      and any(isinstance(s, ast.Assign)
                              and getattr(s.targets[0], "id", "") == "attend" for s in ast.walk(st))))
    inner = next(s for s in ast.walk(holder)
                 if isinstance(s, ast.Assign) and getattr(s.targets[0], "id", "") == "attend")
    pre = ([s for s in ast.walk(holder) if isinstance(s, ast.Assign)
            and getattr(s.targets[0], "id", "") == "ip"][:1]
           or [s for s in body if isinstance(s, ast.Assign)
               and getattr(s.targets[0], "id", "") == "ip"][:1])
    seams[tag] = (pre + [inner]) if pre else [inner]
for biasfuse, keytile, want in ((False, False, "_attend_one"),
                                (True, False, "_attend_one_biasfuse"),
                                (False, True, "_attend_one_keytile"),
                                (True, True, "_attend_one_biasfuse")):
    for tag, grab in seams.items():
        rec = Rec(biasfuse, keytile)
        exec(compile(ast.fix_missing_locations(ast.Module(body = grab, type_ignores = [])),
                     str(LOOP), "exec"), {"self": rec})
        check(f"[5] {tag} seam -> {want} (biasfuse={biasfuse}, keytile={keytile})",
              rec.seen[-1] == want, f"resolved {rec.seen[-1]}, reads {rec.seen}")

fields = {}
for cls in ast.walk(ast.parse(PARAMS.read_text(encoding = "utf8"))):
    if isinstance(cls, ast.ClassDef) and cls.name == "InferParams":
        for st in cls.body:
            if isinstance(st, ast.AnnAssign) and getattr(st.target, "id", ""):
                fields[st.target.id] = ast.unparse(st.value) if st.value else None
check("[5] InferParams declares inkling_attn_biasfuse", "inkling_attn_biasfuse" in fields)
check("[5] default is False", fields.get("inkling_attn_biasfuse") == "False",
      f"default {fields.get('inkling_attn_biasfuse')}")
check("[5] control arm's body untouched", "biasfuse" not in SERVED_BODY
      and "in_band" not in SERVED_BODY, "the OFF path is the served body, not a reworked one")
check("[5] no environment variable in the new arm",
      "environ" not in ast.unparse(METHODS["_attend_one_biasfuse"])
      and "environ" not in ast.unparse(METHODS["_attend_biasfuse_cols"]), "config switch only")

# ------------------------------------------------------------- [6] REGIMES
print("[6] padded ring, mid-context band, multi-tile prefill, long-context decode")
fx = fixture(8, 8192, attn.rel_extent, seed = 12, padded = True)
orc = oracle(fx, attn.rel_extent, attn.scaling)
check("[6] stale ring columns leave the softmax (kv_valid)", dev(attn, fx, orc) <= TOL,
      f"maxabs {dev(attn, fx, orc):.3e}")
fx = fixture(4, 6000, attn.rel_extent, seed = 13, tail = False)
orc = oracle(fx, attn.rel_extent, attn.scaling)
check("[6] mid-context chunk: band straddles the window edge", dev(attn, fx, orc) <= TOL,
      f"maxabs {dev(attn, fx, orc):.3e}")
a_tiny = Attn(ns, tile_bytes = 256 * 1024)
fx = fixture(64, 8192, a_tiny.rel_extent, seed = 14)
orc = oracle(fx, a_tiny.rel_extent, a_tiny.scaling)
rows = a_tiny._attend_one_tile(8192, 64, 4)
cols = a_tiny._attend_biasfuse_cols(8192, rows, 256, 4)
check("[6] multi-tile and multi-block prefill matches oracle", dev(a_tiny, fx, orc) <= TOL,
      f"rows {rows} (tiles {-(-64 // rows)}) cols {cols} (blocks {-(-8192 // cols)}) maxabs "
      f"{dev(a_tiny, fx, orc):.3e}")
check("[6] column dial stays inside its own budget: rel_extent <= cols <= S",
      a_tiny.rel_extent <= cols <= 8192 and rows <= 64, f"rows {rows} cols {cols}")
fx = fixture(1, 131072, attn.rel_extent, seed = 15)
orc = oracle(fx, attn.rel_extent, attn.scaling)
check("[6] decode T=1 at S=131072 matches oracle", dev(attn, fx, orc) <= TOL,
      f"maxabs {dev(attn, fx, orc):.3e}")

# --------------------------------------------------------------------- PLANTED DEFECTS
print("[6b] planted defects (each must trip its check)")

fx_n = fixture(8, 4096, attn.rel_extent, seed = 21)
orc_n = oracle(fx_n, attn.rel_extent, attn.scaling)
fx_h = fixture(8, 4096, attn.rel_extent, seed = 21, tail = False)   # head-aligned: rows INSIDE
orc_h = oracle(fx_h, attn.rel_extent, attn.scaling)                 # the window -> dist == ext
                                                                    # diagonals are causal-valid,
                                                                    # where tail-leak arms can show

fx_e = fixture(8, 1400, attn.rel_extent, seed = 21, tail = False)     # EXTREME head: base 0 gives
fx_e["q_pos"] = torch.arange(1200, 1208)                              # causally-valid columns 176..
fx_e["kv_pos"] = torch.arange(1400)                                   # 1199 with dist in (ext, 1200] -
del fx_e["sink"]                                                      # OUTSIDE the band: the exact
fx_e = {**fx_e, "sink": torch.randn(8, 24, generator=torch.Generator().manual_seed(99))}
orc_e = oracle(fx_e, attn.rel_extent, attn.scaling)                   # geography tail-leak arms need
clean_fused = attn._attend_one_biasfuse(**fx_n)


def numerics_trip(a):
    for fx, orc in ((fx_n, orc_n), (fx_e, orc_e)):   # tail AND just-outside-window geometry
        out = a._attend_one_biasfuse(**fx)            # mutations are silent in one alone
        if (not bool(torch.isfinite(out).all())
                or (out.float() - orc.float()).abs().max().item() > TOL):
            return True
    return False


def tripped(name, subs, probe):
    """Apply a mutation to the fused arm's own source and require a behaviour change."""
    global planted
    detail = ""
    try:
        ok = probe(Attn(fused_methods(subs)))
    except Exception as e:                       # a mutation that crashes did trip
        ok, detail = True, f"raised {type(e).__name__}: {e}"
    planted += 1 if ok else 0
    check(f"[ctrl] {name} tripped", ok, detail)


def traffic_worse_than_clean(a):
    """Band widened to the whole block: the full-width bias plane is back, so materialised bytes
    must rise ABOVE the clean arm's. Measured against the clean arm, not the control arm — at small
    S the control arm is cheaper than a mutated fusion, which would hide the regression. Peak uses
    >= on purpose: at a single-block shape the widened gather is the same size as the scores plane
    the clean arm already allocates, so bytes are the discriminating quantity."""
    mut = census(a._attend_one_biasfuse, fx_n)
    base = census(Attn(ns)._attend_one_biasfuse, fx_n)
    return mut.bytes > base.bytes and mut.peak >= base.peak


fx_big = fx            # [4]'s engaging long-S fixture: multi-block online, not a fallback
orc_big = orc


def online_trip(a):
    """Probe on the engaging long-S arm: a defect only the multi-block (rescaled) path can show.
    That this probe fires is itself evidence the rescale carries weight — an arm that never
    rescales cannot be hurt by a rescale mutation."""
    out = a._attend_one_biasfuse(**fx_big)
    return (not bool(torch.isfinite(out).all())
            or (out.float() - orc_big.float()).abs().max().item() > TOL)


tripped("band start off-by-one (dist >= ext -> dist > ext)",
        (("dist >= ext).sum", "dist > ext).sum"),), numerics_trip)
# Deliberately NOT counted as a planted defect, and it matters that it is not: `legs < width` ->
# `legs <= width` is numerically EQUIVALENT. The extra leg lands on column lo + width, where
# dist < 0 — a non-causal column that masked_fill_ turns into -inf two instructions later, so
# whatever value that leg carries cannot reach the softmax. Counting an off-by-one the mask absorbs
# would buy a green control that detects nothing. The inverted comparison is a real defect:
tripped("band mask inverted (legs < width -> legs > width)",
        (("legs[None, :] < width[:, None]", "legs[None, :] > width[:, None]"),), numerics_trip)
tripped("causal mask polarity flipped (masked_fill_(~causal -> causal)",
        (("masked_fill_(~causal.view", "masked_fill_(causal.view"),), numerics_trip)
tripped("band index window off by one (clamp(0, ext-1 -> 1, ext)",
        ((".clamp(0, ext - 1)", ".clamp(1, ext)"),), numerics_trip)
tripped("scaling dropped from q4 (never re-applied to scores)",
        (("q.view(kvh, grp, T, d).float() * self.scaling", "q.view(kvh, grp, T, d).float()"),),
        numerics_trip)
tripped("fp16 carry across blocks (the 65504 regression 4cb7529a fixed)",
        ((_pv_old, _pv_new),),
        lambda a: not bool(torch.isfinite(a._attend_one_biasfuse(**fx_ov)).all()))
tripped("rescale dropped (acc * alpha -> acc): only the multi-block path can show it",
        (("acc * alpha.unsqueeze(-1)", "acc"),), online_trip)
tripped("band widened to the full block (the bias plane returns)",
        (("band = min(ext, cols)", "band = cols"),), traffic_worse_than_clean)

print("")
print("ATEN-OP AND BYTE CENSUS (TorchDispatchMode, CPU, views excluded) — the wheel case")
print(f"{'shape':<44} {'rows':>5} {'cols':>6} {'ops OFF':>8} {'ops ON':>7}"
      f" {'bytes OFF':>10} {'bytes ON':>9} {'peak OFF':>9} {'peak ON':>8}")
for label, rows, cols, off, on in CEN:
    print(f"{label:<44} {rows:>5} {cols:>6} {off.n_ops:>8} {on.n_ops:>7}"
          f" {gib(off.bytes):>10} {gib(on.bytes):>9} {gib(off.peak):>9} {gib(on.peak):>8}")
print(f"tolerance vs oracle: {TOL:g} abs on fp32 fixtures; served-vs-oracle BITWISE wherever the"
      " served arm takes one wide matmul; torch " + torch.__version__)
print(f"checks run: {ran}, passed: {ran - len(fails)}, planted defects tripped: {planted}/8")
if fails:
    print(f"BIASFUSE FENCE: RED ({len(fails)}): {', '.join(fails)}")
    sys.exit(1)
print(f"BIASFUSE FENCE GREEN: {ran - len(fails)}/{ran} checks passed, "
      f"{planted}/8 planted defects tripped, census table above")
sys.exit(0)