#!/usr/bin/env python3
"""Fence: the hand-written sm_89 flash-style PREFILL attention kernel
(infer_params.inkling_attn_flashprefill, default False).

The change is one CUDA translation unit (exllamav3_ext/inkling/fa_prefill.cu), its torch layer
(fa_prefill_torch.cpp), three pybind bindings, one InferParams field and one attention seam. None of
that is trustworthy from a compile alone, so the fence does two things:

  [1] IT BUILDS AND RUNS THE PRODUCTION TRANSLATION UNITS — fa_prefill.cu and fa_prefill_torch.cpp,
      the shipped sources, compiled by g++ against a CUDA/ATen shim (tests/cpu/fa_prefill_shim/)
      whose kernel launcher is a ucontext SIMT scheduler with fragment-accurate models of
      mma.sync.aligned.m16n8k8 (fp32 and fp16 accumulators), ldmatrix (incl. .trans) and cp.async.
      The judged object is therefore the real call sequence and the real per-lane arithmetic, not a
      mock of it: which device attributes the launcher queries, what it opts the kernel into, the
      grid/block/dynamic-smem/stages of the launch, the fp16 outputs the kernel writes, and what it
      refuses. Pins:
        · the plan for the SERVED geometry (h=24, kvh=8, d=128, rel_extent=1024 — the parent's pack
          read; 2,048-row chunk over a 63,387-key cache) and its shared-memory arithmetic;
        · every shape parameter moves its own derived value independently, so nothing is hard-coded
          in the sizing path (h→grid_y, kvh→grp, d→smem+k_steps, rel_extent→band_tiles,
          S→key_tiles, T→grid_x);
        · declines, never crashes: unsupported head dim, non-integral GQA, rows below one query tile,
          a cache below one key tile, non-positive rel_extent, pre-sm_80 (cp.async), smem above the
          device opt-in or the caller budget, a driver that refuses cudaFuncSetAttribute, and the
          torch layer's dtype / stride / shape refusals;
        · the launch carries exactly the planned grid, block, smem and stage count;
        · NUMERICS: the kernel's fp16 outputs against an independent float64 oracle on five shapes
          (served geometry, partial query/key tiles, the single-stage pipeline, head-dim 256, a
          sink-heavy fixture), with the bias band, the score path and the sink each isolated, plus
          the kernel's own non-finite census, which must read zero.
  [2] IT EXECUTES THE PYTHON SEAM from the served source (inkling2_loop.py parsed by AST, never
      imported as exllamav3) against a recording extension mock: gate OFF resolves no new method and
      touches the extension zero times; a ring-validity mask, a too-short pass, a kernel refusal, a
      missing reason binding, a non-finite census and an audit diff each land on the fused path with
      the INKLING-FLASHDECLINE marker; the bound is measured once per (T,S) SHAPE CLASS rather than
      once per layer lifetime; and the three bindings are pinned in bindings.cpp as required.

COMPILE-VERIFY ONLY / BOOT-PENDING, stated plainly: nvcc's acceptance of the PTX forms, the real
register allocation and occupancy, and any rate on hardware are NOT established here — the wheel
build and the dot26 A/B decide those. What this fence does prove is the arithmetic, the geometry,
the refusals and the gate.

Positive controls: eight planted defects, each a mutation of the shipped source, must trip the check
that exists to catch it; the fence REDs on any survivor. Two of them are defects this kernel was
actually born with and had to be fixed for the oracle to agree: a key tile that was never primed,
and a sink term counted once per lane of the row-group.

Run:  cd <repo> && env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES= \
      PYTHONPATH=. ~/.venvs/tabby-exl3/bin/python tests/cpu/inkling_fa_prefill_fence.py
"""
import ast
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch          # CPU tensors for the seam harness; CUDA_VISIBLE_DEVICES= keeps it off the card

ROOT = pathlib.Path(__file__).resolve().parents[2]
INKL = ROOT / "exllamav3/exllamav3_ext/inkling"
SHIM = ROOT / "tests/cpu/fa_prefill_shim"
LOOP = ROOT / "exllamav3/modules/arch_specific/inkling2_loop.py"
PARAMS = ROOT / "exllamav3/model/config.py"
BINDINGS = ROOT / "exllamav3/exllamav3_ext/bindings.cpp"
KERNEL_CU = INKL / "fa_prefill.cu"
KERNEL_CUH = INKL / "fa_prefill.cuh"
TORCH_CPP = INKL / "fa_prefill_torch.cpp"
FU = ["fa_prefill.cu", "fa_prims.cuh", "fa_prefill.cuh", "fa_prefill_torch.cpp",
      "fa_prefill_torch.cuh"]
SHIM_FILES = ["cuda_runtime.h", "shim.h", "fa_emu.cpp", "api_driver.cpp", "ATen/Tensor.h",
              "ATen/cuda/CUDAContext.h"]

# The served attention geometry, from the parent's read of the served pack's config.json
# (24 q heads / 8 kv heads / head_dim 128 / rel_extent 1024 / 24 layers, 20-local + 4-global), and
# the shapes a 63,387-token ingest presents to the four global layers.
SERVED = dict(h=24, kvh=8, d=128, ext=1024)
ROWS, CACHE = 2048, 63387
TOL = 2e-3                       # fp16 output storage puts the floor near 6e-4
SCRATCH = 2 * 4 * 4              # FA_SCRATCH_BYTES in fa_prefill.cuh: per-warp position extremes

fails = []
ran = 0
passed = 0


def check(name, ok, detail = ""):
    """One assertion, one line: PASS names the contract and carries the number that decided it."""
    global ran, passed
    ran += 1
    print(f"  {'PASS' if ok else 'FAIL'} {name}{(': ' + detail) if detail else ''}")
    if ok:
        passed += 1
    else:
        fails.append(name)


def prepare_tree(tmp, muts = ()):
    """Mirror the repo layout under tmp so the shim's relative includes land on the copy: the judged
    sources are the shipped files, except where a mutation says otherwise. A mutation whose anchor
    is absent raises — a planted defect that misses its mark proves nothing."""
    src = tmp / "exllamav3/exllamav3_ext/inkling"
    shim = tmp / "tests/cpu/fa_prefill_shim"
    src.mkdir(parents = True, exist_ok = True)
    (shim / "ATen/cuda").mkdir(parents = True, exist_ok = True)
    texts = {}
    for f in FU:
        texts[f] = (INKL / f).read_text(encoding = "utf8")
    for old, new in muts:
        hit = [f for f in FU if old in texts[f]]
        if not hit:
            raise AssertionError(f"mutation anchor in no shipped file: {old[:70]}")
        texts[hit[0]] = texts[hit[0]].replace(old, new, 1)
    for f in FU:
        (src / f).write_text(texts[f], encoding = "utf8")
    for f in SHIM_FILES:
        (shim / f).parent.mkdir(parents = True, exist_ok = True)
        shutil.copy(SHIM / f, shim / f)
    return tmp


def build(root, out):
    inc = root / "exllamav3/exllamav3_ext/inkling"
    shim = root / "tests/cpu/fa_prefill_shim"
    cmd = ["g++", "-std=c++17", "-O1", "-x", "c++", "-I", str(shim), "-I", str(inc),
           str(inc / "fa_prefill.cu"), str(inc / "fa_prefill_torch.cpp"),
           str(shim / "fa_emu.cpp"), str(shim / "api_driver.cpp"), "-o", str(out)]
    return subprocess.run(cmd, capture_output = True, text = True)


def parse_result(line):
    """RESULT lines carry a free-text reason, so tokenize around it instead of across it."""
    rest = line[len("RESULT "):]
    out, reason = {}, None
    m = re.search(r"reason=(.*?)(?= nfin=|$)", rest)
    if m:
        reason = m.group(1).strip()
        rest = rest[:m.start()] + rest[m.end():]
    for kv in rest.split():
        k, _, v = kv.partition("=")
        out.setdefault(k, v)
    if reason is not None:
        out["reason"] = reason
    return out


def run_probe(binary, *args):
    r = subprocess.run([str(binary)] + [str(a) for a in args], capture_output = True,
                       text = True, timeout = 1800)
    res, calls, counts = {}, [], {}
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            res.update(parse_result(line))
        elif line.startswith("CALL "):
            calls.append(line[5:])
        elif line.startswith("CALL_COUNT "):
            for kv in line[11:].split():
                k, _, v = kv.partition("=")
                counts[k] = int(v)
    return res, calls, counts, r


def plan_for(binary, **over):
    shape = dict(h = SERVED["h"], kvh = SERVED["kvh"], d = SERVED["d"], ext = SERVED["ext"],
                 T = ROWS, S = CACHE)
    shape.update(over)
    args = ["plan"] + [f"{k}={v}" for k, v in shape.items() if k in ("h", "kvh", "d", "ext", "T", "S")]
    extra = [f"{k}={v}" for k, v in over.items() if k not in ("h", "kvh", "d", "ext", "T", "S")]
    res, _, _, _ = run_probe(binary, *(args + extra))
    return res


print("PART 1 — the production CUDA + torch translation units, built and run against the shim")
TMP = pathlib.Path(tempfile.mkdtemp(prefix = "fa-prefill-"))
BIN = TMP / "probe"
try:
    clean = prepare_tree(TMP / "clean")
    b = build(clean, BIN)
    check("[1a] fa_prefill.cu and fa_prefill_torch.cpp compile and link against the shim",
          b.returncode == 0, (b.stderr.strip().splitlines() or ["ok"])[-1][:180])
    if b.returncode != 0:
        print(f"FA-PREFILL FENCE RED: the shim build of the production sources failed:\n{b.stderr}")
        sys.exit(1)

    # ---- the served geometry: what the plan derives, and what it costs
    sp = plan_for(BIN)
    check("[1b] the served shape plans successfully", sp.get("ok") == "1",
          f"ok={sp.get('ok')} reason={sp.get('reason')}")
    check("[1c] grid is (ceil(rows/bm), heads) and the kv ratio is divided out at plan time",
          int(sp["grid_x"]) == (ROWS + int(sp["bm"]) - 1) // int(sp["bm"])
          and int(sp["grid_y"]) == SERVED["h"] and int(sp["grp"]) == SERVED["h"] // SERVED["kvh"],
          f"grid=({sp['grid_x']},{sp['grid_y']}) bm={sp['bm']} grp={sp['grp']}")
    check("[1d] tile walks come from the cache and the band, not from a literal",
          int(sp["key_tiles"]) == (CACHE + int(sp["bn"]) - 1) // int(sp["bn"])
          and int(sp["band_tiles"]) == (SERVED["ext"] + int(sp["bn"]) - 1) // int(sp["bn"]),
          f"key_tiles={sp['key_tiles']} band_tiles={sp['band_tiles']} bn={sp['bn']} "
          f"(of {sp['key_tiles']} tiles only {sp['band_tiles']} can overlap a row's band)")
    check("[1e] the shared-memory budget is the stated sum, and occupancy follows from it",
          int(sp["smem_q_upper"]) == (SERVED["d"] // 2) * int(sp["bm"]) * 2
          and int(sp["smem_stage"]) == 2 * (int(sp["bn"]) * SERVED["d"] * 2)
          and int(sp["smem_bytes"]) == int(sp["smem_q_upper"])
             + int(sp["stages"]) * int(sp["smem_stage"]) + SCRATCH
          and int(sp["blocks_per_sm"]) == max(1, int(sp["smem_per_sm"]) // int(sp["smem_bytes"])),
          f"q_upper={sp['smem_q_upper']} stage={sp['smem_stage']} stages={sp['stages']} "
          f"total={sp['smem_bytes']} B, per-SM {sp['smem_per_sm']} → {sp['blocks_per_sm']} blk/SM")
    check("[1f] the stage count is the largest that fits the device, within the caller budget",
          int(sp["stages"]) == 2
          and int(plan_for(BIN, budget = int(sp["smem_q_upper"]) + int(sp["smem_stage"])
                           + SCRATCH)["stages"]) == 1,
          f"served={sp['stages']} stages; one-stage budget → "
          f"{plan_for(BIN, budget = int(sp['smem_q_upper']) + int(sp['smem_stage']) + SCRATCH)['stages']}")

    # ---- every shape parameter must move its own derived value
    base = plan_for(BIN, h = 4, kvh = 2, d = 128, ext = 32, T = 64, S = 256)
    moves = dict(
        h=("grid_y", plan_for(BIN, h = 8, kvh = 2, d = 128, ext = 32, T = 64, S = 256)),
        kvh=("grp", plan_for(BIN, h = 4, kvh = 4, d = 128, ext = 32, T = 64, S = 256)),
        d=("smem_bytes", plan_for(BIN, h = 4, kvh = 2, d = 256, ext = 32, T = 64, S = 256)),
        ext=("band_tiles", plan_for(BIN, h = 4, kvh = 2, d = 128, ext = 64, T = 64, S = 256)),
        S=("key_tiles", plan_for(BIN, h = 4, kvh = 2, d = 128, ext = 32, T = 64, S = 512)),
        T=("grid_x", plan_for(BIN, h = 4, kvh = 2, d = 128, ext = 32, T = 128, S = 256)))
    bad = [(k, f, base[f], p[f]) for k, (f, p) in moves.items() if base[f] == p.get(f)]
    check("[1g] each shape parameter moves the derived value on its own axis", not bad,
          "; ".join(f"{k}: {f} {bv}->{pv}" for k, (f, p) in moves.items() for bv, pv in [(base[f], p.get(f))]))
    check("[1h] the head-dim term also moves the reduction depth",
          int(moves["d"][1]["k_steps"]) == 256 // 8 and int(base["k_steps"]) == 128 // 8,
          f"k_steps {base['k_steps']} → {moves['d'][1]['k_steps']} for d 128 → 256")

    # ---- declines: refused, never crashed
    DECLINES = [
        ("d=96 has no instantiation", dict(d = 96), "head_dim"),
        ("non-integral GQA", dict(h = 20, kvh = 3), "GQA ratio"),
        ("rows below one query tile (decode)", dict(T = 32), "below one 64-row tile"),
        ("cache below one key tile", dict(S = 8), "shorter than one"),
        ("non-positive rel_extent", dict(ext = 0), "extent is not positive"),
        ("pre-sm_80 has no cp.async", dict(sm = "75"), "sm_80"),
        ("caller budget below one stage", dict(budget = 1024), "caller budget")]
    for label, over, needle in DECLINES:
        p = plan_for(BIN, **over)
        r, _, cnt, _ = run_probe(BIN, "run", *[f"{k}={v}" for k, v in over.items()])
        check(f"[1i] declines: {label}", p.get("ok") == "0" and needle in p.get("reason", "")
              and r.get("launched") == "0" and cnt.get("launch", -1) == 0,
              f"reason={p.get('reason')!r} launched={r.get('launched')} launches={cnt.get('launch')}")
    small_card = plan_for(BIN, smem_optin = int(sp["smem_q_upper"]) + int(sp["smem_stage"])
                          + SCRATCH - 1)
    check("[1j] a card whose opt-in cannot hold one stage declines instead of launching garbage",
          small_card.get("ok") == "0" and "shared memory" in small_card.get("reason", ""),
          f"reason={small_card.get('reason')!r}")
    refused = run_probe(BIN, "run", "setattr_refuse")
    check("[1k] a driver that refuses the dynamic-smem opt-in is declined, clean exit",
          refused[0].get("launched") == "0"
          and "opt-in" in refused[0].get("reason", "") and refused[3].returncode == 0
          and refused[2].get("launch", -1) == 0,
          f"reason={refused[0].get('reason')!r} rc={refused[3].returncode}")

    # ---- the launch on the wire carries the plan
    _, calls, cnt, _ = run_probe(BIN, "run", f"h={SERVED['h']}", f"kvh={SERVED['kvh']}",
                                 f"d={SERVED['d']}", f"ext={SERVED['ext']}", "T=64", "S=1500")
    launch = next((c for c in calls if c.startswith("launch ")), "")
    want_sp = plan_for(BIN, h = SERVED["h"], kvh = SERVED["kvh"], d = SERVED["d"],
                       ext = SERVED["ext"], T = 64, S = 1500)
    check("[1l] the launch carries the planned grid, block, smem and stage count",
          f"grid=(1,{SERVED['h']})" in launch and f"block={want_sp['threads']}" in launch
          and f"smem={want_sp['smem_bytes']}" in launch and f"stages={want_sp['stages']}" in launch
          and f"grp={want_sp['grp']}" in launch, launch or "no launch recorded")
    check("[1m] the opt-in call carries the same byte count once, and the device is asked once",
          f"cudaFuncSetAttribute attr=8 amount={want_sp['smem_bytes']}" in calls
          and cnt.get("setattr") == 1
          and sum(1 for c in calls if c.startswith("cudaDeviceGetAttribute attr=97 ")) == 1,
          f"setattr={cnt.get('setattr')} attrs={cnt.get('attrs')}")

    # ---- the torch layer's own refusals
    for scene, needle in (("dtype_bad", "dtype"), ("stride_bad", "stride"), ("shape_bad", "output")):
        res, _, cnt, rr = run_probe(BIN, scene)
        check(f"[1n] the torch layer refuses {scene} without launching",
              res.get("launched") == "0" and needle in res.get("reason", "").lower()
              and cnt.get("launch", 0) == 0 and rr.returncode == 0,
              f"launched={res.get('launched')} reason={res.get('reason')!r}")
finally:
    pass

print("")
print("PART 2 — the kernel's numbers, graded against an independent float64 oracle")


def parse_dump(path):
    txt = pathlib.Path(path).read_text(encoding = "utf8").split("\n")
    sh = next(l for l in txt if l.startswith("SHAPE")).split()
    g = lambda k: int(sh[sh.index(k) + 1])
    vec = lambda n: np.array([float(x) for x in
                              next(l for l in txt if l.startswith(n + " ")).split()[1:]],
                             dtype = np.float64)
    return dict(h = g("h"), kvh = g("kvh"), d = g("d"), T = g("T"), S = g("S"), ext = g("ext"),
                scale = float(sh[-1]), nfin = int(next(l for l in txt if l.startswith("NFIN")).split()[1]),
                Q = vec("Q"), K = vec("K"), V = vec("V"), REL = vec("REL"), SINK = vec("SINK"),
                QP = vec("QPOS").astype(np.int64), KP = vec("KPOS").astype(np.int64), O = vec("O"))


def oracle_err(f, use_band = True, sink_mult = 1):
    """Independent float64 attention: fp32 scores + band, exact mask, softmax over scores ++ sink.

    use_band and sink_mult bend the EXPECTATION, not the kernel: with the band removed, or the sink
    term removed (sink_mult=0) or counted per lane as it was before this kernel was fixed
    (sink_mult=4), the gap must open past tolerance. That is how these checks know they exercise the
    band and the sink denominator instead of agreeing with a degenerate answer.
    """
    h, kvh, d, T, S, ext = f["h"], f["kvh"], f["d"], f["T"], f["S"], f["ext"]
    Q = f["Q"].reshape(h, T, d); K = f["K"].reshape(S, kvh, d); V = f["V"].reshape(S, kvh, d)
    REL = f["REL"].reshape(h, T, ext); SINK = f["SINK"].reshape(T, h); O = f["O"].reshape(h, T, d)
    grp = h // kvh
    worst = 0.0
    for hq in range(h):
        hk = hq // grp
        sc = (Q[hq] @ K[:, hk, :].T) * f["scale"]
        dist = f["QP"][:, None] - f["KP"][None, :]
        vis = dist >= 0
        if use_band:
            sc = sc + np.take_along_axis(REL[hq], np.clip(dist, 0, ext - 1), axis=1) * (vis & (dist < ext))
        sc = np.where(vis, sc, -np.inf)
        floor = np.nan_to_num(sc.max(axis = 1), nan = -1e30, posinf = -1e30, neginf = -1e30)
        m = np.maximum(floor, SINK[:, hq]) if sink_mult else floor
        e = np.exp(sc - m[:, None])
        den = e.sum(axis = 1) + sink_mult * np.exp(SINK[:, hq] - m)
        worst = max(worst, float(np.abs((e / den[:, None]) @ V[:, hk, :] - O[hq]).max()))
    return worst


ARMS = [
    ("base", dict()),
    ("served geometry", dict(h = SERVED["h"], kvh = SERVED["kvh"], ext = SERVED["ext"], T = 96,
                             S = 1200)),
    ("partial tiles", dict(T = 70, S = 257)),
    ("single-stage pipeline", dict(budget = 30000)),
    ("head_dim 256", dict(d = 256)),
    ("sink-heavy", dict(S = 128, ext = 16, seed = 7)),
    ("band isolated", dict(rel0 = 1)),
    ("scores isolated", dict(q0 = 1)),
    ("sink isolated", dict(sink0 = 1))]
NUM = {}
for label, over in ARMS:
    dump = TMP / f"dump_{label.replace(' ', '_')}.txt"
    res, _, cnt, r = run_probe(BIN, "run", f"dump={dump}",
                               *[f"{k}={v}" for k, v in over.items()])
    if res.get("launched") != "1":
        NUM[label] = None
        check(f"[2a] numerics: {label}", False, f"not launched: {res.get('reason')!r}")
        continue
    f = parse_dump(dump)
    err = oracle_err(f)
    NUM[label] = err
    check(f"[2a] numerics: {label} matches float64 within {TOL:g}",
          err <= TOL and res.get("nfin") == "0" and f["nfin"] == 0 and cnt.get("launch") == 1,
          f"max|oracle-kernel|={err:.3e} nonfinite={f['nfin']} launched={res.get('launched')}")
base_f = parse_dump(TMP / "dump_base.txt")
check("[2b] the band is really folded in: removing it from the expectation opens a gap",
      oracle_err(base_f, use_band = False) > TOL,
      f"graded {NUM['base']:.3e} vs no-band expectation {oracle_err(base_f, use_band = False):.3e}")
check("[2c] the sink is really in the denominator: dropping it opens a gap",
      oracle_err(base_f, sink_mult = 0) > TOL,
      f"graded {NUM['base']:.3e} vs sinkless expectation {oracle_err(base_f, sink_mult = 0):.3e}")
check("[2d] the grading can SEE a sink counted once per lane: sink_mult=4 opens a gap",
      oracle_err(base_f, sink_mult = 4) > TOL,
      f"four-counted expectation {oracle_err(base_f, sink_mult = 4):.3e} — the defect this kernel "
      f"shipped with before the seed moved to one lane")
check("[2e] every graded arm reports a zero non-finite census",
      all(parse_dump(TMP / f"dump_{l.replace(' ', '_')}.txt")["nfin"] == 0
          for l, v in NUM.items() if v is not None),
      f"{sum(1 for v in NUM.values() if v is not None)} arms, nonfinite=0 each")

print("")
print("PART 3 — the Python seam, executed from the served source")
TREE = ast.parse(LOOP.read_text(encoding = "utf8"))
ATT = next(n for n in ast.walk(TREE) if isinstance(n, ast.ClassDef) and n.name == "InklingAttention")
METH = {n.name: n for n in ATT.body if isinstance(n, ast.FunctionDef)}
CONST = {}
for stmt in TREE.body:
    if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name) \
            and stmt.targets[0].id.startswith("_FA_"):
        CONST[stmt.targets[0].id] = ast.literal_eval(stmt.value)
want = ("_FA_MIN_ROWS", "_FA_AUDIT_BOUND", "_FA_AUDIT_CLASSES", "_FA_SMEM_BUDGET")
check("[3a] the module carries the kernel contract constants", all(k in CONST for k in want),
      ", ".join(f"{k}={CONST.get(k)}" for k in want))
cuh = KERNEL_CUH.read_text(encoding = "utf8")
fa_bm = int(re.search(r"#define FA_BM\s+(\d+)", cuh).group(1))
check("[3b] the Python row floor mirrors the kernel's FA_BM/FA_MIN_ROWS",
      CONST.get("_FA_MIN_ROWS") == fa_bm and re.search(r"#define FA_MIN_ROWS\s+FA_BM", cuh) is not None,
      f"python={CONST.get('_FA_MIN_ROWS')} kernel FA_BM={fa_bm}")
check("[3c] the caller imposes no smem budget: fa_prefill_plan sizes against the device",
      CONST.get("_FA_SMEM_BUDGET") == 0, f"_FA_SMEM_BUDGET={CONST.get('_FA_SMEM_BUDGET')}")
for name in ("_attend_one_flashprefill", "_attend_fallback", "_fa_decline", "_fa_reason",
             "_fa_audit_class"):
    check(f"[3d] InklingAttention defines {name}", name in METH, "")
seams = {m: ast.unparse(METH[m]) for m in ("forward_attn", "_sliding_attend") if m in METH}
check("[3e] both seams resolve the arm behind its own gate, ahead of biasfuse/keytile",
      len(seams) == 2 and all("self._attend_one_flashprefill if ip.inkling_attn_flashprefill" in s
                              and s.index("inkling_attn_flashprefill") < s.index("inkling_attn_biasfuse")
                              for s in seams.values()), f"{sorted(seams)}")

exec("import torch", CONST)                    # CPU tensors only; exllamav3 is never imported
CALLS: list = []
METHODS = ("_attend_one_flashprefill", "_attend_fallback", "_fa_decline", "_fa_reason",
           "_fa_audit_class")


class RecExt:
    """Recording stand-in for the compiled extension. rc / nfin / has_reason / o_val are dials;
    o_val matches the fused-arm stub's answer by default so the audit's bound can PASS."""

    rc = 1
    nfin = 0
    has_reason = True
    o_val = 0.5

    def exl3_inkling_fa_prefill(self, q, k, v, rel, sink, qpos, kpos, o, nfin_t, scale, ext, budget):
        CALLS.append(("prefill", tuple(q.shape), int(k.shape[0]), float(scale), int(ext), int(budget)))
        o.fill_(self.o_val)
        nfin_t.fill_(self.nfin)
        return self.rc

    def exl3_inkling_fa_reason(self):
        if not self.has_reason:
            raise AttributeError("exl3_inkling_fa_reason")
        return "shim reason: cache shorter than one tile"


REC = RecExt()


class Log:
    lines: list = []

    def warning(self, msg):
        Log.lines.append(msg)

    info = warning


LOG = Log()


class Params:
    inkling_attn_flashprefill = False
    inkling_attn_biasfuse = False
    inkling_attn_keytile = False


class Cfg:
    def __init__(self):
        self.infer_params = Params()


RENAMES = (("self.config", "self.cfg"), ("self.layer_idx", "self.layer"),
           ("self.num_q_heads", "self.h"), ("self.head_dim", "self.d"))


def method_src(name, muts = ()):
    src = ast.unparse(METH[name])
    for old, new in RENAMES:
        src = src.replace(old, new)
    for old, new in muts:
        if old not in src:
            raise AssertionError(f"mutation anchor missing in {name}: {old[:60]}")
        src = src.replace(old, new, 1)
    return src


def make_attn(muts = ()):
    """A live stand-in carrying the shipped seam methods; a mutation raises if it misses its mark."""
    cls = type("Attn", (), {})
    ns = dict(CONST, ext = REC, logger = LOG)
    for name in METHODS:
        exec(method_src(name, [(o, n) for m, o, n in muts if m == name]), ns)
        setattr(cls, name, ns[name])

    def arm(which):
        def fn(self, q, k, v, rel, sink, qpos, kpos, kv_valid = None):
            CALLS.append(("fallback", which))
            return torch.full(q.shape, 0.5, dtype = torch.float32)
        return fn

    for which in ("_attend_one", "_attend_one_keytile", "_attend_one_biasfuse"):
        setattr(cls, which, arm(which))
    a = cls()
    a.cfg = Cfg()
    a.layer, a.h, a.d = 5, 2, 2
    a.scaling, a.rel_extent = 2 ** -0.5, 8
    a._fa_refused = False
    a._fa_audited = set()
    return a


def call(attn, T = 96, S = 1500, kv_valid = None):
    q = torch.zeros(2, T, 2, dtype = torch.float16)
    return attn._attend_one_flashprefill(
        q, torch.zeros(S, 1, 2, dtype = torch.float16), torch.zeros(S, 1, 2, dtype = torch.float16),
        torch.zeros(2, T, 8, dtype = torch.float32), torch.zeros(T, 2, dtype = torch.float32),
        torch.arange(T, dtype = torch.long), torch.arange(S, dtype = torch.long), kv_valid)


REC.rc, REC.nfin, REC.has_reason, REC.o_val = 1, 0, True, 0.5
attn = make_attn()
CALLS.clear(); Log.lines = []
out = call(attn)
check("[3f] a clean pass returns the kernel's fp16 output (not the fused path's fp32 answer)",
      out.dtype is torch.float16 and float(out[0, 0, 0]) == 0.5
      and CALLS[0] == ("prefill", (2, 96, 2), 1500, float(attn.scaling), 8, 0),
      f"{CALLS[0]} returned {out.dtype} calls={len(CALLS)}")
check("[3g] the first qualifying shape pays an audit against the fused path and prints its bound",
      any("INKLING-FLASHAUDIT" in l and "bound=" in l for l in Log.lines)
      and ("fallback", "_attend_one") in CALLS,
      next((l for l in Log.lines if "FLASHAUDIT" in l), "-")[:130])
CALLS.clear()
out = call(attn)
check("[3h] the same shape class pays no second audit and returns the kernel's answer",
      out.dtype is torch.float16 and ("fallback", "_attend_one") not in CALLS, f"calls={CALLS}")
CALLS.clear(); Log.lines.clear()
out = call(attn, T = 8192, S = 63387)
check("[3i] a different shape class re-audits: the bound is not inherited from a small pass",
      ("fallback", "_attend_one") in CALLS
      and any("INKLING-FLASHAUDIT" in l and "T=8192 S=63387" in l for l in Log.lines),
      f"calls={CALLS}")
exp_classes = {(t.bit_length(), s.bit_length()) for t, s in ((96, 1500), (8192, 63387))}
check("[3j] audit classes are (T,S) buckets, capped per layer",
      attn._fa_audited == exp_classes and len(attn._fa_audited) <= CONST["_FA_AUDIT_CLASSES"],
      f"classes={sorted(attn._fa_audited)} expected={sorted(exp_classes)} "
      f"cap={CONST['_FA_AUDIT_CLASSES']}")
REC.rc = 0
attn = make_attn(); CALLS.clear(); Log.lines = []
out = call(attn)
check("[3k] a kernel refusal falls back and prints INKLING-FLASHDECLINE with the reason",
      out.dtype is torch.float32 and float(out[0, 0, 0]) == 0.5
      and any("INKLING-FLASHDECLINE" in l and "shim reason" in l for l in Log.lines),
      next((l for l in Log.lines if "FLASHDECLINE" in l), "-")[:130])
CALLS.clear()
call(attn)
check("[3l] a refused layer stops asking: no further kernel calls, fused path only",
      not any(c[0] == "prefill" for c in CALLS) and any(c[0] == "fallback" for c in CALLS),
      f"calls={CALLS[:2]}")
REC.rc, REC.has_reason = 0, False
attn = make_attn(); CALLS.clear(); Log.lines = []
try:
    out = call(attn)
    tolerant = (out.dtype is torch.float32 and float(out[0, 0, 0]) == 0.5
                and any("reason unavailable" in l for l in Log.lines))
    detail = next((l for l in Log.lines if "FLASHDECLINE" in l), "-")[:130]
except Exception as err:
    tolerant, detail = False, f"the decline path raised {type(err).__name__}: {err}"
REC.has_reason = True
check("[3m] a missing reason binding cannot break the decline path", tolerant, detail)
REC.rc, REC.nfin = 1, 3
attn = make_attn(); CALLS.clear(); Log.lines = []
out = call(attn)
check("[3n] a non-finite census pulls the round off the kernel and out of service",
      out.dtype is torch.float32 and attn._fa_refused
      and any("INKLING-FLASHNONFINITE 3" in l for l in Log.lines),
      next((l for l in Log.lines if "NONFINITE" in l), "-")[:130])
REC.nfin = 0
attn = make_attn(); CALLS.clear(); Log.lines = []
out = call(attn, kv_valid = torch.ones(1500, dtype = torch.bool))
check("[3o] a ring-validity pass declines with its scope stated and uses the fused path",
      out.dtype is torch.float32 and attn._fa_refused
      and any("ring-validity mask" in l for l in Log.lines),
      next((l for l in Log.lines if "FLASHDECLINE" in l), "-")[:130])
attn = make_attn(); CALLS.clear(); Log.lines = []
out = call(attn, T = 8)
check("[3p] a pass below the row floor uses the fused path without burning a decline",
      out.dtype is torch.float32 and not attn._fa_refused
      and not any("FLASHDECLINE" in l for l in Log.lines), f"refused={attn._fa_refused}")
attn = make_attn(); attn.cfg.infer_params.inkling_attn_biasfuse = True
REC.rc = 0; CALLS.clear()
out = call(attn)
check("[3q] the fallback lands on the strongest shipped arm the gates select (biasfuse here)",
      ("fallback", "_attend_one_biasfuse") in CALLS, f"calls={CALLS[:2]}")
REC.rc, REC.nfin, REC.o_val = 1, 0, 0.25
attn = make_attn(); CALLS.clear(); Log.lines = []
out = call(attn)                    # kernel says 0.25, the fused path says 0.5 → diff 0.25 > bound
check("[3r] an audit diff above the bound returns the fused answer, not the kernel's",
      out.dtype is torch.float32 and float(out[0, 0, 0]) == 0.5
      and any("above the" in l for l in Log.lines), f"out={float(out[0,0,0])} "
      f"{next((l for l in Log.lines if 'above' in l), '-')[:110]}")
REC.o_val = 0.5

cfg_txt = PARAMS.read_text(encoding = "utf8")
cfg = ast.parse(cfg_txt)
pc = next(n for n in cfg.body if isinstance(n, ast.ClassDef) and n.name == "InferParams")
gate = [s for s in pc.body if isinstance(s, ast.AnnAssign)
        and getattr(s.target, "id", "") == "inkling_attn_flashprefill"]
check("[3s] InferParams declares inkling_attn_flashprefill: bool = False",
      len(gate) == 1 and ast.unparse(gate[0].annotation) == "bool"
      and ast.unparse(gate[0].value) == "False", ast.unparse(gate[0]) if gate else "absent")
head = ast.unparse(pc).split("def __init__")[0]
check("[3t] no environment variable or yaml key sneaks the switch on", "os.environ" not in head,
      "field defaults only")
bind_txt = BINDINGS.read_text(encoding = "utf8")
need = ("m.def(\"exl3_inkling_fa_plan\"", "m.def(\"exl3_inkling_fa_prefill\"",
        "m.def(\"exl3_inkling_fa_reason\"")
check("[3u] bindings.cpp DEFINES all three entry points, the reason lookup included (required)",
      all(n in bind_txt for n in need), f"{[n for n in need if n not in bind_txt] or 'all three'}")

print("")
print("PART 4 — positive controls: planted defects must trip")
planted = 0


def tripped(name, caught, detail = ""):
    global planted
    planted += 1 if caught else 0
    check(f"[ctrl] {name} tripped", caught, detail)


def nat_probe(muts, over = ()):
    """Rebuild the shipped sources with a planted defect and grade the base shape again."""
    tmp = pathlib.Path(tempfile.mkdtemp(prefix = "fa-mut-"))
    try:
        b = build(prepare_tree(tmp, muts), tmp / "probe")
        if b.returncode != 0:
            return True, f"would not compile: {(b.stderr.strip().splitlines() or [''])[-1][:90]}"
        res, _, _, _ = run_probe(tmp / "probe", "run", f"dump={tmp / 'd.txt'}", *over)
        if res.get("launched") != "1":
            return True, f"kernel declined: {res.get('reason')!r}"
        err = oracle_err(parse_dump(tmp / "d.txt"))
        return err > TOL, f"max|oracle-kernel|={err:.3e} against the {TOL:g} bound"
    finally:
        shutil.rmtree(tmp, ignore_errors = True)


caught, why = nat_probe((("    if (n_tiles > 0)", "    if (false)"),))
tripped("first key tile never primed (the bug this kernel shipped with)", caught, why)

caught, why = nat_probe((("lrow[hh] = (cq == 0) ? 1.0f : 0.0f;", "lrow[hh] = 1.0f;"),))
tripped("sink seeded into every lane of the row-group (the 4x denominator bug)", caught, why)

caught, why = nat_probe((("* A.log2e;", "* 0.0f;"),))
tripped("relative-bias gather multiplied to zero (band dropped)", caught, why)

caught, why = nat_probe((("p.smem_bytes = fa_smem_bytes(s.d, stages);\n    p.smem_per_sm = per_sm;",
                          "p.smem_bytes = fa_smem_bytes(s.d, stages) - 32;\n    p.smem_per_sm = per_sm;"),))
smem_sp = plan_for(BIN)
rebuilt = None
tmp = pathlib.Path(tempfile.mkdtemp(prefix = "fa-mut-"))
try:
    bb = build(prepare_tree(tmp, (("p.smem_bytes = fa_smem_bytes(s.d, stages);\n    p.smem_per_sm = per_sm;",
                                   "p.smem_bytes = fa_smem_bytes(s.d, stages) - 32;\n    p.smem_per_sm = per_sm;"),)),
               tmp / "probe")
    if bb.returncode == 0:
        rebuilt = plan_for(tmp / "probe", budget = 0, T = 64, S = 256)
finally:
    shutil.rmtree(tmp, ignore_errors = True)
tripped("smem accounting loses the scratch term",
        bool(rebuilt) and int(rebuilt["smem_bytes"]) != int(rebuilt["smem_q_upper"])
        + int(rebuilt["stages"]) * int(rebuilt["smem_stage"]) + SCRATCH,
        f"planned {rebuilt['smem_bytes'] if rebuilt else '-'} B vs "
        f"q_upper {smem_sp['smem_q_upper']} + stages*stage + {SCRATCH}")

tmp = pathlib.Path(tempfile.mkdtemp(prefix = "fa-mut-"))
try:
    bb = build(prepare_tree(tmp, (("if (sm_major < 8)", "if (false)"),)), tmp / "probe")
    p = plan_for(tmp / "probe", sm = "75") if bb.returncode == 0 else {"ok": "0", "reason": "no build"}
    r, _, cnt, _ = (run_probe(tmp / "probe", "run", "sm=75") if bb.returncode == 0
                    else ({"launched": "0"}, [], {"launch": 0}, None))
    caught = p.get("ok") == "1" or "sm_80" not in p.get("reason", "") or r.get("launched") != "0"
    why = f"mutant on sm_75: ok={p.get('ok')} reason={p.get('reason')!r} launched={r.get('launched')}"
finally:
    shutil.rmtree(tmp, ignore_errors = True)
tripped("pre-sm_80 gate deleted (cp.async issued on Pascal)", caught, why)


def py_probe(muts, dial, action):
    """Mutate a shipped seam method, run the scenario, and report whether the check still holds."""
    REC.rc, REC.nfin, REC.has_reason, REC.o_val = dial
    attn = make_attn(muts)
    CALLS.clear(); Log.lines = []
    try:
        return action(attn), ""
    except Exception as err:
        return True, f"raised {type(err).__name__}: {err}"
    finally:
        REC.rc, REC.nfin, REC.has_reason, REC.o_val = 1, 0, True, 0.5


def violation(kind):
    """Run the scenario on the mutant and report whether a SHIPPED invariant broke (True = caught)."""
    def probe(a):
        out = call(a)
        if kind == "refuse":                     # a declined pass must land on the fused path
            return not (out.dtype is torch.float32 and float(out[0, 0, 0]) == 0.5)
        if kind == "nonfinite":                  # a non-finite census must retire the arm
            call(a)
            return not a._fa_refused
        if kind == "audit":                      # a diff above the bound must return the fused answer
            return not (out.dtype is torch.float32 and float(out[0, 0, 0]) == 0.5)
        raise AssertionError(kind)
    return probe


DECINE = "if not rc:"
caught, why = py_probe([("_attend_one_flashprefill", DECINE, "if False:")], (0, 0, True, 0.5),
                       violation("refuse"))
tripped("kernel refusal ignored (a declined pass would serve garbage)", caught, why)

caught, why = py_probe([("_attend_one_flashprefill", "if nf:", "if False:")], (1, 3, True, 0.5),
                       violation("nonfinite"))
tripped("non-finite census ignored", caught, why)

caught, why = py_probe([("_attend_one_flashprefill", "if diff > _FA_AUDIT_BOUND:",
                         "if diff < _FA_AUDIT_BOUND:")], (1, 0, True, 0.25),
                       violation("audit"))
tripped("audit bound comparison inverted", caught, why)

caught, why = py_probe([("_fa_reason", "except Exception as err:", "except ZeroDivisionError as err:")],
                       (0, 0, False, 0.5), violation("refuse"))
tripped("reason lookup made strict (review point 1)", caught, why or "")

mut_cfg = cfg_txt.replace("inkling_attn_flashprefill: bool = False",
                          "inkling_attn_flashprefill: bool = True", 1)
mpc = next(n for n in ast.parse(mut_cfg).body if isinstance(n, ast.ClassDef)
           and n.name == "InferParams")
mgate = [s for s in mpc.body if isinstance(s, ast.AnnAssign)
         and getattr(s.target, "id", "") == "inkling_attn_flashprefill"]
tripped("InferParams default flipped to True",
        len(mgate) == 1 and ast.unparse(mgate[0].value) != "False",
        f"mutated default reads {ast.unparse(mgate[0]) if mgate else '-'}")

missing = "\n".join(l for l in bind_txt.splitlines()
                   if 'm.def("exl3_inkling_fa_reason"' not in l)
tripped("reason binding dropped from bindings.cpp",
        'm.def("exl3_inkling_fa_reason"' not in missing and "exl3_inkling_fa_reason" in bind_txt,
        f"{bind_txt.count('m.def(\"exl3_inkling_fa_reason\"')} → "
        f"{missing.count('m.def(\"exl3_inkling_fa_reason\"')} definitions")

shutil.rmtree(TMP, ignore_errors = True)
print("")
check("[end] the fence measured something", ran >= 40 and planted >= 6,
      f"{ran} checks, {planted} planted defects")
print(f"FA-PREFILL FENCE {'GREEN' if not fails else 'RED'}: {passed}/{ran} checks passed"
      + (f", failed: {fails}" if fails else ""))
sys.exit(1 if fails else 0)