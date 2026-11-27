#!/usr/bin/env python3
"""Fence: the persisting-L2 window for the decode row span (infer_params.inkling_attn_l2win).

The change is two CUDA calls and the decisions around them: cudaDeviceSetLimit(cudaLimitPersisting
L2CacheSize) carves a slice of L2 per device at context creation, and cudaStreamSetAttribute
(accessPolicyWindow) steers one stream's accesses over a computed VA range onto that slice, with
missProp=Streaming so the routed-expert weight traffic that streams through L2 is evicted first.
Neither call can be exercised on this box (no CUDA context may be created here, and L2 effects do
not reproduce off-card), so the fence does two things instead of one:

  [1] IT BUILDS AND RUNS THE PRODUCTION TRANSLATION UNIT — quant/exl3_l2win.cu is compiled with g++
      against a recording CUDA shim (tests/cpu/l2win_shim/) and driven scenario by scenario, so the
      JUDGED OBJECT IS THE REAL CALL SEQUENCE, not a mock of it: which calls happen, in what order,
      with what arguments, and what is refused. Pins:
        · gate OFF issues NEITHER cudaDeviceSetLimit NOR cudaStreamSetAttribute — and does not even
          query a device attribute (the armed check is the first statement of the carve path);
        · the carve clamps: 49.5 MiB on a 72 MB-L2 card, 32 MiB on a 32 MiB card, and a request
          above the ceiling clamps to the ceiling — refused never, crash never;
        · the window length is the served shapes times the rows, capped by the carve and by the
          device's max window size; each of h/kvh/d/rel_extent moves it independently (so nothing
          is hard-coded in the sizing path), and the bytes in the real cudaStreamSetAttribute call
          are exactly those derived bytes;
        · hitProp=Persisting, missProp=Streaming, hitRatio=1.0 on the wire;
        · a span crossing unallocated VA and a driver that answers cudaErrorInvalidValue each
          produce a DECLINED print, no window, an unchanged device limit and a clean exit;
        · one SetLimit per device, and one SetAttribute per (stream, base, length) — a repeat round
          costs nothing, a changed row count costs one.
  [2] IT EXECUTES THE PYTHON SEAM from the served source (inkling2_loop.py parsed by AST, never
      imported as exllamav3) against a recording extension mock: gate OFF must not touch the
      extension at all; gate ON must arm with the MODULE'S shapes and re-ask no more after a
      refusal. The two call sites (global forward_attn, sliding _sliding_attend) and the DevCtx
      wiring (one carve call in prepare_ctx, the carve cached per device) are pinned structurally.

COMPILE-VERIFY ONLY / BOOT-PENDING, stated plainly: nvcc is not installed on this box, so the CUDA
build of exl3_l2win.cu (real cuda_runtime.h, -lcuda-free driver calls, ROCm branch) is not compiled
here; the host shim proves the logic, not that nvcc accepts the file. The behavioural claim — that a
carved slice makes decode rows survive the expert traffic — is an on-card measurement (dot26 A/B)
and is NOT claimed by this fence. Off-card the shim is the only answer the two calls ever get.

Positive controls: six planted defects, each a mutation of the shipped source, must trip the check
that exists to catch it; the fence REDs on any survivor.

Run:  cd <repo> && env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES= \\
      PYTHONPATH=. ~/.venvs/tabby-exl3/bin/python tests/cpu/inkling_l2win_fence.py
"""
import ast
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
QUANT = ROOT / "exllamav3/exllamav3_ext/quant"
SHIM = ROOT / "tests/cpu/l2win_shim"
LOOP = ROOT / "exllamav3/modules/arch_specific/inkling2_loop.py"
PARAMS = ROOT / "exllamav3/model/config.py"
DEVCTX_CU = QUANT / "exl3_devctx.cu"
DEVCTX_CUH = QUANT / "exl3_devctx.cuh"
L2WIN_CU = QUANT / "exl3_l2win.cu"
L2WIN_CUH = QUANT / "exl3_l2win.cuh"

CEILING = 99 * 1024 * 1024 // 2           # 49.5 MiB, the Xolotl same-card carve
# The PACK's own numbers, read from /home/davan/models/inkling-2-mini-exl3/config.json on 2026-11-27
# (text_config: num_attention_heads=24, num_key_value_heads=8, head_dim=128) and confirmed by the
# on-card arm line `shapes=h24/kvh8/d128/rel1024`. The fixture earlier carried kvh=2/d=256, which no
# served Inkling 2 pack has — sizing checks derive from this dict, so the fix keeps them honest
# instead of merely re-numbering them.
SERVED = dict(h=24, kvh=8, d=128, rel=1024)

fails = []
ran = 0
passed = 0
planted = 0


def check(name, ok, detail = ""):
    """One assertion, one line: PASS names the contract and carries the number that decided it."""
    global ran, passed
    ran += 1
    print(f"  {'PASS' if ok else 'FAIL'} {name}{(': ' + detail) if detail else ''}")
    if ok:
        passed += 1
    else:
        fails.append(name)


def mib(nbytes):
    return f"{nbytes / 2 ** 20:.2f} MiB"


def build(sources, out):
    cmd = ["g++", "-std=c++17", "-Wall", "-x", "c++", "-I", str(SHIM), "-I", str(sources),
           str(sources / "exl3_l2win.cu"), str(SHIM / "shim.cpp"), str(SHIM / "l2win_driver.cpp"),
           "-o", str(out)]
    r = subprocess.run(cmd, capture_output = True, text = True)
    return r


def run_scenario(binary, *args):
    r = subprocess.run([str(binary)] + list(args), capture_output = True, text = True, timeout = 30)
    res, calls, declines = {}, [], []
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            for kv in line[7:].split():
                k, _, v = kv.partition("=")
                res.setdefault(k, v)
        elif line.startswith("CALL "):
            calls.append(line[5:])
        elif "DECLINED" in line:
            declines.append(line)
    return res, calls, declines, r.returncode


def prepare_tree(tmp, muts = ()):
    """Copy the CUDA TU aside so a mutation never edits the tracked file, under the same relative
    layout (ext/quant/… with ext/arch.cuh) that exl3_devctx.cuh's ../arch.cuh include expects."""
    src = tmp / "ext/quant"
    src.mkdir(parents = True)
    shutil.copy(ROOT / "exllamav3/exllamav3_ext/arch.cuh", tmp / "ext/arch.cuh")
    os.symlink(DEVCTX_CUH, src / "exl3_devctx.cuh")
    cuh = L2WIN_CUH.read_text(encoding = "utf8")
    cu = L2WIN_CU.read_text(encoding = "utf8")
    for old, new in muts:
        # A mutation lands on whichever file holds its anchor; an anchor in neither file is a
        # harness error, never a silent pass — a planted defect that misses its mark proves nothing.
        if old in cuh:
            cuh = cuh.replace(old, new, 1)
        elif old in cu:
            cu = cu.replace(old, new, 1)
        else:
            raise AssertionError(f"mutation anchor in neither file: {old[:70]}")
    (src / "exl3_l2win.cuh").write_text(cuh, encoding = "utf8")
    (src / "exl3_l2win.cu").write_text(cu, encoding = "utf8")
    return src


print("PART 1 — the production CUDA translation unit, built and run against a recording shim")
TMP = pathlib.Path(tempfile.mkdtemp(prefix = "l2win-fence-"))
try:
    clean_src = prepare_tree(TMP / "clean")
    binary = TMP / "l2win_probe"
    b = build(clean_src, binary)
    check("[1a] exl3_l2win.cu compiles host-side against the shim", b.returncode == 0,
          (b.stderr.strip().splitlines() or [""])[0][:200])
    if b.returncode != 0:
        print(f"L2WIN FENCE RED: the shim build of the CUDA site failed:\n{b.stderr}")
        sys.exit(1)

    # ---- [1] gate OFF: neither call, not even an attribute query
    res, calls, declines, rc = run_scenario(binary, "gate_off")
    check("[1b] gate OFF issues cudaDeviceSetLimit zero times", res.get("setlimit") == "0",
          f"setlimit={res.get('setlimit')}")
    check("[1c] gate OFF issues cudaStreamSetAttribute zero times", res.get("setattr") == "0",
          f"setattr={res.get('setattr')}")
    check("[1d] gate OFF queries no device attribute either", calls == [], f"{len(calls)} calls")
    check("[1e] gate OFF: carve 0, stream refused with a reason",
          res.get("carve") == "0" and res.get("stream_rc") == "-1" and res.get("refusal"),
          f"carve={res.get('carve')} rc={res.get('stream_rc')} refusal={res.get('refusal')!r}")

    # ---- sizing and clamp: what the real calls must carry
    cres, _, _, _ = run_scenario(binary, "clamp_pure")
    check("[1f] the carve ceiling is the Xolotl figure, 49.5 MiB",
          int(cres["ceiling"]) == CEILING, f"{cres['ceiling']} B == {mib(CEILING)}")
    sres, _, _, _ = run_scenario(binary, "sizing")
    row_b = int(sres["row_bytes_served"])
    check("[1g] row span is sized from the served shapes, not a literal",
          all(int(sres[k]) != row_b for k in
              ("row_bytes_h_only", "row_bytes_kvh_only", "row_bytes_d_only", "row_bytes_rel_only")),
          f"served={row_b} h={sres['row_bytes_h_only']} kvh={sres['row_bytes_kvh_only']} "
          f"d={sres['row_bytes_d_only']} rel={sres['row_bytes_rel_only']}")
    check("[1h] the rel_extent term is exactly the fp32 bias band (h*rel*4)",
          int(sres["row_bytes_rel_only"]) - row_b == SERVED["h"] * SERVED["rel"] * 4,
          f"delta={int(sres['row_bytes_rel_only']) - row_b} "
          f"expected={SERVED['h'] * SERVED['rel'] * 4}")
    check("[1i] window scales exactly with the row count",
          [int(sres[f"window_rows{n}"]) for n in (1, 2, 8)]
          == [row_b, 2 * row_b, 8 * row_b],
          f"{sres['window_rows1']}/{sres['window_rows2']}/{sres['window_rows8']}")
    check("[1j] window is capped by the carve and by the device's max window size",
          int(sres["window_capped"]) == 1 << 20 and int(sres["window_maxwin"]) == 4096,
          f"carve-capped={mib(int(sres['window_capped']))} maxwin={sres['window_maxwin']}")

    check("[1k] a carve above the ceiling clamps to the ceiling (no crash)",
          int(cres["clamp_64mib_on_72mib"]) == CEILING, mib(int(cres["clamp_64mib_on_72mib"])))
    check("[1l] a ceiling above the device's L2 clamps to the device",
          int(cres["clamp_ceiling_on_32mib"]) == 32 << 20, mib(int(cres["clamp_ceiling_on_32mib"])))
    check("[1m] a request below both is honored as asked",
          int(cres["clamp_8mib_on_72mib"]) == 8 << 20, mib(int(cres["clamp_8mib_on_72mib"])))

    # ---- gate ON on a 72 MB-L2 card: the real call sequence
    ares, acalls, _, arc = run_scenario(binary, "arm_ok")
    check("[1n] gate ON sets the device carve once, to the ceiling",
          ares.get("setlimit") == "1" and int(ares["carve"]) == CEILING,
          f"setlimit={ares.get('setlimit')} carve={mib(int(ares['carve']))}")
    check("[1o] the SetLimit carries cudaLimitPersistingL2CacheSize",
          any(c.startswith("cudaDeviceSetLimit limit=6") for c in acalls),
          next((c for c in acalls if c.startswith("cudaDeviceSetLimit")), "-"))
    first_window = next((c for c in acalls if c.startswith("cudaStreamSetAttribute")), "")
    check("[1p] the window bytes are the derived span for those rows",
          f"num_bytes={row_b * 4}" in first_window and int(ares["planned"]) == row_b * 4,
          first_window or "no window set")
    check("[1q] hitProp=Persisting, missProp=Streaming, hitRatio=1.0",
          "hitProp=2" in first_window and "missProp=1" in first_window and "hit=1.0" in first_window,
          first_window)
    check("[1r] a repeat round on the same stream+span re-issues nothing",
          ares.get("again") == "0", f"again={ares.get('again')} setattr={ares.get('setattr')}")
    check("[1s] a changed row count re-applies the window (one call)",
          ares.get("grew") == "1" and ares.get("setattr") == "2",
          f"grew={ares.get('grew')} setattr={ares.get('setattr')}")
    check("[1t] the wider window is exactly the derived bytes for the new rows",
          f"num_bytes={row_b * 8}" in [c for c in acalls if c.startswith("cudaStreamSetAttribute")][-1],
          [c for c in acalls if c.startswith("cudaStreamSetAttribute")][-1])

    # ---- the field sequence: 24 layers arm the same window on one device, then a round asks for it.
    #      Added after the on-card failure (dot26 boot 190495): 24 "armed" lines, 24 rounds refused
    #      with "device carve declined", zero windows set — the arm reset the carve per layer while
    #      the caller trusted the bytes its first arm had been handed.
    rres, rcalls, _, _ = run_scenario(binary, "rearm_layers")
    check("[1ac] 24 same-shape arms keep the carve and the round gets its window",
          int(rres["layers_zeroed"]) == 0 and int(rres["carve"]) == CEILING
          and int(rres["stream_rc"]) > 0 and rres.get("refusal", "") == "",
          f"layers_zeroed={rres.get('layers_zeroed')} rc={rres.get('stream_rc')} "
          f"refusal={rres.get('refusal')!r}")
    check("[1ad] those 24 arms cost exactly ONE cudaDeviceSetLimit (counted before the deliberate "
          "re-arm below)",
          rres.get("setlimit_after_24") == "1",
          f"setlimit_after_24={rres.get('setlimit_after_24')}")
    check("[1ae] a re-arm with different shapes re-carves instead of trusting the old window",
          int(rres["carve_rearm"]) == CEILING and rres.get("setlimit") == "2",
          f"carve_rearm={rres.get('carve_rearm')} setlimit={rres.get('setlimit')}")

    # ---- the stream handle the SERVED decode really carries: torch's default stream is handle 0.
    #      Every other scenario passes a fake non-zero handle, so this is the one that could have
    #      caught the on-card refusal (dot26 boot 199198: 24 armed, 24 "missing stream handle …").
    dres, dcalls, _, _ = run_scenario(binary, "default_stream")
    check("[1af] handle 0 (cudaStreamLegacy, what torch runs decode on) gets its window",
          dres.get("setattr") == "1" and int(dres["stream_rc"]) >= 0
          and dres.get("refusal", "") == "",
          f"rc={dres.get('stream_rc')} setattr={dres.get('setattr')} "
          f"refusal={dres.get('refusal')!r}")
    check("[1ag] the window set on handle 0 carries the derived bytes and the reuse split",
          (lambda w: "stream=(nil)" in w and f"num_bytes={row_b * 4}" in w
                     and "hitProp=2" in w and "missProp=1" in w)(
              next((c for c in dcalls if c.startswith("cudaStreamSetAttribute")), "")),
          next((c for c in dcalls if c.startswith("cudaStreamSetAttribute")), "no window"))

    # ---- a smaller card: clamp through the real call site
    sres2, scalls2, _, _ = run_scenario(binary, "arm_ok", f"l2={32 << 20}")
    check("[1u] on a 32 MiB-L2 card the carve clamps to the card, SetLimit still succeeds",
          int(sres2["carve"]) == 32 << 20 and sres2.get("setlimit") == "1"
          and "-> UNSUPPORTED_LIMIT" not in "".join(scalls2),
          f"carve={mib(int(sres2['carve']))}")

    # ---- refusals
    vres, vcalls, vdecl, vrc = run_scenario(binary, "va_crossing", f"span={1024}")
    check("[1v] a window crossing unallocated VA is refused, never applied",
          vres.get("stream_rc") == "-1" and vres.get("setattr") == "0",
          f"rc={vres.get('stream_rc')} setattr={vres.get('setattr')}")
    check("[1w] the refusal prints a DECLINED line naming the VA",
          any("DECLINED" in d and "unallocated VA" in d for d in vdecl),
          (vdecl or ["-"])[0])
    check("[1x] the refusal exits clean — the engine stays eager, no crash", vrc == 0, f"rc={vrc}")

    ures, ucalls, udecl, urc = run_scenario(binary, "setattr_rejects")
    check("[1y] a driver that answers cudaErrorInvalidValue is refused with DECLINED",
          ures.get("stream_rc") == "-1"
          and any("DECLINED" in d and "cudaStreamSetAttribute" in d for d in udecl),
          (udecl or ["-"])[0])
    check("[1z] a refused window leaves the device carve intact (error cleared, no poisoning)",
          urc == 0 and int(ures["carve_after"]) == int(ures["carve"]),
          f"carve={ures.get('carve')} after={ures.get('carve_after')} rc={urc}")

    lres, _, ldecl, lrc = run_scenario(binary, "setlimit_fails")
    check("[1aa] a device that refuses the carve declines once and does not retry it",
          int(lres["carve"]) == 0 and int(lres["carve_again"]) == 0 and lres.get("setlimit") == "1"
          and lres.get("setattr") == "0" and lrc == 0,
          f"carve={lres.get('carve')} again={lres.get('carve_again')} "
          f"setattr={lres.get('setattr')} rc={lrc}")
    check("[1ab] the refused carve names itself in a DECLINED line",
          any("DECLINED" in d and "SetLimit" in d for d in ldecl), (ldecl or ["-"])[0])
finally:
    shutil.rmtree(TMP, ignore_errors = True)

print("")
print("PART 2 — the DevCtx site and the two Python seams, from the served source")
devctx = DEVCTX_CU.read_text(encoding = "utf8")
body = devctx.split("size_t DevCtx::get_l2win_carve(int device)", 1)
fn_body = body[1].split("\n}")[0] if len(body) == 2 else ""
check("[2a] DevCtx exposes the carve and takes it through exl3_l2win_ctx_carve exactly once",
      len(body) == 2 and fn_body.count("exl3_l2win_ctx_carve") == 1,
      f"{fn_body.count('exl3_l2win_ctx_carve')} call(s) in the accessor")
check("[2b] the accessor reads the owner every call (a second cache here went stale on-card)",
      len(body) == 2
      and "l2win_carve[device] = exl3_l2win_ctx_carve(device)" in body[1].split("\n}")[0]
      and "if (!l2win_carve[device])" not in body[1].split("\n}")[0],
      "owner-read" if len(body) == 2 else "-")
prep = devctx.split("void prepare_ctx(int device)", 1)
check("[2c] prepare_ctx takes the carve once, at context creation",
      len(prep) == 2 and prep[1].split("\n}")[0].count("get_l2win_carve") == 1,
      f"{prep[1].split()[0] if len(prep) == 2 else '-'}")
l2win_src = L2WIN_CU.read_text(encoding = "utf8")
check("[2d] the whole tree has exactly one cudaDeviceSetLimit call site, in exl3_l2win.cu",
      devctx.count("cudaDeviceSetLimit(") == 0
      and l2win_src.count("= cudaDeviceSetLimit(") == 1,
      f"devctx={devctx.count('cudaDeviceSetLimit(')} l2win={l2win_src.count('= cudaDeviceSetLimit(')}")
check("[2e] exactly one cudaStreamSetAttribute call site, and it sits in the stream function",
      l2win_src.count("= cudaStreamSetAttribute(") == 1,
      f"{l2win_src.count('= cudaStreamSetAttribute(')} stream site(s)")
bound = (ROOT / "exllamav3/exllamav3_ext/bindings.cpp").read_text(encoding = "utf8")
check("[2f] the three entry points are bound, and the stream handle is the current stream",
      all(n in bound for n in ("exl3_l2win_arm", "exl3_l2win_stream", "exl3_l2win_refusal"))
      and "at::cuda::getCurrentCUDAStream()" in bound, "bound")

TREE = ast.parse(LOOP.read_text(encoding = "utf8"))
ATT = next(n for n in ast.walk(TREE) if isinstance(n, ast.ClassDef) and n.name == "InklingAttention")
METH = {n.name: n for n in ATT.body if isinstance(n, ast.FunctionDef)}
for name in ("_l2win_arm", "_l2win_round"):
    check(f"[2g] InklingAttention defines {name}", name in METH, "")
seams = []
for meth in ("forward_attn", "_sliding_attend"):
    src = ast.unparse(METH[meth])
    seams.append((meth, "_l2win_round(" in src))
check("[2h] both decode seams call _l2win_round (global forward_attn, sliding _sliding_attend)",
      all(ok for _, ok in seams), f"{[(m, ok) for m, ok in seams]}")

# The two helpers, executed from the served source against a recording extension.
NS = {}
exec("import torch", NS)                                  # torch.device only; CUDA never initialised
ext_calls = []


class RecExt:
    def exl3_l2win_arm(self, device, h, kvh, d, rel):
        ext_calls.append(("arm", device, h, kvh, d, rel))
        return getattr(self, "carve", CEILING)

    def exl3_l2win_stream(self, device, base, rows):
        ext_calls.append(("stream", device, base, rows))
        return getattr(self, "stream_rc", 1)

    def exl3_l2win_refusal(self, device):
        return "shim reason"


class Log:
    def __init__(self):
        self.lines = []

    def warning(self, msg):
        self.lines.append(msg)

    info = warning


class Params:
    inkling_attn_l2win = False


class Cfg:
    infer_params = Params()


class Hot:
    def data_ptr(self):
        return 0x7000_0000_0000


HOT = Hot()                             # one fixture instance: data_ptr() is a bound call


RENAMES = (("self.config", "self.cfg"), ("self.layer_idx", "self.layer"),
           ("self.num_q_heads", "self.h"), ("self.num_kv_heads", "self.kvh"),
           ("self.head_dim", "self.d"))


def method_src(name, muts = ()):
    """The shipped method's own source, with only the harness's attribute names swapped in. A
    mutation raises if its anchor is gone, so a planted defect can never silently miss its mark."""
    src = ast.unparse(METH[name])
    for old, new in RENAMES:
        src = src.replace(old, new)
    for old, new in muts:
        if old not in src:
            raise AssertionError(f"mutation anchor missing in {name}: {old[:70]}")
        src = src.replace(old, new, 1)
    return src


def make_attn(harness):
    """A live InklingAttention stand-in carrying the two shipped methods, unmodified."""
    ns = dict(NS, ext = harness["ext"], logger = harness["log"])

    class Attn:
        pass

    for name in ("_l2win_arm", "_l2win_round"):
        exec(method_src(name), ns)
        setattr(Attn, name, ns[name])
    a = Attn()
    a.cfg = Cfg()
    a.h, a.kvh, a.d, a.rel_extent = 32, 8, 128, 512       # NOT the served shapes: the window must
    a.layer, a.device = 7, NS["torch"].device("cuda", 0)   # follow the module, so these differ
    a._l2win_armed = a._l2win_refused = False
    a._l2win_dev_idx = -1
    return a


harness = {"ext": RecExt(), "log": Log()}
attn = make_attn(harness)
harness["ext"].carve = 0
attn.cfg.infer_params = Params()
attn.cfg.infer_params.inkling_attn_l2win = False
ext_calls.clear()
attn._l2win_round(HOT, 4)                               # gate OFF on a round
check("[2i] gate OFF: the extension is never touched by a decode round", ext_calls == [],
      f"{ext_calls}")
check("[2j] gate OFF: load-path arming is behind the same gate (source-checked)",
      "if self.config.infer_params.inkling_attn_l2win:" in ast.unparse(METH["load"]),
      "load() gates _l2win_arm")

harness["ext"].carve = CEILING
attn.cfg.infer_params.inkling_attn_l2win = True
ext_calls.clear()
attn._l2win_arm(attn.device)                              # explicit call: no lazy re-arm
check("[2k] arm passes the MODULE'S shapes, not a hard-coded served tuple",
      ext_calls == [("arm", 0, 32, 8, 128, 512)], f"{ext_calls}")
ext_calls.clear()
attn._l2win_round(HOT, 4)
check("[2l] gate ON: a round sets the window on the current stream, with its row count",
      ext_calls == [("stream", 0, HOT.data_ptr(), 4)] and attn._l2win_armed, f"{ext_calls}")
ext_calls.clear()
attn._l2win_round(HOT, 8)
check("[2m] a later round keeps asking while the window holds", ext_calls[-1][3] == 8, f"{ext_calls}")

harness["ext"].stream_rc = -1
attn2 = make_attn(harness)
attn2.cfg.infer_params.inkling_attn_l2win = True
harness["log"].lines.clear()
attn2._l2win_round(HOT, 1)
ext_calls.clear()
attn2._l2win_round(HOT, 1)
check("[2n] a refused stream window is not retried on later rounds",
      ext_calls == [] and attn2._l2win_refused, f"{ext_calls} refused={attn2._l2win_refused}")
check("[2o] the refusal reaches the engine log as an INKLING-L2WIN DECLINED line",
      any("INKLING-L2WIN DECLINED" in l for l in harness["log"].lines),
      (harness["log"].lines or ["-"])[0][:90])

harness["ext"].carve = 0
attn3 = make_attn(harness)
attn3.cfg.infer_params.inkling_attn_l2win = True
harness["log"].lines.clear()
attn3._l2win_arm(attn3.device)
ext_calls.clear()
attn3._l2win_round(HOT, 1)
check("[2p] a refused carve stops the layer asking, once paid",
      attn3._l2win_refused and ext_calls == [], f"{ext_calls}")
check("[2q] the refused carve logs INKLING-L2WIN DECLINED with the extension's reason",
      any("INKLING-L2WIN DECLINED" in l and "shim reason" in l for l in harness["log"].lines),
      (harness["log"].lines or ["-"])[0][:90])

cfg_src = PARAMS.read_text(encoding = "utf8")
cfg = ast.parse(cfg_src)
params_cls = next(n for n in cfg.body if isinstance(n, ast.ClassDef) and n.name == "InferParams")
gate = [stmt for stmt in params_cls.body if isinstance(stmt, ast.AnnAssign)
        and getattr(stmt.target, "id", "") == "inkling_attn_l2win"]
check("[2r] InferParams declares inkling_attn_l2win: bool = False",
      len(gate) == 1 and ast.unparse(gate[0].annotation) == "bool"
      and ast.unparse(gate[0].value) == "False",
      ast.unparse(gate[0]) if gate else "absent")
check("[2s] no environment variable or yaml key sneaks the switch on",
      "os.environ" not in ast.unparse(params_cls).split("def __init__")[0],
      "field defaults only")

print("")
print("PART 3 — positive controls: planted defects must trip")


def tripped(name, muts, probe):
    """Plant a defect in a copy of the shipped source and require the matching check to fail."""
    global planted
    detail = ""
    try:
        ok = probe(muts)
    except AssertionError as e:
        # The mutation missed its mark, so the control never ran: that is a harness fault, and
        # counting it as a trip would buy a green that detects nothing.
        check(f"[ctrl] {name} tripped", False, f"harness: {e}")
        return
    except Exception as e:                       # a mutation that crashes did trip
        ok, detail = True, f"raised {type(e).__name__}: {e}"
    planted += 1 if ok else 0
    check(f"[ctrl] {name} tripped", ok, detail)


def run_mutated(muts, scene):
    tmp = pathlib.Path(tempfile.mkdtemp(prefix = "l2win-mut-"))
    try:
        src = prepare_tree(tmp, muts)
        binp = tmp / "probe"
        b = build(src, binp)
        if b.returncode != 0:
            return {"compile_failed": b.stderr.strip().splitlines()[-1][:120] if b.stderr else ""}
        return run_scenario(binp, scene)[0]
    finally:
        shutil.rmtree(tmp, ignore_errors = True)


def guard_removed(muts):
    res = run_mutated(muts, "gate_off")
    return res.get("setlimit") != "0" or res.get("setattr") != "0" or "compile_failed" in res


tripped("armed short-circuit deleted (gate OFF would carve the device)",
        (("    if (!st.armed)\n        return 0;", "    if (false)\n        return 0;"),),
        guard_removed)


def normal_miss(muts):
    tmp = pathlib.Path(tempfile.mkdtemp(prefix = "l2win-mut-"))
    try:
        src = prepare_tree(tmp, muts)
        binp = tmp / "probe"
        if build(src, binp).returncode != 0:
            return True
        _, calls, _, _ = run_scenario(binp, "arm_ok")
        return not any("missProp=1" in c for c in calls if c.startswith("cudaStreamSetAttribute"))
    finally:
        shutil.rmtree(tmp, ignore_errors = True)


tripped("missProp set to Normal (experts stop being evicted first)",
        (("val.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;",
          "val.accessPolicyWindow.missProp = cudaAccessPropertyNormal;"),), normal_miss)


def ceiling_raised(muts):
    tmp = pathlib.Path(tempfile.mkdtemp(prefix = "l2win-mut-"))
    try:
        src = prepare_tree(tmp, muts)
        binp = tmp / "probe"
        if build(src, binp).returncode != 0:
            return True
        res, _, _, _ = run_scenario(binp, "arm_ok")
        return int(res.get("carve", 0)) != CEILING
    finally:
        shutil.rmtree(tmp, ignore_errors = True)


tripped("carve ceiling raised past the comparator figure",
        (("#define L2_CARVE_CEILING_BYTES (size_t(99) * 1024 * 1024 / 2)",
          "#define L2_CARVE_CEILING_BYTES (size_t(64) * 1024 * 1024)"),), ceiling_raised)


def sizing_hardcoded(muts):
    tmp = pathlib.Path(tempfile.mkdtemp(prefix = "l2win-mut-"))
    try:
        src = prepare_tree(tmp, muts)
        binp = tmp / "probe"
        if build(src, binp).returncode != 0:
            return True
        res, _, _, _ = run_scenario(binp, "sizing")
        row_b = int(res["row_bytes_served"])
        return any(int(res[k]) == row_b for k in
                   ("row_bytes_h_only", "row_bytes_kvh_only", "row_bytes_d_only", "row_bytes_rel_only"))
    finally:
        shutil.rmtree(tmp, ignore_errors = True)


tripped("sizing frozen to the served tuple (rel_extent ignored)",
        (("const size_t rel_band = size_t(s.h) * size_t(s.rel_extent) * 4;",
          "const size_t rel_band = size_t(24) * size_t(1024) * 4;"),), sizing_hardcoded)

# Python-side mutations: the seam's own source, mutated in memory and re-executed.
GUARD_OFF = ("if not self.cfg.infer_params.inkling_attn_l2win or self._l2win_refused "
             "or hot is None or (rows <= 0):", "if False:")


def gate_check_removed(muts):
    ns = dict(NS, ext = RecExt(), logger = Log())
    ns["ext"].carve = CEILING
    exec(method_src("_l2win_round", muts), ns)

    class A:
        pass

    a = A()
    a.cfg = Cfg()                                   # gate OFF, the served default
    a._l2win_armed, a._l2win_refused, a._l2win_dev_idx = True, False, 0
    ext_calls.clear()
    ns["_l2win_round"](a, HOT, 4)
    return len(ext_calls) != 0            # gate OFF must make zero calls; a defect makes one


tripped("gate check removed from the per-round seam", (GUARD_OFF,), gate_check_removed)


def prepare_ctx_call_dropped(muts):
    text = DEVCTX_CU.read_text(encoding = "utf8")
    for old, _ in muts:
        text = text.replace(old, "", 1)
    prep = text.split("void prepare_ctx(int device)", 1)
    return len(prep) != 2 or prep[1].split("\n}")[0].count("get_l2win_carve") != 1


def rearm_drops_carve(muts):
    """Plant the lost idempotence guard: every layer's arm resets the carve again. The check that
    must trip is the one that survives the on-card fault — a round that gets no window."""
    tmp = pathlib.Path(tempfile.mkdtemp(prefix = "l2win-mut-"))
    try:
        src = prepare_tree(tmp, muts)
        binp = tmp / "probe"
        if build(src, binp).returncode != 0:
            return True
        res, _, _, _ = run_scenario(binp, "rearm_layers")
        return res.get("setlimit_after_24") != "1" or int(res.get("layers_zeroed", 0)) != 0 \
            or res.get("refusal", "") != ""
    finally:
        shutil.rmtree(tmp, ignore_errors = True)


tripped("per-layer re-arm drops the carve again (the dot26 fault returns)",
        (("    if (st.armed && st.shapes.h == shapes.h && st.shapes.kvh == shapes.kvh &&",
          "    if (false && st.armed && st.shapes.h == shapes.h && st.shapes.kvh == shapes.kvh &&"),),
        rearm_drops_carve)


def zero_handle_refused(muts):
    """Restore the bug that treated handle 0 as a missing stream: the served decode must then lose
    its window, which is what the card showed. Proves [1af] is the check that guards the default
    stream, rather than a check that passes because no scenario ever passed handle 0."""
    tmp = pathlib.Path(tempfile.mkdtemp(prefix = "l2win-mut-"))
    try:
        src = prepare_tree(tmp, muts)
        binp = tmp / "probe"
        if build(src, binp).returncode != 0:
            return True
        res, _, _, _ = run_scenario(binp, "default_stream")
        return res.get("setattr") == "0" or int(res.get("stream_rc", 0)) < 0
    finally:
        shutil.rmtree(tmp, ignore_errors = True)


tripped("legacy default stream (handle 0) refused again — the dot26 fault returns",
        (("    if (!hot_base || rows <= 0)", "    if (!stream || !hot_base || rows <= 0)"),),
        zero_handle_refused)


tripped("carve call dropped from prepare_ctx",
        (("    DevCtx::instance().get_l2win_carve(device);", ""),), prepare_ctx_call_dropped)

print("")
print(f"checks run: {ran}, passed: {ran - len(fails)}, planted defects tripped: {planted}/6")
print("not proved off-card (boot-pending): nvcc acceptance of exl3_l2win.cu (no CUDA toolkit here),"
      " whether a carved slice actually keeps decode rows resident (dot26 A/B), and the driver's"
      " own answer to a window spanning unallocated VA — the shim answers it here.")
if fails:
    print(f"L2WIN FENCE: RED ({len(fails)}): {', '.join(fails)}")
    sys.exit(1)
print(f"L2WIN FENCE GREEN: {ran - len(fails)}/{ran} checks passed, {planted}/6 planted defects tripped")
sys.exit(0)