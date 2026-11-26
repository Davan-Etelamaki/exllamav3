#!/usr/bin/env python3
"""Fence: the coop-drive wiring plan tracks the C++ interfaces it will call.

CPU-only (workstation law: never import exllamav3 here). moe_coop_plan is imported BY
PATH — it is pure python with no package deps. Checks:
  1. Activation/cap constants == the #defines/constexprs parsed FROM THE HEADERS.
  2. CALL_ARGS (positional order for pybind) == the exl3_moe_coop parameter names,
     IN ORDER, parsed from exl3_moe_coop.cuh — arity and order, not just count.
  3. ctr_len mirrors the C++ inline body verbatim (whitespace-normalised) + samples.
  4. classify_experts semantics + REFUSED LEGS NAMING THEIR REASON on fake module-likes
     (the reason is the fence-3 validity gate: a zero A/B delta must be explainable).
  5. Bucket rule (§U): DECODE_BUCKETS == the shared ladder, arena key = padded bucket,
     never live rows; refusal reasons for every ineligible leg.
  6. build_call_args order + loud missing-key failure.
  7. The SERVED module (parsed by AST, never imported) implements the shared rule: bucket
     keying, pinned-staging refill, the declared InferParams switch — and the retired
     live-rows / torch.tensor-refill constructs are gone from the tree.
POSITIVE CONTROLS: a planted rename in the parsed signature trips 2; a planted bucket-key
rename in the AST body trips 7.
"""

import ast
import importlib.util
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
PLAN = ROOT / "exllamav3/modules/arch_specific/moe_coop_plan.py"
MODULE = ROOT / "exllamav3/modules/arch_specific/inkling2_loop.py"
PARAMS = ROOT / "exllamav3/model/config.py"
COOP_CUH = ROOT / "exllamav3/exllamav3_ext/quant/exl3_moe_coop.cuh"
KERNEL_CUH = ROOT / "exllamav3/exllamav3_ext/quant/exl3_moe_coop_kernel.cuh"
COMMON_CUH = ROOT / "exllamav3/exllamav3_ext/moe_group/moe_sm89_common.cuh"

_spec = importlib.util.spec_from_file_location("moe_coop_plan", PLAN)
plan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plan)

fails = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{(': ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


def parse_define(text, name):
    m = re.search(r"#define\s+" + name + r"\s+(\S+)", text)
    return m.group(1) if m else None


def coop_signature(text):
    m = re.search(r"void exl3_moe_coop\s*\((.*?)\)\s*;", text, re.S)
    if m is None:
        raise AssertionError("exl3_moe_coop signature not found in " + str(COOP_CUH))
    params = []
    for chunk in m.group(1).split(","):
        chunk = re.sub(r"/\*.*?\*/", "", chunk).strip()
        if not chunk:
            continue
        name = re.split(r"[\s*&]+", chunk)[-1].strip()
        params.append(name)
    return params


class FakeLin:
    def __init__(self, K, mcg=False, mul1=False, bias=None):
        self.K, self.mcg, self.mul1, self.bias = K, mcg, mul1, bias


def assert_refusal(name, groups, needle):
    desc, reason = plan.classify_experts(groups)
    ok = desc is None and needle in reason
    detail = f" (reason: {reason})" if ok else f" (got desc={desc} reason={reason!r})"
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{detail}")
    if not ok:
        fails.append(name)


def main():
    coop = COOP_CUH.read_text()
    kern = KERNEL_CUH.read_text()
    common = COMMON_CUH.read_text()

    print("[1] constants mirror the headers")
    check("MOE_COOP_ACT_SOFTCAP",
          parse_define(coop, "MOE_COOP_ACT_SOFTCAP") == str(plan.MOE_COOP_ACT_SOFTCAP),
          parse_define(coop, "MOE_COOP_ACT_SOFTCAP") or "missing")
    check("MOE_MAX_DECODE_TOKENS",
          parse_define(common, "MOE_MAX_DECODE_TOKENS") == str(plan.MOE_MAX_DECODE_TOKENS),
          parse_define(common, "MOE_MAX_DECODE_TOKENS") or "missing")
    m = re.search(r"constexpr int MAX_SLOTS\s*=\s*(\d+)", kern)
    check("MOE_COOP_MAX_SLOTS == coop MAX_SLOTS", m and m.group(1) == str(plan.MOE_COOP_MAX_SLOTS),
          m.group(1) if m else "missing")

    print("[2] positional call order == the pybind signature (parsed from header)")
    sig = coop_signature(coop)
    check("arity", len(sig) == len(plan.CALL_ARGS), f"header {len(sig)} vs plan {len(plan.CALL_ARGS)}")
    check("order", sig == list(plan.CALL_ARGS),
          "first divergence: " + (next((f"{a}!={b}" for a, b in zip(sig, plan.CALL_ARGS) if a != b), "-")))
    planted = list(sig)
    planted[6], planted[9] = planted[9], planted[6]  # pretend g_trellis and u_trellis swapped in C++
    check("POSITIVE CONTROL: planted rename caught", planted != list(plan.CALL_ARGS))

    print("[3] ctr_len mirrors the C++ inline")
    m = re.search(r"inline int64_t exl3_moe_coop_ctr_len\([^)]*\)\s*\{\s*return ([^;]+);", coop)
    body = re.sub(r"\s+", " ", m.group(1)).replace("(int64_t)", "").strip() if m else None
    mine = "slots_max * (I / 128) + bsz_max * (Ho / 128) + 2 + (slots_max + 1) + slots_max"
    check("formula text", body == mine, body or "signature not found")
    check("sample rows=1", plan.ctr_len(8, 1, 9216, 2304) == 8 * 72 + 18 + 2 + 9 + 8)
    check("sample cap", plan.ctr_len(256, 32, 9216, 2304) == 256 * 72 + 32 * 18 + 2 + 257 + 256)

    print("[4] classify_experts admits the homogeneous pack and names every refused leg")
    uni = [(FakeLin(3), FakeLin(3), FakeLin(3))] * 128
    desc, reason = plan.classify_experts(uni)
    check("uniform pack classifies",
          desc == {"Kg": 3, "Ku": 3, "Kd": 3, "mcg": False, "mul1": False} and reason == "")
    frac, reason = plan.classify_experts([(FakeLin(3.05, mul1 = True),) * 3])
    check("fractional bitrate admitted", frac is not None and reason == "")
    assert_refusal("gate/up bit split refused", [(FakeLin(3), FakeLin(4), FakeLin(3))], "gate K")
    assert_refusal("per-expert bitrate drift refused",
                   [(FakeLin(3), FakeLin(3), FakeLin(3))] * 4 + [(FakeLin(3), FakeLin(3), FakeLin(4))],
                   "baseline")
    assert_refusal("mixed codebooks refused", [(FakeLin(3), FakeLin(3), FakeLin(3, mcg = True))],
                   "codebook families")
    assert_refusal("biased pack refused", [(FakeLin(3, bias = object()), FakeLin(3), FakeLin(3))], "bias")
    assert_refusal("bitrate outside [1,8] refused", [(FakeLin(0), FakeLin(0), FakeLin(0))], "outside")

    print("[5] bucket rule (graph lane §U): arena key = padded bucket, never live rows")
    check("ladder == the shared decode universe", plan.DECODE_BUCKETS == (1, 2, 4, 8, 16, 32),
          str(plan.DECODE_BUCKETS))
    check("ladder cap == MOE_MAX_DECODE_TOKENS",
          max(plan.DECODE_BUCKETS) == plan.MOE_MAX_DECODE_TOKENS)
    check("ladder slots cap == MOE_COOP_MAX_SLOTS at top_k 8",
          max(plan.DECODE_BUCKETS) * 8 == plan.MOE_COOP_MAX_SLOTS)
    for rows, want in [(1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (32, 32),
                       (33, None), (4096, None)]:
        got = plan.padded_bucket_rows(rows)
        check(f"padded_bucket_rows({rows}) -> {want}", got == want, str(got))
    check("POSITIVE CONTROL: live-row keying retired (3 -> 4, not 3)",
          plan.padded_bucket_rows(3) != 3)
    check("refusal: a below-bucket live count is admitted",
          plan.refusal(3, 8, True, True, True) is None and plan.eligible(3, 8, True, True, True))
    for name, leg, needle in [
        ("cpu x", (3, 8, False, True, True), "not a CUDA tensor"),
        ("fp32 x", (3, 8, True, False, True), "not fp16"),
        ("no ext entry", (3, 8, True, True, False), "exl3_moe_coop"),
        ("above the ladder", (33, 8, True, True, True), "above the decode ladder"),
        ("slots beyond cap", (32, 9, True, True, True), "exceeds MOE_COOP_MAX_SLOTS"),
    ]:
        r = plan.refusal(*leg)
        ok = r is not None and needle in r and not plan.eligible(*leg)
        print(f"  {'ok  ' if ok else 'FAIL'} refusal names {name}"
              + (f" (reason: {r})" if ok else f" (got {r!r})"))
        if not ok:
            fails.append("refusal " + name)
    check("eligible == (refusal is None) over a rows x top_k sweep",
          all(plan.eligible(r, k, True, True, True)
              == (plan.refusal(r, k, True, True, True) is None)
              for r in range(0, 40) for k in (1, 8, 9)))

    print("[6] build_call_args pins order and fails loudly on a missing key")
    vals = {n: n for n in plan.CALL_ARGS}
    check("round-trip order", plan.build_call_args(vals) == tuple(plan.CALL_ARGS))
    vals.pop("mcg")
    try:
        plan.build_call_args(vals)
        check("missing key raises", False)
    except KeyError:
        check("missing key raises", True)

    print("[7] the served module implements the shared rule (AST parse, never imported)")
    tree = ast.parse(MODULE.read_text())
    methods = {n.name: n for cls in tree.body if isinstance(cls, ast.ClassDef)
               for n in cls.body if isinstance(n, ast.FunctionDef)}
    decode = ast.unparse(methods["_coop_decode"])
    state = ast.unparse(methods["_coop_state"])
    check("_coop_decode keys by padded_bucket_rows",
          "padded_bucket_rows(rows)" in decode and "_coop_by_bucket.get(bucket)" in decode)
    check("_coop_state sizes from the bucket",
          "bucket * self.top_k" in state and "ctr_len(slots, bucket, I, Ho)" in state)
    check("refill reads the pinned staging buffer",
          "stage_np[lo:hi] = cur_lists[k]" in decode and "copy_(st[" in decode)
    check("declared switch gates the tier",
          "self.config.infer_params.moe_coop_decode" in decode)
    check("pointer tables cached, not rebuilt per decode step (census 2026-11-26)",
          "self._coop_ptrs_cache is None" in decode)
    planted = decode.replace("self._coop_ptrs_cache is None", "True")
    check("POSITIVE CONTROL: planted per-step rebuild caught",
          planted != decode and "self._coop_ptrs_cache is None" not in planted)
    check("output sliced to the live rows", "out[:rows]" in decode)
    whole = MODULE.read_text()
    check("live-rows dict retired tree-wide", "_coop_by_rows" not in whole)
    check("per-call torch.tensor refill retired tree-wide", "copy_(torch.tensor(" not in whole)
    check("InferParams declares the switch, coop on by default",
          re.search(r"moe_coop_decode:\s*bool\s*=\s*True", PARAMS.read_text()) is not None)
    planted = decode.replace("_coop_by_bucket.get(bucket)", "_coop_by_bucket.get(rows)")
    check("POSITIVE CONTROL: planted live-row key caught",
          planted != decode and "_coop_by_bucket.get(bucket)" not in planted)

    print("[8] boot-shape regression: Linear WRAPPERS (no .bias attr) classify, not raise")
    class LinWrap:  # mirrors modules/linear.py Linear: exposes inner, NOT bias/K/trellis
        def __init__(self, inner):
            self.inner = inner
    tri = [LinWrap(FakeLin(3)) for _ in range(3)]
    try:
        desc, reason = plan.classify_experts([tuple(tri)])
        check("wrapper trio classifies to a desc", desc is not None and desc["Kg"] == 3,
              f"desc={desc} reason={reason!r}")
    except AttributeError as e:  # pre-patch boot failure: 'Linear' has no attribute 'bias'
        check("wrapper trio classifies to a desc", False, f"AttributeError: {e}")
    biased = LinWrap(FakeLin(3, bias="b"))
    desc, reason = plan.classify_experts([(biased, tri[1], tri[2])])
    check("inner bias still refused", desc is None and "bias" in reason, f"reason={reason!r}")
    check("bare inner trio unchanged (fence's own fixtures)",
          plan.classify_experts([(FakeLin(3), FakeLin(3), FakeLin(3))])[0]["Ku"] == 3)
    check("_coop_ptr_lists unwraps before data_ptr",
          "moe_coop_plan.unwrap_linear(m)" in ast.unparse(methods["_coop_ptr_lists"]))

    print("[9] type fidelity: the wrapper fixture must match the REAL served Linear")
    try:  # the boot crash came from the served class, not from a lookalike fixture
        from exllamav3.modules.linear import Linear as RealLinear
        lin_src = pathlib.Path(sys.modules[RealLinear.__module__].__file__).read_text()
    except Exception as e:
        check("real served Linear importable CPU-only", False, f"{type(e).__name__}: {e}")
        RealLinear, lin_src = None, ""
    if RealLinear is not None:
        check("real Linear exposes no .bias at class level (the boot crash shape)",
              not hasattr(RealLinear, "bias"))
        check("wrapper fixture mirrors the real class on the .bias axis",
              hasattr(tri[0], "bias") == hasattr(RealLinear, "bias"),
              f"fixture_bias={hasattr(tri[0], 'bias')} real_bias={hasattr(RealLinear, 'bias')}")
        check("real Linear assigns self.inner in source (unwrap target exists)",
              "self.inner" in lin_src)
        grown = LinWrap(FakeLin(3)); grown.bias = None
        check("POSITIVE CONTROL: a fixture that grows .bias is caught",
              hasattr(grown, "bias") != hasattr(RealLinear, "bias"))

    print("[10] pointer-table epoch: a cached table must die with the weights it pins (law 33)")
    cls = next(c for c in tree.body if isinstance(c, ast.ClassDef)
               and any(isinstance(n, ast.FunctionDef) and n.name == "_coop_decode" for n in c.body))
    own = {n.name: ast.unparse(n) for n in cls.body if isinstance(n, ast.FunctionDef)}
    unloaded = own.get("unload", "")
    check("unload clears the pointer-table cache",
          "self._coop_ptrs_cache = None" in unloaded, f"class={cls.name} unloaded_len={len(unloaded)}")
    check("unload drops the arenas that pin device addresses", "_coop_by_bucket = {}" in unloaded)
    check("bucket creation seeds the fingerprint from a FRESH read, not from the cache",
          "ptr_lists = self._coop_ptr_lists()" in state
          and "'fingerprint': tuple((p for k in names for p in ptr_lists[k]))" in state)
    check("served path rebuilds only on a cold cache or a moved identity",
          "self._coop_ptrs_cache is None" in decode
          and ("fingerprint != st[" in decode and "_coop_ptrs_probe() != st" in decode))
    check("refill recomputes fresh, then advances cache and fingerprint together",
          "st['ptr_lists'] = cur_lists = self._coop_ptr_lists()" in decode
          and "self._coop_ptrs_cache = cur_lists" in decode
          and "st['fingerprint'] = fingerprint" in decode)
    forgotten = unloaded.replace("self._coop_ptrs_cache = None", "pass")
    check("POSITIVE CONTROL: an unload that keeps the cache is caught",
          forgotten != unloaded and "self._coop_ptrs_cache = None" not in forgotten)
    cache_seed = state.replace("for p in ptr_lists[k]))", "for p in self._coop_ptrs_cache[k]))")
    check("POSITIVE CONTROL: seeding the fingerprint from the cache is caught",
          cache_seed != state and "for p in self._coop_ptrs_cache[k])" in cache_seed)

    print("[11] epoch arm: a moved tensor without unload() must not serve cached pointers")
    import numpy as _np
    fn2 = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_coop_decode")
    keep = lambda n: ("_coop_ptrs_cache" in ast.unparse(n) or "cur_lists = " in ast.unparse(n)
                      or ast.unparse(n).startswith("fingerprint = ") or "_coop_ptrs_probe" in ast.unparse(n))
    stmts = [n for n in fn2.body if keep(n)]

    class _Live:
        def __init__(s, v): s.v = v
        def data_ptr(s): return s.v

    class _Buf:
        def __init__(s): s.src = None
        def copy_(s, src, non_blocking=False): s.src = src

    class _Shim:
        def __init__(s):
            s._coop_ptrs_cache = None
            s.live = {"g": _Live(1000), "u": _Live(2000), "d": _Live(3000)}
        def _coop_ptr_lists(s):
            return {k: [m.data_ptr()] for k, m in (("g", s.live["g"]), ("u", s.live["u"]), ("d", s.live["d"]))}
        def _coop_ptrs_probe(s):
            l = s._coop_ptr_lists()
            return None if not l else tuple(l[k][0] for k in ("g", "u", "d"))

    def _run(body):
        code = compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), "<arm>", "exec")
        sh = _Shim(); lists = sh._coop_ptr_lists(); names = sorted(lists)
        stage = _np.array([v for k in names for v in lists[k]], dtype=_np.int64)
        offs, pos = {}, 0
        for k in names:
            offs[k] = (pos, pos + len(lists[k])); pos = offs[k][1]
        st = {"names": names, "offsets": offs, "stage": stage, "stage_np": stage,
              "ptr_lists": lists, "fingerprint": tuple(v for k in names for v in lists[k]),
              "probe": sh._coop_ptrs_probe(), "args": {k: _Buf() for k in names}}
        sh._coop_ptrs_cache = lists
        for k in sh.live: sh.live[k].v += 8000  # weights move; unload() never fires
        ex = {"self": sh, "st": st, "args": st["args"]}
        exec(code, ex)
        return st["ptr_lists"]["g"][0], sh._coop_ptr_lists()["g"][0]

    served, live = _run(stmts)
    check("a moved tensor is caught: the coop step serves the LIVE pointers",
          served == live, f"served_g={served} live_g={live}")
    _dq, _sq = chr(34), chr(39)
    _frag_dq = ' or self._coop_ptrs_probe() != st.get(' + _dq + 'probe' + _dq + ')'
    _frag_sq = ' or self._coop_ptrs_probe() != st.get(' + _sq + 'probe' + _sq + ')'
    planted = [ast.parse(ast.unparse(n).replace(_frag_dq, '').replace(_frag_sq, '')).body[0] for n in stmts]
    check("planted edit actually removed the live-probe test",
          all("_coop_ptrs_probe" not in ast.unparse(n) for n in planted if "self._coop_ptrs_cache is None" in ast.unparse(n)))
    served_p, live_p = _run(planted)
    check("POSITIVE CONTROL: dropping the live probe reintroduces the stale serve",
          served_p != live_p, f"served={served_p} live={live_p}")

    if fails:
        print("MOE COOP PLAN FENCE RED:", ", ".join(fails))
        return 1
    print("MOE COOP PLAN FENCE GREEN: wiring mirrors the headers; planted drift caught.")
    return 0


if __name__ == "__main__":
    sys.exit(main())