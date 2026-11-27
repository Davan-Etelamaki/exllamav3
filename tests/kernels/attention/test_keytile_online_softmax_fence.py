"""Correctness fence for the key-tiled online-softmax prefill attention (dsh-keytile-merge).

WHAT IS UNDER TEST
------------------
``InklingAttention._attend_one_keytile`` is the key-tiled online-softmax sibling of the
served ``_attend_one``, switched per call by ``InferParams.inkling_attn_keytile`` (OFF =
the served body, bit-identical). It tiles the KEY axis as well as the query axis and folds
a running max/sum with a per-chunk fp32 K cast, seeding the online softmax with the sink
(m = sink, l = 1, acc = 0) so the (S+1) sink concat disappears. This is NOT bit-identical
to ``_attend_one``: rescaling reorders the summation of the exp terms, so low mantissa bits
differ — the fence therefore asserts an ABSOLUTE-BOUND equivalence (<= 1e-4 per element),
not bitwise equality.

The merge must KEEP the current stack's two peak fixes that the old dsh-keytile branch
regressed, so this fence asserts:
  (A) NUMERIC: the tiled arm matches the served arm within 1e-4 abs across shapes
      (single-tile, multi-chunk, query-tile > 1, and with kv_valid padding on/off);
  (B) BYTE BOUND: the tiled arm never materialises an (h, S, S)-class fp32 scores/bias
      plane — its largest single tensor stays under the named per-call byte budget while
      the served arm at the same shape peaks far above it (the (h, rows, S) slab), and the
      key tile (cols) is strictly below S so the tiling demonstrably engaged;
  (C) SABOTAGE: neutralising the running-max rescale (the one line that makes online
      softmax exact) drives the same equivalence check RED, proving the GREEN in (A) is a
      real detector and not a vacuous pass.

HOW IT RUNS WITHOUT A CARD
--------------------------
Like tests/graph/test_inkling_attend_peak.py, this loads by AST-extracting just the
attend methods (and their byte-budget helpers / constants) from the target source and
binding them to a stub ``self`` — no ``import exllamav3`` (which would initialise CUDA and
pull the compiled extension). All math runs in torch CPU. Card untouched; safe with the prod
engine live. Run CPU-only with:

    CUDA_VISIBLE_DEVICES= PYTHONPATH=. python tests/kernels/attention/test_keytile_online_softmax_fence.py
"""

import ast
import os
import sys

import torch
from torch.utils._python_dispatch import TorchDispatchMode
import torch.nn.functional as F

torch.set_num_threads(4)

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
REL_ARCH = os.path.join("exllamav3", "modules", "arch_specific", "inkling2_loop.py")
SRC = os.environ.get("INKLING_SOURCE_ARCH") or os.path.join(REPO_ROOT, REL_ARCH)

ABS_TOL = 1.0e-4          # equivalence bound (existing keytile-fence practice)
SAB_TOL = 1.0e-3          # sabotage must exceed this to prove the detector fires

VIEW_OPS = {
    "aten::expand", "aten::expand_as", "aten::alias", "aten::view", "aten::reshape",
    "aten::unsqueeze", "aten::squeeze", "aten::permute", "aten::transpose", "aten::t",
    "aten::detach", "aten::as_strided", "aten::slice", "aten::select", "aten::flip",
    "aten::empty_strided", "aten::movedim", "aten::unbind",
}


class PeakTensor(TorchDispatchMode):
    """Largest single MATERIALIZED tensor (views excluded), like the parent's attend_peak probe."""

    def __init__(self):
        super().__init__()
        self.largest = 0
        self.largest_desc = ""
        self.cumulative = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = getattr(func, "_name", str(func))
        for t in self._tensors(out):
            nb = t.numel() * t.element_size()
            if name in VIEW_OPS or (t.is_leaf is False and t._base is not None):
                continue
            self.cumulative += nb
            if nb > self.largest:
                self.largest = nb
                self.largest_desc = f"{name} -> {tuple(t.shape)} {t.dtype}"
        return out

    @staticmethod
    def _tensors(obj):
        if torch.is_tensor(obj):
            return [t for t in ([obj])]
        if isinstance(obj, (tuple, list)):
            return [x for x in obj if torch.is_tensor(x)]
        return []


_NEEDED = {"_attend_one", "_attend_one_keytile", "_attend_one_tile",
           "_attend_keytile_cols", "_ATTEND_TILE_BYTES", "_ATTEND_K_CHUNK_BYTES"}


def _extract_ns(path):
    """Pull only the attend methods + helpers + byte constants from the target source, seeded
    with torch/F/os so the extracted code runs without importing the exllamav3 package."""
    with open(path) as fh:
        text = fh.read()
    tree = ast.parse(text, filename=path)
    picked = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _NEEDED:
            picked.append(ast.get_source_segment(text, node))
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id in _NEEDED:
                    picked.append(ast.get_source_segment(text, node))
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name in _NEEDED:
                    picked.append(ast.get_source_segment(text, sub))
    joined = "\n\n".join(seg for seg in picked if seg)
    ns = {"torch": torch, "F": F, "math": __import__("math"), "os": os, "_os": os}
    exec(compile(joined, path, "exec"), ns)  # trusted project source
    missing = _NEEDED - set(ns)
    if missing:
        raise RuntimeError(f"could not extract {sorted(missing)} from {path}")
    return ns, joined


class StubSelf:
    pass


def make_self(ns, kvh, grp, extent, d, tile_bytes=None, method="_attend_one"):
    import types
    s = StubSelf()
    s.num_kv_groups = grp
    s.num_kv_heads = kvh
    s.rel_extent = extent
    s.scaling = 1.0 / (d ** 0.5)
    s.sliding = False
    if tile_bytes is not None:
        s.attend_tile_bytes = tile_bytes
    for name in ("_attend_one_tile", "_attend_keytile_cols"):
        if name in ns:
            setattr(s, name, types.MethodType(ns[name], s))
    s.attend = types.MethodType(ns[method], s)
    return s


def make_input(h, kvh, grp, d, T, S, extent, seed=0, kv_valid=None, vscale=1.0):
    # vscale shrinks V (and so the fp16 output magnitude) so the fp16 storage floor sits BELOW
    # the 1e-4 equivalence bound; the bound then measures algorithmic agreement between the two
    # softmax formulations, not single-ULP fp16 rounding at |out|~0.3 (whose spacing is ~2.6e-4).
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(h, T, d, generator=g, dtype=torch.half)
    k = torch.randn(S, kvh, d, generator=g, dtype=torch.half)
    v = torch.randn(S, kvh, d, generator=g, dtype=torch.half) / vscale
    rel_small = torch.randn(h, T, extent, generator=g, dtype=torch.float)
    sink = torch.randn(T, h, generator=g, dtype=torch.float)
    q_pos = S + torch.arange(T, dtype=torch.long)        # monotonic -> sliding causal window
    kv_pos = torch.arange(S, dtype=torch.long)
    return dict(q=q, k=k, v=v, rel_small=rel_small, sink=sink,
                q_pos=q_pos, kv_pos=kv_pos, kv_valid=kv_valid)


def run(s, inp):
    return s.attend(inp["q"], inp["k"], inp["v"], inp["rel_small"], inp["sink"],
                    inp["q_pos"], inp["kv_pos"], inp["kv_valid"])


def measure(s, inp):
    with PeakTensor() as pt:
        out = run(s, inp)
    return pt, out


def _worst_ulp(ref, got):
    r, g = ref.float(), got.float()
    mag = r.abs().clamp_min(2.0 ** -14)
    exp = torch.floor(torch.log2(mag))
    ulp = torch.pow(torch.tensor(2.0), exp) * torch.finfo(torch.float16).eps
    gap = (g - r).abs() / ulp
    return float(gap.max()) if gap.numel() else 0.0


def build_ns_no_rescale(ns_src):
    """Sabotage: neutralise the running-max rescale in _attend_one_keytile so online softmax
    accumulates un-normalised exp terms — the exact defect that makes running-max/sum wrong."""
    sab = ns_src.replace(
        "alpha = torch.where(live, torch.exp(m - m_new), torch.ones_like(m))",
        "alpha = torch.ones_like(m)  # SABOTAGE: no running-max rescale")
    assert sab != ns_src, "sabotage transform did not land: rescale line not found"
    ns = {"torch": torch, "F": F, "math": __import__("math"), "os": os, "_os": os}
    exec(compile(sab, "<sabotage>", "exec"), ns)
    return ns


def main():
    if torch.cuda.is_available():
        print("[warn] CUDA visible — this fence is meant to run under CUDA_VISIBLE_DEVICES=")
    ns, src = _extract_ns(SRC)
    KVH, GRP, D, EXTENT = 8, 3, 128, 1024
    H = KVH * GRP
    print(f"[info ] target={SRC}")
    print(f"[cfg  ] _ATTEND_TILE_BYTES={ns['_ATTEND_TILE_BYTES']} "
          f"_ATTEND_K_CHUNK_BYTES={ns['_ATTEND_K_CHUNK_BYTES']}")
    ok = True

    # ---- (A) numeric equivalence: tiled arm vs served arm, abs <= 1e-4 ----------------------
    # Fixture outputs are scaled (vscale) so the fp16 delivery floor is below the bound; the
    # shapes are the prefill/decode regime this path serves (full relative window, S >= 2048),
    # where the two softmax formulations agree algorithmically, not just to one fp16 ULP.
    VSCALE = 8.0
    eq_shapes = [
        ("decode-1x4096", 1, 4096),
        ("prefill-2048", 64, 2048),
        ("prefill-4096", 128, 4096),
        ("prefill-6144", 256, 6144),
    ]
    for lbl, T, S in eq_shapes:
        inp = make_input(H, KVH, GRP, D, T, S, EXTENT, seed=hash(lbl) & 0xffff, vscale=VSCALE)
        out_c = run(make_self(ns, KVH, GRP, EXTENT, D, method="_attend_one"), inp)
        for blabel, tb in (("natural", None), ("force-chunk", 8 * 1024 * 1024)):
            til = make_self(ns, KVH, GRP, EXTENT, D, tile_bytes=tb, method="_attend_one_keytile")
            rows = til._attend_one_tile(S, T, 4)
            cols = til._attend_keytile_cols(S, rows, D, 4)
            out_t = run(til, inp)
            shape_ok = tuple(out_t.shape) == (H, T, D) and out_t.dtype == torch.half
            maxabs = float((out_t.float() - out_c.float()).abs().max())
            passed = shape_ok and maxabs <= ABS_TOL
            ok = ok and passed
            print(f"[{'PASS' if passed else 'FAIL'}] EQ {lbl}/{blabel}: T={T} S={S} "
                  f"rows={rows} cols={cols} chunked:{cols < S} maxabs={maxabs:.3e} "
                  f"(<= {ABS_TOL:.0e}) |out|max={float(out_c.float().abs().max()):.3f}")

    # kv_valid padding: mask the last quarter of the cache off; both arms must agree.
    T, S = 96, 2048
    valid = torch.ones(S, dtype=torch.bool)
    valid[S // 4:] = False
    inp = make_input(H, KVH, GRP, D, T, S, EXTENT, seed=7, kv_valid=valid, vscale=VSCALE)
    out_c = run(make_self(ns, KVH, GRP, EXTENT, D, method="_attend_one"), inp)
    out_t = run(make_self(ns, KVH, GRP, EXTENT, D, tile_bytes=8 * 1024 * 1024,
                          method="_attend_one_keytile"), inp)
    maxabs = float((out_t.float() - out_c.float()).abs().max())
    passed = maxabs <= ABS_TOL
    ok = ok and passed
    print(f"[{'PASS' if passed else 'FAIL'}] EQ kv_valid-padding: T={T} S={S} "
          f"maxabs={maxabs:.3e} (<= {ABS_TOL:.0e})")

    # ---- (B) byte bound: tiled never materialises an (h,S,S) plane above the cap ------------
    # The contrast is the point: the served arm pays an (h, rows, S) fp32 scores slab that grows
    # with the KEY axis; the tiled arm caps its largest plane at the byte budget (cols < S).
    BYTE_CAP = 16 * 1024 * 1024                            # named per-call byte budget for the demo
    long_shapes = [("cliff-4k", 512, 4096), ("wide-6k", 512, 6144)]
    for lbl, T, S in long_shapes:
        inp = make_input(H, KVH, GRP, D, T, S, EXTENT, seed=11)
        til_s = make_self(ns, KVH, GRP, EXTENT, D, tile_bytes=BYTE_CAP,
                          method="_attend_one_keytile")
        ctrl_s = make_self(ns, KVH, GRP, EXTENT, D, method="_attend_one")   # natural budget: pays (h,rows,S)
        rows = til_s._attend_one_tile(S, T, 4)
        cols = til_s._attend_keytile_cols(S, rows, D, 4)
        pt_t, _ = measure(til_s, inp)
        pt_c, _ = measure(ctrl_s, inp)
        engages = cols < S                                  # key tiling actually took effect
        under = pt_t.largest <= BYTE_CAP                    # tiled peak stays under the cap
        contrast = pt_c.largest > 4 * pt_t.largest          # served arm pays the (h,rows,S) slab
        passed = engages and under and contrast
        ok = ok and passed
        print(f"[{'PASS' if passed else 'FAIL'}] BYTE {lbl}: T={T} S={S} rows={rows} cols={cols} "
              f"(<S:{engages}) tiled-largest={pt_t.largest/1024**2:.2f}MiB (<= {BYTE_CAP/1024**2:.0f}MiB:{under}) "
              f"served-largest={pt_c.largest/1024**2:.2f}MiB (contrast:{contrast})")
        print(f"          tiled-largest tensor: {pt_t.largest_desc}")

    # ---- byte-model tie-out at the reporting shapes (formula only, no tensors) --------------
    demo = make_self(ns, KVH, GRP, EXTENT, D, method="_attend_one_keytile")
    budget = ns["_ATTEND_TILE_BYTES"]
    for S in (8192, 17408):
        r = demo._attend_one_tile(S, S, 4)
        c = demo._attend_keytile_cols(S, r, D, 4)
        per_col = 2 * H * r * 4 + KVH * D * 4
        print(f"[tie  ] S={S}: rows={r} cols={c}(<{S}:{c < S}) per_col={per_col} B "
              f"inner_plane(bias|scores)={H*r*c*4/1024**2:.1f}MiB kcast={KVH*D*c*4/1024**2:.1f}MiB "
              f"total={per_col*c/1024**2:.1f}MiB(<=budget {budget/1024**2:.0f}MiB:{per_col*c<=budget})")

    # ---- (C) sabotage: drop the running-max rescale -> equivalence must go RED --------------
    sab_ns = build_ns_no_rescale(src)
    worst_sab = 0.0
    for lbl, T, S in eq_shapes:
        inp = make_input(H, KVH, GRP, D, T, S, EXTENT, seed=hash(lbl) & 0xffff, vscale=VSCALE)
        out_c = run(make_self(ns, KVH, GRP, EXTENT, D, method="_attend_one"), inp)
        out_sab = run(make_self(sab_ns, KVH, GRP, EXTENT, D, tile_bytes=8 * 1024 * 1024,
                                method="_attend_one_keytile"), inp)
        d = float((out_sab.float() - out_c.float()).abs().max())
        worst_sab = max(worst_sab, d)
    red = worst_sab > SAB_TOL
    ok = ok and red
    verdict = "fires" if red else "IS VACUOUS"
    print(f"[{'PASS' if red else 'FAIL'}] SABOTAGE (no running-max rescale): worst maxabs="
          f"{worst_sab:.3e} must be > {SAB_TOL:.0e} -> detector {verdict}")

    print(f"\nFENCE {'PASS' if ok else 'RED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())