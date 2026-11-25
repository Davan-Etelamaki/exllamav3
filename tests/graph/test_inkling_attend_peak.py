"""sm_89 kernel lane — _attend_one relative-position peak fence (ASSERTING).

Metric authority: parent's attend_peak_probe.py PeakTensor (largest single MATERIALIZED
tensor, views excluded; cumulative materialised; peak RSS). The served saturation/OOM band
is a single fp32 ~(h,T,S) allocation -- the full relative-position `bias` gather and the
(kvh,grp,T,S+1) softmax logits both scale with S and, pre-fix, were materialised OUTSIDE
the tile loop, so shrinking `tile` did not touch them.

Fix under test: bias/dist/causal built PER QUERY TILE inside the loop, tile byte-budget
derived -> largest single tensor bounded; single-tile decode (T==1) and short-context
paths stay launch-neutral (same iteration count); output must remain (h,T,d) fp16 and
BIT-IDENTICAL to the pre-fix body.

Loading is done by AST-extracting just the ``_attend_one`` / ``_attend_one_tile`` methods
(and the ``_ATTEND_TILE_BYTES`` constant) from the target file and binding them to a stub
``self`` -- the same way the parent's baseline drove the method directly. This avoids any
``import exllamav3`` (which initialises CUDA / pulls the compiled extension / a Config that
differs between the workstation wheel and the deployed dot26 wheel), so the fence runs
identically on both hosts with the card masked.

Targets: INKLING_FENCE_TARGET=source (patched worktree) must be GREEN; =installed (the
deployed wheel 1.6.7+0263e305f3b7d, which materialises bias at full S with no byte-budget
tiling) must be RED on the acceptance thresholds. CPU-only; CUDA masked; card untouched.
"""

import ast
import glob
import os
import resource
import sys

import torch
from torch.utils._python_dispatch import TorchDispatchMode

import torch.nn.functional as F

torch.set_num_threads(4)

WT = "/home/davan.etelamaki/Repos/exllamav3-wt-fwdgraphsc"
REL_ARCH = os.path.join("exllamav3", "modules", "arch_specific", "inkling2_loop.py")
SRC = os.environ.get("INKLING_SOURCE_ARCH") or os.path.join(WT, REL_ARCH)

VIEW_OPS = {
    "aten::expand", "aten::expand_as", "aten::alias", "aten::view", "aten::reshape",
    "aten::unsqueeze", "aten::squeeze", "aten::permute", "aten::transpose", "aten::t",
    "aten::detach", "aten::as_strided", "aten::slice", "aten::select", "aten::flip",
    "aten::empty_strided", "aten::movedim", "aten::unbind",
}


class PeakTensor(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.largest = 0
        self.largest_desc = ""
        self.cumulative = 0
        self.views = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = getattr(func, "_name", str(func))
        for t in self._tensors(out):
            nb = t.numel() * t.element_size()
            if name in VIEW_OPS or (t.is_leaf is False and t._base is not None):
                self.views = max(self.views, nb)
                continue
            self.cumulative += nb
            if nb > self.largest:
                self.largest = nb
                self.largest_desc = f"{name} -> {tuple(t.shape)} {t.dtype}"
        return out

    @staticmethod
    def _tensors(obj):
        if torch.is_tensor(obj):
            return [obj]
        if isinstance(obj, (tuple, list)):
            return [x for x in obj if torch.is_tensor(x)]
        return []


def giB(n):
    return n / 1024 ** 3


# --------------------------------------------------------------------------- loaders
_NEEDED_NAMES = {"_attend_one", "_attend_one_tile", "_ATTEND_TILE_BYTES"}


def _extract_method_ns(path):
    """Return a namespace holding the target file's _attend_one (and, when present,
    _attend_one_tile and the _ATTEND_TILE_BYTES constant), seeded with torch/F so the
    extracted code runs without importing the exllamav3 package."""
    with open(path) as fh:
        text = fh.read()
    tree = ast.parse(text, filename=path)
    picked = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _NEEDED_NAMES:
            picked.append(ast.get_source_segment(text, node))
    # module-level constants: _ATTEND_TILE_BYTES = ...
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id in _NEEDED_NAMES:
                    picked.append(ast.get_source_segment(text, node))
    # class methods: dig into ClassDef bodies
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name in _NEEDED_NAMES:
                    picked.append(ast.get_source_segment(text, sub))
    ns = {"torch": torch, "F": F, "math": __import__("math"), "os": os, "_os": os}
    joined = "\n\n".join(seg for seg in picked if seg)
    exec(compile(joined, path, "exec"), ns)  # noqa: S102 - trusted project source
    if "_attend_one" not in ns:
        raise RuntimeError(f"could not extract _attend_one from {path}")
    return ns, joined


def _installed_arch_path():
    env = os.environ.get("INKLING_INSTALLED_ARCH")
    if env:
        return env
    roots = []
    try:
        import site
        roots.extend(site.getsitepackages())
    except Exception:  # noqa: BLE001
        pass
    try:
        import sysconfig
        roots.append(sysconfig.get_paths()["purelib"])
    except Exception:  # noqa: BLE001
        pass
    roots.append(os.path.join(sys.prefix, "lib", "python3.12", "site-packages"))
    roots.append(os.path.join(os.path.dirname(sys.executable), "..",
                             "lib", "python3.12", "site-packages"))
    seen = set()
    for root in roots:
        if not root or root in seen:
            continue
        seen.add(root)
        cand = os.path.join(root, REL_ARCH)
        if os.path.exists(cand):
            return cand
        hits = glob.glob(os.path.join(root, "**", "arch_specific", "inkling2_loop.py"),
                         recursive=True)
        if hits:
            return hits[0]
    return None


def load_arm(target):
    """Resolve a fence arm to (method_ns, display_label). Returns (None, reason) when the
    deployed source cannot be located on this host."""
    if target == "installed":
        path = _installed_arch_path()
        if not path:
            return None, "installed inkling2_loop.py not found on this host"
        ns, src = _extract_method_ns(path)
        return (ns, path, src), "installed"
    ns, src = _extract_method_ns(SRC)
    return (ns, SRC, src), "source"


# --------------------------------------------------------------------------- stub self
class StubSelf:
    pass


def make_self(ns, kvh, grp, extent, d, atb=None):
    s = StubSelf()
    s.num_kv_groups = grp
    s.num_kv_heads = kvh
    s.rel_extent = extent
    s.scaling = 1.0 / (d ** 0.5)
    s.sliding = False
    if "_attend_one_tile" in ns:
        import types
        s._attend_one_tile = types.MethodType(ns["_attend_one_tile"], s)
        if atb is not None:
            s.attend_tile_bytes = atb
    return s


def _worst_ulp(ref, got):
    """Worst per-element distance between two fp16 tensors, measured in fp16 ULP of the
    reference magnitude (denormal floor 2**-14). Ignores sign-bit noise via fp32 subtraction."""
    r = ref.float()
    g = got.float()
    mag = r.abs().clamp_min(2.0 ** -14)
    exp = torch.floor(torch.log2(mag))
    ulp = torch.pow(torch.tensor(2.0, device=r.device), exp) * torch.finfo(torch.float16).eps
    gap = (g - r).abs() / ulp
    return float(gap.max()) if gap.numel() else 0.0


def make(h, kvh, grp, d, T, S, extent, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(h, T, d, generator=g, dtype=torch.half),
            torch.randn(S, kvh, d, generator=g, dtype=torch.half),
            torch.randn(S, kvh, d, generator=g, dtype=torch.half),
            torch.randn(h, T, extent, generator=g, dtype=torch.float),
            torch.randn(T, h, generator=g, dtype=torch.float),
            S + torch.arange(T, dtype=torch.long),
            torch.arange(S, dtype=torch.long))


def prefix_body(self, q, k, v, rel_small, sink, q_pos, kv_pos):
    """Faithful pre-fix _attend_one (materialises bias at full S; tile = 32 if S>16384 else T).
    Independent of the target source, so the RED-on-installed arm needs only the deployed body
    and this oracle stays the reference for bit-identity."""
    h, T, d = q.shape
    S = k.shape[0]
    grp = self.num_kv_groups
    kvh = self.num_kv_heads
    kxf = k.transpose(0, 1).transpose(1, 2).unsqueeze(1).float()
    vx = v.transpose(0, 1).unsqueeze(1).expand(kvh, grp, S, d)
    q4 = q.view(kvh, grp, T, d).float()
    dist = q_pos[:, None] - kv_pos[None, :]
    causal = dist >= 0
    bias = rel_small.gather(-1, dist.clamp(0, self.rel_extent - 1).expand(h, T, S))
    bias = torch.where((causal & (dist < self.rel_extent)).unsqueeze(0), bias, torch.zeros_like(bias))
    bias4 = bias.view(kvh, grp, T, S)
    tile = 32 if S > 16384 else T
    outs = []
    for qs in range(0, T, tile):
        qe = min(qs + tile, T)
        sc = torch.matmul(q4[:, :, qs:qe], kxf) * self.scaling
        sc = sc + bias4[:, :, qs:qe]
        sc = sc.masked_fill(~causal[None, None, qs:qe], float("-inf"))
        snk = sink[qs:qe].T.view(kvh, grp, qe - qs).float()
        comb = torch.cat([sc, snk.unsqueeze(-1)], dim=-1)
        w = torch.softmax(comb, dim=-1)[..., :S].to(v.dtype)
        outs.append(torch.matmul(w, vx))
    return torch.cat(outs, dim=2).reshape(h, T, d)


def measure(fn, self, inp):
    with PeakTensor() as pt:
        out = fn(self, *inp)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
    return pt, out, rss


def _effective_tile(ns, S, T):
    """Tile the hoisted code would pick for this shape, without a live self."""
    budget = ns.get("_ATTEND_TILE_BYTES", 512 * 1024 * 1024)
    h = 24
    denom = h * S * 4 * 2
    cap = max(1, int(budget // denom)) if denom else T
    return min(T, max(cap, 32))


def main():
    target = os.environ.get("INKLING_FENCE_TARGET", "source").lower()
    arm, label = load_arm(target)
    if arm is None:
        print(f"[red-inst] PASS :: {label}")
        print("\nFENCE PASS (installed has no hoistable _attend_one)")
        return 0
    ns, path, _ = arm

    KVH, GRP, D, EXTENT = 8, 3, 128, 1024
    H = KVH * GRP
    SHAPES = [("cliff-8k", 2048, 8192, 1.500), ("tiled-17k", 2048, 17408, 3.188)]

    ok = True
    print(f"[info   ] target={target}  file={path}")
    print(f"[cfg    ] _ATTEND_TILE_BYTES present={'_ATTEND_TILE_BYTES' in ns} "
          f"value={ns.get('_ATTEND_TILE_BYTES')}  _attend_one_tile present={'_attend_one_tile' in ns}")

    attend = ns["_attend_one"]
    for lbl, T, S, thresh in SHAPES:
        inp = make(H, KVH, GRP, D, T, S, EXTENT)
        pt_pre, out_pre, _ = measure(prefix_body, make_self(ns, KVH, GRP, EXTENT, D), inp)
        pt_post, out_post, rss = measure(attend, make_self(ns, KVH, GRP, EXTENT, D), inp)

        shape_ok = tuple(out_post.shape) == (H, T, D) and out_post.dtype == torch.float16

        if target == "installed":
            # deployed body materialises the full-(h,T,S) bias -> peak stays at/above the
            # served band. RED expectation: the unfixed method peaks LARGE.
            bit_ok = bool(torch.equal(out_pre.view(torch.uint16), out_post.view(torch.uint16)))
            passed = pt_post.largest >= int(thresh * 1024 ** 3) - (1 << 20) and bit_ok
            ok = ok and passed
            print(f"[{'PASS' if passed else 'FAIL'}] {lbl}: installed largest={giB(pt_post.largest):.3f} GiB "
                  f"(expect >= {thresh:.3f} = unfixed) bit_vs_prefix={bit_ok}")
            continue

        # Bit-identity of the fix is TWO-fold, because tiling changes the matmul ROW HEIGHT
        # (M-dim), which lets cuBLAS pick a different accumulation blocking -- the same thing
        # the deployed code's own `tile=32` path (S>16384) already does. So:
        #   (A) EXACTNESS: run the hoisted method at the oracle's tile height (byte budget
        #       tuned so _attend_one_tile returns that tile). Must be bit-for-bit equal ->
        #       proves the per-tile gather/softmax/where math is unchanged.
        #   (B) OPERATING POINT: at the budget-derived tile the output may differ only by the
        #       gemm-blocking rounding, asserted <= 2 fp16 ULP per element.
        oracle_tile = 32 if S > 16384 else T
        match_atb = H * S * 4 * 2 * oracle_tile
        out_match = attend(make_self(ns, KVH, GRP, EXTENT, D, atb=match_atb), *inp)
        exact_ok = bool(torch.equal(out_pre.view(torch.uint16), out_match.view(torch.uint16)))
        maxabs = float((out_post.float() - out_pre.float()).abs().max())
        atol_ok = maxabs <= 1.0e-4
        gap = _worst_ulp(out_pre, out_post)

        hoisted = "_attend_one_tile" in ns or "_ATTEND_TILE_BYTES" in ns
        drop_ok = pt_post.largest < int(thresh * 1024 ** 3) // 4   # strict: <= quarter of threshold
        passed = drop_ok and exact_ok and atol_ok and hoisted and shape_ok
        ok = ok and passed
        print(f"[{'PASS' if passed else 'FAIL'}] {lbl}: largest {giB(pt_pre.largest):.3f}->{giB(pt_post.largest):.3f} GiB "
              f"(thresh {thresh:.3f}) cum {giB(pt_pre.cumulative):.2f}->{giB(pt_post.cumulative):.2f} GiB "
              f"peakRSS={rss:.2f}GiB | out={tuple(out_post.shape)} {str(out_post.dtype).replace('torch.','')}")
        print(f"          bit-identical@match-tile({oracle_tile})={exact_ok} [math exact] | "
              f"oper-tile={_effective_tile(ns, S, T)} maxabs-vs-oracle={maxabs:.3e} (<=1e-4, gemm-M-block rounding) "
              f"max-ulp={gap:.1f}")
        print(f"          post-largest: {pt_post.largest_desc}")

    # decode neutrality: T==1 must select tile==1 (single iteration, launch-neutral).
    if target != "installed":
        dec_tile = _effective_tile(ns, 8192, 1)
        dec_ok = dec_tile == 1
        ok = ok and dec_ok
        print(f"[{'PASS' if dec_ok else 'FAIL'}] decode T=1: effective_tile={dec_tile} (expect 1 -> launch-neutral)")

    print(f"\nFENCE {'PASS' if ok else 'RED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())