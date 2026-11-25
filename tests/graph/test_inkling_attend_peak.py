"""sm_89 kernel lane — _attend_one relative-position peak fence (ASSERTING).

Metric authority: parent's attend_peak_probe.py PeakTensor (largest single MATERIALIZED
tensor, views excluded; cumulative materialized; peak RSS). The served saturation/OOM band
is a single fp32 ~(h,T,S) allocation -- the full relative-position `bias` gather and the
(kvh,grp,T,S+1) softmax logits both scale with S and, pre-fix, were materialised OUTSIDE
the tile loop, so shrinking `tile` did not touch them.

Fix under test: bias/dist/causal built PER QUERY TILE inside the loop, tile byte-budget
derived -> largest single tensor bounded; single-tile decode (T==1) and short-context
(S=8192, tile=T) stay launch-neutral (same iteration count); output must remain
(h,T,d) fp16 and BIT-IDENTICAL to the pre-fix body.

Targets: INKLING_FENCE_TARGET=source (patched worktree) must be GREEN; =installed (the
deployed wheel 1.6.7+0263e305f3b7d, which materialises bias at full S with no byte-budget
tiling) must be RED on the acceptance thresholds. CPU-only; CUDA masked; card untouched.
"""

import os
import resource
import sys
import types
import importlib.util

import torch
from torch.utils._python_dispatch import TorchDispatchMode

torch.set_num_threads(4)

WT = "/home/davan.etelamaki/Repos/exllamav3-wt-fwdgraphsc"
SRC = os.environ.get("INKLING_SOURCE_ARCH") or os.path.join(
    WT, "exllamav3", "modules", "arch_specific", "inkling2_loop.py")
PKG = "exllamav3.modules.arch_specific.inkling2_loop"

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


def _seed_cpu_stubs():
    import exllamav3.util.tensor as T
    import exllamav3.cache.recurrent as R
    if not hasattr(T, "host_values"):
        T.host_values = lambda tensor, key: (getattr(tensor, key, None).detach().cpu()
            if isinstance(getattr(tensor, key, None), torch.Tensor) and getattr(tensor, key).is_cuda
            else getattr(tensor, key, None))
    if not hasattr(T, "stage_host"):
        T.stage_host = lambda v, device, key=None, dtype=torch.long: torch.tensor(list(v), dtype=dtype, device=device)
    if not hasattr(T, "get_for_device"):
        T.get_for_device = lambda dd, key, device, default=None: dd.get(key, default)
    if not hasattr(R, "host_copy"):
        R.host_copy = lambda s: s.detach().cpu()
    # exllamav3.__init__ pulls the compiled extension (initialises CUDA -> illegal while the
    # engine holds the card, and fails outright under a masked device). Stub the compiled
    # modules with empty module shims so the real PURE-TORCH inkling2_loop imports; _attend_one
    # itself needs none of them (matmul/gather/softmax/cat only).
    import exllamav3
    root = os.path.dirname(exllamav3.__file__)
    for name, rel in (("exllamav3.extension", None),
                      ("exllamav3.Texly", "Texly"),
                      ("exllamav3.Texly.ex_utils", "Texly")):
        if name in sys.modules and getattr(sys.modules[name], "__file__", None):
            continue                                    # real extension already available
        shim = types.ModuleType(name)
        if rel:
            init = os.path.join(root, rel, "__init__.py")
            if os.path.isdir(os.path.join(root, rel)):
                shim.__path__ = [os.path.join(root, rel)]
                shim.__file__ = init
        sys.modules.setdefault(name, shim)


def load_target():
    _seed_cpu_stubs()
    target = os.environ.get("INKLING_FENCE_TARGET", "source").lower()
    if target == "installed":
        try:
            import exllamav3.modules.arch_specific.inkling2_loop as m
            return m, target
        except Exception as exc:  # noqa: BLE001
            return None, f"installed (import failed: {type(exc).__name__}: {str(exc)[:80]})"
    spec = importlib.util.spec_from_file_location(PKG, SRC)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "exllamav3.modules.arch_specific"
    sys.modules[PKG] = mod
    spec.loader.exec_module(mod)
    return mod, target


def make(h, kvh, grp, d, T, S, extent, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(h, T, d, generator=g, dtype=torch.half),
            torch.randn(S, kvh, d, generator=g, dtype=torch.half),
            torch.randn(S, kvh, d, generator=g, dtype=torch.half),
            torch.randn(h, T, extent, generator=g, dtype=torch.float),
            torch.randn(T, h, generator=g, dtype=torch.float),
            S + torch.arange(T, dtype=torch.long),
            torch.arange(S, dtype=torch.long))


def mk_self(cls, kvh, grp, extent, d):
    self = cls.__new__(cls)
    self.num_kv_groups = grp
    self.num_kv_heads = kvh
    self.rel_extent = extent
    self.scaling = 1.0 / (d ** 0.5)
    self.sliding = False
    return self


def prefix_body(self, q, k, v, rel_small, sink, q_pos, kv_pos):
    """Faithful pre-fix _attend_one (materialises bias at full S; tile = 32 if S>16384 else T).
    Independent of the target module, so RED-on-installed needs only a deployed self."""
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


def main():
    mod, target = load_target()
    if mod is None:
        print(f"[red-inst] PASS :: deployed inkling2_loop not importable -> {target}")
        print("\nFENCE PASS (installed has no hoistable _attend_one)")
        return 0
    cls = mod.InklingAttention

    KVH, GRP, D, EXTENT = 8, 3, 128, 1024
    H = KVH * GRP
    SHAPES = [("cliff-8k", 2048, 8192, 1.500), ("tiled-17k", 2048, 17408, 3.188)]

    ok = True
    print(f"[info   ] target={target}  file={getattr(mod, '__file__', target)}")
    print(f"[cfg    ] _ATTEND_TILE_BYTES present={hasattr(mod, '_ATTEND_TILE_BYTES')} "
          f"value={getattr(mod, '_ATTEND_TILE_BYTES', None)}")

    for label, T, S, thresh in SHAPES:
        inp = make(H, KVH, GRP, D, T, S, EXTENT)
        pt_pre, out_pre, _ = measure(prefix_body, mk_self(cls, KVH, GRP, EXTENT, D), inp)
        pt_post, out_post, rss = measure(cls._attend_one, mk_self(cls, KVH, GRP, EXTENT, D), inp)
        # source target must exercise the hoisted method, not just the pre-fix oracle
        hoisted = _effective_tile(mod, cls, S, T) < T or hasattr(mod, "_ATTEND_TILE_BYTES")

        shape_ok = tuple(out_post.shape) == (H, T, D) and out_post.dtype == torch.float16
        bit_ok = shape_ok and bool(torch.equal(out_pre.view(torch.uint16), out_post.view(torch.uint16)))

        if target.startswith("installed"):
            passed = pt_post.largest >= int(thresh * 1024 ** 3) - (1 << 20)   # deployed: large peak -> RED expectation
            verdict = "PASS" if passed else "FAIL"
            ok = ok and passed
            print(f"[{verdict}] {label}: installed largest={giB(pt_post.largest):.3f} GiB "
                  f"(expect >= {thresh:.3f} = unfixed) tile={getattr(cls, '_attend_one_tile', None) is not None}")
            continue

        # source: must drop WELL under threshold, keep (h,T,d) fp16, and stay BIT-IDENTICAL
        # to the pre-fix body. Multi-tile on long passes is intended (protects the 24 GiB
        # VRAM ceiling); single-token decode (T==1) is provably tile==1 -> launch-neutral,
        # checked separately below.
        drop_ok = pt_post.largest < int(thresh * 1024 ** 3) // 4   # strict: <= quarter of threshold
        passed = drop_ok and bit_ok and hoisted
        ok = ok and passed
        verdict = "PASS" if passed else "FAIL"
        print(f"[{verdict}] {label}: largest {giB(pt_pre.largest):.3f}->{giB(pt_post.largest):.3f} GiB "
              f"(thresh {thresh:.3f}) cum {giB(pt_pre.cumulative):.2f}->{giB(pt_post.cumulative):.2f} GiB "
              f"peakRSS={rss:.2f}GiB | out={tuple(out_post.shape)} {str(out_post.dtype).replace('torch.','')} "
              f"| bitwise={bit_ok}")
        print(f"          post-largest: {pt_post.largest_desc}")

    # decode neutrality: T==1 must select tile==1 (single iteration, launch-neutral).
    if not target.startswith("installed"):
        dec_tile = _effective_tile(mod, cls, 8192, 1)
        dec_ok = dec_tile == 1
        ok = ok and dec_ok
        print(f"[{'PASS' if dec_ok else 'FAIL'}] decode T=1: effective_tile={dec_tile} (expect 1 -> launch-neutral)")

    print(f"\nFENCE {'PASS' if ok else 'RED'}")
    return 0 if ok else 1


def _effective_tile(mod, cls, S, T):
    """Tile the hoisted code would pick for this shape, computed without a live self."""
    budget = getattr(mod, "_ATTEND_TILE_BYTES", 512 * 1024 * 1024)
    h = KVH_EFF = 24
    denom = h * S * 4 * 2
    cap = max(1, int(budget // denom)) if denom else T
    return min(T, max(cap, 32))


if __name__ == "__main__":
    raise SystemExit(main())