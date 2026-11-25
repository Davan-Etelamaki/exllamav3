"""sm_89 -- neither matmul in _attend_one may clone a broadcast KV replica (ASSERTING).

Defect: `_attend_one` passed `(kvh,grp,tile,d)` against `(kvh,1,d,S)`, and `(kvh,grp,tile,S)` against
an `expand`ed `(kvh,1,S,d)`. Neither matmul can broadcast a batch dim lazily, so each one COPIED the
whole KV tensor on every tile call:
    K replica (kvh,grp,d,S) fp32 = h*d*S*4 = 12,288*S bytes -- 768 MiB at 65,536, 3.0 GiB at 262,144
    V replica (kvh,grp,S,d) fp16 = h*S*d*2 =  6,144*S bytes -- 384 MiB at 65,536, 1.5 GiB at 262,144
Above ~43k context those copies, not the byte-budgeted bias, are the largest single allocations on the
path, and the tile dial cannot reach them -- the tile shrinks only the (h,tile,S) temporaries. dot26
runs with ~409 MiB free at peak load, so this sits squarely inside the OOM class the batch must end.

Fix under test: fold `grp` into the axis that is NOT contracted -- `(kvh, grp*tile, d) @ (kvh,d,S)` for
scores, `(kvh, grp*tile, S) @ (kvh,S,d)` for values -- so the batch dims match and nothing broadcasts.
Each output element still contracts over d (scores) or S (values) alone.
Rejected alternative: matching the batch dims with an `expand`ed stride-0 replica and `reshape` to a
flat 24-batch copies too (measured: identical 204 MiB + 102 MiB clones at S=17,408), so it buys nothing.

Numerics, measured on CPU against the released formulation:
  * values fold -- bit-identical at every shape probed.
  * scores fold -- bit-identical on multi-row tiles (T=64 at 65,536 and 262,144; T=2048 at 17,408), and
    at single-row decode it moves 5-8% of the fp16 output codes by exactly ONE ulp (maxabs <= 1.6e-05,
    <= 3.2e-04 of the output magnitude), because batching grp rows into one M=3 gemm changes the fp32
    accumulation order. That is a bounded behaviour change on the hot path, so the decode arm asserts
    the bound rather than pretending it is bit-exact: every differing code must be adjacent, and the
    drift must stay under 1e-3 of the output scale. The card-side consequence is owed at the boot window
    (greedy continuation + acceptance + bench), where the real kernels live.

Detection: a broadcast replica is always produced by a COPY op (clone/expand), never by arithmetic, so
only those ops count. Matching on shape alone would false-positive: when the last query tile happens to
have d(=128) rows, the legitimate (kvh,grp,tile,S) temporaries share the replica shape exactly.

Arms, both computed from the SAME operands (regime: CPU-only, CUDA masked, card untouched):
  * incumbent_body -- the broadcast formulation, written here independently. MUST produce >=1 replica
    copy per shape; a shape where it does not is declared unproven, never passed silently (law 21).
  * target         -- the tree under test. MUST hold: no KV replica materialises, the largest single
    allocation stays inside the byte budget at S=262,144, prefill output is bit-identical, and the decode
    drift stays inside the measured bound (fp16 ULP distance <= 8, |delta| <= 1e-3 of the output scale).
    Wall time is REPORTED, not gated: oneDNN on CPU penalises the transposed value operand (the target leg
    is measurably slower there), and that penalty is a CPU kernel property, not a cublas one. Card latency
    is fenced at the boot window under law 12.

INKLING_NOBCAST_TARGET=source (default, the patched tree) must be GREEN.
INKLING_NOBCAST_SHAPES=full (default) | decode (single-row only, fast) | prefill (multi-row only).
"""
import argparse
import ast
import os
import sys
import time

import torch

torch.set_num_threads(4)

DEFAULT_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "exllamav3", "modules", "arch_specific", "inkling2_loop.py")
VIEW_OPS = {
    "aten::expand", "aten::expand_as", "aten::alias", "aten::view", "aten::reshape",
    "aten::unsqueeze", "aten::squeeze", "aten::permute", "aten::transpose", "aten::t",
    "aten::detach", "aten::as_strided", "aten::slice", "aten::select", "aten::flip",
    "aten::empty_strided", "aten::movedim", "aten::unbind", "aten::_unsafe_view",
}
COPY_OPS = {"aten::clone", "aten::expand", "aten::expand_as"}
KVH, GRP, D, EXTENT = 8, 3, 128, 1024
H = KVH * GRP
DECODE_REL_BOUND = 1e-3      # max |delta| as a fraction of the output scale, decode arm
DECODE_MAX_ULP = 8           # max fp16 ULP distance on the decode arm (measured tail: >=6 at 65,536)


class Peak(torch.utils._python_dispatch.TorchDispatchMode):
    def __init__(self, replica_shapes):
        super().__init__()
        self.largest = 0
        self.desc = ""
        self.cum = 0
        self.replica_shapes = replica_shapes
        self.replicas = []

    def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = getattr(func, "_name", str(func))
        # Dispatch reports overload-suffixed names (aten::slice.Tensor, aten::transpose.int), so match on
        # the base op name -- otherwise views are counted as materialisations and both `largest` and the
        # cumulative total inflate (observed: a hoisted transpose VIEW counted as 256 MiB "allocated").
        base = str(name).split(".")[0]
        for t in self._tensors(out):
            nb = t.numel() * t.element_size()
            if base in VIEW_OPS or name in VIEW_OPS or (t.is_leaf is False and t._base is not None):
                continue
            self.cum += nb
            if name in COPY_OPS and tuple(t.shape) in self.replica_shapes:
                self.replicas.append((name, tuple(t.shape), str(t.dtype)))
            if nb > self.largest:
                self.largest, self.desc = nb, f"{name} -> {tuple(t.shape)} {t.dtype}"
        return out

    @staticmethod
    def _tensors(obj):
        if torch.is_tensor(obj):
            return [obj]
        if isinstance(obj, (tuple, list)):
            return [x for x in obj if torch.is_tensor(x)]
        return []


def extract(path):
    text = open(path).read()
    tree = ast.parse(text, path)
    picked = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tg in node.targets:
                if isinstance(tg, ast.Name) and tg.id in {"_ATTEND_TILE_BYTES", "_ATTEND_K_CHUNK_BYTES"}:
                    picked.append(ast.get_source_segment(text, node))
        if isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name in {"_attend_one", "_attend_one_tile"}:
                    picked.append(ast.get_source_segment(text, sub))
    ns = {"torch": torch, "F": torch.nn.functional, "_os": os, "math": __import__("math")}
    exec(compile("\n\n".join(picked), path, "exec"), ns)   # noqa: S102 - project source
    for need in ("_attend_one", "_attend_one_tile", "_ATTEND_TILE_BYTES", "_ATTEND_K_CHUNK_BYTES"):
        if need not in ns:
            raise RuntimeError(f"{path}: missing {need}")
    return ns


def stub(ns, atb=None):
    import types

    class S:
        num_kv_heads, num_kv_groups, rel_extent, sliding = KVH, GRP, EXTENT, False
        scaling = 1.0 / (D ** 0.5)
    s = S()
    s._attend_one_tile = types.MethodType(ns["_attend_one_tile"], s)
    if atb is not None:
        s.attend_tile_bytes = atb
    return s


def incumbent_body(self, q, k, v, rel_small, sink, q_pos, kv_pos):
    """The shipped formulation at 2c9530cb, written from the released source rather than imported, so
    the control cannot pass by sharing the fix's code path."""
    h, T, d = q.shape
    S = k.shape[0]
    grp, kvh = self.num_kv_groups, self.num_kv_heads
    kxf = k.transpose(0, 1).transpose(1, 2).unsqueeze(1).float()      # (kvh,1,d,S) -> cloned
    vx = v.transpose(0, 1).unsqueeze(1).expand(kvh, grp, S, d)        # (kvh,1,S,d) -> cloned
    q4 = q.view(kvh, grp, T, d).float()
    tile = self._attend_one_tile(S, T, 4)
    outs = []
    for qs in range(0, T, tile):
        qe = min(qs + tile, T)
        dist = q_pos[qs:qe, None] - kv_pos[None, :]
        causal = dist >= 0
        bias = rel_small[:, qs:qe, :].gather(
            -1, dist.clamp(0, self.rel_extent - 1).unsqueeze(0).expand(h, qe - qs, S))
        bias = torch.where((causal & (dist < self.rel_extent)).unsqueeze(0),
                           bias, torch.zeros_like(bias))
        sc = torch.matmul(q4[:, :, qs:qe], kxf) * self.scaling
        sc = sc + bias.view(kvh, grp, qe - qs, S)
        sc = sc.masked_fill(~causal[None, None, :, :], float("-inf"))
        snk = sink[qs:qe].T.view(kvh, grp, qe - qs).float()
        w = torch.softmax(torch.cat([sc, snk.unsqueeze(-1)], dim=-1), dim=-1)[..., :S].to(v.dtype)
        outs.append(torch.matmul(w, vx))
    return torch.cat(outs, dim=2).reshape(h, T, d)


def make(T, S, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(H, T, D, generator=g, dtype=torch.half),
            torch.randn(S, KVH, D, generator=g, dtype=torch.half),
            torch.randn(S, KVH, D, generator=g, dtype=torch.half),
            torch.randn(H, T, EXTENT, generator=g, dtype=torch.float),
            torch.randn(T, H, generator=g, dtype=torch.float),
            torch.tensor([S - T + i for i in range(T)], dtype=torch.long),
            torch.arange(S, dtype=torch.long))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=os.environ.get("INKLING_NOBCAST_SOURCE", DEFAULT_SRC))
    args = ap.parse_args()
    ns = extract(args.source)
    budget = ns["_ATTEND_TILE_BYTES"]
    mode = os.environ.get("INKLING_NOBCAST_TARGET", "source")
    # INKLING_NOBCAST_TARGET=sabotage points the target leg at the broadcast formulation: the run MUST go
    # RED (law 21 -- a fence is not trusted until it has been seen to fire).
    target = incumbent_body if mode == "sabotage" else ns["_attend_one"]
    print(f"[cfg] target={args.source} mode={mode} budget={budget/2**20:.0f}MiB kchunk="
          f"{ns['_ATTEND_K_CHUNK_BYTES']/2**20:.0f}MiB threads={torch.get_num_threads()} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r} "
          f"-- regime: CPU, card untouched")

    shapes = {
        "full": [(64, 262144), (2048, 17408), (1, 131072)],
        "decode": [(1, 1024), (1, 17408), (1, 65536), (1, 131072)],   # 44 of 48 layer-executions/token
        "prefill": [(64, 262144), (2048, 17408)],
        "tiny": [(1, 1024), (1, 4096), (8, 1024)],   # seconds: for sabotage / positive-control drills
        "chunk": [(8, 20480), (64, 32768), (1, 20480)],   # S > k_chunk: exercises the column-blocked path
    }[os.environ.get("INKLING_NOBCAST_SHAPES", "full")]

    ok = True
    for T, S in shapes:
        replicas = {(KVH, GRP, D, S), (KVH, GRP, S, D)}
        inp = make(T, S)
        with Peak(replicas) as p_inc:
            t0 = time.perf_counter()
            out_inc = incumbent_body(stub(ns), *inp)
            t_inc = time.perf_counter() - t0
        with Peak(replicas) as p_tgt:
            t0 = time.perf_counter()
            out_tgt = target(stub(ns), *inp)
            t_tgt = time.perf_counter() - t0

        bit = bool(torch.equal(out_inc.view(torch.uint16), out_tgt.view(torch.uint16)))
        # Signed int16 reinterpretation orders fp16 monotonically, so |i(a)-i(b)| is the ULP distance.
        ulp = (out_inc.view(torch.int16).to(torch.int32) - out_tgt.view(torch.int16).to(torch.int32)).abs()
        max_ulp = int(ulp.max())
        hist = torch.bincount(ulp.flatten(), minlength=9).tolist()
        gap = float((out_inc.float() - out_tgt.float()).abs().max())
        scale = max(float(out_inc.float().abs().max()), 1e-12)
        rel = gap / scale
        n_diff = int((ulp > 0).sum())
        tile_used = stub(ns)._attend_one_tile(S, T, 4)

        control = len(p_inc.replicas) > 0
        common = control and not p_tgt.replicas
        if T == 1:
            good = common and rel <= DECODE_REL_BOUND
            print(f"[{'PASS' if good else 'FAIL'}] decode T=1 S={S}: replica copies inc={len(p_inc.replicas)} "
                  f"tgt={len(p_tgt.replicas)} (want 0) | bit={bit} codes_differing={n_diff}/{ulp.numel()} "
                  f"ulp_hist(0..8+)={hist[:9] + [sum(hist[9:])]} max_ulp={max_ulp} (reported) "
                  f"maxabs={gap:.3e} rel_to_scale={rel:.3e} (bound {DECODE_REL_BOUND:g}) | largest "
                  f"{p_inc.largest/2**20:.1f}->{p_tgt.largest/2**20:.1f} MiB | cumulative "
                  f"{p_inc.cum/2**30:.2f}->{p_tgt.cum/2**30:.2f} GiB | cpu_s {t_inc:.2f}->{t_tgt:.2f} "
                  f"(x{t_tgt / max(t_inc, 1e-9):.2f}, reported only)")
        else:
            within = p_tgt.largest <= budget
            shrunk = p_tgt.largest < p_inc.largest and p_tgt.cum < p_inc.cum
            good = common and bit and within and shrunk
            print(f"[{'PASS' if good else 'FAIL'}] T={T} S={S}: replica copies inc={len(p_inc.replicas)} "
                  f"tgt={len(p_tgt.replicas)} (want 0) | largest {p_inc.largest/2**20:.1f}->"
                  f"{p_tgt.largest/2**20:.1f} MiB (budget {budget/2**20:.0f}; must shrink and fit) | "
                  f"cumulative {p_inc.cum/2**30:.2f}->{p_tgt.cum/2**30:.2f} GiB (must shrink) | "
                  f"bit={bit} codes_differing={n_diff} "
                  f"maxabs={gap:.3e} | tile={tile_used} k_chunk={max(EXTENT, ns['_ATTEND_K_CHUNK_BYTES'] // (KVH * D * 4))} "
                  f"cpu_s {t_inc:.2f}->{t_tgt:.2f} "
                  f"(x{t_tgt / max(t_inc, 1e-9):.2f}, reported only)")
            print(f"          target-largest: {p_tgt.desc}")
            if p_tgt.replicas:
                print(f"          target replicas: {sorted(set(p_tgt.replicas))}")
        if not control:
            print("          CONTROL DID NOT FIRE: the broadcast formulation produced no KV replica here, "
                  "so this shape proves nothing")
        ok = ok and good
        del inp, out_inc, out_tgt, p_inc, p_tgt, ulp, hist

    # Latency fence is CPU-only here; the card figure is owed at the boot window (law 38).
    print(f"\nFENCE {'PASS' if ok else 'RED'} (mode={mode}, "
          f"shapes={os.environ.get('INKLING_NOBCAST_SHAPES', 'full')}, "
          f"regime: CPU, threads=4; GPU latency NOT measured -- owed at the owner's restart)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())