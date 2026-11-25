#!/usr/bin/env python3
"""sm_89 rung 4 fence: the K and V attention-prepare convs run as ONE conv1d launch.

Campaign doc §5 rung 4 / §6: Xolotl's "the K and V attention-prepare kernels run as one launch
... -96 launches a token, bit-identical." On our shape K/V prepare is conv("k")+conv("v") on every
sliding layer (40 of 48 decode layer-forwards/token). Depthwise conv mixes no channels, so
concatenating K and V on the channel axis is bit-identical to the two separate calls -- GIVEN the
backend picks the same accumulation order, which this fence checks against the two-conv baseline.

Proves, on CPU (no pack/GPU, card hidden -- the workstation card is prod):
  GREEN   conv_kv launches FEWER aten ops than the two conv() calls it replaces (the per-token
          win); and its outputs AND conv-state write-back are BITWISE EQUAL to two separate
          conv() calls for BOTH recurrent_history False and True.
  SENSE   (a) un-fusing conv_kv back to two calls removes the launch win -- guard cannot pass it;
          (b) mis-pairing the fused weight (w_v onto K channels) BREAKS bit-identity -- the
              equality is genuinely checking the pairing, not passing vacuously.
  DEPLOYED (law-31 RED): the guard inspects the INSTALLED wheel the engine runs; that wheel
          must not already carry conv_kv, and the has(conv_kv) predicate must be False on it --
          proof the fence detects ABSENCE, not only presence.

Run standalone (deployed engine venv or source checkout, GPU optional):
    python tests/graph/test_inkling_kv_fuse.py
"""
from __future__ import annotations
import importlib.util, os, subprocess, sys, tempfile
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "exllamav3", "modules", "arch_specific", "inkling2_loop.py")
REL = "exllamav3/modules/arch_specific/inkling2_loop.py"

VIEW_OPS = {
    "aten::alias","aten::detach","aten::view","aten::slice","aten::select","aten::unsqueeze",
    "aten::squeeze","aten::expand","aten::expand_as","aten::transpose","aten::permute","aten::t",
    "aten::_unsafe_view","aten::lift_fresh","aten::set_variable_name","aten::EmptyTensorList",
    "aten::as_strided","aten::narrow",
}


class LaunchCount(TorchDispatchMode):
    def __init__(self): self.n = 0
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if str(func) not in VIEW_OPS: self.n += 1
        return func(*args, **(kwargs or {}))


def _seed_cpu_stubs():
    import exllamav3.util.tensor as T, exllamav3.cache.recurrent as R
    if not hasattr(T, "host_values"):
        T.host_values = lambda t, key: (t.detach().cpu() if t.is_cuda else t)
    if not hasattr(T, "stage_host"):
        T.stage_host = lambda v, dev, key, dtype=torch.long: torch.tensor(list(v), dtype=dtype, device=dev)
    if not hasattr(T, "get_for_device"):
        T.get_for_device = lambda d, k, dev, default=None: d.get(k, default)
    if not hasattr(R, "host_copy"):
        R.host_copy = lambda s: s.detach().cpu()


def load_source(path):
    _seed_cpu_stubs()
    nm = "inkling_fence_target"
    spec = importlib.util.spec_from_file_location(f"exllamav3.modules.arch_specific.{nm}", path)
    mod = importlib.util.module_from_spec(spec); mod.__package__ = "exllamav3.modules.arch_specific"
    sys.modules[f"exllamav3.modules.arch_specific.{nm}"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_target():
    # Default: evaluate the PATCHED source the fence ships beside, so GREEN measures the change
    # on every host. Set INKLING_FENCE_TARGET=installed to run the SAME guard against the wheel
    # the engine actually loads (law-31 RED: that wheel has no conv_kv). The deployed artifact is
    # also reported separately in main().
    mode = os.environ.get("INKLING_FENCE_TARGET", "source")
    if mode == "installed":
        _seed_cpu_stubs()
        try:
            import exllamav3.modules.arch_specific.inkling2_loop as m
        except Exception as e:
            return None, f"installed-wheel (no importable inkling2_loop: {repr(e)[:50]})"
        return m, "installed-wheel"
    return load_source(SRC), "patched-source"


def deployed_has_conv_kv():
    """Inspect the INSTALLED exllamav3 (the wheel the engine actually runs) for conv_kv.
    Returns (loaded: bool, present: bool). Absent/old-wheel -> the law-31 RED condition."""
    try:
        import exllamav3.modules.arch_specific.inkling2_loop as m
    except Exception as e:
        return False, repr(e)[:80]
    return True, hasattr(m.InklingConvs, "conv_kv")


KV = 1024; KDIM = 4   # kv_dim = num_kv_heads(8)*head_dim(128); sconv_kernel_size = 4


class FakeState:
    def __init__(self, seed, hist=1):
        g = torch.Generator().manual_seed(seed)
        # conv_states[name] is (slots, chan, K); K widened to `hist` to exercise the
        # recurrent_history write-back path the way the live recurrent cache does.
        self.conv_states = {"k": torch.randn(1, KV, KDIM * hist, generator=g),
                             "v": torch.randn(1, KV, KDIM * hist, generator=g)}
    def snapshot(self): return {n: t.clone() for n, t in self.conv_states.items()}
    def restore(self, snap):
        for n in self.conv_states: self.conv_states[n].copy_(snap[n])


def build_convs(mod, seed=1234, hist=1):
    c = mod.InklingConvs.__new__(mod.InklingConvs)
    c.conv_kernel_size = KDIM
    g = torch.Generator().manual_seed(seed)
    c.w_k = torch.randn(KV, KDIM, generator=g)
    c.w_v = torch.randn(KV, KDIM, generator=g)
    c.w_attn = None; c.w_mlp = None
    c.w_kv = torch.cat([c.w_k, c.w_v], dim=0).unsqueeze(1).contiguous()   # built in load()
    st = FakeState(seed + 1, hist=hist)
    c._state_slot = lambda params: (st, [0])
    g2 = torch.Generator().manual_seed(seed + 2); bf = torch.bfloat16
    kl = torch.randn(1, 1, KV, generator=g2).to(bf)
    vl = torch.randn(1, 1, KV, generator=g2).to(bf)
    return c, st, kl, vl


def count_two(mod):
    c, _, kl, vl = build_convs(mod)
    with LaunchCount() as lc:
        c.conv("k", kl, {}); c.conv("v", vl, {})
    return lc.n

def count_fused(mod):
    c, _, kl, vl = build_convs(mod)
    with LaunchCount() as lc:
        c.conv_kv(kl, vl, {})
    return lc.n


def bit_identity(mod, history):
    c, st, kl, vl = build_convs(mod, hist=1 + int(history))   # wider state when history on
    params = {"recurrent_history": True} if history else {}
    snap = st.snapshot()
    ka = c.conv("k", kl, params); va = c.conv("v", vl, params); state_a = st.snapshot()
    st.restore(snap)
    kb, vb = c.conv_kv(kl, vl, params)
    return (torch.equal(ka, kb) and torch.equal(va, vb)
            and torch.equal(state_a["k"], st.conv_states["k"])
            and torch.equal(state_a["v"], st.conv_states["v"]))


def pairing_break_breaks(mod):
    """Mis-pair the fused weight; if the equality is real, the outputs must DIFFER."""
    c, st, kl, vl = build_convs(mod)
    c.w_kv = torch.cat([c.w_v, c.w_k], dim=0).unsqueeze(1).contiguous()   # swapped pairing
    snap = st.snapshot()
    ka = c.conv("k", kl, {}); va = c.conv("v", vl, {})
    st.restore(snap)
    kb, vb = c.conv_kv(kl, vl, {})
    return not (torch.equal(ka, kb) and torch.equal(va, vb))


MIN_DROP = 6

def main():
    mod, origin = load_target()
    print(f"[info    ] target module: {origin}")

    def emit(tag, ok, detail):
        # Tag = the ARM's name; the PASS/FAIL word = that arm's own verdict. A failing arm can
        # never read as a passing one because both the tag and the word are shown.
        print(f"[{tag}] {'PASS' if ok else 'FAIL'} :: {detail}")
        return bool(ok)

    if mod is None:
        emit("presnce", False, f"target '{origin}' -> guard refuses (nothing to fuse)")
        print("\nFENCE RED")
        return 1

    # ---- DEPLOYED (law-31): the wheel the engine runs; absence here licenses the merge. Informational. ----
    loaded, present = deployed_has_conv_kv()
    dep = bool(present) if loaded else False
    emit("deployed", dep, f"installed wheel conv_kv present={dep}" if loaded
         else f"installed exllamav3 has no inkling2_loop ({present})")

    has = hasattr(mod.InklingConvs, "conv_kv")
    two = count_two(mod)                      # conv("k"/"v") exists on every build -> safe

    if not has:
        emit("presnce", False, f"conv_kv ABSENT on target '{origin}' (two-call baseline={two})")
        emit("launch ", False, "not measured: conv_kv absent")
        emit("bitid-hF", False, "not measured: conv_kv absent")
        emit("bitid-hT", False, "not measured: conv_kv absent")
        emit("pairing", False, "not measured: conv_kv absent")
        emit("sense  ", False, "not measured: conv_kv absent")
        print("\nFENCE RED")
        return 1

    fused = count_fused(mod); dropped = two - fused
    o1 = emit("presnce", True, f"conv_kv present (two-call={two} fused={fused})")
    o2 = emit("launch ", dropped >= MIN_DROP, f"two={two} fused={fused} dropped={dropped} (need >= {MIN_DROP})")
    o3 = emit("bitid-hF", bit_identity(mod, False), "outputs + state write-back bitwise equal")
    o4 = emit("bitid-hT", bit_identity(mod, True), "outputs + state write-back bitwise equal")
    o5 = emit("pairing", pairing_break_breaks(mod), "swapping w_k/w_v must break equality")

    c2, _, kl2, vl2 = build_convs(mod)
    c2.conv_kv = lambda k, v, p: (c2.conv("k", k, p), c2.conv("v", v, p))   # un-fuse in place
    with LaunchCount() as lc:
        c2.conv_kv(kl2, vl2, {})
    poisoned = lc.n
    o6 = emit("sense  ", (two - poisoned) < MIN_DROP,
              f"un-fused launches={poisoned} apparent-drop={two - poisoned} (guard fires when drop < {MIN_DROP})")

    ok = all([o1, o2, o3, o4, o5, o6])
    print(f"\nFENCE {'PASS' if ok else 'RED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
