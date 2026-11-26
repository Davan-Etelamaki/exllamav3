"""sm_89 kernel lane — rung 5 fence: stack q/rel/gate projections into one mat-vec + keep the
index math on device. CPU-dispatch proxy (TorchDispatchMode, views excluded); CPU-fenceable
while both cards are busy -- imports nothing that touches CUDA.

What is python/torch-feasible vs build-bound, established this rung (with citations):

BUILD-BOUND (NOT shippable as a python edit):
  * The "one mat-vec per token" headline. exl3 stores each projection as its own packed Linear:
    Linear.forward delegates wholly to self.inner.forward -- a single ext GEMV over that one
    packed tensor -- exllamav3/modules/linear.py:813/843. wq/wo/gate/wr are separate Linear
    objects (the self.wq/self.wo/self.gate/self.wr Linear ctors in InklingAttention.__init__;
    symbol-bound — inkling2_loop.py line numbers decay with every edit). No plain .weight exists
    to torch.cat in python:
    a stacked mat-vec needs a concatenated exl3/fp16 pack (pack change) or a new grouped-GEMV
    ext entry (ext change). Either way it rides the build, like rungs 1-3, and it is NOT
    bit-identical (concat re-lays quant blocks). Reported, not shipped.

PYTHON-FEASIBLE + BITWISE, MEASURED HERE (the rung-5 python sliver):
  * log_scale computes the position term `tau` from pos_row; on GLOBAL layers project_qkv
    and rel_logits each recompute the SAME tau over the SAME pos_row. Compute tau once and
    pass it into both -> removes one duplicated tau chain per global layer, bit-identical.

RUNG-5 LATENCY HALF (kept, not a kernel-count win; boot-gated):
  * _pos_row(:719) already returns a device tensor, so the only per-step host round-trip is
    _positions(:541).positions.cpu().tolist(). Ring index math (ring_read/ring_write) builds its
    aranges on-device from that ONE list. Deleting the .cpu() removes a D2H SYNC (a CUDA-graph
    capture blocker), NOT a kernel launch -- so it moves latency, not kernels/token. Boot-pending.
"""

import os
import sys
import importlib
import importlib.util
import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ARCH_REL = os.path.join("exllamav3", "modules", "arch_specific", "inkling2_loop.py")
PKG = "exllamav3.modules.arch_specific.inkling2_loop"
GEN = torch.Generator().manual_seed(20261124)

# served inkling-2-mini dims (config.json): hidden 2304, 24 q heads, head_dim 128, d_rel 16
HQ = 24
HD = 128
DREL = 16
THR = 22.6                       # qk_threshold / rel_threshold proxies
ALPHA = 0.1                      # log_alpha
FLOOR = 8192                     # log_floor
# pos_row spans [0, 16384) across t taps: half the taps sit under floor (tau clamped to 1.0),
# half are past floor (log term live). A single tap would pin every tau to 1.0 and make both
# the identity and the sabotage degenerate. 1xT row, the shape _pos_row returns for b=1.
POS = torch.arange(0, 16384, 512, dtype=torch.long).unsqueeze(0)   # (1, 32)

# global layers per decode token, from the layer_map (architecture/inkling2_loop.py):
#   core 6..18 x3 passes + tail 23; global iff (lid+1) % 6 == 0 -> lids {5,11,17,23} -> 4 x ... = 8
GLOBALS_PER_TOKEN = 8

VIEW_OPS = {
    "aten::alias", "aten::detach", "aten::view", "aten::slice", "aten::select",
    "aten::unsqueeze", "aten::squeeze", "aten::expand", "aten::expand_as",
    "aten::transpose", "aten::permute", "aten::t", "aten::_unsafe_view",
    "aten::lift_fresh", "aten::set_variable_name", "aten::EmptyTensorList",
    "aten::as_strided", "aten::narrow",
}


class LaunchCount(TorchDispatchMode):
    """CPU dispatch-op counter; proxy for CUDA kernel launches. Views/reshapes excluded."""

    def __init__(self):
        self.n = 0

    def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
        if str(func) not in VIEW_OPS:
            self.n += 1
        return func(*args, **(kwargs or {}))


def count(fn):
    with LaunchCount() as lc:
        out = fn()
    return lc.n, out


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


def load_source():
    _seed_cpu_stubs()
    path = os.environ.get("INKLING_SOURCE_ARCH") or os.path.join(ROOT, ARCH_REL)
    spec = importlib.util.spec_from_file_location(PKG, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "exllamav3.modules.arch_specific"
    sys.modules[PKG] = mod
    spec.loader.exec_module(mod)
    return mod


def _installed_arch_path():
    override = os.environ.get("INKLING_DEPLOYED_ARCH")
    if override:
        return override
    import exllamav3
    return os.path.join(os.path.dirname(exllamav3.__file__),
                        "modules", "arch_specific", "inkling2_loop.py")


def _deployed_has_fix():
    """Textual readback of the DEPLOYED artifact (no CUDA import). rung-5 fix == compute_log_tau."""
    path = _installed_arch_path()
    if not os.path.exists(path):
        return False, f"{path} ABSENT"
    with open(path) as fh:
        return ("def compute_log_tau" in fh.read()), path


def load_installed():
    """Where the engine loads inkling2 from, and whether the fix is there.

    Presence is decided by importlib.util.find_spec alone -- the resolution path proves the
    wheel is the load source (on dot26 MainPID find_spec resolves to site-packages). An
    import EXCEPTION is a DISTINCT, differently-worded condition: under CUDA_VISIBLE_DEVICES=""
    importing the package raises a CUDA-init error, which must never be reported as `absent`.
    """
    spec = importlib.util.find_spec(PKG)
    if spec is None or not spec.origin:
        return f"ABSENT :: {PKG} not found by find_spec on this interpreter"
    path = spec.origin
    _seed_cpu_stubs()
    try:
        import exllamav3.modules.arch_specific.inkling2_loop as m
        return f"PRESENT at {path} :: compute_log_tau present={hasattr(m, 'compute_log_tau')}"
    except Exception as exc:  # noqa: BLE001
        return (f"PRESENT at {path} :: not imported HERE ({type(exc).__name__}: "
                f"{str(exc)[:60]}) -- import masked, NOT absent")


# ---- the measured python sliver: shared-tau log_scale --------------------------------------

def log_scale_current(x, pos, alpha, floor):
    """Mirror of inkling2_loop.log_scale (:83)."""
    input_dtype = x.dtype
    tau = 1.0 + alpha * torch.log(((pos + 1).float() / floor).clamp(min=1.0))
    return (x.float() * tau[..., None, None]).to(input_dtype)


def compute_tau(pos, alpha, floor):
    """The shared position term, hoisted out of log_scale."""
    return 1.0 + alpha * torch.log(((pos + 1).float() / floor).clamp(min=1.0))


def log_scale_with_tau(x, tau):
    """log_scale application only; tau precomputed. Bit-identical to log_scale_current given tau."""
    input_dtype = x.dtype
    return (x.float() * tau[..., None, None]).to(input_dtype)


def tau_fold_is_bitwise(mod):
    """Apply the real module log_scale vs shared-tau path to q and rel tensors; bitwise-equal."""
    q = torch.randn(1, POS.numel(), HQ, HD, generator=GEN, dtype=torch.bfloat16)
    rs = torch.randn(1, POS.numel(), HQ, DREL, generator=GEN, dtype=torch.bfloat16)
    a_q = mod.log_scale(q, POS, ALPHA, FLOOR).view(torch.uint16)
    a_rs = mod.log_scale(rs, POS, ALPHA, FLOOR).view(torch.uint16)
    tau = compute_tau(POS, ALPHA, FLOOR)
    b_q = log_scale_with_tau(q, tau).view(torch.uint16)
    b_rs = log_scale_with_tau(rs, tau).view(torch.uint16)
    return bool(torch.equal(a_q, b_q)) and bool(torch.equal(a_rs, b_rs))


def tau_sabotage_breaks(mod):
    """Sabotage: a tau computed from doubled positions must break bitwise equality.
    Doubled (not +k) because for pos<FLOOR-1 the clamp(min=1) pins small-shift taus to
    1+alpha*log(1); doubling lands pos+1>=FLOOR so the log term genuinely changes."""
    q = torch.randn(1, POS.numel(), HQ, HD, generator=GEN, dtype=torch.bfloat16)
    real = mod.log_scale(q, POS, ALPHA, FLOOR).view(torch.uint16)
    bad_tau = compute_tau(2 * POS + 1, ALPHA, FLOOR)       # positions differ in the log-active region
    sab = log_scale_with_tau(q, bad_tau).view(torch.uint16)
    return not bool(torch.equal(real, sab))


def main():
    mod = load_source()
    print(f"[stack  ] INFO :: projected q/wr/gate are packed Linear.ext GEMVs (linear.py:843); "
          f"python cannot cat packed weights -> stacked mat-vec = PACK+EXT (build), not shipped")

    ok = True

    # law 31: against the DEPLOYED wheel this same fence must go RED -- the artifact carries
    # no compute_log_tau, so the fused/shared-tau path does not exist there. Never a false GREEN.
    target = os.environ.get("INKLING_FENCE_TARGET", "source").lower()
    dep_has, dep_path = _deployed_has_fix()
    if target == "installed":
        gate_ok = dep_has is False
        print(f"[red-inst] {'PASS' if gate_ok else 'FAIL'} :: deployed wheel compute_log_tau "
              f"present={dep_has} ({dep_path}) -> target=installed expects ABSENT")
        print(f"\nFENCE {'PASS' if gate_ok else 'RED'}")
        return 0 if gate_ok else 1

    # [bitid ] shared-tau log_scale is bit-identical to the module's log_scale (q and rel)
    bitok = tau_fold_is_bitwise(mod)
    ok = ok and bitok
    print(f"[bitid  ] {'PASS' if bitok else 'FAIL'} :: shared-tau log_scale == module log_scale, bitwise (q+rel)")

    # [pairing] sabotage: a wrong-position tau must break the equality (proves the guard is live)
    broke = tau_sabotage_breaks(mod)
    ok = ok and broke
    print(f"[pairing] {'PASS' if broke else 'FAIL'} :: shifted-position tau breaks bitwise equality={broke}")

    # [stack  ] measured launch drop: two log_scale recomputing tau vs one shared tau (per global layer)
    q = torch.randn(1, POS.numel(), HQ, HD, generator=GEN, dtype=torch.bfloat16)
    rs = torch.randn(1, POS.numel(), HQ, DREL, generator=GEN, dtype=torch.bfloat16)
    n_cur = count(lambda: (log_scale_current(q, POS, ALPHA, FLOOR),
                           log_scale_current(rs, POS, ALPHA, FLOOR)))[0]
    n_shr = count(lambda: ((lambda tau: (log_scale_with_tau(q, tau), log_scale_with_tau(rs, tau)))
                          (compute_tau(POS, ALPHA, FLOOR))))[0]
    per_layer_drop = n_cur - n_shr
    per_token_drop = per_layer_drop * GLOBALS_PER_TOKEN
    stack_ok = per_layer_drop > 0
    ok = ok and stack_ok
    print(f"[stack  ] {'PASS' if stack_ok else 'FAIL'} :: q-path+rel-path log_scale: current={n_cur} "
          f"shared={n_shr} drop/layer={per_layer_drop} x globals={GLOBALS_PER_TOKEN} "
          f"=> {per_token_drop} aten/token (python sliver; the mat-vec itself is build)")

    print(f"[info   ] target module: worktree source {ARCH_REL}")
    # Presence = find_spec (proves the wheel is the load path). The running engine maps the
    # wheel's .so (dot26 MainPID /proc/<pid>/maps, 5 mappings, cwd is the checkout but the
    # resolved module is site-packages), NOT the live checkout. A masked import (CUDA-init under
    # CUDA_VISIBLE_DEVICES="") is worded 'not imported HERE', never 'absent'.
    print(f"[deploy ] {load_installed()}")

    print(f"\nFENCE {'PASS' if ok else 'RED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())