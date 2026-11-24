"""No pageable host<->device copy inside the inkling2 whole-forward capture region (blocker #3).

Live witness (dot26, wheel 1.6.7+038204d31a713, journal 2026-11-24 03:34:47):

    Whole-forward CUDA graph disabled for this process: RuntimeError: Cannot copy between CPU and
    CUDA tensors during CUDA graph capture unless the CPU tensor is pinned. Please use
    tensor.pin_memory() or allocate the tensor with pin_memory=True.

torch raises that eagerly from aten's copy_cpu_to_cuda whenever a stream is capturing and the host
side of the copy is pageable. It is the third killer of this model's whole-forward graph: #1 was a
blocking param upload (5004ce2e), #2 was readbacks that synchronized the recording stream (38204d31).

Where it comes from. The captured region is generator.py -> ForwardGraphRunner.run -> model.forward
for one steady-state decode round. Every module that region runs was scanned (SCAN below) for
constructors that build a tensor FROM HOST DATA and aim it at a non-CPU device - the shape a
pageable H2D copy takes in Python. Two survive the allowlist, both in inkling2_loop.py:

    InklingMoE.forward        entries = torch.tensor(order[n:m], device = x.device, dtype = torch.long)
    InklingAttention._pos_row   torch.tensor(params.get("position", 0), device = self.device)

The MoE one runs in every layer of every pass, so it is the copy the driver hit. Its neighbour four
lines up, `sel_flat.tolist()`, is the identical copy in the other direction - `.tolist()` moves a
CUDA tensor to freshly allocated PAGEABLE host memory, which is why the census's deferred item A
surfaced as this RuntimeError and not as the 900 synchronization error.

Fixed at the source, in one place: util/tensor.py adds pinned_stage / host_values / stage_host.
Outside a capture they behave exactly as before (host_values returns the same blocking .cpu()
result; stage_host uploads a fresh temporary, which CUDA stages internally). Inside a capture the
transfer goes through a reusable PINNED buffer with a non-blocking copy, which the driver accepts
and records; the buffer's address is stable, so a replay re-reads it - the same contract
Generator._staging and Embedding._pinned_staging already provide. What capture still cannot do is
branch on a value it has not produced yet: the recorded region replays the capture round's expert
selection, so routing fidelity across rounds stays deferred item A's job (device routing tables +
a grouped/stacked-weight expert tier), not this patch's.

Two claims, both CPU-only (no CUDA context at collection or at run):
  * static census - every host-data constructor in the region's modules, each with its reason;
  * behaviour - with a fake pinned allocator and a forced capture flag standing in for the driver,
    the transfers really do travel through the pinned slot, really do stay synchronous when eager,
    and the MoE still computes the same mixture.

Named blind spot, so the next reader does not read this census as the whole capture story: a scan
of constructors cannot see a host tensor moved by a positional .to(device). One such move sits on
the region's input path and is NOT a pageable copy, so it is not this blocker -
modules/embedding.py:263 `x.to(w_dev)` uploads the staged (pinned) ids to a resident embedding's
device, pinned source but a BLOCKING copy, i.e. blocker #1's class, and it only runs for packs
whose embedding is a resident table rather than a RowTable. The packs that boot on this path use
the table (row_table.py uploads from pinned buffers, non-blocking). Deferred with that reason,
next to items B/C/D.

Run standalone (no pytest, works in the deployed engine venv):
    python tests/graph/test_capture_region_host_uploads.py
"""

from __future__ import annotations

import ast
import collections
import os
import tempfile
import types

import torch

try:
    import pytest

    pytestmark = pytest.mark.nogpu
except ImportError:  # the deployed engine venv ships no pytest; the file still runs standalone
    pytestmark = ()

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir))

# Every module the captured forward runs, in execution order: the region root and its input
# preparation, the token-embedding gather, the inkling2 layers, the KV cache layers it writes and
# reads, the norms and Linears each layer calls, and the movers themselves. A host-data constructor
# aimed at a device in any of these files is a pageable copy inside the capture.
SCAN = [
    "exllamav3/model/model.py",
    "exllamav3/model/model_ls.py",
    "exllamav3/generator/fwd_graph.py",
    "exllamav3/cache/recurrent_util.py",
    "exllamav3/modules/module.py",
    "exllamav3/modules/embedding.py",
    "exllamav3/modules/row_table.py",
    "exllamav3/modules/rmsnorm.py",
    "exllamav3/modules/linear.py",
    "exllamav3/modules/quant/exl3.py",
    "exllamav3/modules/arch_specific/inkling2_loop.py",
    "exllamav3/cache/fp16.py",
    "exllamav3/cache/quant.py",
    "exllamav3/cache/nvfp4.py",
    "exllamav3/util/tensor.py",
    "exllamav3/util/device_copy.py",
]

# Constructors that copy HOST DATA into a tensor. torch.zeros/ones/full/arange with device= are
# NOT here: they generate on the device and never stage through host memory. Stated limit of a
# syntax-only scan: a host tensor moved by a positional .to(device) is invisible to it, which is
# why the behavioural tests below check the transfers the region's code actually issues.
HOST_DATA_CTORS = {"tensor", "as_tensor", "from_numpy", "frombuffer", "fromiter", "fromlist"}

# The census, keyed exactly as _uploads reports a site: {(file, enclosing scope): {ctor: count}}.
# Each entry names why that copy is outside every capture. Everything else in SCAN must be clean -
# a new pageable copy fails the test, and a site that leaves the region must be deleted from here
# so the table stays the truth.
ALLOWED = {
    # FP4/FP8 checkpoint tensors dequantized while the weight loads. Both constructors sit in
    # functions reached only while loading (modules/linear.py load paths, lines 241-558); load
    # runs once before the generator exists, so no capture is ever recording.
    ("exllamav3/modules/linear.py", "_mxfp4_dequant"): {"tensor": 1},
    ("exllamav3/modules/linear.py", "Linear.dequant_e8m0_blocks_"): {"tensor": 1},
    # The reconstructed-column readout mirror (modules/quant/exl3.py), called only from
    # modules/sparse_head.py:382 (SparseHead.mirror). The inkling2 readout is InklingOutputRead,
    # so this is not inside the captured region, and sparse-head models are not graph-eligible.
    ("exllamav3/modules/quant/exl3.py", "LinearEXL3.reconstruct_columns"): {"tensor": 2},
}


def _parse(path: str) -> ast.Module:
    with open(path, encoding = "utf-8") as fh:
        return ast.parse(fh.read(), filename = path)


def _scopes(tree: ast.Module) -> dict:
    """{id(node): dotted scope} for every function/class, so a hit reports the frame it sits in"""
    scopes = {}

    def visit(node, scope):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                child_scope = f"{scope}.{child.name}" if scope else child.name
                scopes[id(child)] = child_scope
                visit(child, child_scope)
            else:
                visit(child, scope)

    visit(tree, "")
    return scopes


def _uploads(path: str) -> list:
    """[(line, scope, ctor, device expression)] for host-data constructions aimed at a non-CPU
    device, plus any bare .cuda() call (an explicit upload of an existing host tensor)"""
    tree = _parse(path)
    scopes = _scopes(tree)
    hits = []
    stack = []

    def visit(node):
        stack.append(node)
        try:
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.attr if isinstance(fn, ast.Attribute) else (
                    fn.id if isinstance(fn, ast.Name) else None)
                device = None
                for kw in node.keywords:
                    if kw.arg == "device":
                        device = ast.unparse(kw.value)
                if name in HOST_DATA_CTORS and device is not None and "cpu" not in device.lower():
                    scope = next((scopes[id(a)] for a in reversed(stack) if id(a) in scopes),
                                 "<module>")
                    hits.append((node.lineno, scope, name, device))
                elif name == "cuda" and device is None:
                    scope = next((scopes[id(a)] for a in reversed(stack) if id(a) in scopes),
                                 "<module>")
                    hits.append((node.lineno, scope, "cuda", "-"))
        finally:
            for child in ast.iter_child_nodes(node):
                visit(child)
            stack.pop()

    visit(tree)
    return hits


def test_detector_fires_on_a_known_upload():
    # Positive control (law 5): an empty census only means something once the scan has proved it
    # can see the construct it guards.
    src = "\n".join([
        "class InklingProbe:",
        "    def forward(self, x):",
        "        a = torch.tensor([1, 2], device = x.device, dtype = torch.long)",
        "        b = torch.as_tensor([3], device = 'cuda')",
        "        c = torch.zeros((2,), device = x.device)   # device-generated: not an upload",
        "        d = torch.tensor([0], device = 'cpu')      # host: not an upload",
        "        e = x.cpu()",
        "        return a, b, c, d, e",
    ])
    fd, path = tempfile.mkstemp(suffix = ".py")
    try:
        with os.fdopen(fd, "w", encoding = "utf-8") as fh:
            fh.write(src)
        hits = _uploads(path)
    finally:
        os.remove(path)
    assert [(line, scope, ctor) for line, scope, ctor, _ in hits] == [
        (3, "InklingProbe.forward", "tensor"),
        (4, "InklingProbe.forward", "as_tensor"),
    ], hits


def test_capture_region_has_no_pageable_uploads():
    missing = []
    found = collections.Counter()
    detail = {}
    for rel in SCAN:
        path = os.path.join(REPO, rel)
        if not os.path.isfile(path):
            missing.append(rel)
            continue
        for line, scope, ctor, device in _uploads(path):
            found[(rel, scope, ctor)] += 1
            detail.setdefault((rel, scope, ctor), []).append(f"{rel}:{line} (device = {device})")
    assert not missing, f"scan lost a file, so its verdict is vacuous: {missing}"
    allowed = collections.Counter({key + (ctor,): n for key, table in ALLOWED.items()
                                  for ctor, n in table.items()})
    extra = found - allowed
    assert not extra, (
        "host->device copy inside the whole-forward capture region with no census entry: "
        + "; ".join("%s (x%d)" % (", ".join(detail[k]), n) for k, n in sorted(extra.items()))
        + " - a pageable CPU tensor copied to CUDA raises 'Cannot copy between CPU and CUDA tensors "
          "during CUDA graph capture unless the CPU tensor is pinned' and disables the graph for "
          "the process; route it through util.tensor.stage_host (pinned, stream-ordered while "
          "capturing) or add the site to ALLOWED with its reason"
    )
    absent = allowed - found
    assert not absent, (
        "census entry no longer in the source: "
        + ", ".join("%s %s %s x%d" % (*key, n) for key, n in sorted(absent.items()))
        + "; drop it from ALLOWED so the table stays the truth"
    )


# ---------------------------------------------------------------------------
# Behaviour: the pinned staging primitive (tests/graph convention: no CUDA context)
# ---------------------------------------------------------------------------

class _StageTensor(torch.Tensor):
    """One host-backed fixture for BOTH roles the staging primitive talks to, because two different
    tensor subclasses cannot dispatch a copy against each other in torch (only one
    __torch_function__ handler serves a call). In the source role it reports itself device-resident
    (.device -> cuda, every .cpu() counted); in the staging-slot role it reports itself pinned and
    records how each copy into it was issued, so a test can tell a recorded copy (non_blocking)
    from a waited-on one. Storage is host memory: no CUDA context is created."""

    copies = []

    @property
    def device(self):
        return torch.device("cuda")

    def is_pinned(self):
        return True

    def cpu(self, *a, **k):
        self.readbacks = getattr(self, "readbacks", 0) + 1
        return torch.Tensor.cpu(self)

    def copy_(self, src, non_blocking = False):
        _StageTensor.copies.append((tuple(self.shape), tuple(src.shape), bool(non_blocking)))
        return torch.Tensor.copy_(self, src, non_blocking = non_blocking)


def _on_device(values, dtype = torch.int32) -> _StageTensor:
    t = torch.tensor(list(values), dtype = dtype)
    d = torch.Tensor._make_subclass(_StageTensor, t)
    d.readbacks = 0
    return d


def _staged(fn, *args, capturing = True, **kwargs):
    """Run fn with util.tensor's pinned allocator faked and the capture flag forced. The fake
    allocator hands out oversized slots (like Generator._staging) so a test can tell a recorded
    copy from a waited-on one by the shapes it sees."""
    from exllamav3.util import tensor as ut

    state = {"allocations": 0}

    def alloc(numel, dtype):
        state["allocations"] = state.get("allocations", 0) + 1
        state["sizes"] = state.get("sizes", []) + [numel]
        return torch.Tensor._make_subclass(_StageTensor, torch.zeros((numel,), dtype = dtype))

    _StageTensor.copies = []
    ut._pin_stages.clear()
    saved = (ut._pinned_empty, ut._capturing)
    ut._pinned_empty, ut._capturing = alloc, (lambda: capturing)
    try:
        out = fn(*args, **kwargs)
    finally:
        ut._pinned_empty, ut._capturing = saved
    return out, state


def test_host_values_snapshots_into_a_pinned_slot_while_capturing():
    # Capture legality: the readback lands in pinned memory, issued non-blocking (a blocking copy
    # would synchronize the recording stream), and no .cpu() is attempted on the device tensor.
    from exllamav3.util.tensor import host_values

    src = _on_device([9, 4, 12])
    out, state = _staged(host_values, src, ("probe", "lens"), capturing = True)
    assert out.is_pinned(), "capture-time readback did not land in pinned memory: it would raise"
    assert state["allocations"] == 1, state
    assert [c[2] for c in out.copies] == [True], f"copy was waited on inside a capture: {out.copies}"
    assert out[:3].tolist() == [9, 4, 12]


def test_host_values_keeps_eager_semantics():
    # Eager: exactly the blocking .cpu() the code did before - same values, one readback, and no
    # pinned buffer burned on a path that never captures.
    from exllamav3.util.tensor import host_values

    src = _on_device([5, 0, 17], torch.int64)
    out, state = _staged(host_values, src, ("probe", "sel"), capturing = False)
    assert out.tolist() == [5, 0, 17], out.tolist()
    assert src.readbacks == 1, f"eager path stopped reading the device tensor back: {src.readbacks}"
    assert state["allocations"] == 0, f"eager path allocated pinned memory: {state}"


def test_stage_host_uploads_from_a_pinned_slot_and_reuses_it_while_capturing():
    # The direction the driver reported. While capturing, the source must be the persistent pinned
    # slot - a pageable temporary is what raised - and it must be ONE slot that fwd_graph.py's
    # replay can re-read, not a fresh allocation per launch.
    from exllamav3.util.tensor import stage_host

    first, state = _staged(stage_host, [7, 11, 3], "cpu", ("probe", "entries"), torch.long)
    assert first.tolist()[:3] == [7, 11, 3]
    assert first.is_pinned(), "uploaded from a pageable source; capture would raise"
    assert state["allocations"] == 1, state
    second, state = _staged(stage_host, [1, 2, 3, 4, 5, 6], "cpu", ("probe", "entries"),
                            torch.long)
    assert second.tolist()[:6] == [1, 2, 3, 4, 5, 6]
    assert state["allocations"] == 1, f"pinned slot rebuilt per launch: {state}"


def test_stage_host_matches_the_temporary_it_replaces_when_eager():
    # Eager parity with `torch.tensor(values, device = ...)`: same values, same dtype, and the
    # pageable temporary stays pageable (no pinned RAM reserved outside captures).
    from exllamav3.util.tensor import stage_host

    values = [4, 0, 9, 2]
    out, state = _staged(stage_host, values, "cpu", ("probe", "entries"), torch.long,
                         capturing = False)
    assert out.tolist() == values
    assert out.dtype == torch.long
    assert not out.is_pinned()
    assert state["allocations"] == 0, state


def test_moe_dispatch_runs_through_the_staging_and_computes_the_same_mixture():
    """InklingMoE.forward run for real on CPU with stub experts while the capture flag is forced:
    the per-expert row indices go back to the device through the pinned slot (the copy the driver
    reported), and the result still matches a reference that recomputes the mixture token by
    token, so the staging path is shown to carry the right integers rather than merely to exist."""
    from exllamav3.modules.arch_specific.inkling2_loop import InklingMoE

    h, inter, e_n, top_k = 4, 6, 5, 2
    torch.manual_seed(3)
    wg = [torch.randn(inter, h) for _ in range(e_n)]
    wu = [torch.randn(inter, h) for _ in range(e_n)]
    wd = [torch.randn(h, inter) for _ in range(e_n)]

    class _Lin:
        """Stand-in for a Linear: casts its input to the weight dtype the way the real module's
        kernels do, so the fp16 hand-off the MoE makes on the down projection is reproduced."""

        def __init__(self, w):
            self.w = w

        def forward(self, x, params):
            return x.to(self.w.dtype).matmul(self.w.T)

    stub = types.SimpleNamespace(
        top_k = top_k, route_scale = 1.7, gate_softcap = 4.0, up_softcap = 25.0,
        router_w = torch.randn(e_n, h), router_bias = torch.randn(e_n),
        experts_g = [_Lin(w) for w in wg], experts_u = [_Lin(w) for w in wu],
        experts_d = [_Lin(w) for w in wd],
    )
    x = torch.randn(2, 3, h)
    out, state = _staged(InklingMoE.forward, stub, x, {})
    assert state.get("allocations", 0) >= 1, (
        "the MoE used no pinned slot: it is back to a pageable host temporary per launch")

    # The same forward with no capture in progress: no pinned memory reserved, and bit-identical
    # output. That is the eager-semantics claim for the busiest call site in the model.
    out_eager, state_eager = _staged(InklingMoE.forward, stub, x, {}, capturing = False)
    assert state_eager["allocations"] == 0, f"eager dispatch reserved pinned RAM: {state_eager}"
    assert torch.equal(out, out_eager), "capture-path staging changed the numbers"

    # Reference: same router, per-token loop, no sorting and no index tensors, and the same fp16
    # hand-off on the down projection, so agreement means the staged indices landed in the right
    # rows rather than merely that both paths ran.
    scores = torch.sigmoid(x.reshape(-1, h).float().matmul(stub.router_w.T))
    topi = torch.topk(scores + stub.router_bias, top_k, dim = -1, sorted = False).indices
    w = scores.gather(1, topi)
    w = w / (w.sum(dim = -1, keepdim = True) + 1e-20) * 1.7
    ref = torch.zeros(x.shape[0] * x.shape[1], h)
    flat = x.reshape(-1, h)
    for row in range(ref.shape[0]):
        for k in range(top_k):
            e = int(topi[row, k])
            g = flat[row].matmul(wg[e].T).float()
            u = flat[row].matmul(wu[e].T).float()
            act = torch.tanh(g / 4.0) * 4.0 * torch.sigmoid(g) * (torch.tanh(u / 25.0) * 25.0)
            ref[row] += act.half().float().matmul(wd[e].T).float() * w[row, k]
    assert torch.allclose(out.float(), ref.view_as(out.float()), atol = 2e-4), \
        (out.float() - ref.view_as(out.float())).abs().max().item()


def _main() -> int:
    import inspect
    import sys

    # Standalone: pytest normally puts the repo root on sys.path (tests/conftest.py); running this
    # file directly does not, so add it here before the tests import the package.
    if REPO not in sys.path:
        sys.path.insert(0, REPO)

    class _MP:
        def setattr(self, obj, attr, val):
            setattr(obj, attr, val)

    failed = 0
    for name, fn in sorted(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        try:
            if bool(inspect.signature(fn).parameters):
                fn(_MP())
            else:
                fn()
            print("ok   %s" % name)
        except Exception as exc:  # noqa: BLE001 - standalone runner reports every failure mode
            failed += 1
            print("FAIL %s: %s" % (name, exc))
    print("RAN %d tests, %d failed" % (len([k for k in globals() if k.startswith("test_")]), failed))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_main())