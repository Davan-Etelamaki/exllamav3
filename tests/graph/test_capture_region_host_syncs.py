"""No host sync inside the inkling2 whole-forward capture region (blocker #2).

With the capture-time upload fixed (5004ce2e), the whole-forward CUDA graph next dies on the
eager attention/MoE path of inkling2_loop, which reads device values back to the host while
the graph is recording. A readback is a stream synchronization, and the driver refuses it
inside a capture (cuStreamSynchronize -> CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, 900);
fwd_graph.py then disables the runner for the process and decode runs eager.

Fixed here: InklingAttention._gather_global called cache_seqlens.cpu() on the tensor
get_for_device had just uploaded from the generator's pinned staging buffer. The host already
held those lengths, so the readback was pure round trip: _host_lens takes them from the
buffer they were uploaded from, same values and dtype, no sync. The fallback keeps the
readback for callers that hand the model genuinely device-resident lengths.

Inventory test (test_capture_region_host_sync_inventory) freezes the rest of the census so a
new readback cannot sneak in and a deferred one stays named. Every allowlisted entry carries
its reason; the deferred items are listed there with the patch they need.

Run standalone (no pytest, works in the deployed engine venv):
    python tests/graph/test_capture_region_host_syncs.py
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
CAPTURE_REGION = os.path.join(REPO, "exllamav3", "modules", "arch_specific", "inkling2_loop.py")

# Attribute calls that move data to the host (or wait on the stream) and bare builtins that
# force a readback when applied to a device tensor. .float() / .cpu() as METHODS are caught by
# the attribute arm; x.float() (dtype cast, stays on device) is deliberately not a sync.
ATTR_READBACK = {"tolist", "cpu", "item", "numpy", "synchronize", "nonzero"}
NAME_READBACK = {"int", "float", "bool"}

# The census. Key: dotted scope of the enclosing (class.)function, value: {construct: count}.
# Reasons decide fix vs defer; a site whose count changes must update this table, which is the
# point - the guard fails on both directions.
ALLOWED = {
    # Host-resident sources: params["recurrent_slots"] is a CPU int32 tensor built from Python
    # slot ids (cache/recurrent_util._get_slot_tensor), and params["positions"] /
    # params["cache_seqlens"] are the generator's pinned CPU staging buffers. These reads cost
    # no CUDA op, so they are capture-LEGAL. They do freeze their VALUES into the capture
    # (deferred item B: the graph key must cover, or the code must stop depending on, them).
    "InklingConvs._state_slot": {"tolist": 1},
    "InklingConvs.ring_write": {"tolist": 1, "int": 1},
    "InklingConvs.ring_read": {"tolist": 1, "int": 1},
    # Conditional: a no-op while the source is host-resident, which is what the fwd-graph path
    # feeds it (prepare_flash_attn aliases positions to the staged cache_seqlens). A real D2H
    # if a caller ever hands device-resident positions - deferred item C.
    "InklingConvs._positions": {"cpu": 1, "tolist": 1},
    "InklingAttention._attend_one": {"float": 1},        # float("-inf") literal, no readback
    "InklingAttention._gather_global": {"int": 1},       # int: host tensor values (freeze, item B)
    "InklingAttention._gather_global._st": {"float": 2}, # INKLING_DEBUG readback (item D)
    # The fixed site's fallback: reads back ONLY when params holds no host-resident twin, i.e.
    # when the lengths really are device-resident. That caller is outside a capture today
    # (prefill and the draft verifier hand device tensors and never graph-capture).
    "_host_lens": {"cpu": 1},
    # Deferred item A: router selection is computed ON DEVICE, so this tolist() is a genuine
    # D2H sync in the capture region and cannot be replaced without changing the algorithm -
    # per-expert weights are separate Linear modules, so a Python expert index (not a device
    # value) must pick them. Needs the device-side routing tables + a grouped/stacked-weight
    # expert tier (see module docstring of block_sparse_mlp._RoutingGraph for the capture-safe
    # prelude this should reuse).
    "InklingMoE.forward": {"tolist": 1},
    "InklingLayer.forward": {"int": 2, "float": 2},      # INKLING_DEBUG print (item D)
    "InklingLayer.forward._s": {"float": 2},             # INKLING_DEBUG readback (item D)
}


def _inventory(path: str) -> collections.Counter:
    """{(dotted scope, construct): count} for every host-readback construct in a source file"""
    with open(path, encoding = "utf-8") as fh:
        tree = ast.parse(fh.read(), filename = path)
    inv = collections.Counter()

    def visit(node, scope: str):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visit(child, f"{scope}.{child.name}" if scope else child.name)
            else:
                if isinstance(child, ast.Call):
                    fn = child.func
                    if isinstance(fn, ast.Attribute) and fn.attr in ATTR_READBACK:
                        inv[(scope, fn.attr)] += 1
                    elif isinstance(fn, ast.Name) and fn.id in NAME_READBACK:
                        inv[(scope, fn.id)] += 1
                visit(child, scope)

    visit(tree, "")
    return inv


def _flatten(table: dict) -> collections.Counter:
    return collections.Counter({(scope, name): n for scope, names in table.items()
                               for name, n in names.items()})


def test_detector_fires_on_a_known_readback():
    # Positive control: the scan must find a readback it is told about, or an empty inventory
    # would certify any file (law 5: a zero verdict is only meaningful once the measurement ran).
    src = "\n".join([
        "class InklingProbe:",
        "    def forward(self, x, params):",
        "        sel_cpu = x.tolist()",
        "        n = int(params['n'])",
        "        return x.cpu(), sel_cpu, n",
    ])
    fd, path = tempfile.mkstemp(suffix = ".py")
    try:
        with os.fdopen(fd, "w", encoding = "utf-8") as fh:
            fh.write(src)
        inv = _inventory(path)
    finally:
        os.remove(path)
    assert inv == collections.Counter({("InklingProbe.forward", "tolist"): 1,
                                       ("InklingProbe.forward", "int"): 1,
                                       ("InklingProbe.forward", "cpu"): 1}), inv


def test_capture_region_host_sync_inventory():
    assert os.path.isfile(CAPTURE_REGION), CAPTURE_REGION
    found = _inventory(CAPTURE_REGION)
    assert found, "inventory scanned nothing; the guard is vacuous"
    extra = found - _flatten(ALLOWED)
    missing = _flatten(ALLOWED) - found
    assert not extra, (
        "host-readback construct in the capture region with no census entry: "
        + ", ".join(f"{scope}.{name} x{n}" for (scope, name), n in sorted(extra.items()))
        + "; either fix it (keep the values on the host side they came from) or add it to "
          "ALLOWED with its reason"
    )
    assert not missing, (
        "census entry no longer in the source: "
        + ", ".join(f"{scope}.{name} x{n}" for (scope, name), n in sorted(missing.items()))
        + "; drop it from ALLOWED so the table stays the truth"
    )


class _OnDevice(torch.Tensor):
    """A tensor that reports itself as device-resident and counts every readback.

    Storage is host memory (these tests never touch CUDA), but the attribute lookups the
    library makes - .device, .cpu() - hit the subclass overrides, so a sync inside the code
    under test is observable without a device.
    """

    @property
    def device(self):
        return torch.device("cuda")

    def cpu(self, *a, **k):
        self.readbacks = getattr(self, "readbacks", 0) + 1
        return torch.Tensor.cpu(self)


def _on_device(values) -> _OnDevice:
    t = torch.tensor(values, dtype = torch.int32)
    d = torch.Tensor._make_subclass(_OnDevice, t)
    d.readbacks = 0
    return d


def test_host_lens_uses_the_staging_source():
    # The fix: when params holds the pinned CPU buffer the device copy was uploaded from, the
    # host lengths come from it and the device copy is never read back.
    from exllamav3.modules.arch_specific.inkling2_loop import _host_lens

    staged = torch.tensor([1023, 7], dtype = torch.int32)
    dev = _on_device([1023, 7])
    out = _host_lens({"cache_seqlens": staged}, dev)
    assert out is staged, "did not take the host source it was handed"
    assert dev.readbacks == 0, f"read the device copy back {dev.readbacks}x; that sync kills the capture"
    assert [int(out[i]) for i in range(out.shape[0])] == [1023, 7]


def test_host_lens_still_reads_back_a_device_only_source():
    # The fallback must survive: a caller whose lengths really are device-resident has no host
    # twin to take, and gets the same values it got before the fix.
    from exllamav3.modules.arch_specific.inkling2_loop import _host_lens

    dev = _on_device([4095])
    out = _host_lens({"cache_seqlens": dev}, dev)
    assert dev.readbacks == 1, "the device-only path stopped reading back; it now silently freezes"
    assert int(out[0]) == 4095


def test_host_lens_parity_with_the_old_readback():
    # Semantic parity: for the tensor pair the capture path actually holds (the staging buffer
    # plus the device copy get_for_device uploaded from it), the fixed expression returns
    # exactly what cache_seqlens.cpu() returned.
    from exllamav3.modules.arch_specific.inkling2_loop import _host_lens

    staged = torch.arange(4, dtype = torch.int32) * 251 + 13
    upload = staged.clone()                       # what get_for_device handed the module
    old = upload.cpu()                            # the expression this replaces
    counted = _on_device(staged.tolist())
    assert torch.equal(_host_lens({"cache_seqlens": staged}, counted), old)
    assert counted.readbacks == 0


def test_gather_global_reads_nothing_from_the_device():
    """The site itself, run for real on CPU: the global-attention gather must not sync."""
    from exllamav3.modules.arch_specific.inkling2_loop import InklingAttention
    from exllamav3.constants import PAGE_SIZE

    kvh, hd, b, t = 2, 8, 2, 1
    lengths = [3 * PAGE_SIZE + 5, PAGE_SIZE - 3]
    n_pages = sum(-(-l // PAGE_SIZE) for l in lengths) + 2
    k_cache = torch.randn(n_pages, PAGE_SIZE, kvh, hd)
    v_cache = torch.randn(n_pages, PAGE_SIZE, kvh, hd)

    class _Layer:                      # no stage_range -> the paged (fp16-style) branch
        def get_kv(self, cache_seqlens, block_table, sliding_window):
            return k_cache, v_cache

    class _Cache:
        layers = {(0, 0): _Layer()}

    block_table = torch.zeros((b, 4), dtype = torch.int32)
    for row in range(b):
        for p in range(-(-(lengths[row] + t) // PAGE_SIZE)):
            block_table[row, p] = row * 4 + p

    staged = torch.tensor(lengths, dtype = torch.int32)
    params = {"cache_seqlens": staged}
    dev_seqlens = _on_device(lengths)
    kh = torch.randn(b, t, kvh, hd)
    vh = torch.randn(b, t, kvh, hd)

    stub = types.SimpleNamespace(layer_idx = 0)   # _gather_global reads only self.layer_idx
    kk, vv, lens = InklingAttention._gather_global(
        stub, _Cache(), dev_seqlens, block_table, 0, kh, vh, t, params)

    assert dev_seqlens.readbacks == 0, (
        f"_gather_global read cache_seqlens back to the host {dev_seqlens.readbacks}x; inside a "
        "CUDA graph capture that is cuStreamSynchronize -> CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED"
    )
    assert lens == [l + t for l in lengths], lens
    assert kk.shape == (b, max(lens), kvh, hd) and vv.shape == kk.shape

    # Independent reference: the rows for sequence i are the pages its block-table row addresses,
    # flattened and cut to the window length. forward_attn writes the new K/V into those pages
    # (update_layer_direct) before calling this gather; the stub cache stands in for that with a
    # static pool, so what is checked here is the row counts and the page mapping - the two
    # things the host-side lengths decide.
    for i, ln in enumerate(lens):
        phys = [int(block_table[i, p]) for p in range(-(-ln // PAGE_SIZE))]
        ref = k_cache[phys].reshape(-1, kvh, hd)[:ln]
        assert torch.equal(kk[i, :ln], ref), f"gathered rows differ for sequence {i}"
        assert torch.equal(vv[i, :ln], v_cache[phys].reshape(-1, kvh, hd)[:ln])
    # Rows past a sequence's window are padding in the (b, maxlen) buffer and must stay zero,
    # so a wrong host length would show up as nonzero junk rather than pass silently.
    assert torch.count_nonzero(kk[1, lens[1]:]) == 0
    assert torch.count_nonzero(vv[1, lens[1]:]) == 0


def _main() -> int:
    import inspect
    import sys

    # Standalone: pytest normally puts the repo root on sys.path (tests/conftest.py); running this
    # file directly does not, so add it here before the tests import the package.
    if REPO not in sys.path:
        sys.path.insert(0, REPO)

    failed = 0
    for name, fn in sorted(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        try:
            if bool(inspect.signature(fn).parameters):
                class _MP:
                    def setattr(self, obj, attr, val):
                        setattr(obj, attr, val)

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