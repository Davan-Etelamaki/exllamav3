"""Capture-time param uploads must be stream-ordered, not blocking.

Every whole-forward CUDA graph capture died on every boot (dot26 journal, 2026-11-24):

    File ".../exllamav3/modules/arch_specific/inkling2_loop.py", line 701, in _pos_row
        positions = get_for_device(params, "positions", self.device, None)
    File ".../exllamav3/util/tensor.py", line 171, in get_for_device
        return to_device(v_cap, device, non_blocking = False)
    torch.AcceleratorError: CUDA error: operation not permitted when stream is capturing
    The CUDA driver logged: Returning 900 (CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED) from cuStreamSynchronize

A capture may not synchronize the stream it is recording. `get_for_device` answered the capture
branch with a BLOCKING host->device copy, so the first params tensor that had to be uploaded
(`positions`) invalidated the capture and ForwardGraphRunner disabled itself for the process.
Recorded work must be asynchronous, which is also what fwd_graph.py's contract states: dropping
params["dev_cache"] before capture exists so the graph records pinned H2D copies from the
generator's staging addresses and re-reads them, refilled in place, on every replay.

The eager branch of the same function already uploads host sources asynchronously (host_to_device,
"the copy is stream-ordered ahead of the kernels that consume it"); only the capture branch asked
for a blocking copy. The async copy is only sound because that source outlives the capture:
`Generator._staging` hands out slices of persistent pinned buffers (pin_memory=True), never a fresh
host tensor, so the address the graph records stays valid and stays current.

Run standalone (no pytest, works in the prod venv):
    python tests/graph/test_capture_region_upload_is_async.py
"""

from __future__ import annotations

import os

import torch

try:
    import pytest

    pytestmark = pytest.mark.nogpu
except ImportError:  # the deployed engine venv ships no pytest; the file still runs standalone
    pytestmark = ()

CUDA_DEV = torch.device("cuda:0")
DOWNLOADED = "uploaded"


def _patched(monkeypatch, capturing: bool):
    """Swap the three module globals get_for_device consults; record every upload it makes.

    Nothing here touches CUDA: the capture flag is faked and both copy helpers are intercepted, so
    the test runs with no device and never initializes a context (the engine card stays untouched).
    """
    import exllamav3.util.tensor as ut

    calls = []

    def fake_to_device(t, device, non_blocking = False):
        calls.append({"via": "to_device", "non_blocking": non_blocking})
        return DOWNLOADED

    def fake_host_to_device(t, device):
        calls.append({"via": "host_to_device", "non_blocking": True})  # always stream-ordered
        return DOWNLOADED

    monkeypatch.setattr(ut, "to_device", fake_to_device)
    monkeypatch.setattr(ut, "host_to_device", fake_host_to_device)
    monkeypatch.setattr(ut, "_capturing", lambda: capturing)
    return ut, calls


def _upload(monkeypatch, capturing: bool, key = "positions"):
    """One get_for_device read of a pinned-style staging tensor (a CPU tensor, as the generator
    hands out: Generator._staging returns a slice of a pin_memory=True buffer). `key` is the key
    READ; the staged tensor always sits under "positions", so a test can read a missing key."""
    ut, calls = _patched(monkeypatch, capturing)
    staged = torch.zeros(4, dtype = torch.int32)
    params = {"positions": staged}
    out = ut.get_for_device(params, key, CUDA_DEV, None)
    return params, calls, out, staged


def _is_async(call: dict) -> bool:
    return call["via"] == "host_to_device" or call["non_blocking"] is True


def test_capture_upload_is_non_blocking(monkeypatch):
    """The defect: inside a capture the upload must not be a blocking copy."""
    _, calls, out, _ = _upload(monkeypatch, capturing = True)
    assert len(calls) == 1, "intercepted no upload; the test judges nothing"
    assert _is_async(calls[0]), (
        f"capture branch uploaded via {calls[0]['via']} with non_blocking="
        f"{calls[0].get('non_blocking')}; a blocking copy calls cuStreamSynchronize and "
        "invalidates the capture (cudaErrorStreamCaptureUnsupported)"
    )
    assert out == DOWNLOADED


def test_eager_read_still_uploads_and_caches(monkeypatch):
    # Outside a capture the function keeps its own contract: an async host upload for a CPU source
    # (that is what the shipped eager branch does) plus a dev_cache entry. Guards against a fix
    # that broke the eager path while quieting the capture branch.
    params, calls, out, staged = _upload(monkeypatch, capturing = False)
    assert len(calls) == 1, "intercepted no upload; the test judges nothing"
    assert _is_async(calls[0]), "the eager branch uploads host sources asynchronously already"
    assert out == DOWNLOADED
    assert params["dev_cache"][(id(staged), CUDA_DEV)][1] == DOWNLOADED


def test_capture_never_populates_dev_cache(monkeypatch):
    # Pre-existing invariant (Xid 31 FAULT_PDE): a capture-time copy lives in the graph's private
    # pool and its storage dies with the graph, so it must never be cached for later eager callers.
    params, calls, _, _ = _upload(monkeypatch, capturing = True)
    assert len(calls) == 1, "intercepted no upload; the test judges nothing"
    assert "dev_cache" not in params


def test_capture_absent_key_returns_default_without_upload(monkeypatch):
    params, calls, out, _ = _upload(monkeypatch, capturing = True, key = "absent_key")
    assert calls == [], f"uploaded {calls} for a key the dict does not hold"
    assert out is None, f"absent key returned {out!r} instead of the default"


def test_staging_source_is_a_pinned_persistent_buffer():
    # Precondition of the async copy: the recorded H2D source must outlive the capture and stay
    # current. Generator._staging is the producer of params["positions"]; it must hand out slices
    # of a pinned buffer allocated once per (name, width), not a fresh host tensor per round.
    here = os.path.dirname(os.path.abspath(__file__))
    pkg = os.path.abspath(os.path.join(here, os.pardir, os.pardir, "exllamav3"))
    with open(os.path.join(pkg, "generator", "generator.py"), encoding = "utf-8") as fh:
        text = fh.read()
    start = text.index("    def _staging(self, name, rows: int")
    body = text[start:text.index("\n    def ", start + 10)]
    assert "pin_memory = True" in body, "_staging no longer allocates pinned host memory"
    assert "buf[:rows]" in body, "_staging no longer returns a slice of a persistent buffer"


def _main() -> int:
    import inspect
    import sys

    # Standalone: pytest normally puts the repo root on sys.path (tests/conftest.py); running this
    # file directly does not, so add it here before the tests import the package.
    repo = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        os.pardir, os.pardir))
    if repo not in sys.path:
        sys.path.insert(0, repo)

    failed = 0
    for name, fn in sorted(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        needs_mp = bool(inspect.signature(fn).parameters)
        try:
            if needs_mp:
                class _MP:  # minimal stand-in for pytest.monkeypatch
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