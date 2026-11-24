"""Whole-forward graph: the replay signature must carry the round geometry, and a failed capture
must degrade once and loudly, with the frame.

Live witness (dot26, wheels 38204d31 and 0a19785408ce0, journal 2026-11-24 17:11:46 and 17:22:26):

    Whole-forward CUDA graph disabled for this process: RuntimeError: The expanded size of the
    tensor (881) must match the existing size (256) at non-singleton dimension 0.
    Target sizes: [881, 8, 128].  Tensor sizes: [256, 8, 128]

Two things that log line hid, both fenced here:

* the signature did not carry the geometry. The captured region builds tensor shapes from
  host-side sequence extents (inkling2_loop.py _gather_global takes lens/maxlen from _host_lens,
  ring_read takes them from positions), so replaying a graph captured at one block-table width /
  longest-sequence value attends a frozen window and mismatches the live tensors' shapes. The key
  now carries max_pages_batch and max_seq_len.
* the degrade path discarded the traceback, so the defect could only be seen by re-raising with
  EXL3_FWD_GRAPH_DEBUG=1 - and that flag re-raised on EVERY round (disabled_reason was set only
  after the raise), so each round paid two warmups plus a private capture pool until prefill died
  with torch.OutOfMemoryError instead of serving. The reason is now set before the raise, and the
  warning carries the frame.

CPU-only: a stand-in forward plays the part of the capture region.
"""

import logging
import os

import pytest

from exllamav3.generator.fwd_graph import ForwardGraphRunner, make_key


def test_the_replay_signature_carries_the_round_geometry():
    # Same staged shape, different context: the captured gather extents differ, so they must not
    # share a graph. Blocks the silent case where a 256-token-context graph is replayed at 881.
    base = (1, 1, False, False, ())
    same = make_key(*base, 32, 881)
    assert same == make_key(*base, 32, 881)
    assert make_key(*base, 32, 882) != same, "longest sequence is not in the replay signature"
    assert make_key(*base, 33, 881) != same, "block-table width is not in the replay signature"


def test_a_failed_capture_degrades_once_and_logs_the_frame(caplog):
    calls = {"n": 0}

    def forward_fn():
        calls["n"] += 1
        if calls["n"] == 1:                 # the first warmup execution stands in for the capture
            raise RuntimeError("simulated capture failure")
        return "eager-result"

    runner = ForwardGraphRunner()
    with caplog.at_level(logging.WARNING, logger="exllamav3.generator.fwd_graph"):
        out = runner.run(make_key(1, 1, False, False, (), 8, 64), {}, forward_fn)
    assert out == "eager-result", "a failed capture must return the eager round, not break it"
    assert runner.disabled_reason == "RuntimeError: simulated capture failure"
    text = caplog.text
    assert "Whole-forward CUDA graph disabled" in text, "the degrade was silent"
    assert "Traceback (most recent call last)" in text and "in forward_fn" in text, \
        "the warning named a symptom without its site: the frame must ride with the message"
    # and once disabled, no further capture attempt at all
    n_before = calls["n"]
    assert runner.run(make_key(1, 1, False, False, (), 8, 64), {}, forward_fn) == "eager-result"
    assert calls["n"] == n_before + 1, "the runner kept trying to capture after disabling itself"


def test_debug_mode_raises_once_then_degrades(monkeypatch):
    # EXL3_FWD_GRAPH_DEBUG must buy the frame without turning every later round into a fresh
    # capture attempt: that retry storm is what OOMed prefill on the 2026-11-24 17:19 proof boot.
    monkeypatch.setenv("EXL3_FWD_GRAPH_DEBUG", "1")
    calls = {"n": 0}

    def forward_fn():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated capture failure")
        return "eager-result"

    runner = ForwardGraphRunner()
    with pytest.raises(RuntimeError):
        runner.run(make_key(1, 1, False, False, (), 8, 64), {}, forward_fn)
    assert runner.disabled_reason is not None, "DEBUG re-raised without recording the reason: every later round retries"
    n_before = calls["n"]
    assert runner.run(make_key(1, 1, False, False, (), 8, 64), {}, forward_fn) == "eager-result"
    assert calls["n"] == n_before + 1, "a DEBUG boot re-attempted the capture instead of degrading"


def test_the_generator_builds_its_key_through_the_helper():
    # Static fence: a key built inline would drift from the geometry rule without any test noticing.
    import inspect
    from exllamav3.generator import generator as gen
    src = inspect.getsource(gen)
    assert "_fg_mod.make_key(" in src, "the call site builds its own signature again"
    assert "max_pages_batch, max_seq_len," in src, "geometry is not passed to the signature"
