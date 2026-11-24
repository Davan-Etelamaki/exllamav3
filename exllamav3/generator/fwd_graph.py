"""Shape-keyed CUDA-graph runner for the steady-state decode forward.

Captures self.model.forward for a fixed (rows, ids_width, draft_window, offsets,
embeddings, recurrent) signature. Steady-state rounds (same jobs, no arrivals or
completions) refill the generator's persistent _staging buffers in place and
REPLAY the graph instead of relaunching every layer's kernels; any other round
runs eager. Opt-out: EXL3_FWD_GRAPH=0 (default ON).

Safety contract:
- inputs must be persistent tensors refilled in place (_staging guarantees this);
- replay returns a CLONE of the static output, so results holding views cannot be
  clobbered by the next replay;
- capture happens outside any graph context on a private pool; a capture failure
  disables the runner for the process (logged once), never the round.
"""

import logging
import os
import torch
import traceback

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get("EXL3_FWD_GRAPH", "1") != "0"
_WARMUPS = int(os.environ.get("EXL3_FWD_WARMUPS", "2"))


def make_key(rows, ids_width, use_offsets, has_draft, state_ids, max_pages_batch, max_seq_len):
    """
    Signature a captured round may be replayed for. The staged ids, offsets and recurrent states are
    not enough: the attention region builds its K/V gather from HOST-side sequence extents
    (inkling2_loop.py _gather_global takes lens/maxlen from _host_lens, ring_read takes them from
    positions), so the captured tensors' shapes and the Python loop bounds freeze at capture time.
    A replay at a different context would then attend over the capture round's frozen window, so the
    geometry the staging was built for (block-table width, longest sequence) joins the key.
    """
    return (rows, ids_width, use_offsets, has_draft, state_ids, max_pages_batch, max_seq_len)


class ForwardGraphRunner:
    def __init__(self):
        self.enabled = _ENABLED
        self.graphs = {}          # key -> (graph, static_out)
        self.disabled_reason = None
        self.stats = {"capture": 0, "replay": 0, "eager": 0}

    def run(self, key, forward_kwargs, forward_fn, on_after_warmup = None):
        """key: hashable signature tuple. forward_fn(**forward_kwargs) -> tensor.
        on_after_warmup: called once after the two warmup executions and before
        capture, so callers can undo side effects of warmups (e.g. recurrent advance)."""
        if not self.enabled or self.disabled_reason is not None:
            self.stats["eager"] += 1
            return forward_fn(**forward_kwargs)
        entry = self.graphs.get(key)
        if entry is None:
            try:
                # warmup on side stream (allocator/autotune settle), then capture
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(_WARMUPS):
                        forward_fn(**forward_kwargs)
                torch.cuda.current_stream().wait_stream(s)
                if on_after_warmup is not None:
                    on_after_warmup()
                # get_for_device caches per-device copies in params["dev_cache"]: warmup fills
                # it, and a capture that hits the cache records NO host->device re-upload for
                # the staged tensors, so replays would read the frozen warmup values forever.
                # Drop the cache so the capture records pinned H2D copies from the staging
                # addresses (refilled in place by the generator every round), then drop it
                # again so the post-capture eager run doesn't read pool-resident copies.
                pk = forward_kwargs.get("params")
                if isinstance(pk, dict):
                    pk.pop("dev_cache", None)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, capture_error_mode = "relaxed"):
                    static_out = forward_fn(**forward_kwargs)
                if isinstance(pk, dict):
                    pk.pop("dev_cache", None)
                self.graphs[key] = (g, static_out)
                self.stats["capture"] += 1
                # first post-capture round runs eager so the captured buffers are
                # fully re-populated by the caller before the first replay
                self.stats["eager"] += 1
                return forward_fn(**forward_kwargs)
            except Exception as e:  # capture unsafe here: degrade, never break
                self.disabled_reason = f"{type(e).__name__}: {e}"
                # The frame, not only the message: the 256-vs-881 shape error named a symptom with no
                # site, and EXL3_FWD_GRAPH_DEBUG bought the frame only by aborting every request.
                # Logged once per process because disabled_reason is set before the raise.
                logger.warning(f"Whole-forward CUDA graph disabled for this process: "
                               f"{self.disabled_reason}\n{traceback.format_exc()}")
                if os.environ.get("EXL3_FWD_GRAPH_DEBUG"):
                    raise
                # (previously the reason assignment sat after the DEBUG raise, so a DEBUG boot
                # re-attempted the capture on every round: two warmups plus a private pool each
                # time, until prefill died of an OutOfMemoryError instead of the real defect)
                # The logged-once promise (law 5: no silent zero-emitter) is kept by the warning above,
                # which now carries the frame as well as the message.
                torch.cuda.synchronize()
                self.stats["eager"] += 1
                return forward_fn(**forward_kwargs)
        g, static_out = entry
        g.replay()
        self.stats["replay"] += 1
        return static_out.clone()