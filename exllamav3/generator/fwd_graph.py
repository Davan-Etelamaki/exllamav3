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

import os
import torch

_ENABLED = os.environ.get("EXL3_FWD_GRAPH", "1") != "0"
_WARMUPS = int(os.environ.get("EXL3_FWD_WARMUPS", "2"))


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
                if os.environ.get("EXL3_FWD_GRAPH_DEBUG"):
                    raise
                self.disabled_reason = f"{type(e).__name__}: {e}"
                torch.cuda.synchronize()
                self.stats["eager"] += 1
                return forward_fn(**forward_kwargs)
        g, static_out = entry
        g.replay()
        self.stats["replay"] += 1
        return static_out.clone()