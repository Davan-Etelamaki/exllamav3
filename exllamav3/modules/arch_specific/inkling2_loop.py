"""
Inkling 2 Mini (Inkling2Loop) runtime modules.

Math ported from transformers 5.24.0 modeling_inkling2_loop.py (vendor-faithful reference);
state plumbing follows the ShortConv recurrent-state pattern so the loop's per-pass KV/conv
state rides the engine's (layer_idx, instance) addressing. The 3-pass loop core is driven by
the engine's layer_map relayering: params carries a per-forward flow dict with the pre-loop
snapshots and the block-state bank; each core layer's forward() performs the pass-boundary
reset when layer_instance advances. Attention runs in torch (correctness-first): additive
relative-position bias and per-token sinks are not expressible in the paged kernels yet.
"""
from __future__ import annotations
from typing_extensions import override
from collections import OrderedDict
import logging
import math
import os as _os
import torch
import torch.nn.functional as F

from ...cache import Cache
from ...cache.recurrent import host_copy
from ...model.config import Config
from ...util.tensor import _capturing, get_for_device, host_values, stage_host
from ...constants import PAGE_SIZE
from ...ext import exllamav3_ext as ext
from .. import Module, Linear, RMSNorm
from . import moe_coop_plan

KEY_FLOW = "inkling_flow"

logger = logging.getLogger(__name__)

# Per-call byte budget for the _attend_one relative-position temporaries. Derives the query
# tile so the largest single allocation on the long-S path stays bounded; only multi-row
# passes tile down, so single-token decode (T == 1) and short-context passes stay launch-
# neutral. Tunable per-instance via self.attend_tile_bytes.
_ATTEND_TILE_BYTES = int(_os.environ.get("INKLING_ATTEND_TILE_BYTES", 512 * 1024 * 1024))

# Byte cap for the fp32 K temporary inside the tile loop. Once the broadcast replicas are folded away
# (§48), the largest single allocation on the long-S path is `kxf.float()`: kvh*d*S*4 = 4,096*S bytes,
# which is 1.0 GiB at the served 262,144 -- still an OOM-sized slab on a card that runs with ~409 MiB
# free at peak load. Splitting the scores matmul along the KEY axis caps it at this budget instead.
# Splitting columns is exact: each score is dot(q_row, k_column) over d alone, so a column block
# computes the same values in the same order as the full-width product (the fence proves bit-identity).
# Floor at rel_extent so a chunk still covers one full relative window.
_ATTEND_K_CHUNK_BYTES = int(_os.environ.get("INKLING_ATTEND_K_CHUNK_BYTES", 64 * 1024 * 1024))


def _fresh_flow() -> dict:
    return {"bank": [], "pre_bank": None, "pre_hidden": None, "prev_bank": None, "prev_hidden": None}


def flow_of(params: dict) -> dict:
    # prepare_inputs normally installs the flow; entry paths that bypass it (a load-time warmup
    # forward) must still get a fresh one rather than crash.
    flow = params.get(KEY_FLOW)
    if flow is None:
        flow = _fresh_flow()
        params[KEY_FLOW] = flow
    return flow


def _host_lens(params: dict, seqlens) -> torch.Tensor:
    """Cache lengths as host values, without a device->host sync.

    The generator stages cache_seqlens in a persistent pinned CPU buffer (generator._staging)
    and get_for_device uploads that buffer to the compute device, so params already holds the
    host original of the device copy. Reading the copy back with .cpu() is a blocking D2H:
    harmless when eager, fatal while a CUDA graph records the very stream it would
    synchronize (cuStreamSynchronize -> CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, 900), which is
    how every whole-forward capture of this model died. The two operands hold the same
    values and the same dtype - the device tensor IS that upload - so taking the host side
    changes no result. Callers that hand the model genuinely device-resident lengths (no CPU
    original in params) still pay the copy.
    """
    src = params.get("cache_seqlens")
    if isinstance(src, torch.Tensor) and src.device.type == "cpu":
        return src
    if isinstance(seqlens, torch.Tensor):
        return seqlens.cpu()
    return torch.tensor(list(seqlens))


# CAPTURED-GRAPH COST, ADMISSION AND EVICTION.
# A captured graph keeps its private pool RESIDENT: the padded K/V assembly and the fp32
# relative-bias temporaries do not return to the allocator between rounds. One graph at
# (bucket rows R, extent C) costs about
#   3 · R · C · kv_heads · head_dim · 2 B   (padded K, padded V, the gather output)
#   + 2 · h · R · C · 4 B                   (relative bias + companion softmax logits, fp32)
# → at R=1, C=1024, kvh=8, hd=128, h=24: 6.5 MB; at R=1, C=4096: 26 MB; at R=8, C=4096: 210 MB.
# The 4090 has ONE virtual address space shared by every layer's pools, which is exactly how the
# MoE routegraph capture died (dot26: VA exhaustion while physical memory looked fine), and the
# served card runs at 23.47/23.59 GiB with ~13 MiB free under fleet load — measured 2026-11-28,
# when cap=4 across ~20 sliding layers reserved ~2.1 GiB and the capture's own 20 MiB fp32
# allocation OOMed mid-capture. So: a declared cap dial (infer_params.inkling_attn_graph_max_-
# graphs, default 1), a PRE-CAPTURE MARGIN GATE asking 2x the estimate (attn_vram_margin_ok) that
# refuses with decline="vram-margin" instead of fighting the allocator, and FREQUENCY-FILTERED
# admission with LRU eviction: field result 2026-11-27, dot26 pid 3970261 — a hard cap of 8
# filled with cold (rows-bucket × extent-bucket) corners within a minute of boot and then refused
# the hot key FOREVER, so the capture arm engaged on zero requests.
_ATTN_EXTENT_FLOOR = 1024      # smallest padded extent; collapses the CHEAP low end of the
                               # context axis into one key instead of one key per doubling
_ATTN_GRAPH_LOG_EVERY = 8192   # visits per layer before an ATTN-GRAPH summary line
_ATTN_GRAPH_MARGIN_FACTOR = 2  # free VRAM required per byte of estimated pool, before capturing
_ATTN_GRAPH_CAPTURE_SLACK = 32 * 1024 * 1024   # plus this much for the capture forward's OWN
                                               # temporaries: dot26 2026-11-28 died on a 20 MiB
                                               # fp32 matmul alloc inside _attend_one at 13 MiB
                                               # free, which the pool estimate does not cover
_ATTN_GRAPH_WARMUPS = 2        # side-stream warmups before capture (the estate runner's number)
_ATTN_GRAPH_MARGIN_COOLDOWN = 64   # rounds a layer waits after a vram-margin refusal, so the
                                   # driver round-trip is not paid every step on a full card


def attn_extent_bucket(need: int, ceiling: int, floor: int = _ATTN_EXTENT_FLOOR) -> int:
    """CTX leg of a sliding-layer capture key: the KV extent a captured graph may replay for.

    The next power of two at or above NEED, floored at FLOOR and capped at CEILING (window +
    chunk, the most any row can hold). Rounding up is safe because every column past a row's
    real length carries kv_valid=False (ring_geometry) and is masked out of the softmax; that
    is what makes the extent a BUCKET instead of one capture per context length. The floor is
    the deliberate widening the field result asked for: below it a doubling of context minted a
    new graph while the columns it saved were nearly free, and the sparse high end (where
    padding costs real bandwidth) stays tight. One implementation — the fence
    (tests/cpu/inkling_attn_graph_fence.py) refuses a second."""
    bucket = 1 << max(int(need) - 1, 0).bit_length()
    return min(max(bucket, int(floor)), int(ceiling))


class AttnGraphLedger:
    """Per-layer admission, eviction and telemetry for piecewise attention graphs. Pure python:
    the fence drives it directly, no CUDA and no Config (tests/cpu/inkling_attn_graph_fence.py).

    · frequency filter — a key is admitted only after it has been REQUESTED min_hits times
      since boot, so a fleet corner visited once (a transient joined batch, a one-off context
      crossing) never parks a pool that the hot single-lane key needs;
    · LRU eviction — admitting past cap evicts the LEAST RECENTLY USED resident key and drops
      its slot, which returns that graph's private pool (torch releases it when the CUDAGraph
      object dies). A refused key is not banned: re-admission is one threshold away;
    · counters ride the log so an A/B proves engagement without py-spy: see
      InklingAttention._attn_summary_line for the marker and field names.
    """

    def __init__(self, cap: int, min_hits: int):
        self.resident = OrderedDict()      # key -> slot, oldest first = eviction order
        self.requests = {}                 # key -> requests since boot (never reset)
        self.cap = int(cap)
        self.min_hits = int(min_hits)
        self.visits_since_log = 0
        self.evictions = 0
        self.declines = {}                 # short token -> count
        self.folded = {"capture": 0, "replay": 0}   # runner tallies kept across eviction

    def note(self, key):
        """Record that this round asked for `key`, before any admission decision."""
        self.requests[key] = self.requests.get(key, 0) + 1

    def hit(self, key):
        """Resident slot for `key`, marked most-recently-used; None when not resident."""
        slot = self.resident.get(key)
        if slot is None:
            return None
        self.resident.move_to_end(key)
        return slot

    def eligible(self, key) -> bool:
        return self.requests.get(key, 0) >= max(1, int(self.min_hits))

    def admit(self, key, slot):
        """Install `slot` for `key`, evicting least-recently-used residents to honour cap.
        Returns the evicted key (or None). The cap is read live so a yaml flip takes effect."""
        evicted = None
        while len(self.resident) >= max(0, int(self.cap)):
            evicted, doomed = self.resident.popitem(last = False)
            self._fold(doomed)
            self.evictions += 1
        self.resident[key] = slot
        self.resident.move_to_end(key)
        return evicted

    def drop(self, key):
        """Remove a resident key without counting an eviction (a graph that failed to capture
        is worthless, not merely cold)."""
        doomed = self.resident.pop(key, None)
        if doomed is not None:
            self._fold(doomed)

    def clear_resident(self):
        """Unload path: the pool addresses the graphs read die with the cache layers, so drop
        the graphs but keep the request counts (hot keys re-admit without warming again)."""
        for doomed in self.resident.values():
            self._fold(doomed)
        self.resident.clear()

    def refuse(self, token):
        self.declines[token] = self.declines.get(token, 0) + 1

    def _slot_stats(self, slot):
        """A slot carries its own counters ({"capture", "replay", "eager"}): the layer runs its
        own capture, so the tallies live with the graph, not on a shared runner object."""
        if not isinstance(slot, dict):
            return {}
        return slot.get("stats") or getattr(slot.get("runner"), "stats", None) or {}

    def _fold(self, slot):
        st = self._slot_stats(slot)
        self.folded["capture"] += st.get("capture", 0)
        self.folded["replay"] += st.get("replay", 0)

    def tally(self):
        """Live totals: the counters of resident graphs plus what eviction folded away."""
        live = {"capture": 0, "replay": 0, "eager": 0}
        for slot in self.resident.values():
            st = self._slot_stats(slot)
            for name in live:
                live[name] += st.get(name, 0)
        waiting = sum(1 for key, n in self.requests.items()
                      if key not in self.resident and n < max(1, int(self.min_hits)))
        return {"resident": len(self.resident), "cap": int(self.cap),
                "min_hits": int(self.min_hits), "waiting": waiting, "evictions": self.evictions,
                "captures": self.folded["capture"] + live["capture"],
                "replays": self.folded["replay"] + live["replay"],
                "eager": self.folded.get("eager", 0) + live["eager"],
                "declines": dict(self.declines),
                "requests": {key: n for key, n in self.requests.items()}}


def attn_capture_key(rows: int, extent: int, state_ptr: int) -> tuple | None:
    """Signature one captured sliding-layer decode graph may be replayed for:
    (shared decode bucket, padded KV extent, recurrent-pool identity).

    The rows leg is the SHARED arena rule — moe_coop_plan.padded_bucket_rows — never a re-derived
    bucket loop, so the graph lane and the coop arena agree on the padded row count (§U). The
    pool identity joins the key because a captured gather bakes the pool's ADDRESS: free()/alloc()
    moves it, and a replay against the old VA would read freed storage (law 33).
    None: above the decode ladder (prefill/mixed rows) — keep the eager path."""
    bucket = moe_coop_plan.padded_bucket_rows(rows)
    if bucket is None:
        return None
    return bucket, int(extent), int(state_ptr)


def attn_pool_bytes(rows: int, cols: int, kv_heads: int, head_dim: int, q_heads: int) -> int:
    """Estimated private-pool cost of ONE captured sliding-attention graph — the formula written
    in config.py: padded K, padded V and the ring-gather output (3 · rows · cols · kv_heads ·
    head_dim · 2 B, fp16) plus the fp32 relative-bias and companion softmax logits
    (2 · q_heads · rows · cols · 4 B).

    It deliberately does NOT include the capture forward's own fp32 temporaries: on dot26
    2026-11-28 the allocation that actually failed was a 20 MiB fp32 matmul inside _attend_one
    at 13 MiB of free memory, so the gate below asks for a MARGIN over this estimate, not the
    estimate itself."""
    return 3 * int(rows) * int(cols) * int(kv_heads) * int(head_dim) * 2 \
        + 2 * int(q_heads) * int(rows) * int(cols) * 4


def attn_vram_margin_ok(device, need_bytes: int, factor = _ATTN_GRAPH_MARGIN_FACTOR,
                        slack = _ATTN_GRAPH_CAPTURE_SLACK):
    """Pre-capture margin gate: allow a capture only when the device reports at least
    FACTOR × need_bytes + SLACK free. Returns (ok, free_bytes, reserved_bytes).

    The served 4090 runs at 23.47/23.59 GiB with ~13 MiB free under fleet load, and a capture
    that dies halfway reserves its pool anyway (the routegraph lesson: one VA space per card).
    The estimate passed in covers the pool, so SLACK covers what the capture forward itself
    allocates on top of it — a 20 MiB fp32 matmul was the allocation that actually failed on
    dot26 2026-11-28 — and FACTOR leaves room for the other layers still serving. Asking for a
    margin turns a probable-OOM capture into a logged refusal that keeps the eager path.
    A non-CUDA device reports ok: the graph arm is a CUDA path and a CPU probe would be a false
    refusal."""
    dev = device if isinstance(device, torch.device) else torch.device(device)
    if dev.type != "cuda":
        return True, -1, 0
    free, _total = torch.cuda.mem_get_info(dev)
    reserve = int(factor) * int(need_bytes) + int(slack)
    return int(free) >= reserve, int(free), reserve


def is_oom_error(err: BaseException) -> bool:
    """torch raises torch.cuda.OutOfMemoryError, or a RuntimeError whose text says 'out of
    memory' when the allocator itself fails inside a capture. Both mean the same thing here."""
    oom = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom is not None and isinstance(err, oom):
        return True
    return "out of memory" in str(err).lower()


def ring_geometry(
    pos_first: torch.Tensor,   # (B,) int64 DEVICE: position of the first token of this chunk
    t: int,                    # tokens in this chunk
    window: int,               # sliding attention window
    n_ring: int,               # ring rows per slot (InklingLayerState.k_state dim 1)
    ring_cols: int,            # padded column count (extent - t), fixed per capture bucket
):
    """Sliding-window read geometry as DEVICE tensors — no host scalar enters an index.

    Returns (idx, kv_pos, kv_valid, cnt):
      idx[b, j]      = (lo_b + j) % n_ring, lo_b = max(0, pos_b + t - window) — gather against the
                       fixed-address ring at a DEVICE index (the old code built arange(lo, lo+cnt)
                       from python ints, which a capture freezes);
      kv_pos[b, :]   = cat([lo_b + j for j < ring_cols, pos_b + i for i < t]) — the position of
                       every K/V column (ring columns, then this chunk's own rows);
      kv_valid[b, :] = False where a padded ring column is stale (j >= cnt_b) — required, not
                       cosmetic: lo_b + cnt_b == pos_b, so the first stale column sits at
                       dist = 0 and the causal mask alone would admit it;
      cnt[b]         = pos_b - lo_b = ring rows this sequence really has.
    Values match the scalar path exactly for j < cnt (fence arm 1)."""
    dev = pos_first.device
    lo = (pos_first + t - window).clamp(min = 0)                        # (B,)
    cnt = pos_first - lo                                                # (B,)
    cols = torch.arange(ring_cols, device = dev)                        # (ring_cols,)
    idx = (lo[:, None] + cols[None, :]) % n_ring                        # (B, ring_cols)
    kv_pos = torch.cat([lo[:, None] + cols[None, :],
                        pos_first[:, None] + torch.arange(t, device = dev)[None, :]], dim = 1)
    kv_valid = torch.cat([cols[None, :] < cnt[:, None],
                          torch.ones((pos_first.shape[0], t), dtype = torch.bool, device = dev)],
                         dim = 1)
    return idx, kv_pos, kv_valid, cnt


def rmsnorm_f(x: torch.Tensor, eps: float) -> torch.Tensor:
    # Fused single-kernel form of x.float()*rsqrt(mean(x.float()²)+eps) (was 3 eager kernels,
    # top decode hotspot at 138/360 census samples 2026-11-26); fp32 promotion preserved.
    return torch.nn.functional.rms_norm(x.float(), (x.shape[-1],), eps = eps)


def l2_clip(x: torch.Tensor, threshold: float) -> torch.Tensor:
    input_dtype = x.dtype
    x = x.float()
    norm = torch.linalg.vector_norm(x, dim = -1, keepdim = True)
    return (x * (threshold / (norm + 1e-6)).clamp(max = 1.0)).to(input_dtype)


def scaled_l2_clip(x: torch.Tensor, scale: torch.Tensor, threshold: float) -> torch.Tensor:
    input_dtype = x.dtype
    x = x.float() * scale.float()
    norm = torch.linalg.vector_norm(x, dim = -1, keepdim = True)
    return (x * (threshold / (norm + 1e-6)).clamp(max = 1.0)).to(input_dtype)


def compute_log_tau(pos: torch.Tensor, alpha: float, floor: int) -> torch.Tensor:
    """The position term shared by every log_scale consumer in a forward. Hoisted so global
    layers compute it once, not once per q/rel projection."""
    return 1.0 + alpha * torch.log(((pos + 1).float() / floor).clamp(min = 1.0))


def log_scale(x: torch.Tensor, pos: torch.Tensor, alpha: float, floor: int,
              tau: torch.Tensor | None = None) -> torch.Tensor:
    input_dtype = x.dtype
    if tau is None:
        tau = compute_log_tau(pos, alpha, floor)
    return (x.float() * tau[..., None, None]).to(input_dtype)


def block_read(query: torch.Tensor, bank: list, partial: torch.Tensor, eps: float) -> list:
    """
    Tiny cross-attention over [bank ; partial]: a few learned query vectors, same at every
    position, softmax over the (<= 5) stacked states. Reference: Inkling2LoopBlockAttentionResidual.
    query: (nq, H) fp32; bank: list of (B, T, H); partial: (B, T, H). Returns nq × (B, T, H) fp32.
    """
    srcs = list(bank)
    srcs.append(partial)
    stacked = torch.stack(srcs, dim = -2).float()                     # (B, T, n, H)
    stacked = rmsnorm_f(stacked, eps)
    h = stacked.shape[-1]
    q = query.float().view(-1, h)
    q = q.expand(*stacked.shape[:-2], -1, -1)                         # (B, T, nq, H)
    scores = torch.matmul(q, stacked.transpose(-1, -2)) * (h ** -0.5)
    w = F.softmax(scores.float(), dim = -1)
    out = torch.matmul(w, stacked)                                    # (B, T, nq, H)
    return [out[..., i, :].to(partial.dtype) for i in range(q.shape[-2])]


# ---------------------------------------------------------------------------
# Per-layer recurrent state: depthwise conv windows + sliding K/V ring
# ---------------------------------------------------------------------------

class InklingLayerState:

    def __init__(
        self,
        module: "InklingConvs",
        max_batch_size: int,
        max_history: int,
        cache_id: int,
    ):
        self.module = module
        self.device = None
        self.max_history = max_history
        self.max_batch_size = max_batch_size
        self.cache_id = cache_id

        k = module.conv_kernel_size
        self.conv_states = {
            name: torch.empty((max_batch_size, dim, k + max_history), dtype = torch.float, device = "meta")
            for name, dim in module.conv_dims.items()
        }

        if module.sliding:
            self.k_state = torch.empty(
                (max_batch_size, module.kv_state_size, module.num_kv_heads, module.head_dim),
                dtype = torch.half, device = "meta",
            )
            self.v_state = torch.empty_like(self.k_state)
        else:
            self.k_state = None
            self.v_state = None

    def get_checkpoint_size(self):
        k = self.module.conv_kernel_size
        size = sum(dim * k * 4 for dim in self.module.conv_dims.values())
        if self.module.sliding:
            w = self.module.sliding_window
            size += 2 * w * self.module.num_kv_heads * self.module.head_dim * 2
        return size

    def storage_size(self):
        tensors = list(self.conv_states.values())
        if self.module.sliding:
            tensors += [self.k_state, self.v_state]
        return sum(t.numel() * t.element_size() for t in tensors)

    def get_state_tensors(self):
        return tuple(self.conv_states.values())

    def alloc(self, device):
        self.conv_states = {n: torch.zeros_like(t, device = device) for n, t in self.conv_states.items()}
        if self.module.sliding:
            self.k_state = torch.zeros_like(self.k_state, device = device)
            self.v_state = torch.zeros_like(self.v_state, device = device)
        self.device = device

    def free(self):
        self.conv_states = {n: torch.empty_like(t, device = "meta") for n, t in self.conv_states.items()}
        if self.module.sliding:
            self.k_state = torch.empty_like(self.k_state, device = "meta")
            self.v_state = torch.empty_like(self.v_state, device = "meta")
        self.device = None

    def clear(self, idx: int):
        if self.device is None:
            return
        for t in self.conv_states.values():
            t[idx].zero_()
        if self.module.sliding:
            self.k_state[idx].zero_()
            self.v_state[idx].zero_()

    def rewind(self, slot: int, last_history: int, num_tokens: int):
        # Speculative decoding is off for v1 and the whole-forward CUDA graph is not
        # graph-safe for this eager-torch attention, so the only rewind the lane issues
        # undoes a warmup/probe forward that had no prior real history. Zeroing the slot
        # returns it to the pristine state that forward advanced from.
        assert num_tokens >= 0 and (last_history == 0 or num_tokens <= last_history + 1), \
            f"inkling rewind: {num_tokens} tokens outside recorded history of {last_history}"
        if self.device is None:
            return
        for t in self.conv_states.values():
            t[slot].zero_()
        if self.module.sliding:
            self.k_state[slot].zero_()
            self.v_state[slot].zero_()

    def stash(self, slot, position: int = 0):
        st = {n: host_copy(t[slot, :, :self.module.conv_kernel_size])
              for n, t in self.conv_states.items()}
        if self.module.sliding:
            w = self.module.sliding_window
            st["kv"] = (host_copy(self.k_state[slot, :w]), host_copy(self.v_state[slot, :w]))
        return st

    def unstash(self, slot, stashed, position: int = 0):
        for n, t in self.conv_states.items():
            t[slot, :, :self.module.conv_kernel_size].copy_(stashed[n])
        if self.module.sliding:
            w = self.module.sliding_window
            self.k_state[slot, :w].copy_(stashed["kv"][0])
            self.v_state[slot, :w].copy_(stashed["kv"][1])

    def tp_export(self, plan):
        return {
            "cls": InklingLayerState,
            "args": {
                "cache_id": self.cache_id,
                "max_history": self.max_history,
                "max_batch_size": self.max_batch_size,
            }
        }


class InklingConvs(Module):
    """
    Per-layer recurrent state holder and the depthwise causal convolutions on it:
    K/V (static on sliding layers, per-token-tap dynamic on global layers), post-attention
    and post-MLP convs, plus the K/V ring of sliding layers. Not a chain module - the owning
    InklingLayer calls its ops directly. Registered as a submodule so load/unload/tensor
    bookkeeping and the recurrent_cache cap ride the usual machinery.
    """

    def __init__(
        self,
        config: Config,
        key: str,
        layer_idx: int,
        hidden_size: int,
        kv_dim: int,
        num_kv_heads: int,
        head_dim: int,
        sliding: bool,
        sliding_window: int,
        conv_kernel_size: int,
        dynamic: bool,
        key_k_conv: str,
        key_v_conv: str,
        key_attn_conv: str,
        key_mlp_conv: str,
    ):
        super().__init__(config, key, None)
        self.module_name = "InklingConvs"
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.kv_dim = kv_dim
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding = sliding
        self.sliding_window = sliding_window
        self.conv_kernel_size = conv_kernel_size
        self.dynamic = dynamic

        # Ring holds the full attention window; the current chunk's K/V is attended from
        # registers (read-before-write), so no chunk-size headroom is needed. Page-rounded.
        self.kv_state_size = -(-sliding_window // PAGE_SIZE) * PAGE_SIZE

        self.conv_dims = {"k": kv_dim, "v": kv_dim, "attn": hidden_size, "mlp": hidden_size}

        self.key_k_conv = key_k_conv
        self.key_v_conv = key_v_conv
        self.key_attn_conv = key_attn_conv
        self.key_mlp_conv = key_mlp_conv

        self.w_k = self.w_v = self.w_kv = self.w_attn = self.w_mlp = None
        self.k_static = self.v_static = None
        self.k_wproj = self.v_wproj = None

        self.caps.update({"recurrent_cache": True})
        self.layer_state_cls = InklingLayerState
        self.recurrent_layers = []
        self.tp_recurrent_lookup = {}
        self.tp_reduce = False
        self.has_split_cache = False

    @override
    def optimizer_targets(self):
        return []

    @override
    def forward(self, x, params, out_dtype = None):
        raise RuntimeError("InklingConvs is driven by InklingLayer/InklingAttention")

    def _load_depthwise(self, key: str, chan: int):
        # (C, 1, K) depthwise torch layout -> (C, K) fp32
        w = self.config.stc.get_tensor(key, self.device, optional = False, allow_bf16 = True, no_defer = True).float()
        if w.ndim == 3:
            w = w.squeeze(1)
        assert w.shape[0] == chan and w.shape[1] == self.conv_kernel_size, \
            f"inkling conv {key}: unexpected shape {tuple(w.shape)}"
        return w.contiguous()

    @override
    def load(self, device, **kwargs):
        self.device = device
        if self.dynamic:
            # stored (K, C) fp32; weight_proj (heads*K, hidden) fp32
            self.k_static = self.config.stc.get_tensor(
                self.key_k_conv + ".static_w", device, optional = False, allow_bf16 = True, no_defer = True).float().contiguous()
            self.v_static = self.config.stc.get_tensor(
                self.key_v_conv + ".static_w", device, optional = False, allow_bf16 = True, no_defer = True).float().contiguous()
            self.k_wproj = self.config.stc.get_tensor(
                self.key_k_conv + ".weight_proj", device, optional = False, allow_bf16 = True, no_defer = True).float().contiguous()
            self.v_wproj = self.config.stc.get_tensor(
                self.key_v_conv + ".weight_proj", device, optional = False, allow_bf16 = True, no_defer = True).float().contiguous()
            self.w_k = self.w_v = None
        else:
            self.w_k = self._load_depthwise(self.key_k_conv + ".weight", self.kv_dim)
            self.w_v = self._load_depthwise(self.key_v_conv + ".weight", self.kv_dim)
            # Prebuild the fused K/V weight (rung 4): cat once at load so conv_kv costs no
            # per-token concatenation. (2*kv, 1, K), matches conv_kv's groups = 2*kv.
            self.w_kv = torch.cat([self.w_k, self.w_v], dim = 0).unsqueeze(1).contiguous()
        self.w_attn = self._load_depthwise(self.key_attn_conv + ".weight", self.hidden_size)
        self.w_mlp = self._load_depthwise(self.key_mlp_conv + ".weight", self.hidden_size)
        for rl in self.recurrent_layers:
            rl.alloc(device)

    @override
    def unload(self):
        self.w_k = self.w_v = self.w_kv = self.w_attn = self.w_mlp = None
        self.k_static = self.v_static = self.k_wproj = self.v_wproj = None
        for rl in self.recurrent_layers:
            rl.free()
        super().unload()

    @override
    def get_tensors(self):
        t = super().get_tensors()
        for k, x in [
            (self.key_k_conv + (".weight" if not self.dynamic else ".static_w"), self.w_k if not self.dynamic else self.k_static),
            (self.key_v_conv + (".weight" if not self.dynamic else ".static_w"), self.w_v if not self.dynamic else self.v_static),
            (self.key_attn_conv + ".weight", self.w_attn), (self.key_mlp_conv + ".weight", self.w_mlp),
        ]:
            if x is not None:
                t[k] = x
        return t

    def _state_slot(self, params: dict):
        rsg = params.get("recurrent_states")
        if not rsg:
            return None, None
        if rsg[0].exported:
            st = self.tp_recurrent_lookup[rsg[0].cache]
        else:
            st = rsg[0].cache.get_recurrent_layer((self.layer_idx, params.get("layer_instance") or 0))
        return st, get_for_device(params, "recurrent_slots", "cpu").tolist()

    def conv(self, name: str, x: torch.Tensor, params: dict) -> torch.Tensor:
        """
        Depthwise causal conv with (conv + residual)/sqrt(kernel+1), fp32
        (Inkling2LoopShortConvolution). x: (B, T, C).
        """
        weight = {"k": self.w_k, "v": self.w_v, "attn": self.w_attn, "mlp": self.w_mlp}[name]
        kdim = self.conv_kernel_size
        input_dtype = x.dtype
        x = x.float()
        residual = x
        bsz, seqlen, chan = x.shape
        xt = x.transpose(1, 2)                                          # (B, C, T)

        st, slots = self._state_slot(params)
        if st is None:
            state = torch.zeros((bsz, chan, kdim), dtype = torch.float, device = x.device)
            slots_cpu = list(range(bsz))
            save_history = False
        else:
            state = st.conv_states[name]
            slots_cpu = slots
            save_history = params.get("recurrent_history", False)

        ys = []
        for i, slot in enumerate(slots_cpu):
            full = torch.cat([state[slot, :, :kdim].unsqueeze(0), xt[i].unsqueeze(0)], dim = -1)
            if save_history:
                write_size = min(state.shape[-1], full.shape[-1])
                state[slot][:, -write_size:].copy_(full[0, :, -write_size:])
            else:
                state[slot][:, :kdim].copy_(full[0, :, -kdim:])
            y = F.conv1d(full, weight.unsqueeze(1), None, padding = kdim - 1, groups = chan)
            ys.append(y[:, :, kdim:kdim + seqlen])

        out = (ys[0] if len(ys) == 1 else torch.cat(ys, dim = 0)).transpose(1, 2)   # (B, T, C)
        out = (out + residual) / math.sqrt(kdim + 1)
        return out.to(input_dtype)

    def conv_kv(self, k_lin: torch.Tensor, v_lin: torch.Tensor, params: dict):
        """K and V depthwise convs in ONE conv1d launch (sm_89 rung 4: the K/V attention-prepare
        kernels run as one launch). Depthwise conv mixes no channels, so concatenating K and V on
        the channel axis and sharing the residual/scale is bit-identical to two separate conv()
        calls. The fused weight is prebuilt in load(); the stored K/V conv-state buffers are read
        in place (no full-width per-token copy), so the only added kernel per token is the single
        activation cat the one conv1d needs, against dropping a whole second conv body."""
        kdim = self.conv_kernel_size
        input_dtype = k_lin.dtype
        kv = k_lin.shape[-1]
        x = torch.cat([k_lin, v_lin], dim = -1).float()                   # (B, T, 2*kv)
        residual = x
        bsz, seqlen, chan = x.shape
        xt = x.transpose(1, 2)                                            # (B, 2*kv, T)

        st, slots = self._state_slot(params)
        if st is None:
            state = torch.zeros((bsz, chan, kdim), dtype = torch.float, device = x.device)
            slots_cpu = list(range(bsz))
            save_history = False
            bufs = None
        else:
            sk, sv = st.conv_states["k"], st.conv_states["v"]
            slots_cpu = slots
            save_history = params.get("recurrent_history", False)
            bufs = ((sk, 0, kv), (sv, kv, chan))

        ys = []
        for i, slot in enumerate(slots_cpu):
            if bufs is None:
                lpad = state[slot, :, :kdim].unsqueeze(0)
            else:
                lpad = torch.cat([sk[slot, :, :kdim], sv[slot, :, :kdim]], dim = 0).unsqueeze(0)
            full = torch.cat([lpad, xt[i].unsqueeze(0)], dim = -1)
            if save_history:
                for dst, s0, s1 in bufs:
                    write_size = min(dst.shape[-1], full.shape[-1])
                    dst[slot][:, -write_size:].copy_(full[0, s0:s1, -write_size:])
            elif bufs is not None:
                for dst, s0, s1 in bufs:
                    dst[slot][:, :kdim].copy_(full[0, s0:s1, -kdim:])
            y = F.conv1d(full, self.w_kv, None, padding = kdim - 1, groups = chan)
            ys.append(y[:, :, kdim:kdim + seqlen])

        out = (ys[0] if len(ys) == 1 else torch.cat(ys, dim = 0)).transpose(1, 2)   # (B, T, 2*kv)
        out = (out + residual) / math.sqrt(kdim + 1)
        out = out.to(input_dtype)
        return out[..., :kv], out[..., kv:]

    def conv_dynamic(self, which: str, x: torch.Tensor, dyn_input: torch.Tensor, params: dict) -> torch.Tensor:
        """
        Per-token-tap conv for the global K/V (Inkling2LoopDynamicShortConvolution):
        taps = static (K,C) laid out per head + weight_proj(dyn_input) per token, shared
        within each KV head; (conv + residual)/sqrt(2*kernel+1), fp32. x: (B, T, kv_dim).
        """
        static_w = self.k_static if which == "k" else self.v_static
        wproj = self.k_wproj if which == "k" else self.v_wproj
        kdim = self.conv_kernel_size
        input_dtype = x.dtype
        bsz, seqlen, hidden = x.shape

        dw = torch.matmul(dyn_input.float().reshape(-1, dyn_input.shape[-1]), wproj.T)
        dw = dw.view(bsz, seqlen, self.num_kv_heads, 1, kdim)
        # static is stored (K, C): reference lays it out per head as (heads, C/heads, K)
        weights = static_w.float().T.reshape(self.num_kv_heads, -1, kdim) + dw

        x = x.float()
        residual = x
        ctx = x.transpose(1, 2)                                          # (B, C, T)
        st, slots = self._state_slot(params)
        if st is not None:
            cs = st.conv_states[which]
            slots_cpu = slots
            save_history = params.get("recurrent_history", False)
            ctxs = []
            for i, slot in enumerate(slots_cpu):
                full = torch.cat([cs[slot].unsqueeze(0), ctx[i].unsqueeze(0)], dim = -1)
                if save_history:
                    write_size = min(cs.shape[-1], full.shape[-1])
                    cs[slot][:, -write_size:].copy_(full[0, :, -write_size:])
                else:
                    cs[slot].copy_(full[0, :, -cs.shape[-1]:])
                ctxs.append(full)
            ctx = torch.cat(ctxs, dim = 0)
        ctx = F.pad(ctx, (kdim - 1, 0))[..., -(seqlen + kdim - 1):]
        windows = ctx.unfold(-1, kdim, 1).flip(-1).transpose(1, 2)       # (B, T, C, K)
        windows = windows.reshape(bsz, seqlen, self.num_kv_heads, -1, kdim)
        out = (windows * weights).sum(-1).reshape(bsz, seqlen, hidden)
        out = (out + residual) / math.sqrt(2 * kdim + 1)
        return out.to(input_dtype)

    def ring_write(self, k: torch.Tensor, v: torch.Tensor, params: dict):
        """Append processed K/V (B, T, kvh, hd) at positions (cache_seqlens..) into the ring."""
        st = self._state_slot(params)[0]
        if st is None:
            return
        positions = self._positions(params)
        s = st.k_state
        n = s.shape[1]
        slots = get_for_device(params, "recurrent_slots", "cpu").tolist()
        for b, slot in enumerate(slots):
            pos = int(positions[b])
            idx = (pos + torch.arange(k.shape[1], device = k.device)) % n
            s[slot, idx] = k[b].to(s.dtype)
            st.v_state[slot, idx] = v[b].to(s.dtype)

    def ring_read(self, params: dict, seqlen: int, slots: torch.Tensor,
                  pos_first: torch.Tensor, ring_cols: int):
        """
        Rows [lo_b, lo_b+ring_cols) of the ring per sequence, as DEVICE-indexed gathers, with the
        column positions and validity the attention needs. lo_b = max(0, pos_b + seqlen - window);
        columns past a row's real length (cnt_b) are stale ring cells carried at a fixed shape and
        masked by kv_valid, so the read has one shape per capture bucket instead of one per step.

        Returns (k, v) as (B, ring_cols, kvh, hd) plus (kv_pos, kv_valid, cnt) — see ring_geometry.
        slots and pos_first are device tensors; no python scalar reaches an index, which is what
        froze this read under capture (fwd_graph.unsafe_types names it).
        """
        st = self._state_slot(params)[0]
        s = st.k_state
        n = s.shape[1]
        idx, kv_pos, kv_valid, cnt = ring_geometry(
            pos_first, seqlen, self.sliding_window, n, ring_cols)
        flat_idx = slots.long()[:, None] * n + idx                       # (B, ring_cols)
        tail = tuple(s.shape[2:])
        k = s.reshape(-1, *tail)[flat_idx]
        v = st.v_state.reshape(-1, *tail)[flat_idx]
        # padded columns are ZERO, as the per-row pad() left them: multiplying a valid row by 1.0
        # is exact, so the control arm keeps every bit the old gather produced, and a captured
        # replay never reads a stale ring cell as context. Skipped at ring_cols == 0 (the first
        # step after prefill), where -1 in a zero-width view is ambiguous.
        if ring_cols:
            colmask = kv_valid[:, :ring_cols].to(s.dtype).view(-1, ring_cols, 1, 1)
            k, v = k * colmask, v * colmask
        return k, v, kv_pos, kv_valid, cnt

    def _positions(self, params: dict) -> list:
        positions = params.get("positions")
        if positions is not None:
            if isinstance(positions, torch.Tensor):
                # host_values, not .cpu(): at decode `positions` IS the generator's pinned staging
                # buffer (prepare_flash_attn aliases it to cache_seqlens), so this returns the host
                # original for free; a genuinely device-resident tensor inside a capture would pay
                # a blocking D2H (cuStreamSynchronize -> CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED 900)
                # on the recording stream, which is how every capture of this model died.
                return host_values(positions, ("inkling2", "positions")).tolist()
            return list(positions)
        return [params.get("position", 0)] * params.get("batch_size", 1)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class InklingAttention(Module):
    """
    No-RoPE attention with learned relative-position bias, per-token sinks, Q/K scaled L2
    clip (log-scaled on global layers), unweighted V norm, sigmoid-gated output and short
    convolutions on K/V. Sliding layers keep a ring in their InklingConvs state; global
    layers use the engine's paged KV cache (dequantized to fp16 on read).
    Driven by InklingLayer via forward_attn (not a chain module by itself).
    """

    def __init__(
        self,
        config: Config,
        key: str,
        layer_idx: int,
        hidden_size: int,
        head_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        sliding: bool,
        rel_extent: int,
        d_rel: int,
        qk_threshold: float,
        rel_threshold: float,
        rms_norm_eps: float,
        log_alpha: float,
        log_floor: int,
        convs: InklingConvs,
    ):
        super().__init__(config, key, None)
        self.module_name = "InklingAttention"
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_q_heads // num_kv_heads
        self.sliding = sliding
        self.rel_extent = rel_extent
        self.d_rel = d_rel
        self.qk_threshold = qk_threshold
        self.rel_threshold = rel_threshold
        self.rms_norm_eps = rms_norm_eps
        self.log_alpha = log_alpha
        self.log_floor = log_floor
        self.scaling = head_dim ** -0.5
        self.convs = convs
        self.register_submodule(convs)

        kp = f"{key}."
        self.wq = Linear(config, kp + "wq_du", hidden_size, num_q_heads * head_dim,
                         qmap = "block.attn", out_dtype = torch.half, qgroup = key + ".q")
        self.wk = Linear(config, kp + "wk_dv", hidden_size, num_kv_heads * head_dim,
                         qmap = "block.attn", out_dtype = torch.half, qgroup = key + ".k")
        self.wv = Linear(config, kp + "wv_dv", hidden_size, num_kv_heads * head_dim,
                         qmap = "block.attn", out_dtype = torch.half, qgroup = key + ".v")
        self.wo = Linear(config, kp + "wo_ud", num_q_heads * head_dim, hidden_size,
                         qmap = "block.attn.o", out_dtype = torch.float, qgroup = key + ".o")
        self.gate = Linear(config, kp + "gate_attn", hidden_size, num_q_heads * head_dim,
                           qmap = "block.attn", out_dtype = torch.half, qgroup = key + ".gate")
        self.wr = Linear(config, kp + "wr_du", hidden_size, num_q_heads * d_rel,
                         qmap = "block.attn", out_dtype = torch.half, qgroup = key + ".r")
        for lin in (self.wq, self.wk, self.wv, self.wo, self.gate, self.wr):
            self.register_submodule(lin)

        self.k_q_scale = kp + "q_l2_clip.scale"
        self.k_rel_scale = kp + "rel_l2_clip.scale"
        self.k_rel_proj = kp + "rel_logits_proj.proj"
        self.k_sink = kp + "attn_sink_proj.weight"
        self.q_scale = None
        self.rel_scale = None
        self.rel_proj = None
        self.sink_w = None

        if not sliding:
            self.caps.update({"kv_cache": True})
            self.tp_cache_lookup = {}
            self.has_split_cache = False
            self.kv_head_dim = head_dim
            self.kv_heads = num_kv_heads
        self.cache_layers = []
        # Piecewise-graph state (off unless config.infer_params.inkling_attn_graph): admission,
        # eviction and telemetry for this layer's captured graphs. unload() clears the residents
        # because a graph round bakes the recurrent pool's ADDRESS (law 33: cached device
        # addresses die with the tensors they pin); cap and min_hits are re-read from
        # infer_params every round, so the yaml dials move without a reload.
        self._attn_ledger = AttnGraphLedger(config.infer_params.inkling_attn_graph_max_graphs,
                                            config.infer_params.inkling_attn_graph_min_hits)
        self._attn_graph_failed = None
        self._attn_graph_logged = set()
        self._attn_margin_wait = 0      # rounds to skip the VRAM probe after a margin refusal

    @override
    def optimizer_targets(self):
        return [[p.optimizer_targets() for p in (self.wq, self.wk, self.wv, self.wo, self.gate, self.wr)]]

    @override
    def load(self, device, **kwargs):
        super().load(device, **kwargs)
        for cl in self.cache_layers:
            cl.alloc(device)
        self.q_scale = self.config.stc.get_tensor(self.k_q_scale, device, optional = False, allow_bf16 = True, no_defer = True).float()
        self.rel_scale = self.config.stc.get_tensor(self.k_rel_scale, device, optional = False, allow_bf16 = True, no_defer = True).float()
        self.rel_proj = self.config.stc.get_tensor(self.k_rel_proj, device, optional = False, allow_bf16 = True, no_defer = True).float()
        self.sink_w = self.config.stc.get_tensor(self.k_sink, device, optional = False, allow_bf16 = True, no_defer = True).float()

    @override
    def unload(self):
        self.q_scale = self.rel_scale = self.rel_proj = self.sink_w = None
        for cl in getattr(self, "cache_layers", []):
            cl.free()
        self.cache_layers = []
        # the graphs read the pool addresses cl.free() releases: drop them, never replay dead VA
        self._attn_ledger.clear_resident()
        self._attn_graph_failed = None
        self._attn_margin_wait = 0
        super().unload()

    @override
    def get_tensors(self):
        t = super().get_tensors()
        for k, x in [
            (self.k_q_scale, self.q_scale), (self.k_rel_scale, self.rel_scale),
            (self.k_rel_proj, self.rel_proj), (self.k_sink, self.sink_w),
        ]:
            if x is not None:
                t[k] = x
        return t

    @override
    def forward(self, x, params, out_dtype = None):
        raise RuntimeError("InklingAttention is driven by InklingLayer")

    def project_qkv(self, h_q, h_k, h_v, pos_row, params, tau=None):
        q_lin = self.wq.forward(h_q, params)
        b, t, _ = q_lin.shape
        q = q_lin.view(b, t, self.num_q_heads, self.head_dim).float()
        q = scaled_l2_clip(q, self.q_scale, self.qk_threshold)
        if not self.sliding:
            q = log_scale(q, pos_row, self.log_alpha, self.log_floor, tau)
        q = q.to(torch.half).transpose(1, 2).contiguous()                # (B,h,T,d)

        k_lin = self.wk.forward(h_k, params)
        v_lin = self.wv.forward(h_v, params)
        if self.sliding:
            kc, vc = self.convs.conv_kv(k_lin, v_lin, params)
        else:
            kc = self.convs.conv_dynamic("k", k_lin, h_k, params)
            vc = self.convs.conv_dynamic("v", v_lin, h_v, params)
        kh = kc.view(b, t, self.num_kv_heads, self.head_dim)
        vh = vc.view(b, t, self.num_kv_heads, self.head_dim)
        kh = l2_clip(kh, self.qk_threshold)
        vh = rmsnorm_f(vh, self.rms_norm_eps).to(vh.dtype)               # unweighted V norm
        return q, kh, vh                                                  # (B,h,T,d) / (B,T,kvh,d)

    def rel_logits(self, h_q, pos_row, params, tau=None):
        rs = self.wr.forward(h_q, params)                                 # (B,T,h*d_rel)
        b, t, _ = rs.shape
        rs = rs.view(b, t, self.num_q_heads, self.d_rel).float()
        rs = scaled_l2_clip(rs, self.rel_scale, self.rel_threshold)
        if not self.sliding:
            rs = log_scale(rs, pos_row, self.log_alpha, self.log_floor, tau)
        return torch.matmul(rs.transpose(1, 2), self.rel_proj)            # (B,h,T,extent)

    def _attend_one_tile(self, S, T, itemsize):
        """Largest query-tile whose dominant fp32 temporaries -- the (h, tile, S) relative
        bias and the (kvh, grp, tile, S+1) softmax logits -- stay under the per-call byte
        budget. Kept >= T for the usual single-tile decode (T == 1), so the hoist adds no
        extra query tile and stays launch-neutral; only multi-row passes (T above the byte
        budget) tile down, which is where the (h, T, S) bias was the largest allocation."""
        T = int(T)
        if T <= 1:
            # Single-row decode tiles at 1 whatever the budget says: cap is floored at 1
            # and the return is min(T, cap) == T, so the mem_get_info driver round-trip
            # plus budget maths were pure overhead on every attention layer of every
            # decode step (35/360 py-spy samples 2026-11-26). Tile-invariant by proof:
            # same tile, same allocation, same bits. Multi-row passes keep the live read.
            return T
        budget = getattr(self, "attend_tile_bytes", _ATTEND_TILE_BYTES)
        if torch.cuda.is_available():
            # Headroom, not the ceiling, decides: the served card runs with 0.4-2.8 GiB
            # free depending on fleet load, so a constant budget OOMs exactly when the
            # engine is busiest. Half of what the device reports stays for the rest of
            # the forward; the 8 MiB floor keeps the tile count bounded when free is tiny.
            # Row grouping never changes a row's softmax set, so any value here is
            # bit-identical -- this dial cannot move the output.
            free = int(torch.cuda.mem_get_info()[0])
            budget = min(budget, max(8 * 1024 * 1024, free // 2))
        h = self.num_kv_heads * self.num_kv_groups       # combined q-head count (24)
        denom = h * S * itemsize * 2                     # bias + companion logits, fp32
        if not denom:
            return int(T)
        cap = max(1, int(budget // denom))               # rows that actually fit the budget
        # 32 rows is a preference (it mirrors the old S>16384 path and bounds the loop count),
        # never a licence to overrun the budget. Applied unconditionally it was the latter: at
        # S=262,144 one row costs ~50 MiB, so a 512 MiB budget allows 10 rows and the floor
        # bought 32 -- a ~1.6 GiB allocation under a 512 MiB ceiling, three times over, exactly
        # where the served window lives. Take the floor only while it still fits.
        if 32 * denom <= budget:
            cap = max(cap, 32)
        return min(int(T), cap)

    def _attend_keytile_cols(self, S, rows, d, itemsize):
        """Largest key-column block whose dominant fp32 temporaries — the (h, rows, cols)
        relative bias, the (kvh, grp*rows, cols) scores, and the per-chunk (kvh, d, cols) fp32 K
        and V casts — together fit the per-call byte budget.

        The query-tile dial (_attend_one_tile) bounds rows, but the scores/bias plane grows
        with the KEY axis too, so a long-S prefill keeps an (h, rows, S) slab no matter how
        small rows gets — the same O(S) peak _attend_one pays. Column tiling bounds the part
        the row dial cannot reach. The per-column cost carries BOTH the row-dependent terms
        (bias + scores, 2*h*rows*itemsize) and the row-INDEPENDENT fp32 casts (K and V,
        2*kvh*d*itemsize); the keytile branch's byte model omitted the casts, the term that
        dominates the peak while rows stays small (a defect this merge does not carry), and the
        V cast is the per-chunk fp32 copy the long-context PV accumulation (see _attend_one_
        keytile) requires to stay off fp16 overflow. Floored at rel_extent so one chunk still
        covers a full relative window; capped at S so a short cache is a single chunk (launch-
        neutral, identical to the control arm's iteration count)."""
        S = int(S)
        rows = max(1, int(rows))
        budget = getattr(self, "attend_tile_bytes", _ATTEND_TILE_BYTES)
        if torch.cuda.is_available():
            free = int(torch.cuda.mem_get_info()[0])
            budget = min(budget, max(8 * 1024 * 1024, free // 2))
        h = self.num_kv_heads * self.num_kv_groups
        kvh = self.num_kv_heads
        per_col = 2 * h * rows * itemsize + 2 * kvh * d * itemsize   # bias+scores (row-scaled) + K cast + V cast (row-free)
        if not per_col:
            return S
        cols = budget // per_col
        cols = max(self.rel_extent, cols)                        # floor: one full relative window
        return min(S, max(1, cols))

    def _attend_one(self, q, k, v, rel_small, sink, q_pos, kv_pos, kv_valid = None):
        # single sequence: q (h,T,d) half; k/v (S_,kvh,d) half; rel_small (h,T,extent) fp32;
        # sink (T,h) fp32; q_pos (T,), kv_pos (S_,) int on device;
        # kv_valid (S_,) bool on device: False on columns padded past this row's real extent
        # (None = every column of k is real, which is what the global path and the exact-fit
        # sliding path pass, so their mask tensor is bit-for-bit the one before)
        h, T, d = q.shape
        S = k.shape[0]
        grp = self.num_kv_groups
        kvh = self.num_kv_heads
        kxf = k.transpose(0, 1).transpose(1, 2)                            # (kvh,d,S) fp16 view, no cast
        k_chunk = max(self.rel_extent, _ATTEND_K_CHUNK_BYTES // (kvh * d * 4))
        vx = v.transpose(0, 1)                                            # (kvh,S,d), NOT expanded
        q4 = q.view(kvh, grp, T, d).float()                               # heads grouped by kv head

        # dist/causal/bias are built per query tile, not once for the whole (h, T, S): the
        # fp32 (h, T, S) bias gather was the largest single allocation on the long-S path,
        # and it sat OUTSIDE the tile loop so shrinking the tile never touched it. Tiling the
        # relative-position math keeps its peak at O(tile * S) and is bit-identical -- every
        # value is a pure function of (q_pos[row], kv_pos), so each tile's rows are bit-equal
        # to the old full-matrix rows.
        tile = self._attend_one_tile(S, T, 4)
        outs = []
        for qs in range(0, T, tile):
            qe = min(qs + tile, T)
            dist = q_pos[qs:qe, None] - kv_pos[None, :]                   # (tt, S)
            causal = dist >= 0
            if kv_valid is not None:
                # padded ring columns join the causal refusal, so they leave the softmax exactly as
                # the old per-row slice left it. With every column valid (global path, exact-fit
                # sliding) this AND is with an all-True row and cannot change a bit.
                causal = causal & kv_valid[None, :]
            bias = rel_small[:, qs:qe, :].gather(                         # (h, tt, S)
                -1, dist.clamp(0, self.rel_extent - 1).unsqueeze(0).expand(h, qe - qs, S))
            bias = torch.where((causal & (dist < self.rel_extent)).unsqueeze(0),
                               bias, torch.zeros_like(bias))
            bias4 = bias.view(kvh, grp, qe - qs, S)
            # grp is folded into M so the batch dims match and bmm broadcasts nothing. Passing
            # (kvh,1,d,S) against (kvh,grp,tt,d) made bmm CLONE kxf to (kvh,grp,d,S) on every tile
            # call: h*d*S*4 bytes = 12,288*S, so 768 MiB at S=65,536 and 3.0 GiB at the served
            # 262,144 -- above ~43k context that copy, not the bias, was the largest allocation,
            # and the byte-budget tile could not reach it. Each output element still contracts
            # over d alone, so the arithmetic is unchanged; the fence proves it.
            q4t = q4[:, :, qs:qe].reshape(kvh, grp * (qe - qs), d)
            # Chunk only where the fp32 K temporary is the bigger allocation. The cast costs kvh*d*4 =
            # 4,096 bytes per key column; the scores tensor this tile produces anyway costs h*4 bytes per
            # column. So when tile_rows >= 4096/(h*4) == 128/h (42.67 rows at h=24), the wide single
            # matmul is strictly better -- chunking there would add a block-copy without lowering the peak
            # (measured: cumulative 2.88 -> 2.95 GiB and x3.37 CPU at T=64, S=32,768, where tile=64).
            # At the served shapes the rule selects chunks in both regimes that matter: decode (tile=1) and
            # long-context prefill (tile=10 at S=262,144), because there the K cast dwarfs the tile's scores.
            if S > k_chunk and kvh * d * 4 > h * 4 * (qe - qs):
                sc3 = torch.empty((kvh, grp * (qe - qs), S), dtype=torch.float32, device=q.device)
                for c0 in range(0, S, k_chunk):
                    c1 = min(c0 + k_chunk, S)
                    # Slice the hoisted (kvh,d,S) view, not k itself: slicing k before the transposes was
                    # measured WORSE at decode (T=1, S=131,072: +0.79 GiB of per-block transpose and slice
                    # traffic versus +0.29 GiB here), because each block then re-transposes and torch
                    # materialises those intermediates. The peak cap is identical either way.
                    sc3[:, :, c0:c1] = torch.matmul(q4t, kxf[:, :, c0:c1].float())
                sc = sc3.view(kvh, grp, qe - qs, S) * self.scaling
            else:
                # Column blocks are exact -- a score is dot(q_row, k_column) over d alone, so splitting the
                # key axis computes the same values in the same order as one wide product (asserted, not
                # assumed: bit=True at S=20,480 and S=32,768, both above k_chunk=16,384).
                sc = torch.matmul(q4t, kxf.float()).view(kvh, grp, qe - qs, S) * self.scaling
            sc = sc + bias4
            sc = sc.masked_fill(~causal[None, None, :, :], float("-inf"))
            snk = sink[qs:qe].T.view(kvh, grp, qe - qs).float()           # (kvh,grp,tt)
            comb = torch.cat([sc, snk.unsqueeze(-1)], dim = -1)
            w = F.softmax(comb, dim = -1)[..., :S].to(v.dtype)
            # grp folded into the row axis on this side too. Expanding vx to (kvh,grp,S,d) made
            # matmul CLONE the whole KV value tensor once per tile call: kvh*grp*S*d*2 bytes =
            # 6,291,456*S, so 1.5 GiB at the served 262,144 -- the sibling of the K clone above,
            # and equally out of reach for the tile dial. Contracting (kvh,grp*tt,S) against
            # (kvh,S,d) needs no expand; each row's weighted sum over S is unchanged.
            outs.append(torch.matmul(w.reshape(kvh, grp * (qe - qs), S), vx)
                        .view(kvh, grp, qe - qs, d))                      # (kvh,grp,tt,d)
        attn = torch.cat(outs, dim = 2).reshape(h, T, d)
        return attn

    def _attend_one_keytile(self, q, k, v, rel_small, sink, q_pos, kv_pos, kv_valid = None):
        # Key-tiled online-softmax sibling of _attend_one, behind infer_params.inkling_attn_keytile.
        # Same signature and shapes as _attend_one (each caller swaps only the method name), so the
        # switch is a checked-per-call method choice, exactly as inkling_attn_graph selects arms.
        # Tiles the KEY axis as well as the query axis and folds a running max/sum with a PER-CHUNK
        # fp32 K cast (Strata's ink_attn.cuh formulation):
        #     m' = max(m, max_cols(sc));  a = exp(m - m');
        #     l = l*a + sum(exp(sc - m'));  acc = acc*a + exp(sc - m') @ V
        # dividing once at the end. The sink seeds the init (m = sink, l = 1, acc = 0), which is
        # exact rather than approximate: exp(sink - m) = 1 folds the sink into the denominator as
        # one term and leaves the numerator untouched, so the (S+1) concat disappears. Mathematically
        # the same value as the single pass, NOT bit-identical: rescaling reorders the summation of
        # exp terms, so low mantissa bits differ (the fence asserts <=1e-4 abs, not equality). A row
        # whose keys are all masked keeps m = sink, l = 1 and returns zeros, as _attend_one does.
        # The stack's two peak fixes are KEPT, not regressed: grp folds into the row axis on both
        # sides so matmul broadcasts nothing and clones neither K nor V (the branch's expand of vx
        # to (kvh,grp,S,d) is gone), and kxf is cast per key chunk off the hoisted fp16 view rather
        # than hoisting a full-S fp32 slab (the branch's ~1 GiB @262k hoist is gone). kv_valid folds
        # into the causal refusal exactly as _attend_one does.
        h, T, d = q.shape
        S = k.shape[0]
        grp = self.num_kv_groups
        kvh = self.num_kv_heads
        kxf = k.transpose(0, 1).transpose(1, 2)                            # (kvh,d,S) fp16 view, no cast
        vx = v.transpose(0, 1)                                            # (kvh,S,d), NOT expanded
        q4 = q.view(kvh, grp, T, d).float()                               # heads grouped by kv head
        rows = self._attend_one_tile(S, T, 4)
        cols = self._attend_keytile_cols(S, rows, d, 4)
        outs = []
        for qs in range(0, T, rows):
            qe = min(qs + rows, T)
            tt = qe - qs
            snk = sink[qs:qe].T.view(kvh, grp, tt).float()                # (kvh,grp,tt)
            m = snk.clone()                                               # running max, sink-seeded
            l = torch.ones_like(m)                                        # denominator, sink term = 1
            acc = torch.zeros(kvh, grp, tt, d, device = q.device, dtype = torch.float32)
            q4t = q4[:, :, qs:qe].reshape(kvh, grp * tt, d)
            for ks in range(0, S, cols):
                ke = min(ks + cols, S)
                cc = ke - ks
                dist = q_pos[qs:qe, None] - kv_pos[None, ks:ke]            # (tt, cc)
                causal = dist >= 0
                if kv_valid is not None:
                    # padded ring columns join the causal refusal; all-True when every column is
                    # real (global path, exact-fit sliding), so it cannot change a value.
                    causal = causal & kv_valid[None, ks:ke]
                bias = rel_small[:, qs:qe, :].gather(                     # (h, tt, cc)
                    -1, dist.clamp(0, self.rel_extent - 1).unsqueeze(0).expand(h, tt, cc))
                bias = torch.where((causal & (dist < self.rel_extent)).unsqueeze(0),
                                   bias, torch.zeros_like(bias))
                sc = torch.matmul(q4t, kxf[:, :, ks:ke].float()).view(kvh, grp, tt, cc)
                sc = sc * self.scaling + bias.view(kvh, grp, tt, cc)
                sc = sc.masked_fill(~causal[None, None, :, :], float("-inf"))
                m_new = torch.maximum(m, sc.amax(dim = -1))               # (kvh,grp,tt)
                live = torch.isfinite(m_new)                              # sink-only rows stay finite
                alpha = torch.where(live, torch.exp(m - m_new), torch.ones_like(m))
                p = torch.where(live[..., None], torch.exp(sc - m_new[..., None]),
                                torch.zeros_like(sc))
                l = l * alpha + p.sum(dim = -1)
                # FP32 PV accumulation. The online-softmax numerator is UN-normalized until the
                # final /l, so its partial sum Σ exp(sc-m)·v scales with attention breadth × |v|
                # and overflows fp16 (max 65504) at long context — a broad-attention, RMS~2-value
                # prefill over 63k keys reaches ~127k here. The eager arm escapes it because it
                # normalises weights before its half matmul. Keeping the V product in fp32 (as
                # flash-attention keeps O fp32) bounds the per-chunk partial to fp32 range; the
                # single .to(v.dtype) happens after /l, when the result is back to O(|v|).
                acc = acc * alpha[..., None] + torch.matmul(
                    p.reshape(kvh, grp * tt, cc), vx[:, ks:ke, :].float()
                ).view(kvh, grp, tt, d)                                   # (kvh,grp,tt,d)
                m = m_new
            outs.append((acc / l.clamp(min = 1e-30)[..., None]).to(v.dtype))   # (kvh,grp,tt,d)
        attn = torch.cat(outs, dim = 2).reshape(h, T, d)
        return attn

    def forward_attn(self, h_q, h_k, h_v, params):
        b, t, _ = h_q.shape
        pos_row = self._pos_row(params, b, t)
        # Global layers apply log_scale on both the q path and the rel path over the SAME
        # pos_row; compute the shared position term once instead of inside each projection.
        tau = None if self.sliding else compute_log_tau(pos_row, self.log_alpha, self.log_floor)
        q, kh, vh = self.project_qkv(h_q, h_k, h_v, pos_row, params, tau)
        rel_small = self.rel_logits(h_q, pos_row, params, tau)            # (B,h,T,ext)
        sink = torch.matmul(h_q.float(), self.sink_w.T)                   # (B,T,heads)

        if self.sliding:
            attn = self._attend_sliding(
                {"q": q, "kh": kh, "vh": vh, "rel_small": rel_small, "sink": sink,
                 "pos_row": pos_row}, params, b, t)
        else:
            cache = params["cache"]
            cache_seqlens = get_for_device(params, "cache_seqlens", self.device)
            block_table = get_for_device(params, "block_table", self.device)
            inst = params.get("layer_instance") or 0
            cache.update_layer_direct(self.layer_idx, cache_seqlens, block_table, kh, vh, t, inst)
            kv_cat_k, kv_cat_v, kv_lens = self._gather_global(
                cache, cache_seqlens, block_table, inst, kh, vh, t, params)
            # Checked per call: keytile swaps the attend method, the OFF arm is the served body
            attend = (self._attend_one_keytile
                      if self.config.infer_params.inkling_attn_keytile else self._attend_one)
            outs = []
            for bi in range(b):
                S = kv_lens[bi]
                kv_pos = torch.arange(S, device = q.device)
                o = attend(
                    q[bi], kv_cat_k[bi][:S], kv_cat_v[bi][:S], rel_small[bi], sink[bi], pos_row[bi],
                    kv_pos)
                outs.append(o.transpose(0, 1))                            # (T,h,d)
            attn = torch.stack(outs).reshape(b, t, -1).contiguous()

        gate = torch.sigmoid(self.gate.forward(h_q, params).float())
        attn = (attn.float() * gate).to(torch.half)
        out = self.wo.forward(attn, params)                               # (B,T,H) fp32
        return out

    def _attend_sliding(self, inp, params, b, t):
        """Sliding-window attention for one round: control arm (eager, per-row exact extents —
        the slicing and shapes of the pre-capture path) or capture arm (padded to the bucket,
        one graph per (shared decode bucket, padded extent, pool identity)). Both arms read the
        ring through ring_geometry's DEVICE indices and append with ring_write AFTER the read,
        so the chunk's own K/V stays out of what it attends (read-before-write)."""
        window = self.convs.sliding_window
        pos_host = self.convs._positions(params)[:b]
        cnts = [min(max(int(p), 0), window) for p in pos_host]           # ring rows per sequence
        if self.config.infer_params.inkling_attn_graph and t == 1:
            # pilot scope: single-token decode only. The shared ladder counts ROWS, and at t == 1
            # rows == batch, so padding the batch to the bucket is padding the rows; a verify
            # window (t > 1) would need row-padding instead. Declining it costs nothing: the
            # control arm below runs the same math at the live shapes.
            out = self._sliding_capture(inp, params, b, t, cnts)
            if out is not None:
                self.convs.ring_write(inp["kh"], inp["vh"], params)
                return out[:b]
        inp = dict(inp, slots=get_for_device(params, "recurrent_slots", self.device))
        ring_cols = max(cnts) if cnts else 0
        kr, vr, kv_pos, kv_valid, _ = self.convs.ring_read(
            params, t, inp["slots"], inp["pos_row"][:, 0], ring_cols)
        self.convs.ring_write(inp["kh"], inp["vh"], params)
        return self._sliding_attend(inp, kr, vr, kv_pos, kv_valid, b, t,
                                    [cnts[bi] + t for bi in range(b)])

    def _sliding_capture(self, inp, params, b, t, cnts):
        """Capture arm with HOT-KEY admission: refill this round's inputs into the bucket's
        static buffers, then replay (or capture) the ring-read + attend body. Returns None —
        keep the control arm — when this round may not run a graph; every refusal counts its
        reason, so a zero A/B delta stays explainable from the log instead of mysterious.

        Admission (field result 2026-11-27: a hard cap of 8, filled by cold fleet corners within
        a minute of boot, then refused the hot key forever and the arm engaged on zero requests):
        a key must first be requested inkling_attn_graph_min_hits times since boot, and the
        resident set is an LRU of inkling_attn_graph_max_graphs graphs — admitting past the cap
        evicts the least-recently-used key and drops its slot, freeing that graph's private pool
        (the 4090 has one VA space for all of them; see the ledger's cost note).

        ForwardGraphRunner runs the body twice before it captures and once after, so the buffers
        the graph bakes are the static ones (live q/kh/vh come from the allocator and move every
        step); the body itself is side-effect free — ring_write stays outside it, which is also
        why no on_after_warmup rewind is needed (compare the whole-forward runner's recurrent
        advance)."""
        led = self._attn_ledger
        ip = self.config.infer_params
        led.cap = ip.inkling_attn_graph_max_graphs        # the dials are live: no reload needed
        led.min_hits = ip.inkling_attn_graph_min_hits
        if led.cap <= 0:
            self._record_decline("cap", "inkling_attn_graph_max_graphs is 0")
            return None
        if self._attn_graph_failed is not None:
            self._record_decline("failed", f"capture already failed on this layer: "
                                           f"{self._attn_graph_failed}")
            return None
        st = self.convs._state_slot(params)[0]
        pool_k, pool_v = st.k_state, st.v_state
        extent = attn_extent_bucket(max(cnts) + t, self.convs.sliding_window + t)
        key = attn_capture_key(b, extent, pool_k.data_ptr())
        if key is None:
            self._record_decline("above-ladder",
                                 f"{b} rows above the decode ladder "
                                 f"{moe_coop_plan.DECODE_BUCKETS} (prefill/mixed)")
            return None
        led.note(key)                                     # counted before any admission decision
        if not (pool_k.is_contiguous() and pool_v.is_contiguous()):
            # the gather addresses the pool flat; a non-contiguous pool would make reshape COPY,
            # and a graph that baked that copy's address would read dead storage on replay
            self._record_decline("pool-noncontiguous",
                                 "the flat gather would copy, freezing a dead address in the graph")
            return None
        slot = led.hit(key)
        if slot is None:
            if not led.eligible(key):
                return None                    # frequency filter: this key is still too young
            if self._attn_margin_wait:
                self._attn_margin_wait -= 1    # a full card stays full; skip the driver call
                return None
            need = attn_pool_bytes(key[0], key[1], self.num_kv_heads, self.head_dim,
                                   self.num_q_heads)
            ok, free, reserve = attn_vram_margin_ok(self.device, need)
            if not ok:
                # the card is where serving happens: a capture that cannot fit is a refusal, not
                # an experiment. Dot26 2026-11-28 captured four graphs on a card with ~13 MiB
                # free, reserved ~2.1 GiB of pools, then OOMed mid-capture and spent 640 s.
                self._attn_margin_wait = _ATTN_GRAPH_MARGIN_COOLDOWN
                self._record_decline(
                    "vram-margin",
                    f"free {free / 2 ** 20:.0f} MiB < {reserve / 2 ** 20:.0f} MiB "
                    f"(margin for a {need / 2 ** 20:.0f} MiB pool at rows{key[0]}/cols{key[1]})")
                return None
            slot = self._new_graph_slot(key[0], inp)
            evicted = led.admit(key, slot)
            logger.warning(f"ATTN-GRAPH capture L{self.layer_idx:02d} "
                           f"key=rows{key[0]}/cols{key[1]}/pool{key[2]:x} "
                           f"hits={led.requests[key]} resident={len(led.resident)} cap={led.cap}"
                           + (f" evicted=rows{evicted[0]}/cols{evicted[1]}" if evicted else ""))
        rin = slot["in"]
        rb = key[0]
        for name in ("q", "kh", "vh", "rel_small", "sink", "pos_row"):
            rin[name][:b].copy_(inp[name])
        rin["slots"][:b].copy_(get_for_device(params, "recurrent_slots", self.device))
        rin["slots"][b:].zero_()                                   # padded rows: slot 0, zeros
        ring_cols = extent - t

        def body(params):
            # params is a parameter, not a closure, so the capture can drop params["dev_cache"]
            # around itself: otherwise the upload the graph records is the warmup's cached device
            # copy and every replay re-reads the warmup's positions and slots forever
            return self._sliding_body(rin, params, rb, t, ring_cols)

        out = self._attn_round(slot, key, body, params)
        led.visits_since_log += 1
        if led.visits_since_log >= _ATTN_GRAPH_LOG_EVERY:
            led.visits_since_log = 0
            logger.warning(self._attn_summary_line())
        return out

    def _sliding_body(self, inp, params, rb, t, ring_cols):
        """The captured unit: device-indexed ring read + per-row attend at one fixed shape."""
        kr, vr, kv_pos, kv_valid, _ = self.convs.ring_read(
            params, t, inp["slots"], inp["pos_row"][:, 0], ring_cols)
        return self._sliding_attend(inp, kr, vr, kv_pos, kv_valid, rb, t, None)

    def _sliding_attend(self, inp, kr, vr, kv_pos, kv_valid, rb, t, extents):
        kv_k = torch.cat([kr, inp["kh"]], dim = 1).to(torch.half)
        kv_v = torch.cat([vr, inp["vh"]], dim = 1).to(torch.half)
        # Checked per call, matching the global arm and inkling_attn_graph's seam style
        attend = (self._attend_one_keytile
                  if self.config.infer_params.inkling_attn_keytile else self._attend_one)
        outs = []
        for bi in range(rb):
            if extents is None:
                k, v, pos, valid = kv_k[bi], kv_v[bi], kv_pos[bi], kv_valid[bi]
            else:
                # control arm: the exact per-row column count, so kv_valid[:S] is all True and the
                # mask, the shapes and the bits are the pre-capture path's
                S = extents[bi]
                k, v, pos, valid = kv_k[bi][:S], kv_v[bi][:S], kv_pos[bi][:S], kv_valid[bi][:S]
            outs.append(attend(
                inp["q"][bi], k, v, inp["rel_small"][bi], inp["sink"][bi], inp["pos_row"][bi],
                pos, valid).transpose(0, 1))                             # (T,h,d)
        return torch.stack(outs).reshape(rb, t, -1).contiguous()

    def _attn_round(self, slot, key, body, params):
        """Run one round of a captured key, owning the capture HERE.

        The layer deliberately does not delegate to generator.fwd_graph.ForwardGraphRunner: its
        blanket except prints "Whole-forward CUDA graph disabled for this process", and on
        2026-11-28 that sentence attributed a per-layer OOM to the whole-forward runner. Owning
        the capture keeps the failure local and the log honest — an out-of-memory here degrades
        THIS layer (park it, drop its slot, free the partial pool) and never reaches that
        runner's process-wide disable (fwd_graph.py:131 stays what it always was: the whole
        forward's own capture failure path).

        Sequence matches the estate runner: warm up on a side stream so the allocator and
        autotuning settle, drop params["dev_cache"] so the capture records the pinned H2D copies
        from the staging addresses (not the warmup's cached device copies), capture, then run
        eager once so the static buffers are re-populated before the first replay."""
        if slot["failed"] is not None:
            slot["stats"]["eager"] += 1
            return body(params)                          # parked: eager only, never retry
        if slot["graph"] is None:
            try:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(_ATTN_GRAPH_WARMUPS):
                        body(params)
                torch.cuda.current_stream().wait_stream(stream)
                params.pop("dev_cache", None)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, capture_error_mode = "relaxed"):
                    static_out = body(params)
                params.pop("dev_cache", None)
                slot["graph"] = graph
                slot["out"] = static_out
                slot["stats"]["capture"] += 1
            except Exception as err:                     # including OutOfMemoryError
                reason = f"{'capture out of memory' if is_oom_error(err) else type(err).__name__}" \
                         f": {err}"
                slot["failed"] = reason
                self._attn_graph_failed = reason
                self._attn_ledger.drop(key)              # returns the statics and the pool
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()             # a half-built pool is not left reserved
                self._record_decline("capture-oom" if is_oom_error(err) else "capture-failed", reason)
                logger.warning(
                    f"{'ATTN-GRAPH capture-oom' if is_oom_error(err) else 'ATTN-GRAPH capture-failed'} "
                    f"L{self.layer_idx:02d} key=rows{key[0]}/cols{key[1]} — {reason}")
                slot["stats"]["eager"] += 1
                return body(params)
            slot["stats"]["eager"] += 1
            return body(params)                          # first round after capture: eager
        slot["graph"].replay()
        slot["stats"]["replay"] += 1
        return slot["out"].clone()

    def _new_graph_slot(self, rb, inp):
        """Static input buffers for one bucket (row count rb, everything else this round's
        shapes) plus the graph's own counters. Allocated outside any capture — the first round
        for a key always runs the warmups eager."""
        rin = {name: torch.zeros((rb,) + tuple(src.shape[1:]), dtype = src.dtype,
                                 device = src.device)
               for name, src in inp.items()}
        rin["slots"] = torch.zeros((rb,), dtype = torch.int64, device = self.device)
        return {"in": rin, "graph": None, "out": None, "failed": None,
                "stats": {"capture": 0, "replay": 0, "eager": 0}}

    def _record_decline(self, token, detail = ""):
        """Counts a decline for the telemetry line and logs it once per token — a reason that
        repeats every round is a fact about the traffic, already in the count, not a new event."""
        self._attn_ledger.refuse(token)
        if token not in self._attn_graph_logged:
            self._attn_graph_logged.add(token)
            logger.warning(f"InklingAttention L{self.layer_idx:02d}: sliding-graph arm off — "
                           f"{token}: {detail}")

    def _attn_summary_line(self):
        """The engagement evidence line. Markers, stable across releases:
          ATTN-GRAPH capture …        one per newly admitted key (with what it evicted)
          ATTN-GRAPH capture-oom …    one per layer whose capture ran out of memory on a
                                      nearly-full card; that layer stands down, the process
                                      keeps its other graphs
          ATTN-GRAPH capture-failed … one per layer whose capture raised for another reason
          ATTN-GRAPH summary …        every _ATTN_GRAPH_LOG_EVERY visits on this layer
          sliding-graph arm off — <token>   first occurrence of each decline reason; tokens:
                                      above-ladder, pool-noncontiguous, cap, vram-margin,
                                      capture-oom, capture-failed, failed
        Fields on the summary: resident, cap, min_hits, waiting, captures, replays, eager,
        evictions, declines=<token:count>, requests=<rows/cols:count for the busiest keys>,
        failed=<reason when parked>. Grepping `ATTN-GRAPH` alone answers the only question that
        mattered on 2026-11-27: did the arm ever engage — no py-spy needed."""
        tal = self._attn_ledger.tally()
        declines = "|".join(f"{k}:{v}" for k, v in sorted(tal["declines"].items())) or "-"
        busy = sorted(tal["requests"].items(), key = lambda kv: (-kv[1], str(kv[0])))[:6]
        requests = "|".join(f"rows{k[0]}/cols{k[1]}:{n}" for k, n in busy) or "-"
        line = (f"ATTN-GRAPH summary L{self.layer_idx:02d} resident={tal['resident']} cap={tal['cap']} "
                f"min_hits={tal['min_hits']} waiting={tal['waiting']} "
                f"captures={tal['captures']} replays={tal['replays']} eager={tal['eager']} "
                f"evictions={tal['evictions']} declines={declines} requests={requests}")
        return line if self._attn_graph_failed is None else line + f" failed={self._attn_graph_failed}"

    def attn_graph_stats(self):
        """Capture/replay/eager totals, admission state and per-key runner stats, for the A/B
        that decides whether this ships. Off the hot path: called by the operator, not by a step."""
        led = self._attn_ledger
        tal = led.tally()
        tal["keys"] = {f"rows{key[0]}/cols{key[1]}/pool{key[2]:x}": dict(led._slot_stats(slot))
                       for key, slot in led.resident.items()}
        tal["failed"] = self._attn_graph_failed
        return tal

    def _pos_row(self, params: dict, b: int, t: int) -> torch.Tensor:
        position_ids = get_for_device(params, "position_ids", self.device, None)
        if position_ids is not None:
            return position_ids.to(torch.long)
        positions = get_for_device(params, "positions", self.device, None)
        # The fallback spells a scalar position as a device tensor. stage_host rather than
        # torch.tensor(..., device=...): inside a capture torch refuses a copy whose host side is
        # pageable, which is what generator._fwd_unsafe's "pinned-memory error" boot note was.
        base = positions.long() if positions is not None else stage_host(
            [params.get("position", 0)], self.device, ("inkling2", "position"),
            torch.long).expand(b)
        return base[:, None] + torch.arange(t, device = self.device)[None, :]

    def _gather_global(self, cache, cache_seqlens, block_table, inst, kh, vh, t, params):
        # Host-side lengths only feed Python row counts (lens/maxlen), so they must not cost a
        # stream sync: take them from the staging buffer they were uploaded from (_host_lens).
        host_seqlens = _host_lens(params, cache_seqlens)
        layer = cache.layers[self.layer_idx, inst]
        b = kh.shape[0]
        lens = [int(host_seqlens[i]) + t for i in range(b)]
        maxlen = max(lens)

        _dbg = _os.environ.get("INKLING_DEBUG")
        def _st(x):
            return float(x.float().std()) if (x is not None and x.numel()) else float("nan")
        if _dbg:
            print(f"GATHER L{self.layer_idx:02d} i{inst} cls={type(layer).__name__} "
                  f"kh={_st(kh):.4f}/{kh.dtype} vh={_st(vh):.4f} lens={lens}", flush = True)

        if hasattr(layer, "stage_range"):
            # NVFP4: dequantize only the used pages into a contiguous fp16 staging pool whose
            # per-sequence page runs are contiguous (stage_range's remapped block_table)
            ks, vs, _ = layer.stage_range(cache_seqlens, block_table, t)
            nbs = [max(1, -(-l // PAGE_SIZE)) for l in lens]
            starts = [sum(nbs[:i]) for i in range(b)]
            kk = torch.zeros((b, maxlen) + ks.shape[2:], dtype = ks.dtype, device = ks.device)
            vv = torch.zeros_like(kk)
            for i, (s, n, l) in enumerate(zip(starts, nbs, lens)):
                kk[i, :l] = ks[s:s + n].reshape(-1, ks.shape[2], ks.shape[3])[:l]
                vv[i, :l] = vs[s:s + n].reshape(-1, vs.shape[2], vs.shape[3])[:l]
            if _dbg:
                print(f"  NVFP4 ks={_st(ks):.4f} vs={_st(vs):.4f} kk={_st(kk):.4f} vv={_st(vv):.4f}",
                      flush = True)
            return kk, vv, lens

        k_cache, v_cache = layer.get_kv(cache_seqlens, block_table, -1)
        kk = torch.zeros((b, maxlen) + k_cache.shape[2:], dtype = k_cache.dtype, device = k_cache.device)
        vv = torch.zeros_like(kk)
        for i in range(b):
            nb = max(1, -(-lens[i] // PAGE_SIZE))
            phys = block_table[i, :nb].long()
            kk[i, :lens[i]] = k_cache[phys].reshape(-1, k_cache.shape[2], k_cache.shape[3])[:lens[i]]
            vv[i, :lens[i]] = v_cache[phys].reshape(-1, v_cache.shape[2], v_cache.shape[3])[:lens[i]]
        return kk, vv, lens


def repeat_kv_heads(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, s, kvh, d = x.shape
    return x[:, :, None, :, :].expand(b, s, kvh, n_rep, d).reshape(b, s, kvh * n_rep, d)


# ---------------------------------------------------------------------------
# Gate / block-read / MoE / layer orchestrator
# ---------------------------------------------------------------------------

class InklingMessageGate(Module):
    """
    Inkling2LoopMessageGate: (1 - tanh(down·z proj)) * h + tanh(...) * msg. Inputs rms-normed
    (unweighted, eps=rms_norm_eps), fp32; down projection runs in its stored (bf16) dtype,
    forget/gain projections in fp32.
    """

    def __init__(self, config: Config, key: str, hidden_size: int, eps: float):
        super().__init__(config, key, None)
        self.module_name = "InklingMessageGate"
        self.hidden_size = hidden_size
        self.eps = eps
        self.down_w = self.forget_w = self.gain_w = None

    @override
    def optimizer_targets(self):
        return []

    @override
    def load(self, device, **kwargs):
        super().load(device, **kwargs)
        stc = self.config.stc
        self.down_w = stc.get_tensor(self.key + ".down_weight", device, optional = False, allow_bf16 = True, no_defer = True).contiguous()
        self.forget_w = stc.get_tensor(self.key + ".forget_weight", device, optional = False, allow_bf16 = True, no_defer = True).float().contiguous()
        self.gain_w = stc.get_tensor(self.key + ".gain_weight", device, optional = False, allow_bf16 = True, no_defer = True).float().contiguous()

    @override
    def unload(self):
        self.down_w = self.forget_w = self.gain_w = None
        super().unload()

    @override
    def get_tensors(self):
        t = super().get_tensors()
        for suf, x in (("down_weight", self.down_w), ("forget_weight", self.forget_w), ("gain_weight", self.gain_w)):
            if x is not None:
                t[self.key + "." + suf] = x
        return t

    def forward(self, h: torch.Tensor, msg: torch.Tensor, params = None, out_dtype = None) -> torch.Tensor:
        input_dtype = h.dtype
        h = h.float()
        msg = msg.float()
        cat = torch.cat([rmsnorm_f(h, self.eps), rmsnorm_f(msg, self.eps)], dim = -1)
        z = torch.matmul(cat.to(self.down_w.dtype), self.down_w.T).float()
        forget = torch.tanh(torch.matmul(z, self.forget_w.T))
        gain = torch.tanh(torch.matmul(z, self.gain_w.T))
        return ((1.0 - forget) * h + gain * msg).to(input_dtype)


class InklingBlockRead(Module):
    """Tiny cross-attention over [bank; x] with learned queries (Inkling2LoopBlockAttentionResidual).
    The query tensor is a top-level pack tensor addressed by absolute key."""

    def __init__(self, config: Config, key: str, hidden_size: int, num_queries: int, eps: float):
        super().__init__(config, key, None)
        self.module_name = "InklingBlockRead"
        self.hidden_size = hidden_size
        self.num_queries = num_queries
        self.eps = eps
        self.query = None

    @override
    def optimizer_targets(self):
        return []

    @override
    def load(self, device, **kwargs):
        super().load(device, **kwargs)
        q = self.config.stc.get_tensor(self.key, device, optional = False, allow_bf16 = True, no_defer = True).float()
        q = q.reshape(self.num_queries, self.hidden_size)
        assert q.shape == (self.num_queries, self.hidden_size), \
            f"inkling block read {self.key}: shape {tuple(q.shape)}, expected {(self.num_queries, self.hidden_size)}"
        self.query = q.contiguous()

    @override
    def unload(self):
        self.query = None
        super().unload()

    @override
    def get_tensors(self):
        t = super().get_tensors()
        if self.query is not None:
            t[self.key] = self.query
        return t

    def read(self, bank: list, x: torch.Tensor) -> list:
        return block_read(self.query, bank, x, self.eps)

    @override
    def forward(self, x, params, out_dtype = None):
        raise RuntimeError("InklingBlockRead is driven by InklingLayer")


class InklingOutputRead(InklingBlockRead):
    """Final block read over the whole bank plus the last hidden (attn_res.output_query),
    run as a model-level module in the epilogue chain."""

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype = None) -> torch.Tensor:
        flow = flow_of(params)
        return self.read(flow["bank"], x)[0].to(x.dtype)


# Routed-expert weight-residence ledger (InklingMoE; the config switch is
# infer_params.inkling_moe_expert_residency_ledger, default OFF). These three dials are module
# constants, not env flags: the switch is the interface, the cadence is a property of the log.
_MOE_EXPERTX_RECENSUS = 64        # MoE forwards between full residence censuses
_MOE_EXPERTX_LOG_EVERY = 2048     # MoE forwards between summary lines per layer
_MOE_EXPERTX_TENSORS = ("trellis", "suh", "svh")   # what a routed expert's GEMMs read


def _fresh_expertx_ledger() -> dict:
    """Empty routed-expert residence ledger. One per load epoch: InklingMoE.unload resets it,
    because a reload re-allocates every expert's tensors and the previous addresses are gone."""
    return {"forwards": 0, "censuses": 0, "moves": 0, "fingerprint": (),
            "census": {"experts": 0, "device": 0, "host": 0, "missing": 0, "first_offdevice": -1}}


class InklingMoE(Module):
    """
    Inkling2LoopSparseMoeBlock: sigmoid router with additive selection correction
    (scores + bias top-k, renormalised weights times route_scale), per-expert SoftcappedSwiGLU
    (gate cap 4: tanh(g/4)*4 * sigmoid(g); up cap 25: tanh(u/25)*25). No shared expert
    (n_shared_experts = 0). Accumulation in fp32.
    """

    def __init__(
        self,
        config: Config,
        key: str,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        route_scale: float,
        gate_softcap: float,
        up_softcap: float,
        layer_idx: int,
    ):
        super().__init__(config, key, None)
        self.module_name = "InklingMoE"
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = num_experts_per_tok
        self.route_scale = route_scale
        self.gate_softcap = gate_softcap
        self.up_softcap = up_softcap

        self.router_w = None
        self.router_bias = None
        self.experts_g = []
        self.experts_u = []
        self.experts_d = []
        for e in range(num_experts):
            ek = f"{key}.experts.{e}"
            g = Linear(config, ek + ".gate_proj", hidden_size, intermediate_size,
                       qmap = "block.mlp", out_dtype = torch.half, qgroup = ek + ".g")
            u = Linear(config, ek + ".up_proj", hidden_size, intermediate_size,
                       qmap = "block.mlp", out_dtype = torch.half, qgroup = ek + ".u")
            d = Linear(config, ek + ".down_proj", intermediate_size, hidden_size,
                       out_dtype = torch.float, qgroup = ek + ".d")
            self.experts_g.append(g)
            self.experts_u.append(u)
            self.experts_d.append(d)
            for lin in (g, u, d):
                self.register_submodule(lin)
        # Per-decode-BUCKET fixed-address arenas for the exl3_moe_coop decode tier
        # (moe_group/README.md); empty until the first eligible decode step. Keyed by
        # moe_coop_plan.padded_bucket_rows, never by live rows: one captured graph replays
        # for every live count at or below its bucket (graph lane §U).
        self._coop_by_bucket: dict = {}
        self._coop_ptrs_cache: dict | None = None
        self._coop_refuse_reason = None
        # Residence ledger for the routed-expert weights; never touched while
        # infer_params.inkling_moe_expert_residency_ledger is False (the served default)
        self._expertx = _fresh_expertx_ledger()

    @override
    def optimizer_targets(self):
        return [lin.optimizer_targets() for lin in (*self.experts_g, *self.experts_u, *self.experts_d)]

    @override
    def load(self, device, **kwargs):
        super().load(device, **kwargs)
        stc = self.config.stc
        self.router_w = stc.get_tensor(self.key + ".gate.weight", device, optional = False, allow_bf16 = True, no_defer = True).float().contiguous()
        self.router_bias = stc.get_tensor(self.key + ".gate.bias", device, optional = False, no_defer = True).float().contiguous()

    @override
    def unload(self):
        self.router_w = self.router_bias = None
        self._coop_by_bucket = {}  # pointer tables pin device addresses; they die with the weights
        self._coop_ptrs_cache = None
        self._coop_refuse_reason = None
        self._expertx = _fresh_expertx_ledger()   # new load epoch: old addresses are dead
        super().unload()

    @override
    def get_tensors(self):
        t = super().get_tensors()
        for k, x in ((self.key + ".gate.weight", self.router_w), (self.key + ".gate.bias", self.router_bias)):
            if x is not None:
                t[k] = x
        return t

    def _coop_ptr_lists(self):
        # pointer tables reference the INNER quant tensors (Linear wrappers expose no
        # .trellis/.suh/.svh); see moe_coop_plan.unwrap_linear for why
        return {
            tag + "_" + what: [getattr(moe_coop_plan.unwrap_linear(m), what).data_ptr() for m in mods]
            for tag, mods in (("g", self.experts_g), ("u", self.experts_u), ("d", self.experts_d))
            for what in ("trellis", "suh", "svh")
        }

    def _coop_ptrs_probe(self):
        """Cheapest LIVE identity signal for the pointer tables: the first expert's three
        projections. An offload/promote/reload re-allocates the layer's expert tensors, so
        these device addresses move with them. Needed because a memoised table is its own
        evidence: reading the fingerprint from the cache can never notice that the weights
        moved without unload (arm proven 2026-11-26: served 1000 while live was 9000)."""
        if not self.experts_g:
            return None
        un = moe_coop_plan.unwrap_linear
        # .trellis hop: unwrap lands on the LinearEXL3 MIXIN, which holds tensors but is
        # not one — calling .data_ptr() on it raised AttributeError inside every coop-armed
        # request on the 98cf64d4 wheel (repro RED 2026-11-26). trellis is reallocated on
        # any promote/offload/reload, so it moves exactly when the tables must.
        return (un(self.experts_g[0]).trellis.data_ptr(), un(self.experts_u[0]).trellis.data_ptr(),
                un(self.experts_d[0]).trellis.data_ptr())

    def _coop_state(self, device, bucket: int):
        """Fixed-address arena for one decode BUCKET: nine int64 pointer tables (the
        per-expert LinearEXL3 tensors are referenced BY POINTER at load — never a stacked
        copy), a pinned int64 staging buffer for fingerprint refills, the coop scratch and
        the fp32 output buffer, all sized by the bucket. Built once per (bucket, load
        epoch); reloads invalidate it via unload(). Returns None and records the reason
        when the pack itself refuses (homogeneity legs live in moe_coop_plan)."""
        desc, reason = moe_coop_plan.classify_experts(
            zip(self.experts_g, self.experts_u, self.experts_d))
        if desc is None:
            self._coop_refuse_reason = reason
            return None
        g0, d0 = self.experts_g[0], self.experts_d[0]
        Hi, I = int(g0.in_features), int(g0.out_features)
        Ho = int(d0.out_features)
        h = self.hidden_size
        if Hi % 128 or I % 128 or Ho % 128 or Ho < h or int(d0.in_features) != I:
            self._coop_refuse_reason = \
                f"geometry not a multiple of 128 (Hi {Hi}, I {I}, Ho {Ho}, h {h})"
            return None
        # (expert-bias homogeneity and bit/codebook uniformity are decided by
        # moe_coop_plan.classify_experts above, where the CPU fence guards them)
        ptr_lists = self._coop_ptr_lists()
        names = sorted(ptr_lists)
        stage = torch.empty(sum(len(ptr_lists[k]) for k in names),
                            dtype = torch.int64, pin_memory = True)
        stage_np = stage.numpy()
        offsets = {}
        pos = 0
        for k in names:
            stage_np[pos:pos + len(ptr_lists[k])] = ptr_lists[k]
            offsets[k] = (pos, pos + len(ptr_lists[k]))
            pos += len(ptr_lists[k])
        tables = {k: stage[offsets[k][0]:offsets[k][1]].to(device).contiguous() for k in names}
        slots = bucket * self.top_k
        args = dict(
            min_expert = -1, max_expert = -1, Hi = Hi,
            g_bias = None, u_bias = None, d_bias = None,
            Kg = desc["Kg"], Ku = desc["Ku"], Kd = desc["Kd"],
            mcg = desc["mcg"], mul1 = desc["mul1"],
            act = moe_coop_plan.MOE_COOP_ACT_SOFTCAP,
            act_limit = self.gate_softcap, up_cap = self.up_softcap, gated = True,
            sh_out = None, sh_gate_w = None,
            had_g = torch.zeros((slots, Hi), dtype = torch.half, device = device),
            had_u = torch.zeros((slots, Hi), dtype = torch.half, device = device),
            gu_g = torch.zeros((slots, I), dtype = torch.half, device = device),
            gu_u = torch.zeros((slots, I), dtype = torch.half, device = device),
            act_out = torch.zeros((slots, I), dtype = torch.half, device = device),
            d_out = torch.zeros((slots, Ho), dtype = torch.float, device = device),
            ctr = torch.zeros(moe_coop_plan.ctr_len(slots, bucket, I, Ho), dtype = torch.int, device = device),
            out = torch.zeros((bucket, h), dtype = torch.float, device = device),
            **tables,
        )
        st = {"args": args, "ptr_lists": ptr_lists, "stage": stage, "stage_np": stage_np,
              "offsets": offsets, "names": names,
              "fingerprint": tuple(p for k in names for p in ptr_lists[k]),
              "probe": self._coop_ptrs_probe()}
        self._coop_by_bucket[bucket] = st
        return st

    def _coop_decode(self, flat, sel, w):
        """Grouped decode through exl3_moe_coop: one launch pair for the whole MoE layer,
        routing kept ON DEVICE (the host_values/stage_host roundtrip and the per-expert
        python loop are skipped entirely). Returns None to keep the python dispatch —
        the declared switch off (config.infer_params.moe_coop_decode, the fence-3 A/B
        seam), prefill, unsupported pack geometry, expert weights that vanished or an
        extension without the entry point; a pack-level refusal records its reason in
        self._coop_refuse_reason. Decode-only contract: rows <= 32, slots <= 256
        (exllamav3/exllamav3_ext/moe_group/README.md).

        ARENA RULE (§U): the arena is keyed by padded_bucket_rows, never by live rows.
        At a bucket the capture layer hands padded FIXED-ADDRESS input buffers, so the
        baked bsz equals the bucket and rows above the live count are masked downstream;
        below it (eager) the live-row shapes run against the oversized arena — bsz comes
        from x.size(0) in exl3_moe_coop_run and every scratch index is a superset."""
        self._coop_refuse_reason = None
        if not self.config.infer_params.moe_coop_decode:
            return None
        rows = flat.size(0)
        reason = moe_coop_plan.refusal(rows, self.top_k, flat.is_cuda,
                                       flat.dtype == torch.half,
                                       hasattr(ext, "exl3_moe_coop"))
        if reason is not None:
            self._coop_refuse_reason = reason
            return None
        bucket = moe_coop_plan.padded_bucket_rows(rows)
        st = self._coop_by_bucket.get(bucket) or self._coop_state(flat.device, bucket)
        if st is None:
            return None
        args = st["args"]
        # Pointer tables are pure functions of the LOADED weights: rebuilt only when the
        # layer lifecycle says they moved (unload/load), never per decode step. Before this
        # cache, _coop_ptr_lists + unwrap_linear burned ~36% of decode samples (dot26 census
        # 2026-11-26 17:2x: 100+30 of ~360).
        if self._coop_ptrs_cache is None or self._coop_ptrs_probe() != st.get("probe"):
            self._coop_ptrs_cache = self._coop_ptr_lists()
        cur_lists = self._coop_ptrs_cache
        fingerprint = tuple(p for k in st["names"] for p in cur_lists[k])
        if fingerprint != st["fingerprint"]:
            # Weights moved (offload/reload): refill the tables IN PLACE so the addresses
            # a captured graph round baked in stay meaningful on replay (doorbell rule).
            # The refill reads from the pinned staging buffer: no allocation and no
            # pageable H2D inside a capture step (torch refuses the latter; it is how
            # every capture of this model died, dot26 journal 2026-11-24 03:34:47).
            st["ptr_lists"] = cur_lists = self._coop_ptr_lists()  # cache was stale: weights moved
            self._coop_ptrs_cache = cur_lists
            stage_np = st["stage_np"]
            for k in st["names"]:
                lo, hi = st["offsets"][k]
                stage_np[lo:hi] = cur_lists[k]
                args[k].copy_(st["stage"][lo:hi], non_blocking = True)
            st["fingerprint"] = fingerprint
            st["probe"] = self._coop_ptrs_probe()
        call = dict(args)
        call["x"] = flat
        call["sel"] = sel.reshape(rows, self.top_k)
        call["rw"] = w.reshape(rows, self.top_k).half()
        ext.exl3_moe_coop(*moe_coop_plan.build_call_args(call))
        out = args["out"]
        return out[:rows]

    def _expertx_census(self) -> tuple[dict, tuple]:
        """Residence census of THIS layer's routed-expert weights: for every expert, the device
        type and address of the three tensors its projections read (trellis/suh/svh — the same
        quant tensors the coop pointer tables reference, reached the same way through
        moe_coop_plan.unwrap_linear).

        The census answers one question with a number instead of a reading of the loader: is
        there any routed-expert weight movement left for a prefetch to hide? An expert whose
        tensors sit outside the compute device is a page-in/HtoD/cache-fetch candidate; when every
        expert of the layer is already on the compute device, and its addresses do not move
        between censuses, the amount of transfer a side stream could overlap is zero. A missing
        tensor counts as off-device: while a module is unloaded nothing of it is resident."""
        un = moe_coop_plan.unwrap_linear
        census = {"experts": 0, "device": 0, "host": 0, "missing": 0, "first_offdevice": -1}
        fingerprint = []
        for g, u, d in zip(self.experts_g, self.experts_u, self.experts_d):
            off = 0
            for lin in (g, u, d):
                inner = un(lin)
                for what in _MOE_EXPERTX_TENSORS:
                    t = getattr(inner, what, None)
                    if t is None:
                        census["missing"] += 1
                        off += 1
                        fingerprint.append(0)
                        continue
                    fingerprint.append(int(t.data_ptr()))
                    if torch.device(t.device).type != "cuda":
                        off += 1
            census["experts"] += 1
            if off:
                census["host"] += 1
                if census["first_offdevice"] < 0:
                    census["first_offdevice"] = census["experts"] - 1
            else:
                census["device"] += 1
        return census, tuple(fingerprint)

    def _expertx_verdict(self) -> str:
        """One word for what the last census saw, so a boot log is read with grep instead of
        interpreted. all_device_resident_at_load is the NEGATIVE result a routed-expert prefetch
        needs: nothing to hide, so no side stream is worth its risk on this pack."""
        led = self._expertx
        cen = led["census"]
        if led["moves"]:
            return "weights_moved_without_unload"
        if cen["missing"]:
            return f"missing_tensors:{cen['missing']}"
        if cen["host"]:
            return f"offdevice_experts:{cen['host']}:first_e{cen['first_offdevice']}"
        return "all_device_resident_at_load"

    def _expertx_summary_line(self) -> str:
        led = self._expertx
        cen = led["census"]
        return (f"INKLING-MOE-EXPERTX L{self.layer_idx:02d} "
                f"device_resident={cen['device']}/{cen['experts']} offdevice={cen['host']} "
                f"missing={cen['missing']} first_offdevice={cen['first_offdevice']} "
                f"forwards={led['forwards']} censuses={led['censuses']} moves={led['moves']} "
                f"recensus_every={_MOE_EXPERTX_RECENSUS} log_every={_MOE_EXPERTX_LOG_EVERY} "
                f"verdict={self._expertx_verdict()}")

    def _expertx_note(self):
        """One forward's bookkeeping for the residence ledger. Sync-free and allocation-free by
        requirement: the full census walks the layer's experts on the first forward (so a boot
        reports residence immediately) and then every _MOE_EXPERTX_RECENSUS forwards, and a
        summary line leaves the log every _MOE_EXPERTX_LOG_EVERY forwards. Reading data_ptr/device
        never touches the stream, so arming the ledger cannot reorder work, cannot add a host
        sync to the decode step, and leaves the coop arena's row-padding contract untouched
        (tests/cpu/inkling_moe_expertx_fence.py arms all three)."""
        led = self._expertx
        led["forwards"] += 1
        if led["censuses"] == 0 or led["forwards"] % _MOE_EXPERTX_RECENSUS == 0:
            cen, fingerprint = self._expertx_census()
            led["censuses"] += 1
            if led["fingerprint"] and fingerprint != led["fingerprint"]:
                # Addresses moved with no unload() between: the weights really did move, and a
                # prefetch lane would have work to hide (the pointer tables also refill here).
                led["moves"] += 1
            led["census"] = cen
            led["fingerprint"] = fingerprint
        if led["forwards"] == 1 or led["forwards"] % _MOE_EXPERTX_LOG_EVERY == 0:
            logger.warning(self._expertx_summary_line())

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype = None) -> torch.Tensor:
        if self.config.infer_params.inkling_moe_expert_residency_ledger:
            self._expertx_note()
        b, t, h = x.shape
        flat = x.reshape(-1, h)
        hs = flat.float()
        scores = torch.sigmoid(torch.matmul(hs, self.router_w.T))
        sel = torch.topk(scores + self.router_bias, self.top_k, dim = -1, sorted = False).indices
        w = scores.gather(1, sel)
        w = w / (w.sum(dim = -1, keepdim = True) + 1e-20)
        w = w * self.route_scale

        coop = self._coop_decode(flat, sel, w)
        if coop is not None:
            return coop.view(b, t, h)
        # One-shot surface of the grouped-decline reason: an inert coop tier is otherwise
        # invisible (witness 2026-11-27: eager python dispatch at 6.2 tok/s/lane with no
        # log naming why the grouped tier declined this pack)
        if self._coop_refuse_reason is not None:
            # The ladder refusal at load/prefill shapes is BY DESIGN (the tier is
            # decode-only); it must not consume the one-shot that names a real
            # DECODE-shape decline (boot 2026-11-27: the 2048-row dummy forward ate the
            # single print and hid the decode-side verdict).
            expected_prefill = "above the decode ladder" in self._coop_refuse_reason
            flag = "_coop_prefill_logged" if expected_prefill else "_coop_decline_logged"
            if not getattr(type(self), flag, False):
                setattr(type(self), flag, True)
                print(
                    f" !! moe coop decode DECLINED: {self._coop_refuse_reason} "
                    f"(layer {self.layer_idx})"
                    + ("" if expected_prefill else "; decode runs python-dispatch")
                )

        sel_flat = sel.reshape(-1)
        w_flat = w.reshape(-1)
        out = torch.zeros((b * t, h), dtype = torch.float, device = x.device)
        # Expert dispatch happens in Python, so the selection has to reach the host and the chosen
        # row indices have to go back. Neither may be a pageable copy: while the whole-forward CUDA
        # graph records, torch refuses a CPU<->CUDA copy whose host side is not pinned, which is how
        # every capture of this model died (dot26 journal 2026-11-24 03:34:47). host_values /
        # stage_host take both directions through a pinned slot - synchronous and byte-identical
        # when eager, recorded and re-readable when capturing. Routing stays the capture round's
        # (census deferred item A: device routing tables + a grouped expert tier own that next).
        #
        # Two arms, split by shape and by capture state, not by a dial:
        #   t > 1 and not capturing → GROUPING ON THE DEVICE. The prefill census (§CC, 2026-11-27)
        #     put the ingest's critical path on this call site: one blocking read per executed MoE
        #     layer, then Python sorting chunk_size × top_k entries before the next launch.
        #     argsort(stable=True) is the SAME total order the host sort below produces (Python's
        #     sorted is stable and keyed on the expert id), bincount+cumsum are the group
        #     boundaries, and the host reads only the ≤2 × num_routed_experts boundary pair
        #     instead of top_k × rows values plus a Python sort plus an order upload. Measured on
        #     CPU at rows 2048 (n = 16,384): 2.696 ms of host work per MoE layer (tolist 0.044 +
        #     sort 1.493 + upload 0.654 + group scan 0.505) → 0.448 ms including the read-back;
        #     over the census's ~50 executed layers × ~46 chunks that is 6.2 s → 1.0 s of host
        #     time per 93k-token ingest. Its GPU-side price is NOT measured here and is not
        #     assumed: the sync frame's share absorbs the compute it waits for (the §CC correction),
        #     and only the queued cuda-event census prices that. Group membership, group order and
        #     the per-token index_add_ sequence are unchanged, so the result is bit-identical —
        #     tests/cpu/inkling_moe_devdispatch_fence.py executes both arms over random routings.
        #   t == 1 or capturing → THE HOST ROUND, verbatim. A decoded step carries top_k × batch
        #     entries, where device grouping would buy three extra launches to save microseconds
        #     (measured at rows 8: 0.010 ms), and a capture must keep baking the routing it
        #     recorded: boundaries frozen from the capture round would otherwise meet a fresh
        #     argsort on replay and hand tokens to the wrong expert.
        if t > 1 and not _capturing():
            order_all = torch.argsort(sel_flat, stable = True)
            counts = torch.bincount(sel_flat, minlength = self.num_routed_experts)
            chosen = torch.nonzero(counts).flatten()
            ends = torch.cumsum(counts[chosen], 0)
            bounds = host_values(torch.stack([chosen, ends], 0), ("inkling2", "moe_bounds"))
            groups = list(zip(bounds[0].tolist(), bounds[1].tolist()))
        else:
            sel_cpu = host_values(sel_flat, ("inkling2", "moe_sel")).tolist()
            order = sorted(range(len(sel_cpu)), key = lambda i: sel_cpu[i])
            # Diet (2026-11-26): one upload of the whole sorted order, groups become device
            # views. Decode never reaches this path (_coop_decode returns above), so no
            # recorded graph depends on per-group source addresses; the eager group order is
            # identical by construction, so index_add_ accumulates bitwise the same.
            order_all = stage_host(order, x.device, ("inkling2", "moe_order_all"), torch.long)
            groups = []
            n = 0
            while n < len(order):
                e = sel_cpu[order[n]]
                m = n
                while m < len(order) and sel_cpu[order[m]] == e:
                    m += 1
                groups.append((e, m))
                n = m
        tokens_all = torch.div(order_all, self.top_k, rounding_mode = "floor")
        start = 0
        for e, end in groups:
            entries = order_all[start:end]
            tokens = tokens_all[start:end]
            start = end
            xs = flat.index_select(0, tokens)
            g = self.experts_g[e].forward(xs, params).float()
            u = self.experts_u[e].forward(xs, params).float()
            act = torch.tanh(g / self.gate_softcap) * self.gate_softcap
            act = act * torch.sigmoid(g)
            act = act * (torch.tanh(u / self.up_softcap) * self.up_softcap)
            y = self.experts_d[e].forward(act.to(torch.half), params)
            out.index_add_(0, tokens, y.float() * w_flat.index_select(0, entries).unsqueeze(-1))
        return out.view(b, t, h)


class InklingLayer(Module):
    """
    One Inkling decoder layer (any of the four variants: sliding/full x non-loop/loop),
    driven for one of its (layer_map) pass instances. Reads/writes the per-forward flow
    dict (params[KEY_FLOW]) for the block-state bank, the pre-loop snapshot and cross-pass
    messages; the pass boundary reset runs when a core instance >0 is entered.
    """

    def __init__(
        self,
        config: Config,
        key_prefix: str,
        layer_idx: int,
        *,
        hidden_size: int,
        head_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        sliding: bool,
        sliding_window: int,
        is_loop: bool,
        loop_j: int,
        loop_start: int,
        loop_end: int,
        num_pre_loop_blocks: int,
        gate_rank: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        route_scale: float,
        gate_softcap: float,
        up_softcap: float,
        rms_norm_eps: float,
        rel_extent: int,
        d_rel: int,
        qk_threshold: float,
        rel_threshold: float,
        log_alpha: float,
        log_floor: int,
        conv_kernel_size: int,
        n_layers_per_block: int,
    ):
        p = f"{key_prefix}layers.{layer_idx}."
        super().__init__(config, p.rstrip("."), None)
        self.module_name = "InklingLayer"
        self.layer_idx = layer_idx
        self.sliding = sliding
        self.is_loop = is_loop
        self.loop_j = loop_j
        self.loop_start = loop_start
        self.loop_end = loop_end
        self.num_pre_loop_blocks = num_pre_loop_blocks
        self.is_boundary = layer_idx % n_layers_per_block == 0
        self.hidden_size = hidden_size
        self.rms_norm_eps = rms_norm_eps

        self.attn_read = None
        if layer_idx > 0:
            self.attn_read = InklingBlockRead(
                config, f"{key_prefix}attn_res.attn_queries.{layer_idx - 1}", hidden_size,
                1 if sliding else 3, rms_norm_eps,
            )
            self.register_submodule(self.attn_read)
        self.mlp_read = InklingBlockRead(
            config, f"{key_prefix}attn_res.mlp_queries.{layer_idx}", hidden_size, 1, rms_norm_eps,
        )
        self.register_submodule(self.mlp_read)

        self.input_norm = None
        self.q_norm = self.k_norm = self.v_norm = None
        if sliding:
            self.input_norm = RMSNorm(config, p + "attn_norm", rms_norm_eps = rms_norm_eps, out_dtype = torch.half)
            self.register_submodule(self.input_norm)
        else:
            self.q_norm = RMSNorm(config, p + "q_input_norm", rms_norm_eps = rms_norm_eps, out_dtype = torch.half)
            self.k_norm = RMSNorm(config, p + "k_input_norm", rms_norm_eps = rms_norm_eps, out_dtype = torch.half)
            self.v_norm = RMSNorm(config, p + "v_input_norm", rms_norm_eps = rms_norm_eps, out_dtype = torch.half)
            for n in (self.q_norm, self.k_norm, self.v_norm):
                self.register_submodule(n)
        self.mlp_norm = RMSNorm(config, p + "mlp_norm", rms_norm_eps = rms_norm_eps, out_dtype = torch.half)
        self.register_submodule(self.mlp_norm)

        self.convs = InklingConvs(
            config = config,
            key = p.rstrip("."),
            layer_idx = layer_idx,
            hidden_size = hidden_size,
            kv_dim = num_kv_heads * head_dim,
            num_kv_heads = num_kv_heads,
            head_dim = head_dim,
            sliding = sliding,
            sliding_window = sliding_window,
            conv_kernel_size = conv_kernel_size,
            dynamic = not sliding,
            key_k_conv = p + "attn.k_sconv",
            key_v_conv = p + "attn.v_sconv",
            key_attn_conv = p + "attn_sconv",
            key_mlp_conv = p + "mlp_sconv",
        )
        self.attn = InklingAttention(
            config = config,
            key = p + "attn",
            layer_idx = layer_idx,
            hidden_size = hidden_size,
            head_dim = head_dim,
            num_q_heads = num_q_heads,
            num_kv_heads = num_kv_heads,
            sliding = sliding,
            rel_extent = rel_extent,
            d_rel = d_rel,
            qk_threshold = qk_threshold,
            rel_threshold = rel_threshold,
            rms_norm_eps = rms_norm_eps,
            log_alpha = log_alpha,
            log_floor = log_floor,
            convs = self.convs,
        )
        self.register_submodule(self.attn)

        self.gate_q = self.gate_k = self.gate_v = self.gate_mlp = None
        self.msg_query = None
        if is_loop:
            self.gate_q = InklingMessageGate(config, p + "controllers.attn", hidden_size, rms_norm_eps)
            self.gate_k = InklingMessageGate(config, p + "controllers.k", hidden_size, rms_norm_eps)
            self.gate_v = InklingMessageGate(config, p + "controllers.v", hidden_size, rms_norm_eps)
            self.gate_mlp = InklingMessageGate(config, p + "controllers.mlp", hidden_size, rms_norm_eps)
            for g in (self.gate_q, self.gate_k, self.gate_v, self.gate_mlp):
                self.register_submodule(g)
            self.key_cp_q = f"{key_prefix}middle_cycle_cross_pass_queries"
            self.key_cp_kv = f"{key_prefix}middle_cycle_cross_pass_kv_queries"

        self.moe = InklingMoE(
            config, p + "mlp", hidden_size, intermediate_size, num_experts,
            num_experts_per_tok, route_scale, gate_softcap, up_softcap,
            layer_idx = layer_idx,
        )
        self.register_submodule(self.moe)

    @override
    def optimizer_targets(self):
        return [*self.attn.optimizer_targets(), *self.moe.optimizer_targets()]

    @override
    def load(self, device, **kwargs):
        super().load(device, **kwargs)
        if self.is_loop:
            cp = self.config.stc.get_tensor(self.key_cp_q, device, optional = False, allow_bf16 = True, no_defer = True).float()
            kvp = self.config.stc.get_tensor(self.key_cp_kv, device, optional = False, allow_bf16 = True, no_defer = True).float()
            n_loop = self.loop_end - self.loop_start
            cp = cp.reshape(2 * n_loop, self.hidden_size)
            kvp = kvp.reshape(2 * n_loop, self.hidden_size)
            j = self.loop_j
            # two tensors hold the (n_loop, 4, H) message queries as (q, mlp) and (k, v) row pairs,
            # interleaved per loop layer
            self.msg_query = torch.stack([cp[2 * j], kvp[2 * j], kvp[2 * j + 1], cp[2 * j + 1]]).contiguous()

    @override
    def unload(self):
        self.msg_query = None
        super().unload()

    @override
    def get_tensors(self):
        t = super().get_tensors()
        if self.msg_query is not None:
            t[f"{self.key}.msg_query.{self.loop_j}"] = self.msg_query
        return t

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype = None) -> torch.Tensor:
        flow = flow_of(params)
        inst = params.get("layer_instance") or 0
        dtype = x.dtype
        # Draft-model tap: the shared export_state_layers path keys on layer_idx with instance 0,
        # which a loop model's later-pass taps cannot reach - a drafter that taps a specific
        # (layer_idx, instance) pair sets export_state_layer_instances instead (see
        # Inkling2MTPModel.draft_verifier_params, threaded through model_ls._load_autosplit)
        export_inst = params.get("export_state_layer_instances")
        export_state = export_inst is not None and (self.layer_idx, inst) in export_inst

        if self.is_loop and self.layer_idx == self.loop_start and inst > 0:
            x = flow["pre_hidden"]
            flow["bank"] = list(flow["pre_bank"])

        block_residual = x
        if self.attn_read is None:
            hs_q = hs_k = hs_v = x
        else:
            reads = self.attn_read.read(flow["bank"], x)
            if self.sliding:
                hs_q = hs_k = hs_v = reads[0]
            else:
                hs_q, hs_k, hs_v = reads

        messages = None
        if self.is_loop and inst > 0:
            messages = block_read(self.msg_query, flow["prev_bank"], flow["prev_hidden"], self.rms_norm_eps)

        if self.sliding:
            if messages is None:
                h_norm = self.input_norm.forward(hs_q, params)
                hq = hk = hv = h_norm
            else:
                m_q, m_k, m_v, _ = messages
                hq = self.input_norm.forward(self.gate_q.forward(hs_q, m_q), params)
                hk = self.input_norm.forward(self.gate_k.forward(hs_k, m_k), params)
                hv = self.input_norm.forward(self.gate_v.forward(hs_v, m_v), params)
        else:
            if messages is None:
                hq = self.q_norm.forward(hs_q, params)
                hk = self.k_norm.forward(hs_k, params)
                hv = self.v_norm.forward(hs_v, params)
            else:
                m_q, m_k, m_v, _ = messages
                hq = self.q_norm.forward(self.gate_q.forward(hs_q, m_q), params)
                hk = self.k_norm.forward(self.gate_k.forward(hs_k, m_k), params)
                hv = self.v_norm.forward(self.gate_v.forward(hs_v, m_v), params)

        attn_out = self.attn.forward_attn(hq, hk, hv, params)
        ao_raw = attn_out
        attn_out = self.convs.conv("attn", attn_out, params)
        ao_conv = attn_out

        attn_out = attn_out.float()
        if self.is_boundary:
            flow["bank"].append(block_residual)
            residual = attn_out
        else:
            residual = block_residual.float() + attn_out

        m_in = self.mlp_read.read(flow["bank"], residual)[0]
        if messages is not None:
            m_in = self.gate_mlp.forward(m_in, messages[3])
        m_in = self.mlp_norm.forward(m_in, params)
        m_out = self.moe.forward(m_in, params)
        mo_raw = m_out
        m_out = self.convs.conv("mlp", m_out, params)
        mo_conv = m_out
        out = (residual + m_out.float()).to(dtype)

        import os as _os
        if _os.environ.get("INKLING_DEBUG"):
            def _s(t):
                return float(t.float().std()) if (t is not None and t.numel()) else float("nan")
            _o = out[0].detach().float()
            _cos = float(torch.nn.functional.cosine_similarity(_o[:-1], _o[1:], dim=-1).mean())
            _tv = float(_o.var(dim=0).sum())
            print(f"DBG L{self.layer_idx:02d} i{inst} sl={int(self.sliding)} bd={int(self.is_boundary)} "
                  f"x{_s(x)} hq{_s(hq)} ao_raw{_s(ao_raw)} ao_cv{_s(ao_conv)} m_in{_s(m_in)} "
                  f"mo_raw{_s(mo_raw)} mo_cv{_s(mo_conv)} out{_s(out)} adj_cos={_cos:.3f} tok_var={_tv:.2f} "
                  f"bank{len(flow['bank'])}", flush=True)

        if not self.is_loop and self.layer_idx == self.loop_start - 1:
            flow["pre_hidden"] = out
            flow["pre_bank"] = list(flow["bank"])
        if self.is_loop and self.layer_idx == self.loop_end - 1 and inst < 2:
            flow["prev_bank"] = flow["bank"][self.num_pre_loop_blocks + 1:]
            flow["prev_hidden"] = out

        if export_state:
            s = params.get("export_states")
            if not s:
                s = params["export_states"] = []
            out_ = out.half() if out.dtype != torch.half else out
            out_ = out_.clamp_(-65504.0, 65504.0)
            s.append(out_)

        return out