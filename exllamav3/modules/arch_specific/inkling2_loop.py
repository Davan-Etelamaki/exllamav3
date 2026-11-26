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
import math
import os as _os
import torch
import torch.nn.functional as F

from ...cache import Cache
from ...cache.recurrent import host_copy
from ...model.config import Config
from ...util.tensor import get_for_device, host_values, stage_host
from ...constants import PAGE_SIZE
from ...ext import exllamav3_ext as ext
from .. import Module, Linear, RMSNorm
from . import moe_coop_plan

KEY_FLOW = "inkling_flow"

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

    def ring_read(self, params: dict, seqlen: int):
        """
        Rows [lo_b, pos_b) of the ring per sequence (strict history before this chunk),
        lo_b = max(0, pos_b + seqlen - window). Returns (k, v) as (B, R, kvh, hd) with a
        per-row valid-length tensor and per-row lo offsets.
        """
        st = self._state_slot(params)[0]
        s = st.k_state
        n = s.shape[1]
        positions = self._positions(params)
        slots = get_for_device(params, "recurrent_slots", "cpu").tolist()
        ks, vs, los, lens = [], [], [], []
        for b, slot in enumerate(slots):
            pos = int(positions[b])
            lo = max(0, pos + seqlen - self.sliding_window)
            cnt = max(0, pos - lo)
            if cnt > 0:
                idx = (lo + torch.arange(cnt, device = s.device)) % n
                ks.append(s[slot, idx])
                vs.append(st.v_state[slot, idx])
            else:
                ks.append(s[slot, :0])
                vs.append(st.v_state[slot, :0])
            los.append(lo)
            lens.append(cnt)
        mlen = max(lens)
        def pad(rows):
            if all(r.shape[0] == mlen for r in rows):
                return torch.stack(rows)
            out = torch.zeros((len(rows), mlen) + tuple(rows[0].shape[1:]), dtype = s.dtype, device = s.device)
            for i, r in enumerate(rows):
                out[i, :r.shape[0]] = r
            return out
        return pad(ks), pad(vs), los, lens

    def _positions(self, params: dict) -> list:
        positions = params.get("positions")
        if positions is not None:
            if isinstance(positions, torch.Tensor):
                return positions.cpu().tolist()
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

    def _attend_one(self, q, k, v, rel_small, sink, q_pos, kv_pos):
        # single sequence: q (h,T,d) half; k/v (S_,kvh,d) half; rel_small (h,T,extent) fp32;
        # sink (T,h) fp32; q_pos (T,), kv_pos (S_,) int on device
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
            kr, vr, los, lens = self.convs.ring_read(params, t)           # (B,R,kvh,hd)
            self.convs.ring_write(kh, vh, params)
            kv_cat_k = torch.cat([kr, kh], dim = 1).to(torch.half)
            kv_cat_v = torch.cat([vr, vh], dim = 1).to(torch.half)
            kv_lens = [lens[i] + t for i in range(b)]
        else:
            cache = params["cache"]
            cache_seqlens = get_for_device(params, "cache_seqlens", self.device)
            block_table = get_for_device(params, "block_table", self.device)
            inst = params.get("layer_instance") or 0
            cache.update_layer_direct(self.layer_idx, cache_seqlens, block_table, kh, vh, t, inst)
            kv_cat_k, kv_cat_v, kv_lens = self._gather_global(
                cache, cache_seqlens, block_table, inst, kh, vh, t, params)

        outs = []
        for bi in range(b):
            S = kv_lens[bi]
            if self.sliding:
                kv_pos = torch.arange(los[bi], los[bi] + S, device = q.device)
            else:
                kv_pos = torch.arange(S, device = q.device)
            q_pos = pos_row[bi]
            o = self._attend_one(
                q[bi], kv_cat_k[bi][:S], kv_cat_v[bi][:S], rel_small[bi], sink[bi], q_pos, kv_pos)
            outs.append(o.transpose(0, 1))                                # (T,h,d)
        attn = torch.stack(outs).reshape(b, t, -1).contiguous()

        gate = torch.sigmoid(self.gate.forward(h_q, params).float())
        attn = (attn.float() * gate).to(torch.half)
        out = self.wo.forward(attn, params)                               # (B,T,H) fp32
        return out

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
    ):
        super().__init__(config, key, None)
        self.module_name = "InklingMoE"
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
        return (un(self.experts_g[0]).data_ptr(), un(self.experts_u[0]).data_ptr(),
                un(self.experts_d[0]).data_ptr())

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
        if not self.config.infer_params.moe_coop_decode:
            return None
        rows = flat.size(0)
        reason = moe_coop_plan.refusal(rows, self.top_k, flat.is_cuda,
                                       flat.dtype == torch.half,
                                       hasattr(ext, "exl3_moe_coop"))
        if reason is not None:
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

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype = None) -> torch.Tensor:
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
        sel_cpu = host_values(sel_flat, ("inkling2", "moe_sel")).tolist()
        order = sorted(range(len(sel_cpu)), key = lambda i: sel_cpu[i])
        n = 0
        gi = 0
        while n < len(order):
            e = sel_cpu[order[n]]
            m = n
            while m < len(order) and sel_cpu[order[m]] == e:
                m += 1
            # One pinned slot per group: while capturing, every group's upload reads its own
            # address, so the recorded graph cannot have two launches sharing one source that the
            # next group's fill overwrites. Slot index is stable across rounds, which is what makes
            # the recorded addresses meaningful on replay.
            entries = stage_host(order[n:m], x.device, ("inkling2", "moe_entries", gi),
                                 torch.long)
            gi += 1
            tokens = torch.div(entries, self.top_k, rounding_mode = "floor")
            n = m
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

        return out