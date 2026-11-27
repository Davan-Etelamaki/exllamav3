from __future__ import annotations
from typing_extensions import override
import torch
import weakref

from ..cache import Cache
from ..model.config import Config
from ..model.model import Model
from ..modules import RMSNorm, TransformerBlock, Attention, GatedMLP, Module, Linear
from ..modules.attn import prepare_for_attn
from ..util.rope import RopeStyle
from ..util.device_copy import to_device
from ..util.tensor import get_for_device

# Inkling-2-mini MTP drafter (component "mtp"), contract decoded from the same-shape oracle
# Indras-Mirror/Xolotl src/program/inkling_dspark.cpp and the pack header of mtp.safetensors
# (141 root-namespace tensors). Registration: Inkling2LoopConfig declares this class in
# model_classes["mtp"] (popped when the pack lacks the drafter); tabbyAPI draft_mode "mtp"
# loads it via Model.from_config(main_config, component="mtp") over the SAME config object and
# model dir, so all keys are read at the pack ROOT (no prefix) and dimensions come from the
# config.json top-level "mtp_config" stanza.
#
# Contract facts (parent-verified):
#  - mtp_config: block_size 8, hidden 2304, heads 24 q / 8 kv, head_dim 128, rope_theta 1e6,
#    sliding_window_size 1024 on local layers {0,2}, layer 1 global, rms_norm_eps 1e-6,
#    markov_head_type rnn_hidden, markov_rank 256, target_layer_ids [5, 23, 41, 48].
#    (The stanza carries no num_hidden_layers; the pack census fixes it at 3.)
#  - target_layer_ids are LAYER-VISITS over the unrolled loop map (0..5, 6..17 x3, 18..23):
#    visit 5 = (layer 5, instance 0), visit 23 = (layer 11, instance 1), visit 41 = (layer 17,
#    instance 2), visit 48 = (layer 23, instance 2) output - raw, NOT final-normed (dflash
#    prefill never runs the final RMSNorm; laguna parity). Instance taps fire through the
#    export_state_layer_instances seam in modules/arch_specific/inkling2_loop.py (append landed
#    with this commit). Execution order equals tap order.
#  - Per draft layer L the block input mixes the taps: c_L = rms(sum_j softmax(fuse_weight)[L][j]
#    * taps[j], layers.L.dflare_norm.weight) (cpp:631-640), feeding the block ahead of its
#    input_layernorm. Block tokens: row 0 = the anchor (seed) token embedded through the TARGET's
#    input module (its embed-side norm belongs to the main model, cpp:777/:1052-1069 - there is
#    no embed norm in mtp.safetensors), rows 1..7 = mask_embed (cpp).
#  - "norm.weight" in this pack is the DRAFTER'S FINAL norm before the borrowed lm_head
#    (cpp:363 output_norm), NOT an embed norm.
#  - Context rings (cross-round): per drafter layer L the target taps project through
#    layers.L.self_attn.{k,v}_proj_ctx (+ k_norm_ctx) and the k/v rows are written into L's
#    cache like dflash_laguna.update_kv_from_target (rope at cache_seqlens offset, k_norm_ctx).
#  - Exit: fuse row 3 (+ markov_head.dflare_norm) feeds the rnn markov head (full-cat rnn_proj,
#    gate/cand/latent thirds, state = gate*old + (1-gate)*cand); logits = 30*tanh((base_lm +
#    markov fold)/30), fold in TORCH ORDER (no transpose), prev = argmax, tau 0.35 truncation.
#
# REMAINING AFTER C1 (this commit): block-forward tap/dflare injection into the residual stream,
# update_kv_from_target ctx-ring writes, the markov sampling loop + confidence truncation,
# rnn_proj input width + Linear bias handling verified against the pack, embed-side norm
# inclusion verified against Inkling2InputRead. C1 ships the module tree, caps and seams.

MTP_NUM_LAYERS = 3
MTP_TAP_TARGETS = ((5, 0), (11, 1), (17, 2), (23, 2))


class Inkling2MTPModel(Model):
    # config is the SHARED Inkling2LoopConfig (see header); no separate MTP config class

    def __init__(
        self,
        config: Config,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        m = config.config_dict.get("mtp_config")
        assert m, "Inkling2MTPModel: config.json carries no mtp_config stanza"
        self.mtp_config = m
        h = int(m["hidden_size"])
        hd = int(m["head_dim"])
        eps = float(m["rms_norm_eps"])
        self.block_size = int(m["block_size"])
        self.n_taps = len(m["target_layer_ids"])
        self.markov_rank = int(m["markov_rank"])
        # The drafter ropes with its own theta (1e6), not the target's
        self.rope_settings = config.read_rope_settings_default(
            RopeStyle.NEOX,
            default_rope_theta = float(m.get("rope_theta", 1_000_000.0)),
            override_head_dim = hd,
            config_dict = m,
        )

        self.input_layer = Inkling2MTPInputLayer(
            config = config,
            key = "mtp_block_input",
            hidden_size = h,
            rms_norm_eps = eps,
            block_size = self.block_size,
            key_mask_embedding = "mask_embed",
        )
        self.modules = [self.input_layer]
        self.first_block_idx = 1
        self.attn_modules = []

        for idx in range(MTP_NUM_LAYERS):
            sliding = idx in set(m["local_layer_ids"])
            p = f"layers.{idx}"
            attn = Attention(
                config = config,
                key = f"{p}.self_attn",
                layer_idx = idx,
                hidden_size = h,
                head_dim = hd,
                num_q_heads = int(m["num_attention_heads"]),
                num_kv_heads = int(m["num_key_value_heads"]),
                rope_settings = self.rope_settings,
                key_q = "q_proj",
                key_k = "k_proj",
                key_v = "v_proj",
                key_o = "o_proj",
                qmap = "block.attn",
                sliding_window = int(m["sliding_window_size"]) if sliding else -1,
                q_norm = RMSNorm(config = config, key = f"{p}.self_attn.q_norm", rms_norm_eps = eps),
                k_norm = RMSNorm(config = config, key = f"{p}.self_attn.k_norm", rms_norm_eps = eps),
                out_dtype = torch.float,
            )
            self.attn_modules.append(attn)
            self.modules += [Inkling2MTPBlock(
                config = config,
                key = p,
                layer_idx = idx,
                hidden_size = h,
                head_dim = hd,
                attn_norm = RMSNorm(config = config, key = f"{p}.input_layernorm", rms_norm_eps = eps),
                dflare_norm = RMSNorm(config = config, key = f"{p}.dflare_norm", rms_norm_eps = eps),
                attn = attn,
                mlp_norm = RMSNorm(config = config, key = f"{p}.post_attention_layernorm", rms_norm_eps = eps),
                mlp = GatedMLP(
                    config = config,
                    key = f"{p}.mlp",
                    hidden_size = h,
                    intermediate_size = int(m["intermediate_size"]),
                    key_up = "up_proj",
                    key_gate = "gate_proj",
                    key_down = "down_proj",
                    qmap = "block.mlp",
                    interm_dtype = torch.half,
                    out_dtype = torch.float,
                ),
                rms_norm_eps = eps,
            )]

        self.last_kv_module_idx = len(self.modules) - 1
        self.blocks = [mm for mm in self.modules if isinstance(mm, Inkling2MTPBlock)]
        self._anchor_taps = None

        # rnn markov head: rnn_proj / hidden_out Linears + raw markov_w1 bigram selector,
        # passthrough in the linear stream, consumed via member attributes by the sampling loop
        self.markov_head = Inkling2MTPMarkovHead(
            config = config,
            key = "markov_head",
            hidden_size = h,
            markov_rank = self.markov_rank,
            rms_norm_eps = eps,
        )
        self.modules += [self.markov_head]

        # Drafter's final norm before the borrowed lm_head (pack key "norm" == cpp output_norm)
        self.final_norm = RMSNorm(config = config, key = "norm", rms_norm_eps = eps, out_dtype = torch.half)
        self.modules += [self.final_norm]

        self.logit_layer_idx = None
        self.caps.update({
            "uncalibrated_quantize": True,
            "supports_tp": False,
            "attach_target": True,
            "dflash_draft": True,
            "default_draft_size": self.block_size - 1,   # 7 drafts under the 8-row block
            "autosplit_load_fwd": False,
        })

        self.attached_model = None
        # Target-side tap requests: RAW hidden states at the logical target_layer_ids
        # [5,23,41,48] mapped onto (layer_idx, instance) pairs. Visit 48 = the last physical
        # instance's output, NOT the final-normed hidden: dflash prefill stops before the
        # final RMSNorm (job.py to_output is mtp_draft-only), so a norm-key tap never fires
        # here; the norm is the drafter's own job (per-layer dflare_norm post-mix, cpp:631).
        # Same contract as the native dflash_laguna.py (export_state_layers = raw target ids).
        self.draft_verifier_params.update({
            "export_state_layer_instances": set(MTP_TAP_TARGETS),
        })


    @override
    def prepare_layer_map(self):
        # The target's loop relayering (config.layer_map = "0..5,6..17 x3,18..23") belongs to
        # the TARGET's module list; this drafter owns a flat 3-layer stack. Without this
        # override, Model.prepare_layer_map indexes inner[] with text-map ids and raises
        # IndexError at model.py:159 on boot (observed dot26 2026-11-26 21:04 EST).
        self.fwd_modules = [(m, 0, idx) for idx, m in enumerate(self.modules)]
        if self.last_kv_module_idx is not None:
            self.last_kv_module_idx_instance = (self.last_kv_module_idx, 0)


    @override
    def attach_to(self, target):
        self.attached_model = weakref.ref(target)
        self.input_layer.attached_model = self.attached_model


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        # Round contract: input_ids is the seed column only; the sampling loop needs it back
        # as the markov prev-token for slot 0
        params["inkling_seed_ids"] = input_ids.clone()
        return prepare_for_attn(input_ids, params)


    def default_load_shape_dtype(self, chunk_size):
        return (1, 1), torch.long


    def default_load_params(self, max_chunk_size):
        return {}


    def _mix_taps(self, layer_idx: int, taps: list[torch.Tensor]) -> torch.Tensor:
        # softmax(fuse_weight)[layer_idx] over the four taps (cpp:631); row 3 serves the exit
        alpha = torch.softmax(self.input_layer.fuse_weight.float(), dim = -1)[layer_idx]
        acc = None
        for a, t in zip(alpha, taps):
            v = t * a.to(t.dtype)
            acc = v if acc is None else acc + v
        return acc


    def update_kv_from_target(
        self,
        target_hidden: list,
        cache: Cache,
        params: dict,
        lengths: list[int] = None,
    ):
        """
        Fill each drafter layer's context ring from the target taps, mirroring
        dflash_laguna.update_kv_from_target: per-layer mix softmax(fuse_weight)[L] over the four
        taps, THAT layer's dflare_norm on the mix (cpp:631-640), k/v ctx projection, rope at the
        cache_seqlens offset with k_norm_ctx applied, then cache.update_layer_direct rows. The
        block forward then attends these rows as its context - the taps enter the draft sequence
        through the ring, which persists across rounds.

        Ring K/V only (laguna-parity): the query side runs the block's normal q_proj —
        mtp.safetensors carries no q_proj_ctx (pack census + oracle cpp:7).
        """
        if lengths is not None:
            max_length = max(lengths)
            target_hidden = [t[:, :max_length] for t in target_hidden]
        device = self.input_layer.device
        target_hidden = [to_device(t, device) for t in target_hidden]
        assert len(target_hidden) == self.n_taps, \
            f"Inkling2MTP: expected {self.n_taps} exported taps, got {len(target_hidden)}"

        # Last-position taps anchor the markov exit (cpp:792 fuse row 3 + markov dflare_norm)
        self._anchor_taps = [t[:, -1:, :] for t in target_hidden]

        for L in range(MTP_NUM_LAYERS):
            block = self.blocks[L]
            attn = block.attn
            mix = self._mix_taps(L, target_hidden)
            mix = block.dflare_norm.forward(mix, params, out_dtype = torch.half)
            k = block.k_proj_ctx.forward(mix, params)
            v = block.v_proj_ctx.forward(mix, params)
            bsz, seqlen, _ = k.shape
            k = k.view(bsz, seqlen, attn.num_kv_heads, attn.head_dim)
            v = v.view(bsz, seqlen, attn.num_kv_heads, attn.head_dim)
            block_table = get_for_device(params, "block_table", device)
            cache_seqlens = get_for_device(params, "cache_seqlens", device)
            # rope at per-row cache offsets, k_norm_ctx as the pre-rope norm (laguna call shape)
            k, _ = attn.rope.apply(
                k, None, 0, cache_seqlens, None, True,
                block.k_norm_ctx.weight, None, block.k_norm_ctx.rms_norm_eps,
                block.k_norm_ctx.constant_bias, None,
            )
            cache.update_layer_direct(
                attn.layer_idx, cache_seqlens, block_table, k, v, seqlen, 0
            )


    def sample_from_state(self, state: torch.Tensor, params: dict) -> torch.Tensor:
        """
        Sequential markov draft over the block (deepseek_v4_mtp's loop in spirit): base logits
        through the target's lm_head (the drafter's final norm already ran in the module
        stream), bigram bias 30*tanh((base + fold)/30) with the rnn_hidden state update
        gate/cand/latent (task contract: gate=sigmoid(first third), cand=tanh(second third),
        latent=tanh(third third), state = gate*old + (1-gate)*cand), fold in TORCH ORDER
        (hidden_out @ lm_head, no transpose where the storage allows it), prev = argmax feeding
        markov_w1[prev]. tau = 0.35 truncates params["draft_conf"] at the first low-confidence
        slot. Returns (b, block_size) with column 0 = the seed (generator crops [:,1:]).

        Deviation flags: f_rnn exit vector = rms(softmax(fuse_weight)[3] . anchor taps,
        markov_head.dflare_norm) per cpp:792; rnn bias vectors applied only if loaded; the
        block's hidden rows are computed in parallel from masked inputs and the markov chain
        post-processes them (the oracle re-feeds prev embeddings per slot - acceptance-tuning
        wheel can revisit).
        """
        tgt = self.attached_model()
        lm = tgt.modules[tgt.logit_layer_idx]
        vocab = self.config.vocab_size
        rnn = self.markov_head
        rank = self.markov_rank
        bsz, blk = state.shape[0], state.shape[1]
        device = state.device

        # Exit conditioning vector (fuse row 3 over the anchor taps), cpp:792
        if self._anchor_taps is not None and len(self._anchor_taps) == self.n_taps:
            f_rnn = rnn.dflare_norm.forward(
                self._mix_taps(3, self._anchor_taps), params, out_dtype = torch.float
            )
        else:
            f_rnn = torch.zeros((bsz, 1, self.input_layer.hidden_size), device = device)

        # fold_w in TORCH ORDER: hidden_out.weight (rank, hidden) composed with the target's
        # lm_head into (rank, vocab); orientation resolved once from the storage shapes
        lw = lm.weight
        hw = rnn.hidden_out.weight
        if hw.shape[1] == lw.shape[0]:
            fold_w = hw.mm(lw)                    # lm stored (hidden, vocab): pure torch order
        elif hw.shape[1] == lw.shape[1]:
            fold_w = hw.mm(lw.t())                 # lm stored (vocab, hidden): documented
        else:
            raise RuntimeError(
                f"Inkling2MTP: cannot compose fold_w from hidden_out {tuple(hw.shape)} and lm_head {tuple(lw.shape)}")
        if fold_w.shape[-1] != vocab:
            fold_w = fold_w.t()

        mw1 = rnn.markov_w1                          # bigram table (rank, rank) or (vocab, rank)
        markov_state = torch.zeros((bsz, rank), device = device)
        prev = params.get("inkling_seed_ids")
        prev = prev[:, 0] if prev is not None and prev.dim() > 1 else (prev if prev is not None else torch.zeros(bsz, dtype = torch.long, device = device))
        prev = prev.to(device).long().clamp(0, mw1.shape[0] - 1)
        if mw1.shape[1] != rank and mw1.shape[0] == rank:
            mw1 = mw1.t()

        ids = torch.zeros((bsz, blk), dtype = torch.long, device = device)
        ids[:, 0] = prev
        conf = torch.zeros((bsz, blk - 1), device = device)
        for i in range(blk - 1):
            hi = state[:, i, :].float()
            base = lm.forward(lm.prepare_for_device(state[:, i, :], params), params)
            base = base[..., :vocab].float()
            prev_emb = mw1[prev][:, :rank] if mw1.shape[0] != rank else mw1[prev]
            rnn_in = torch.cat(
                [markov_state, prev_emb, hi, f_rnn[:, 0, :]], dim = -1)
            raw = rnn.rnn_proj.forward(rnn_in, params).float()
            if rnn.rnn_bias is not None and rnn.rnn_bias.numel() == raw.shape[-1]:
                raw = raw + rnn.rnn_bias
            gate = torch.sigmoid(raw[:, :rank])
            cand = torch.tanh(raw[:, rank:2 * rank])
            latent = torch.tanh(raw[:, 2 * rank:3 * rank])
            markov_state = gate * markov_state + (1.0 - gate) * cand
            bias = (latent + (prev_emb @ mw1[:, :rank] if prev_emb.shape[-1] == rank else 0.0)).mm(fold_w)
            if rnn.hidden_bias is not None and rnn.hidden_bias.numel() == vocab:
                bias = bias + rnn.hidden_bias
            logits = 30.0 * torch.tanh((base + bias) / 30.0)
            c, p = torch.max(logits, dim = -1)
            conf[:, i] = c
            ids[:, i + 1] = p
            prev = p
        # Constant-confidence truncation: everything after the first sub-tau draft is dropped
        alive = torch.cumprod((conf >= 0.35).float(), dim = -1)
        params["draft_conf"] = conf * alive
        for b in range(bsz):
            keep = int(alive[b].sum().item())
            ids[b, keep + 1:] = ids[b, keep]
        return ids


    @staticmethod
    @override
    def get_additional_compiled_tensors(config: Config) -> dict:
        # Root tensors owned outside module key prefixes: the tap mixer and the block-mask rows
        # (Inkling2MTPInputLayer loads them raw; listed here for the compile / load census)
        tensors = {}
        tensors.update(config.stc.list_tensors(prefix = "fuse_weight"))
        tensors.update(config.stc.list_tensors(prefix = "mask_embed"))
        return tensors


class Inkling2MTPBlock(TransformerBlock):
    """
    One drafter block. Adds what TransformerBlock has no slot for: dflare_norm (the post tap-mix
    norm feeding the block, cpp:631-640 - applied by the C2 block forward ahead of attn_norm) and
    the context-ring projections k/v_proj_ctx (+ k_norm_ctx) owned by update_kv_from_target.
    Pack census 2026-11-26: mtp.safetensors carries NO q_proj_ctx (oracle cpp:7 confirms
    k/v ctx only) - a q_proj_ctx Linear here fails load with "No tensors found".
    Registered as submodules so they load and compile with the block; the inherited forward is
    the usual norm-attn-mlp residual over whatever input the mix hands it.
    """

    def __init__(
        self,
        config: Config,
        key: str,
        layer_idx: int,
        hidden_size: int,
        head_dim: int,
        attn_norm: RMSNorm,
        dflare_norm: RMSNorm,
        attn: Attention,
        mlp_norm: RMSNorm,
        mlp: GatedMLP,
        rms_norm_eps: float,
        **kwargs,
    ):
        super().__init__(
            config = config,
            key = key,
            layer_idx = layer_idx,
            attn_norm = attn_norm,
            attn = attn,
            mlp_norm = mlp_norm,
            mlp = mlp,
            **kwargs,
        )
        self.dflare_norm = dflare_norm
        self.register_submodule(self.dflare_norm)
        kv_dim = attn.num_kv_heads * head_dim
        self.k_proj_ctx = Linear(
            config = config,
            key = f"{key}.self_attn.k_proj_ctx",
            in_features = hidden_size,
            out_features = kv_dim,
            qmap = "block.attn",
            out_dtype = torch.float,
        )
        self.v_proj_ctx = Linear(
            config = config,
            key = f"{key}.self_attn.v_proj_ctx",
            in_features = hidden_size,
            out_features = kv_dim,
            qmap = "block.attn",
            out_dtype = torch.float,
        )
        self.k_norm_ctx = RMSNorm(
            config = config,
            key = f"{key}.self_attn.k_norm_ctx",
            rms_norm_eps = rms_norm_eps,
        )
        for m in (self.k_proj_ctx, self.v_proj_ctx, self.k_norm_ctx):
            self.register_submodule(m)


class Inkling2MTPMarkovHead(Module):
    """
    rnn_hidden markov head (pack: markov_head.rnn_proj.{trellis,suh,svh,mul1} + .bias,
    markov_head.hidden_out.{trellis,suh,svh,mul1} + .bias, markov_head.markov_w1.weight,
    markov_head.dflare_norm.weight). rnn consumes the FULL concatenation
    cat([state(rank), markov_w1[prev](rank), hidden_i(hidden), f_rnn(?)]); a column-split is
    algebraically identical (parent F6). Out width = 3 * rank (gate | cand | latent, confirmed
    by rnn_proj.bias length when C3 reads the header).
    TODO(C3): exact in_features (f_rnn width) and Linear bias wiring (rnn_proj.bias /
    hidden_out.bias) verified against the pack; wrong in_features fails loudly at load, not
    silently.
    """

    def __init__(
        self,
        config: Config,
        key: str,
        hidden_size: int,
        markov_rank: int,
        rms_norm_eps: float,
    ):
        super().__init__(config, key, None)
        self.module_name = "Inkling2MTPMarkovHead"
        self.hidden_size = hidden_size
        self.markov_rank = markov_rank
        self.rnn_proj = Linear(
            config = config,
            key = f"{key}.rnn_proj",
            in_features = 3 * markov_rank + hidden_size,
            out_features = 3 * markov_rank,
            qmap = "block.attn",
            out_dtype = torch.float,
        )
        self.hidden_out = Linear(
            config = config,
            key = f"{key}.hidden_out",
            in_features = hidden_size,
            out_features = markov_rank,
            qmap = "block.attn",
            out_dtype = torch.float,
        )
        self.dflare_norm = RMSNorm(
            config = config,
            key = f"{key}.dflare_norm",
            rms_norm_eps = rms_norm_eps,
        )
        for m in (self.rnn_proj, self.hidden_out, self.dflare_norm):
            self.register_submodule(m)
        self.markov_w1 = None
        self.rnn_bias = None
        self.hidden_bias = None


    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        self.markov_w1 = self.config.stc.get_tensor(
            f"{self.key}.markov_w1.weight", device, optional = False,
            allow_bf16 = True, no_defer = True
        ).float()
        # Biases are applied manually in the sampling loop (the pack carries them even where
        # Linear modules don't own them); optional in case the converter folded them elsewhere
        self.rnn_bias = self.config.stc.get_tensor(
            f"{self.key}.rnn_proj.bias", device, optional = True,
            allow_bf16 = True, no_defer = True
        )
        self.rnn_bias = None if self.rnn_bias is None else self.rnn_bias.float()
        self.hidden_bias = self.config.stc.get_tensor(
            f"{self.key}.hidden_out.bias", device, optional = True,
            allow_bf16 = True, no_defer = True
        )
        self.hidden_bias = None if self.hidden_bias is None else self.hidden_bias.float()


    @override
    def unload(self):
        self.markov_w1 = None
        self.rnn_bias = None
        self.hidden_bias = None
        super().unload()


    def optimizer_targets(self):
        raise NotImplementedError()


    @override
    def get_tensors(self):
        t = super().get_tensors()
        if self.markov_w1 is not None:
            t[f"{self.key}.markov_w1.weight"] = self.markov_w1
        return t


    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype = None) -> torch.Tensor:
        # Passthrough: consumed through member modules from the sampling loop (C3)
        return x


class Inkling2MTPInputLayer(Module):
    """
    Block-token stream builder + raw root-tensor owner (fuse_weight 4x4 tap mixer, mask_embed
    placeholder rows). The dflare tap-mix itself is applied per block by the C2 forward; this
    module owns the tensors and grows the seed column into the block embedding.
    """

    def __init__(
        self,
        config: Config,
        key: str,
        hidden_size: int,
        rms_norm_eps: float,
        block_size: int,
        key_mask_embedding: str,
        out_dtype: torch.dtype | None = torch.float,
    ):
        super().__init__(config, key, None)
        self.module_name = "Inkling2MTPInputLayer"
        self.hidden_size = hidden_size
        self.out_dtype = out_dtype
        self.block_size = block_size
        self.rms_norm_eps = rms_norm_eps
        self.key_mask_embedding = key_mask_embedding
        self.attached_model = None
        self.fuse_weight = None
        self.mask_embed = None
        self.caps.update({"x_cpu": True})


    @override
    def optimizer_targets(self):
        return []


    @override
    def load(self, device: torch.device, **kwargs):
        self.device = device
        self.fuse_weight = self.config.stc.get_tensor(
            "fuse_weight", device, optional = False, allow_bf16 = True, no_defer = True
        ).float()
        # rows 0..2 feed the three blocks, row 3 conditions the markov exit (cpp:792)
        assert self.fuse_weight.shape[0] >= 4, \
            f"Inkling2MTP: fuse_weight needs >=4 rows (3 blocks + exit), got {tuple(self.fuse_weight.shape)}"
        self.mask_embed = self.config.stc.get_tensor(
            self.key_mask_embedding, device, optional = False, allow_bf16 = True, no_defer = True
        ).float()


    @override
    def unload(self):
        self.fuse_weight = None
        self.mask_embed = None


    @override
    def get_tensors(self):
        t = {}
        if self.fuse_weight is not None:
            t["fuse_weight"] = self.fuse_weight
        if self.mask_embed is not None:
            t[self.key_mask_embedding] = self.mask_embed
        return t


    def prepare_for_device(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        return x


    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype = None) -> torch.Tensor:
        # x: seed column only (b,1) under the dflash round contract. Grow to (b, block_size) and
        # embed through the ATTACHED target's input module - Inkling2InputRead owns the embed and
        # its embed-side norm (cpp:777/:1052-1069; no embed norm in mtp.safetensors). Mask rows
        # 1..block_size-1 are then overwritten from this pack's mask_embed.
        # TODO(C2): confirm target.modules[0] is the embed reader and norm-inclusive for a bare
        # token id; verify against Inkling2InputRead.forward.
        bsz = x.shape[0]
        pad = torch.full((bsz, self.block_size - 1), 0, dtype = torch.long, device = x.device)
        x = torch.cat((x, pad), dim = -1)
        target = self.attached_model()
        emb = target.modules[0].forward(x, params)
        me = self.mask_embed
        if me.dim() == 1:
            me = me.view(1, -1)
        emb = emb.clone()
        emb[:, 1:, :] = me[:1].to(emb.dtype)   # single mask row broadcast to rows 1..7
        return emb.to(self.out_dtype)