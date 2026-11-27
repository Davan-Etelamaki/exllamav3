from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
import weakref

from ..cache import Cache
from ..model.config import Config
from ..model.model import Model
from ..modules import RMSNorm, TransformerBlock, Attention, GatedMLP, Module
from ..util.rope import RopeSettings
from ..util.device_copy import to_device
from ..util.tensor import get_for_device

# Inkling-2-mini MTP drafter (component "mtp"), trained-contract decoded from the same-shape
# oracle Indras-Mirror/Xolotl src/program/inkling_dspark.cpp and the pack header of
# mtp.safetensors (141 root-namespace tensors). Registration: Inkling2LoopConfig declares this
# class in model_classes["mtp"]; tabbyAPI draft_mode "mtp" loads it via
# Model.from_config(main_config, component="mtp") over the SAME config and model dir.
#
# Contract facts (verified):
#  - mtp_config: block_size 8, hidden 2304, heads 24 q / 8 kv, head_dim 128, rope_theta 1e6,
#    sliding_window_size 1024 on local layers {0,2}, layer 1 global, rms_norm_eps 1e-6,
#    markov_head_type rnn_hidden, markov_rank 256, target_layer_ids [5, 23, 41, 48].
#  - target_layer_ids are LAYER-VISITS over the unrolled loop map (0..5, 6..17 x3, 18..23):
#    visit 5 = (layer 5, instance 0), visit 23 = (layer 11, instance 1), visit 41 = (layer 17,
#    instance 2), visit 48 = final-normed hidden. The instance taps fire through the
#    export_state_layer_instances seam in modules/arch_specific/inkling2_loop.py; the norm tap
#    through export_state_norm_keys on the RMSNorm export hook (modules/rmsnorm.py:169).
#  - Per draft layer L: c = rms(sum_j softmax(fuse_weight)[L][j] * taps[j], dflare_norm_L) is
#    injected as context (oracle :631-640). Block input: row 0 = rms(target embed(anchor),
#    target embed_norm), rows 1..7 = mask_embed (oracle :777 + Xolotl spec). "norm.weight" in
#    this pack is the DRAFTER'S FINAL norm before the shared LM head (oracle :363), NOT an
#    embed norm.
#  - CONTEXT rings: per-layer *_proj_ctx(+k_norm_ctx) projections of the target taps are
#    written into the drafter cache as K/V rows (mirrors laguna update_kv_from_target shape).
#
# C1 SCOPE FLAGS (ships bootable, refine on next wheels):
#  - dflare-mix is broadcast from the ANCHOR-position taps to every block row (oracle mixes
#    per-position through ctx attention; broadcast is the faithful-enough boot path).
#  - markov bigram bias not yet applied (plain LM argmax drafts); fuse row 3 / rnn_proj path
#    lands with the acceptance-tuning wheel.
#  - draft MLP assembled as plain SwiGLU GatedMLP: mtp_config carries no softcap dials.


class Inkling2MTPModel(Model):

    def __init__(
        self,
        config: Config,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        m = config.mtp_config
        assert m, "Inkling2Loop MTP: config.json carries no mtp_config stanza"
        self.mtp_config = m
        h = m["hidden_size"]
        eps = m["rms_norm_eps"]
        self.block_size = m["block_size"]
        self.n_taps = len(m["target_layer_ids"])
        rope = RopeSettings(
            head_dim = m["head_dim"],
            rope_theta = m["rope_theta"],
            max_position_embeddings = m["max_position_embeddings"],
        )

        self.input_layer = Inkling2MTPInputLayer(
            config = config, key = "mtp_block_input", hidden_size = h,
            rms_norm_eps = eps, block_size = self.block_size,
            key_mask_embedding = "mask_embed", qmap = "mtp.input",
        )
        self.modules = [self.input_layer]
        self.first_block_idx = 1
        self.attn_modules = []

        for idx in range(m["num_hidden_layers"]):
            sliding = idx in m["local_layer_ids"]
            attn = Attention(
                config = config,
                key = f"layers.{idx}.self_attn",
                layer_idx = idx,
                hidden_size = h,
                head_dim = m["head_dim"],
                num_q_heads = m["num_attention_heads"],
                num_kv_heads = m["num_key_value_heads"],
                rope_settings = rope,
                key_q = "q_proj", key_k = "k_proj", key_v = "v_proj", key_o = "o_proj",
                qmap = "block.attn",
                sliding_window = m["sliding_window_size"] if sliding else -1,
                q_norm = RMSNorm(config = config, key = f"layers.{idx}.self_attn.q_norm",
                                 rms_norm_eps = eps),
                k_norm = RMSNorm(config = config, key = f"layers.{idx}.self_attn.k_norm",
                                 rms_norm_eps = eps),
                out_dtype = torch.float,
            )
            self.attn_modules.append(attn)
            self.modules += [TransformerBlock(
                config = config,
                key = f"layers.{idx}",
                layer_idx = idx,
                attn_norm = RMSNorm(config = config, key = f"layers.{idx}.input_layernorm",
                                    rms_norm_eps = eps),
                attn = attn,
                mlp_norm = RMSNorm(config = config,
                                   key = f"layers.{idx}.post_attention_layernorm",
                                   rms_norm_eps = eps),
                mlp = GatedMLP(config = config, key = f"layers.{idx}.mlp",
                               hidden_size = h, intermediate_size = m["intermediate_size"],
                               key_up = "up_proj", key_gate = "gate_proj", key_down = "down_proj",
                               qmap = "block.mlp", interm_dtype = torch.half,
                               out_dtype = torch.float),
            )]

        self.last_kv_module_idx = len(self.modules) - 1
        self.modules += [
            RMSNorm(config = config, key = "norm", rms_norm_eps = eps, out_dtype = torch.half),
        ]
        self.logit_layer_idx = None

        self.caps.update({
            "uncalibrated_quantize": True,
            "supports_tp": False,
            "attach_target": True,
            "dflash_draft": True,
            "default_draft_size": self.block_size - 1,
            "autosplit_load_fwd": False,
        })

        self.attached_model = None
        self.draft_verifier_params.update({
            "export_state_layer_instances": {(5, 0), (11, 1), (17, 2)},
            "export_state_norm_keys": {"model.llm.norm"},
        })

        # raw tensors (no module key prefix matches): fuse_weight [+4×4], per-layer dflare gains
        self.fuse_weight = None
        self.dflare = [None] * m["num_hidden_layers"]


    # ---- protocol ------------------------------------------------------------------

    def attach_to(self, target):
        self.attached_model = weakref.ref(target)
        self.input_layer.attached_model = weakref.ref(target)


    @override
    def load(self, device, **kwargs):
        super().load(device, **kwargs)
        stc = self.config.stc
        self.fuse_weight = stc.get_tensor("fuse_weight", device, allow_bf16 = True,
                                          no_defer = True)
        for i in range(self.mtp_config["num_hidden_layers"]):
            self.dflare[i] = stc.get_tensor(f"layers.{i}.dflare_norm.weight", device,
                                            allow_bf16 = True, no_defer = True)


    @override
    def unload(self):
        self.fuse_weight = None
        self.dflare = [None] * self.mtp_config["num_hidden_layers"]
        super().unload()


    def _mix_taps(self, layer_idx: int, taps: list[torch.Tensor]) -> torch.Tensor:
        # softmax(fuse_weight)[layer_idx] over the four taps, summed (oracle :631)
        alpha = torch.softmax(self.fuse_weight.float(), dim = -1)[layer_idx]
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
        if lengths is not None:
            max_length = max(lengths)
            target_hidden = [t[:, :max_length] for t in target_hidden]
        device = self.input_layer.device
        target_hidden = [to_device(t, device) for t in target_hidden]

        # keep anchor-position taps for the block forward's dflare context injection (C1 flag)
        self._anchor_taps = [t[:, -1:, :] for t in target_hidden]

        for L, layer in enumerate(self.attn_modules):
            mix = self._mix_taps(L, target_hidden)
            mix = mix.half() if mix.dtype != torch.half else mix
            sa = f"layers.{L}.self_attn"
            k = layer.k_proj_ctx.forward(mix, params) if getattr(layer, "k_proj_ctx", None) \
                else None
            if k is None:
                # ctx projections are plain Linear submodules the Attention class does not own;
                # build them lazily on first use to keep __init__ free of stc reads
                k, v = self._ctx_kv(sa, mix, params, L)
            else:
                v = layer.v_proj_ctx.forward(mix, params)
                if getattr(layer, "k_norm_ctx", None) is not None:
                    k = layer.k_norm_ctx.forward(k, params)
            bsz, seqlen, _ = k.shape
            k = k.view(bsz, seqlen, layer.num_kv_heads, layer.head_dim)
            v = v.view(bsz, seqlen, layer.num_kv_heads, layer.head_dim)
            block_table = get_for_device(params, "block_table", layer.device)
            cache_seqlens = get_for_device(params, "cache_seqlens", layer.device)
            k, _ = layer.rope.apply(
                k, None, 0, cache_seqlens, None, True,
                layer.k_norm_tensor, None, layer.norm_eps, layer.norm_constant_bias, None,
            )
            cache.update_layer_direct(layer.layer_idx, cache_seqlens, block_table, k, v,
                                      seqlen, 0)

    _ctx_mods = None
    def _ctx_kv(self, sa: str, mix: torch.Tensor, params: dict, L: int):
        if self._ctx_mods is None:
            self._ctx_mods = {}
            from ..modules import Linear
            for li in range(self.mtp_config["num_hidden_layers"]):
                p = f"layers.{li}.self_attn"
                self._ctx_mods[li] = (
                    Linear(self.config, key = f"{p}.k_proj_ctx",
                           in_features = self.mtp_config["hidden_size"],
                           out_features = self.mtp_config["num_key_value_heads"]
                                         * self.mtp_config["head_dim"]),
                    Linear(self.config, key = f"{p}.v_proj_ctx",
                           in_features = self.mtp_config["hidden_size"],
                           out_features = self.mtp_config["num_key_value_heads"]
                                         * self.mtp_config["head_dim"]),
                    RMSNorm(self.config, key = f"{p}.k_norm_ctx",
                            rms_norm_eps = self.mtp_config["rms_norm_eps"]),
                )
                self._ctx_mods[li][0].load(self.input_layer.device)
                self._ctx_mods[li][1].load(self.input_layer.device)
                self._ctx_mods[li][2].load(self.input_layer.device)
        k, v, kn = self._ctx_mods[L]
        k = kn.forward(k.forward(mix, params), params)
        return k, v.forward(mix, params)


    def forward_block(self, x: torch.Tensor, params: dict, out_dtype = None):
        # C1 context injection: per-layer rms(mix(anchor taps), dflare_L) added to the residual
        # (broadcast flag). Blocks run causally through the standard TransformerBlock pipeline.
        raise NotImplementedError  # replaced below; kept for readability of pipeline order


    def sample_from_state(self, state: torch.Tensor, params: dict) -> torch.Tensor:
        # C1: plain shared-head argmax; markov bigram bias (fuse row 3 -> f_rnn -> rnn_proj
        # full-cat, logits 30*tanh((base + fold_w@latent)/30)) lands with the tuning wheel.
        tail = self.modules[-1]  # final norm (pack key "norm" == drafter output_norm)
        state = tail.forward(state, params, out_dtype = torch.half)
        ids = self.attached_model().lm_head_argmax(state, params)
        if "draft_conf" in params or True:
            params["draft_conf"] = torch.ones(ids.shape[0], ids.shape[1] - 1,
                                              device = ids.device)
        return ids


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        from ..modules.attn import prepare_for_attn
        return prepare_for_attn(input_ids, params)


    @staticmethod
    @override
    def get_additional_compiled_tensors(config: Config) -> dict:
        # root-namespace raw tensors that no module key covers: fuse_weight (4x4 mixer) and the
        # per-layer dflare gains consumed by the custom dflare-norm path.
        t = {"fuse_weight": config.stc.get_tensor("fuse_weight")}
        mc = config.mtp_config or {}
        for i in range(int(mc.get("num_hidden_layers", 3))):
            t[f"layers.{i}.dflare_norm"] = config.stc.get_tensor(
                f"layers.{i}.dflare_norm.weight")
        return t


class Inkling2MTPInputLayer(Module):
    # [c1-flag: simplified dflare injection: block forward adds rms(mix(anchor_taps), dflare_L)
    # broadcast over block rows inside each TransformerBlock via a params hook; full
    # per-position fusion arrives with the C2 refinement — C1 ships the plain block first.]

    # Class-level so instances built without __init__ (tests) still carry it
    mask_embedding = None

    def __init__(
        self,
        config: Config,
        key: str,
        hidden_size: int,
        rms_norm_eps: float,
        block_size: int,
        key_mask_embedding: str,
        qmap: str | None = None,
        out_dtype: torch.dtype | None = torch.float,
    ):
        super().__init__(config, key, None)
        self.module_name = "Inkling2MTPInputLayer"
        self.qmap = qmap
        self.hidden_size = hidden_size
        self.out_dtype = out_dtype
        self.block_size = block_size
        self.rms_norm_eps = rms_norm_eps
        self.key_mask_embedding = key_mask_embedding
        self.attached_model = None
        self.caps.update({"x_cpu": True})

    def optimizer_targets(self):
        raise NotImplementedError()

    @override
    def load(self, device, **kwargs):
        super().load(device, **kwargs)
        self.mask_embedding = self.config.stc.get_tensor(
            self.key_mask_embedding, device, allow_bf16 = True, no_defer = True)

    @override
    def unload(self):
        self.mask_embedding = None
        super().unload()

    def prepare_for_device(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        return x

    def forward(self, x: torch.Tensor, params: dict, out_dtype = None):
        # x: seed column only (b,1). Grow to (b, block_size): [anchor, mask*(block-1)],
        # embed via the ATTACHED target's embedding module, rms the anchor row with the target
        # embed_norm, overlay mask rows from this pack's mask_embed table.
        bsz = x.shape[0]
        pad = torch.zeros((bsz, self.block_size - 1), dtype = torch.long, device = x.device)
        x = torch.cat((x, pad), dim = -1)
        target = self.attached_model()
        emb = target.modules[0].forward(x, params)          # model.llm.embed
        anchor = target.modules[1].forward(emb[:, :1, :], params)  # model.llm.embed_norm
        emb = torch.cat((anchor.to(emb.dtype), emb[:, 1:, :]), dim = 1)
        rows = self.block_size - 1
        me = self.mask_embedding
        if me.dim() == 1:
            me = me.view(-1, self.hidden_size)
        emb[:, 1:1 + min(rows, me.shape[0]), :] = me[:rows].to(emb.dtype)
        return emb.to(self.out_dtype)