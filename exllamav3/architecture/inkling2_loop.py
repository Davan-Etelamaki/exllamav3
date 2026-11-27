"""
Inkling 2 Mini (Inkling2Loop) architecture.

Relayering drives the loop: with loop_config core 6..18 x 3 passes the layer map becomes
"0..5,6..17,6..17,6..17,18..23", so the engine runs the 12 core layers three times and
addresses each pass's KV/conv state by (layer_idx, instance). The intra-forward loop state
(block-state bank, pre-loop snapshot, cross-pass message sources) rides params[KEY_FLOW],
initialized by prepare_inputs and read/written by InklingLayer. Text-only for v1: the vision
tower and the MTP drafter are not registered.
"""
from __future__ import annotations
from typing_extensions import override
import json
import os
import torch

from ..model.config import Config, no_default
from ..model.model import Model
from ..modules import RMSNorm, Embedding, Linear
from ..modules.attn import prepare_for_attn
from ..modules.short_conv import ShortConvState
from ..cache.recurrent_util import prepare_for_recurrence
from ..modules.arch_specific.inkling2_loop import (
    InklingLayer, InklingOutputRead, KEY_FLOW,
)


class Inkling2LoopConfig(Config):
    arch_string = "Inkling2LoopForConditionalGeneration"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        with open(os.path.join(directory, "config.json"), encoding = "utf8") as f:
            peek = json.load(f)
        tc = peek.get("text_config", peek)
        lc = tc["loop_config"]
        core_start = int(lc["core_start"])
        core_end = int(lc["core_end"])
        num_passes = int(lc["num_passes"])
        num_passes = int(os.environ.get("INKLING_PASSES", num_passes))
        num_layers = int(tc["num_hidden_layers"])
        assert not lc["is_varloop"], "Inkling: variable-depth loop is not supported"
        assert num_passes >= 1 and 0 < core_start < core_end <= num_layers, \
            f"Inkling: bad loop_config {lc}"
        layer_map = ",".join(
            ([f"0..{core_start - 1}"] if core_start > 0 else []) +
            [f"{core_start}..{core_end - 1}"] * num_passes +
            ([f"{core_end}..{num_layers - 1}"] if core_end < num_layers else [])
        )

        # This architecture derives its own layer_map from loop_config; a caller-supplied
        # layer_map (e.g. tabbyAPI forwarding one) is overridden, not collided with.
        kwargs.pop("layer_map", None)
        kwargs.pop("config_key", None)
        from .inkling2_mtp import Inkling2MTPModel
        super().__init__(
            directory,
            {"text": Inkling2LoopTextModel, "mtp": Inkling2MTPModel},
            layer_map = layer_map,
            config_key = "text_config",
            **kwargs,
        )
        # MTP drafter component: config.json carries its dial set at top level and the pack
        # ships mtp.safetensors next to the model shards (root key namespace, no prefix).
        # A pack without the file serves text-only, exactly like the qwen3_5 pop-guard.
        self.mtp_config = peek.get("mtp_config")
        if not os.path.isfile(os.path.join(directory, "mtp.safetensors")) or not self.mtp_config:
            self.model_classes.pop("mtp", None)

        # The readout lives in a head/ subdirectory with its own .safetensors files
        head_dir = os.path.join(directory, "head")
        if os.path.isdir(head_dir):
            self.stc.add_tensor_files(head_dir)

        # Attention dims (identical for sliding and full: swa_* mirror the top-level dials)
        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.head_dim = self.read_cfg(int, "head_dim", None) or \
            self.hidden_size // self.read_cfg(int, "num_attention_heads", no_default)
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "num_key_value_heads", self.num_q_heads)
        assert self.read_cfg(int, "swa_head_dim", self.head_dim) == self.head_dim, \
            "Inkling: sliding and full attention must share head_dim"
        assert self.read_cfg(int, "swa_num_attention_heads", self.num_q_heads) == self.num_q_heads
        assert self.read_cfg(int, "swa_num_key_value_heads", self.num_kv_heads) == self.num_kv_heads

        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", no_default)
        self.assert_cfg(bool, "o_bias", False)
        self.assert_cfg(bool, "q_bias", False)

        # Sliding set is the complement of global layers; local_layer_ids lists the sliding ones
        local_layer_ids = self.read_cfg(list, "local_layer_ids", no_default)
        self.local_layer_ids = [int(i) for i in local_layer_ids]
        assert len(self.local_layer_ids) <= self.num_hidden_layers
        self.sliding_layer_ids = set(self.local_layer_ids)

        # Loop core
        self.loop_start = core_start
        self.loop_end = core_end
        self.loop_passes = num_passes
        self.loop_gate_rank = int(lc["gate_rank"])
        self.n_layers_per_block = self.read_cfg(int, "n_layers_per_attn_res_block", no_default)
        self.num_pre_loop_blocks = core_start // self.n_layers_per_block

        # Relative-position attention
        self.sliding_window = self.read_cfg(int, ["sliding_window_size", "sliding_window"], no_default)
        self.conv_kernel_size = self.read_cfg(int, ["sconv_kernel_size", "conv_kernel_size"], no_default)
        self.rel_extent = self.read_cfg(int, "rel_extent", no_default)
        self.d_rel = self.read_cfg(int, "d_rel", no_default)
        self.qk_l2_clip_threshold = self.read_cfg(float, "qk_l2_clip_threshold", no_default)
        self.rel_l2_clip_threshold = self.read_cfg(float, "rel_l2_clip_threshold", no_default)
        self.log_scaling_alpha = self.read_cfg(float, "log_scaling_alpha", no_default)
        self.log_scaling_n_floor = self.read_cfg(int, "log_scaling_n_floor", no_default)
        self.assert_cfg(bool, "use_value_norm_no_gamma", True)
        self.assert_cfg(bool, "attn_res_normalize_values", True)

        # MoE
        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)
        self.num_experts = self.read_cfg(int, ["n_routed_experts", "num_local_experts"], no_default)
        self.num_experts_per_tok = self.read_cfg(int, "num_experts_per_tok", no_default)
        self.assert_cfg(int, "n_shared_experts", 0)
        self.route_scale = self.read_cfg(float, ["route_scale", "routed_scaling_factor"], no_default)
        self.mlp_gate_softcap = self.read_cfg(float, "mlp_gate_softcap", no_default)
        self.mlp_up_softcap = self.read_cfg(float, "mlp_up_softcap", no_default)

        # Readout
        self.final_logit_softcapping = self.read_cfg(float, "final_logit_softcapping", 0.0)

        # No rotary embeddings anywhere in this model
        self.rope_settings = None
        self.max_position_embeddings = self.read_cfg(
            int, ["model_max_length", "max_position_embeddings"], self.default_max_position_embeddings()
        )

    def is_sliding(self, layer_idx: int) -> bool:
        return layer_idx in self.sliding_layer_ids

    def default_max_position_embeddings(self):
        return 1048576


class Inkling2LoopTextModel(Model):
    config_class = Inkling2LoopConfig

    def __init__(
        self,
        config: Inkling2LoopConfig,
        key_prefix: str = "model.llm",
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        p = f"{key_prefix}."

        self.modules += [
            Embedding(
                config = config,
                key = p + "embed",
                vocab_size = config.vocab_size,
                hidden_size = config.hidden_size,
            ),
            RMSNorm(
                config = config,
                key = p + "embed_norm",
                rms_norm_eps = config.rms_norm_eps,
            ),
        ]

        self.first_block_idx = len(self.modules)

        for idx in range(config.num_hidden_layers):
            is_loop = config.loop_start <= idx < config.loop_end
            self.modules += [
                InklingLayer(
                    config = config,
                    key_prefix = p,
                    layer_idx = idx,
                    hidden_size = config.hidden_size,
                    head_dim = config.head_dim,
                    num_q_heads = config.num_q_heads,
                    num_kv_heads = config.num_kv_heads,
                    sliding = config.is_sliding(idx),
                    sliding_window = config.sliding_window,
                    is_loop = is_loop,
                    loop_j = idx - config.loop_start,
                    loop_start = config.loop_start,
                    loop_end = config.loop_end,
                    num_pre_loop_blocks = config.num_pre_loop_blocks,
                    gate_rank = config.loop_gate_rank,
                    intermediate_size = config.intermediate_size,
                    num_experts = config.num_experts,
                    num_experts_per_tok = config.num_experts_per_tok,
                    route_scale = config.route_scale,
                    gate_softcap = config.mlp_gate_softcap,
                    up_softcap = config.mlp_up_softcap,
                    rms_norm_eps = config.rms_norm_eps,
                    rel_extent = config.rel_extent,
                    d_rel = config.d_rel,
                    qk_threshold = config.qk_l2_clip_threshold,
                    rel_threshold = config.rel_l2_clip_threshold,
                    log_alpha = config.log_scaling_alpha,
                    log_floor = config.log_scaling_n_floor,
                    conv_kernel_size = config.conv_kernel_size,
                    n_layers_per_block = config.n_layers_per_block,
                )
            ]

        self.last_kv_module_idx = len(self.modules) - 1

        self.modules += [
            InklingOutputRead(
                config = config,
                key = p + "attn_res.output_query",
                hidden_size = config.hidden_size,
                num_queries = 1,
                eps = config.rms_norm_eps,
            ),
            RMSNorm(
                config = config,
                key = p + "norm",
                rms_norm_eps = config.rms_norm_eps,
                out_dtype = torch.half,
            ),
            Linear(
                config = config,
                key = p + "unembed",
                in_features = config.hidden_size,
                out_features = config.vocab_size,
                qmap = "block",
                softcap = config.final_logit_softcapping,
                caps = {"logits_output": True},
            ),
        ]

        self.logit_layer_idx = len(self.modules) - 1

        self.caps.update({
            "recurrent_states": True,
            "default_recurrent_checkpoint_interval": 2048,
        })
        self.recurrent_state_cls = ShortConvState

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        prepare_for_recurrence(input_ids, params, self)
        input_ids = prepare_for_attn(input_ids, params)
        params[KEY_FLOW] = {
            "bank": [],
            "pre_bank": None,
            "pre_hidden": None,
            "prev_bank": None,
            "prev_hidden": None,
        }
        return input_ids