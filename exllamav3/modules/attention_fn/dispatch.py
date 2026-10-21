import torch
from ...cache import CacheLayer, Cache, CacheLayer_quant, CacheLayer_fp8, CacheLayer_nvfp4
from .common import AttnArgs, AttnFn
from .bighead_scalar import fn_bighead_scalar_attn
from .torch import fn_torch_sdpa_fallback_cache, fn_torch_sdpa_fallback_nocache
from .xformers import fn_xformers_cutlass_fallback_cache, fn_xformers_cutlass_fallback_nocache
from .triton_paged import (
    _qc_staging,
    fn_triton_paged_attn,
    fn_triton_paged_attn_longq,
    fn_triton_paged_attn_decode,
    fn_triton_paged_attn_prefill,
    fn_triton_varlen_attn,
    fn_triton_paged_attn_decode_qc,
    fn_triton_paged_attn_prefill_qc,
    fn_triton_attn_nocache,
)

# Candidate attn functions in order of preference: the Triton decode/prefill/varlen kernels
# serve every shape they support (any head_dim <= 512, zero-padded to a power of two), then the
# older Triton kernels and the torch/xformers fallbacks for the rest
_fns_triton_fast: list[AttnFn] = [
    fn_triton_paged_attn_decode,
    fn_triton_paged_attn_prefill,
    fn_triton_varlen_attn,
]

# Quant-direct calls carry the packed cache in q_cache and leave k_cache/v_cache as None, which makes them
# indistinguishable from cache-less attention to any backend that only checks has_kv_cache(). Such a backend
# would silently attend over just the new K/V rows and ignore the cached context, so quant-direct calls only
# ever dispatch over the qc-aware functions
_fns_qc: list[AttnFn] = [
    fn_triton_paged_attn_decode_qc,
    fn_triton_paged_attn_prefill_qc,
]

# NVFP4/FP8 cache layers hand their raw paged tensors (packed uint8 + E4M3 scales for
# nvfp4, plain fp8 pages for fp8) to the general Triton paged kernels, which append the
# new K/V in-kernel and dequantize online. Other backends only ever see these modes
# through full dequantized fp16 copies, so such calls dispatch over these two only
_fns_fp8: list[AttnFn] = [
    fn_triton_paged_attn,
    fn_triton_paged_attn_longq,
]

# Quantized caches feed the attention kernels directly (online dequant or prefill staging by
# EXL3_QC_STAGING level, see triton_paged); level 2 restores the dequantize-then-attend path
# with full-size fp16 temporaries for A/B testing
_qc_attn = _qc_staging < 2

attn_fns: list[AttnFn] = _fns_triton_fast + [
    fn_triton_attn_nocache,
    fn_triton_paged_attn,
    fn_triton_paged_attn_longq,
    fn_bighead_scalar_attn,
    fn_xformers_cutlass_fallback_cache,
    fn_xformers_cutlass_fallback_nocache,
    fn_torch_sdpa_fallback_cache,
    fn_torch_sdpa_fallback_nocache
]

def _tensor_desc(t: torch.Tensor | None) -> str:
    if t is None:
        return "None"
    return f"shape={tuple(t.shape)} dtype={t.dtype} device={t.device} contiguous={t.is_contiguous()}"


def _print_no_attn_match_report(args: AttnArgs):
    tried = ", ".join(fn.__name__ for fn in attn_fns)
    print(
        "No matching attention function found.\n"
        f"  shape: bsz={args.bsz} q_len={args.q_len} kv_len={args.kv_len} "
        f"q_heads={args.num_q_heads} kv_heads={args.num_kv_heads} dim={args.dim}\n"
        f"  flags: cache={args.has_kv_cache()} varlen={args.is_varlen()} gqa={args.is_gqa()} "
        f"causal={args.causal} window_size={args.window_size} softcap={args.softcap} "
        f"non_causal_spans={args.non_causal_spans is not None} sinks={args.sinks is not None}\n"
        f"  scale: sm_scale={args.sm_scale} max_seqlen={args.max_seqlen}\n"
        f"  q: {_tensor_desc(args.q)}\n"
        f"  k: {_tensor_desc(args.k)}\n"
        f"  v: {_tensor_desc(args.v)}\n"
        f"  k_cache: {_tensor_desc(args.k_cache)}\n"
        f"  v_cache: {_tensor_desc(args.v_cache)}\n"
        f"  block_table: {_tensor_desc(args.block_table)}\n"
        f"  cache_seqlens: {_tensor_desc(args.cache_seqlens)}\n"
        f"  cu_seqlens: {_tensor_desc(args.cu_seqlens)}\n"
        f"  tried: {tried}"
    )


def attn_dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache: CacheLayer | Cache | None = None,
    cache_idx: int | None = None,
    cache_instance: int | None = None,
    causal: bool = True,
    sm_scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    window_size: int | None = None,
    window_right: int = 0,
    sink_key0: bool = False,
    softcap: float = 0.0,
    block_table: torch.Tensor | None = None,
    cache_seqlens: torch.Tensor | None = None,
    non_causal_spans: list | None = None,
    sinks: torch.Tensor | None = None,
    dispatch_cache: dict | None = None,
    max_kv_len: int | None = None,
    staging_pages: int | None = None,
):
    """
    Select and run the first compatible attention implementation for the supplied tensors.

    The dispatcher builds an AttnArgs description covering regular, varlen and paged-cache attention modes, obtains
    K/V cache tensors when a Cache or CacheLayer is provided, and tries registered attention backends in preference
    order. After a cached attention call, any updated K/V tensors are written back through the same cache interface.
    """
    bsz, q_len, num_q_heads, dim = q.shape
    _, kv_len, num_kv_heads, _ = k.shape

    # Get cache tensors. Quantized layers pass their packed tensors straight to the attention
    # kernels when possible: new K/V are quantized into the cache up front and never
    # materialized as full fp16 cache-sized temporaries
    q_cache = None
    fp8_fast = False
    k_scales = None
    v_scales = None
    if cache is not None:
        assert block_table is not None
        assert cache_seqlens is not None
        layer = cache if isinstance(cache, CacheLayer) else cache.layers[cache_idx, cache_instance or 0]
        if (
            _qc_attn and
            isinstance(layer, CacheLayer_quant) and
            layer.compand_a == 0.0 and
            q.dtype == torch.float16 and
            dim <= 512 and dim % 32 == 0 and   # packed groups of 32; non-pow2 dims run zero-padded
            cu_seqlens is None
        ):
            layer.update_kv_direct(cache_seqlens, block_table, k, v, q_len)
            q_cache = layer.get_qkv()
            k_cache, v_cache = None, None
        elif (
            isinstance(layer, CacheLayer_nvfp4) and
            layer.k is not None and
            q.dtype == torch.float16 and
            cu_seqlens is None
        ):
            # NVFP4 path: new K/V are quantized (E2M1 + E4M3 scale/16) and appended in-kernel
            # by the nvfp4 update kernel; attention reads the packed pages with online dequant
            k_cache, v_cache = layer.k, layer.v
            k_scales, v_scales = layer.ks, layer.vs
            fp8_fast = True
        elif (
            isinstance(layer, CacheLayer_fp8) and
            layer.k is not None and
            q.dtype == torch.float16 and
            cu_seqlens is None
        ):
            # Raw fp8 pages straight into the Triton kernels; new K/V are appended in-kernel
            # (dtype-aware update kernel), same flow as fp16 caches
            k_cache, v_cache = layer.k, layer.v
            fp8_fast = True
        else:
            k_cache, v_cache = layer.get_kv(cache_seqlens, block_table, window_size if window_size is not None else -1)
            if isinstance(layer, CacheLayer_fp8) and k_cache is not None:
                # No Triton fast path: give the generic backends dequantized fp16 copies. They
                # append the new K/V into these temporaries; the post-hook writes fp8 pages back
                k_cache = k_cache.to(torch.half)
                v_cache = v_cache.to(torch.half)
            elif isinstance(layer, CacheLayer_nvfp4) and k_cache is not None:
                # Generous fp16 copies; the post-hook re-quantizes the new K/V into the pages
                k_cache, v_cache = layer.dequant_full()
    else:
        k_cache, v_cache = None, None

    # Defaults
    if sm_scale is None:
        sm_scale = dim ** (-0.5)

    # Dispatch
    args = AttnArgs(
        bsz, q_len, num_q_heads, dim,
        kv_len, num_kv_heads,
        q, k, v,
        k_cache, v_cache,
        causal,
        sm_scale,
        cu_seqlens, max_seqlen,
        window_size,
        softcap,
        block_table, cache_seqlens,
        non_causal_spans,
        q_cache,
        sinks,
        max_kv_len = max_kv_len,
        window_right = window_right,
        sink_key0 = sink_key0,
        staging_pages = staging_pages,
        k_scales = k_scales,
        v_scales = v_scales,
    )
    # Quant-direct calls select among the qc-aware backends only; a separate hint slot keeps a function that
    # won a cache-less or fp16-cache call from being retried on quant-direct arguments (it cannot see q_cache
    # and would accept them as cache-less)
    if fp8_fast:
        candidates = _fns_fp8
        hint_key = "fn_fp8"
    elif q_cache is not None:
        candidates = _fns_qc
        hint_key = "fn_qc"
    else:
        candidates = attn_fns
        hint_key = "fn"

    # Retry the backend that matched last time for this caller before scanning the full list.
    # Candidate functions return None on incompatible arguments, so a stale hint self-corrects
    fn = dispatch_cache.get(hint_key) if dispatch_cache is not None else None
    o = fn(args) if fn is not None else None

    if o is None:
        args.sanity_check()
        for fn in candidates:
            o = fn(args)
            if o is not None:
                break
        else:
            _print_no_attn_match_report(args)
            raise ValueError("No matching attention function")
        if dispatch_cache is not None:
            dispatch_cache[hint_key] = fn

    # Update cache (quant-direct mode wrote the new K/V before the attention call; the
    # nvfp4/fp8 Triton mode appended them in-kernel during it)
    if cache is not None and q_cache is None and not fp8_fast:
        if isinstance(cache, CacheLayer):
            cache.update_kv(cache_seqlens, block_table, k_cache, v_cache, q_len)
        elif isinstance(cache, Cache):
            cache.update_layer(cache_idx, cache_seqlens, block_table, k_cache, v_cache, q_len, cache_instance)

    return o
