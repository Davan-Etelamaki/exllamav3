"""
CPU numerical simulation for fence 1 of the sm89 MoE grouped-dispatch lane (dsh-moe, 2026-11-25).

Two implementations of the same MoE expert dispatch, compared on identical dequant operands:

- reference_dispatch: the CURRENT python dispatch loop of Inkling2LoopSparseMoeBlock.forward,
  moved verbatim (comments stripped) from exllamav3/modules/arch_specific/inkling2_loop.py:1109-1160
  into this CPU simulation. host_values(...) becomes .tolist(); stage_host(...) becomes a plain CPU
  tensor; self.experts_{g,u,d}[e].forward(xs, params) becomes a Linear-equivalent matmul against a
  FakeExpert's stored dequant operands (gate/up round their output to half exactly as
  out_dtype = torch.half does; down takes half input and returns float exactly as out_dtype =
  torch.float does).
- grouped_dispatch: the numerical shape of the phase-1 grouped kernel: device-side expert
  segmentation (count -> pad each segment to a block multiple -> one expert id per kernel block,
  the vLLM fused_moe.py:69-162 shape), a POINTER TABLE built at load (table[e] -> weight bundle,
  never a stacked weight copy), a fused swiglu-softcap epilogue, and a weighted accumulate into the
  output in segment-descending order — deliberately a different summation order than the reference's
  ascending expert groups, so the comparison exercises fp accumulation-order sensitivity.

The down-projection leg is the one mathematical simplification: the real kernel stages the
intermediate through an fp16 scratch between its two grids, which is bit-identical to the
reference's act.to(torch.half) before experts_d — so both legs here share the same epilogue and
the same operands. What differs is ONLY the dispatch structure and accumulation order, which is
what fence 1 certifies.

No exllamav3 import here: importing the package initializes CUDA (workstation law: the prod engine
holds that GPU), so this module and its test run on CPU torch only.
"""

import torch

# Fake geometry: expert-count and top_k match the served model (dot26 config 2026-11-25);
# hidden/intermediate are shrunk for CPU speed. Caps/route_scale mirror the in-file defaults
# (gate cap 4, up cap 25, docstring at inkling2_loop.py:1041-1046).
NUM_EXPERTS = 128
TOP_K = 8
GATE_SOFTCAP = 4.0
UP_SOFTCAP = 25.0
ROUTE_SCALE = 1.0


class FakeExpert:
    """One routed expert's three INDEPENDENT quantized Linear modules (per-expert qgroups),
    stood in by their dequantized fp16 operand matrices. Storage layout matches the Linear ctors
    at inkling2_loop.py:1072-1080: gate/up are (intermediate x hidden), down is (hidden x intermediate)."""

    def __init__(self, w_g, w_u, w_d):
        self.w_g = w_g  # half (intermediate, hidden)
        self.w_u = w_u  # half (intermediate, hidden)
        self.w_d = w_d  # half (hidden, intermediate)

    def __repr__(self):
        return f"FakeExpert(id={self.expert_id})"


def gu_forward(expert, xs, which):
    """experts_g[e]/experts_u[e].forward(xs, params).float() — fp32-accumulate GEMV against dequant
    fp16 operands, output rounded to out_dtype = half, then widened to float by the caller line."""
    w = expert.w_g if which == "g" else expert.w_u
    return (xs.float() @ w.float().T).half().float()


def down_forward(expert, act_half):
    """experts_d[e].forward(act.to(torch.half), params) — input half, output fp32 (out_dtype float)."""
    return act_half.float() @ expert.w_d.float().T


def act_epilogue(g, u):
    """Fused swiglu-softcap epilogue, verbatim from Inkling2LoopSparseMoeBlock.forward
    (bound by the SERVED_ANCHORS strings in test_moe_grouped_dispatch; bare line numbers
    decay with edits)."""
    act = torch.tanh(g / GATE_SOFTCAP) * GATE_SOFTCAP
    act = act * torch.sigmoid(g)
    act = act * (torch.tanh(u / UP_SOFTCAP) * UP_SOFTCAP)
    return act


def route(hs, router_w, router_bias, top_k = TOP_K, route_scale = ROUTE_SCALE):
    """Router, verbatim from inkling2_loop.py:1112-1116.

    SPEC (supervisor charge C, 2026-11-25): where biased scores tie, the DEVICE rule is
    authoritative — lowest expert id wins (moe_route.cu's butterfly picks min index among
    equal maxima). torch.topk's tie order is unspecified, so this fence asserts the case
    is TIE-FREE: selection membership is then deterministic and a parity failure can
    never be an unspecified torch tie posing as one.
    """
    scores = torch.sigmoid(torch.matmul(hs, router_w.T))
    biased = scores + router_bias
    srt = biased.sort(dim = -1).values
    tied = (srt[:, 1:] == srt[:, :-1]).any(dim = -1)
    if bool(tied.any()):
        rows = tied.nonzero().flatten().tolist()[:8]
        raise AssertionError(f"malformed fence case: tied biased scores at token rows {rows}; "
                             "device tie-break (lowest expert id) is the spec")
    sel = torch.topk(biased, top_k, dim = -1, sorted = False).indices
    w = scores.gather(1, sel)
    w = w / (w.sum(dim = -1, keepdim = True) + 1e-20)
    w = w * route_scale
    return sel, w


def reference_dispatch(flat, sel_flat, w_flat, table):
    """The current python dispatch loop, verbatim from inkling2_loop.py:1131-1160 modulo the
    substitutions named in the module docstring. This IS the fence's reference implementation."""
    top_k = TOP_K
    sel_cpu = sel_flat.tolist()                      # host_values(sel_flat, (...)).tolist()
    order = sorted(range(len(sel_cpu)), key = lambda i: sel_cpu[i])
    out = torch.zeros((flat.shape[0], flat.shape[1]), dtype = torch.float)
    n = 0
    while n < len(order):
        e = sel_cpu[order[n]]
        m = n
        while m < len(order) and sel_cpu[order[m]] == e:
            m += 1
        entries = torch.tensor(order[n:m], dtype = torch.long)   # stage_host(order[n:m], ...)
        tokens = torch.div(entries, top_k, rounding_mode = "floor")
        n = m
        xs = flat.index_select(0, tokens)
        g = gu_forward(table[e], xs, "g")
        u = gu_forward(table[e], xs, "u")
        act = act_epilogue(g, u)
        y = down_forward(table[e], act.to(torch.half))
        out.index_add_(0, tokens, y.float() * w_flat.index_select(0, entries).unsqueeze(-1))
    return out


def build_segments(sel_flat, num_experts, block_multiple):
    """Device-side segmentation in the vLLM fused_moe.py:69-162 shape, computed on CPU here.
    Returns (sorted_token_ids, per_block_expert_ids, num_blocks); invalid padding slots carry the
    sentinel num_entries (= out-of-range entry id), exactly like vLLM's num_valid_tokens masking."""
    num_entries = sel_flat.numel()
    sort_by_expert = torch.argsort(sel_flat, stable = True)       # histogram+scatter on device
    sel_sorted = sel_flat[sort_by_expert]
    counts = torch.bincount(sel_sorted, minlength = num_experts)
    blocks_per_e = (counts + block_multiple - 1) // block_multiple
    padded_counts = blocks_per_e * block_multiple
    seg_start = torch.cumsum(padded_counts, dim = 0) - padded_counts
    num_blocks = int(blocks_per_e.sum())
    sorted_token_ids = torch.full((num_blocks * block_multiple,), num_entries, dtype = torch.long)
    expert_for_block = torch.repeat_interleave(torch.arange(num_experts, dtype = torch.long), blocks_per_e)
    for e in range(num_experts):
        k = int(counts[e])
        if k:
            sorted_token_ids[int(seg_start[e]):int(seg_start[e]) + k] = sort_by_expert[:k]
            sort_by_expert = sort_by_expert[k:]
    return sorted_token_ids, expert_for_block, num_blocks


def grouped_dispatch(flat, sel_flat, w_flat, table, block_multiple = 4):
    """Phase-1 numerical shape: one pass over kernel blocks addressed through the pointer table;
    blocks are processed in DESCENDING order (the kernel's atomic weighted accumulate races the
    reference's ascending index_add_), padding slots are masked by the sentinel."""
    top_k = TOP_K
    num_entries = sel_flat.numel()
    out = torch.zeros((flat.shape[0], flat.shape[1]), dtype = torch.float)
    sorted_token_ids, expert_for_block, num_blocks = build_segments(
        sel_flat, len(table), block_multiple)
    for blk in reversed(range(num_blocks)):
        e = int(expert_for_block[blk])
        lo = blk * block_multiple
        window = sorted_token_ids[lo:lo + block_multiple]
        entries = window[window < num_entries]        # token_mask: offs_token < num_valid_tokens
        if entries.numel() == 0:
            continue
        tokens = torch.div(entries, top_k, rounding_mode = "floor")
        xs = flat.index_select(0, tokens)
        g = gu_forward(table[e], xs, "g")             # <-- pointer-table deference
        u = gu_forward(table[e], xs, "u")
        act = act_epilogue(g, u)                      # fused epilogue
        y = down_forward(table[e], act.to(torch.half))
        contrib = y.float() * w_flat.index_select(0, entries).unsqueeze(-1)
        out.index_add_(0, tokens, contrib)            # atomic weighted accumulate
    return out


def compare_outputs(ref, got, sel, w_flat, atol, rtol):
    """Returns (ok, info). info names the worst token, its routed experts, and every routed expert
    whose contribution carries a tolerance breach, so a RED line is attributable."""
    diff = (ref - got).abs()
    tol = atol + rtol * ref.abs()
    breach = diff > tol
    if not bool(breach.any()):
        return True, ""
    exceed = (diff / tol).max()
    worst = int(diff.argmax())
    worst_token = worst // got.shape[1]
    routed = sorted(set(sel[worst_token].tolist()))
    bad_tokens = breach.any(dim = 1).nonzero().flatten().tolist()
    implicated = sorted({int(e) for tk in bad_tokens for e in sel[tk].tolist()})
    info = (f"max rel exceedance {exceed:.3e}; worst token {worst_token} routed experts {routed}; "
            f"{len(bad_tokens)} tokens over tolerance; breached rows route through experts {implicated}")
    return False, info


def make_experts(gen, num_experts, hidden, inter):
    table = []
    for _ in range(num_experts):
        g = torch.randn(inter, hidden, generator = gen).mul_(0.35).half()
        u = torch.randn(inter, hidden, generator = gen).mul_(0.35).half()
        d = torch.randn(hidden, inter, generator = gen).mul_(0.35).half()
        table.append(FakeExpert(g, u, d))
    for e, ex in enumerate(table):
        ex.expert_id = e
    return table


def make_case(seed, tokens, hidden = 48, inter = 96, num_experts = NUM_EXPERTS, perm = None):
    """Seeded routing case. perm (a permutation of expert indices) SHUFFLES THE EXPERT SET: weight
    bundle e is moved to perm[e] and the router bias is permuted with it, so the chosen set and the
    per-expert operands both change between shuffle rounds."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(tokens, hidden, generator = gen).half()
    router_w = torch.randn(num_experts, hidden, generator = gen).mul_(0.2)
    router_bias = torch.randn(num_experts, generator = gen).mul_(0.5)
    table = make_experts(gen, num_experts, hidden, inter)
    if perm is not None:
        shuffled = [table[int(perm[e])] for e in range(num_experts)]
        table = shuffled
        router_bias = router_bias[perm.long()]
    sel, w = route(x.float(), router_w, router_bias)
    return dict(x = x, sel = sel, w = w, table = table)