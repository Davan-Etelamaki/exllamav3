"""Fence for the route-graph on CPU-split MoE modules (cut 8).

Live py-spy (prod, 2026-10-25): block_sparse_mlp.py:1253 bincount held 20.06% of
decode samples while the tolist sync at 1257 was near-absent -> fused no-sync mode
is on but graph_route is False every layer: the CPU-split model failed the
`cpu_split_first is None` gate. The gate is now removed; this fence proves the
graph path produces EXACTLY the eager tensors and that replay consumes fresh
inputs (positive control per law 5).
"""
import os
import time

import pytest
import torch


def _eager(flat_expert_global, flat_weight, num_tokens, top_k, E, first):
    # Mirrors block_sparse_mlp.forward eager branch (lines 1236-1253)
    if first is None or first == 0 and flat_expert_global.max() < E:
        local = flat_expert_global
    else:
        local = flat_expert_global - first
        valid = (local >= 0) & (local < E)
        local = torch.where(valid, local, torch.full_like(local, E))
    order = local.argsort()
    from exllamav3.modules.block_sparse_mlp import buffered_interleaved_arange
    flat_token = buffered_interleaved_arange(num_tokens, top_k, device = local.device)
    token_sorted = flat_token[order]
    weight_sorted = flat_weight[order]
    expert_count = torch.bincount(local, minlength = E + 1)
    return local, token_sorted, weight_sorted, expert_count


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA")
@pytest.mark.parametrize("first", [None, 4])
def test_route_graph_matches_eager_and_replays_fresh(first):
    from exllamav3.modules.block_sparse_mlp import _RoutingGraph
    dev = torch.device("cuda")
    torch.manual_seed(7)
    num_tokens, top_k, E = 8, 4, 16
    if first is None:
        sel = torch.randint(0, E, (num_tokens, top_k), device = dev, dtype = torch.int64)
    else:
        # CPU-split shape: globals span [first, first+E) plus out-of-slice picks
        sel = (torch.randint(0, E + 4, (num_tokens, top_k), device = dev, dtype = torch.int64) + first - 2).clamp_min(0)
    w = torch.rand((num_tokens, top_k), device = dev, dtype = torch.float32)

    rg = _RoutingGraph(sel.reshape(-1), w.reshape(-1), num_tokens, top_k, E, first, dev)

    # Fresh inputs AFTER capture: replay must reflect THESE, not capture-time data
    sel2 = sel.clone()
    sel2[0, 0] = (first if first is not None else 0)  # in-slice value
    w2 = w.clone()
    w2[0, 0] = 0.123456
    local_g, ts_g, ws_g, ec_g, inv_g, tab_g = rg.run(sel2.reshape(-1), w2.reshape(-1))
    local_e, ts_e, ws_e, ec_e = _eager(sel2.reshape(-1), w2.reshape(-1), num_tokens, top_k, E, first)

    assert torch.equal(local_g, local_e)
    assert torch.equal(ts_g, ts_e)
    assert torch.equal(ws_g, ws_e)
    assert torch.equal(ec_g[:E + 1], ec_e)
    A = num_tokens * top_k
    order = local_e.argsort()
    assert torch.equal(inv_g[order], torch.arange(A, device = dev, dtype = inv_g.dtype))
    start = torch.cumsum(ec_e, 0) - ec_e
    assert torch.equal(tab_g[0], start.to(tab_g.dtype))
    assert torch.equal(tab_g[2], (ec_e > 0).long().to(tab_g.dtype))

    # Positive control (law 5): perturbed input must change outputs -> replay is live
    ws_snap = ws_g.clone()  # graph outputs are STATIC buffers; snapshot before next replay
    w3 = w2.clone()
    w3[0, :] = w3[0, :] + 10.0  # deterministic: shifted row must reorder sorted weights
    _, _, ws_g3, _, _, _ = rg.run(sel2.reshape(-1), w3.reshape(-1))
    assert not torch.equal(ws_g3, ws_snap)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA")
def test_graph_replay_beats_eager_launch_cost():
    from exllamav3.modules.block_sparse_mlp import _RoutingGraph, buffered_interleaved_arange
    dev = torch.device("cuda")
    num_tokens, top_k, E = 32, 10, 512
    sel = torch.randint(0, E, (num_tokens, top_k), device = dev, dtype = torch.int64)
    w = torch.rand((num_tokens, top_k), device = dev, dtype = torch.float32)
    rg = _RoutingGraph(sel.reshape(-1), w.reshape(-1), num_tokens, top_k, E, None, dev)

    def eager():
        local = sel.reshape(-1)
        order = local.argsort()
        ft = buffered_interleaved_arange(num_tokens, top_k, device = dev)
        _ = ft[order]
        _ = w.reshape(-1)[order]
        _ = torch.bincount(local, minlength = E + 1)

    for _ in range(20):
        eager(); rg.run(sel.reshape(-1), w.reshape(-1))
    torch.cuda.synchronize()
    N = 200
    t0 = time.perf_counter()
    for _ in range(N):
        eager()
    torch.cuda.synchronize()
    t_eager = (time.perf_counter() - t0) / N
    t0 = time.perf_counter()
    for _ in range(N):
        rg.run(sel.reshape(-1), w.reshape(-1))
    torch.cuda.synchronize()
    t_graph = (time.perf_counter() - t0) / N
    print(f"route-glue us/step eager={t_eager*1e6:.1f} graph={t_graph*1e6:.1f}", flush = True)
    assert t_graph < t_eager  # regime: shared-GPU workstation sm_120