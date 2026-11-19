"""_RoutingGraph must survive the disk tier's event waits (prod 2026-11-19).

Fault: `_RoutingGraph.__init__` captured with torch's DEFAULT capture_error_mode, which is
cudaStreamCaptureModeGlobal. Under a global capture, EVERY other thread is forbidden from touching
CUDA events. exllamav3's disk tier writer thread blocks on exactly such an event
(generator/disk_cache.py `_do_write_page` -> `ev.synchronize()`), so the two collide:

  * the writer gets back 900 CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, and upstream's handler spent
    the whole L3 tier after 3 such pages ("writes disabled for the rest of this session");
  * our own capture was invalidated, so `_rgc[key] = False` stuck and MoE routing silently
    returned to ~8 eager launches per layer per step.

Fix: capture ThreadLocal, the mode exllamav3_ext/graph.cu:63 already uses. Measured on dot26 with
Launchers/repro_capture_event_sync_2026-11-19.py: global -> CAPTURE=INVALIDATED WAIT=CONFLICT;
thread_local -> CAPTURE=ok WAIT=ok; relaxed -> CAPTURE=ok WAIT=ok.

Run standalone (no pytest needed on the GPU host):
    python tests/modules/test_routing_graph_capture.py       # exit 0 = both arms pass
Also pytest-collectable: pytest tests/modules/test_routing_graph_capture.py -s
RED-arm proof: run it against the pre-fix build -- capture_failures must be > 0 and the racer must
report conflicts.
"""

import threading
import time

import torch

NUM_TOKENS, TOP_K, E = 8, 4, 16
A_RT = NUM_TOKENS * TOP_K
RACE_SECONDS = 1.0
CONSTRUCTIONS = 6


def _require_gpu():
    if not torch.cuda.is_available():
        print("NO-CUDA: run this on a GPU host (dot26)")
        raise SystemExit(2)
    torch.cuda.init()


def _import_routing_graph():
    from exllamav3.modules.block_sparse_mlp import _RoutingGraph
    print(f"UNDER TEST {_RoutingGraph.__module__}: {torch.cuda.get_device_name(0)}")
    import exllamav3.modules.block_sparse_mlp as m
    print(f"MODULE {m.__file__}")
    return _RoutingGraph


def _make_inputs(device):
    gen = torch.Generator(device="cpu").manual_seed(20261119)
    sel = torch.randint(0, E, (A_RT,), generator=gen, dtype=torch.int64).to(device)
    w = torch.rand(A_RT, generator=gen, dtype=torch.float32).to(device)
    return sel, w


def _event_racer(box, stop):
    """Stands in for the disk tier writer thread: records an event and blocks on it, continuously"""
    a = torch.zeros(64, device="cuda")
    conflicts = 0
    syncs = 0
    while not stop.is_set():
        ev = torch.cuda.Event()
        a.add_(1.0, non_blocking=True)
        ev.record()
        try:
            ev.synchronize()
        except Exception as e:  # noqa: BLE001 - the error text is the measurement
            conflicts += 1
            box.setdefault("first_conflict", f"{type(e).__name__}: {str(e).splitlines()[0]}")
        syncs += 1
        if syncs % 64 == 0:
            torch.cuda.synchronize()      # keeps the CUDA context tidy and the loop honest
            box["syncs"] = syncs
            box["conflicts"] = conflicts
    box["syncs"] = syncs
    box["conflicts"] = conflicts


def test_routing_graph_capture_survives_concurrent_event_waits():
    """Build the graph while another thread blocks on CUDA events, the way the disk tier does"""
    _require_gpu()
    _RoutingGraph = _import_routing_graph()
    device = torch.device("cuda")
    sel, w = _make_inputs(device)

    box = {}
    failures = []
    for i in range(CONSTRUCTIONS):
        stop = threading.Event()
        racer = threading.Thread(target=_event_racer, args=(box, stop), daemon=True)
        racer.start()
        deadline = time.monotonic() + RACE_SECONDS
        try:
            rg = _RoutingGraph(sel, w, NUM_TOKENS, TOP_K, E, None, device)
            ok = True
        except Exception as e:  # noqa: BLE001
            rg, ok = None, False
            failures.append(f"construction {i}: {type(e).__name__}: {str(e).splitlines()[0]}")
        while time.monotonic() < deadline:
            pass
        stop.set()
        racer.join(5.0)
        if rg is not None:
            rg.g.replay()
            torch.cuda.synchronize()
        print(f"ARM construction-{i} capture_ok={ok} racer_syncs={box.get('syncs')} "
              f"racer_conflicts={box.get('conflicts')}")
        if box.get("first_conflict"):
            print(f"    racer detail: {box['first_conflict']}")
        box.pop("first_conflict", None)

    print(f"VERDICT capture_failures={len(failures)}/{CONSTRUCTIONS} "
          f"racer_conflicts_total={box.get('conflicts', 'n/a')}")
    assert not failures, ("routing-graph capture died under a concurrent event wait: "
                          + "; ".join(failures))


def test_routing_graph_replay_matches_eager_prelude():
    """The captured prelude must return what the same code returns eager (graph correctness)"""
    _require_gpu()
    _RoutingGraph = _import_routing_graph()
    device = torch.device("cuda")
    sel, w = _make_inputs(device)
    rg = _RoutingGraph(sel, w, NUM_TOKENS, TOP_K, E, None, device)

    out_sel, out_tok, out_w, out_cnt, out_inv, out_tables = rg.run(sel, w)
    torch.cuda.synchronize()

    order = sel.argsort()
    tok = order.div(TOP_K, rounding_mode="floor")
    wt = w[order]
    cnt = torch.zeros(E + 1, dtype=torch.int64, device=device)
    cnt.scatter_add_(0, sel, torch.ones_like(sel))
    start = torch.cumsum(cnt, 0) - cnt
    tables = torch.stack([start, start, (cnt > 0).long()])
    inv = torch.empty_like(order).scatter_(0, order, torch.arange(A_RT, device=device))

    torch.testing.assert_close(out_sel, sel)
    torch.testing.assert_close(out_tok, tok)
    torch.testing.assert_close(out_w, wt)
    torch.testing.assert_close(out_cnt, cnt)
    torch.testing.assert_close(out_inv, inv)
    torch.testing.assert_close(out_tables, tables)
    print("VERDICT replay==eager ok")


if __name__ == "__main__":
    rc = 0
    for fn in (test_routing_graph_capture_survives_concurrent_event_waits,
               test_routing_graph_replay_matches_eager_prelude):
        try:
            fn()
        except SystemExit as e:
            raise
        except Exception as e:  # noqa: BLE001
            rc = 1
            print(f"FAILED {fn.__name__}: {e}")
    print("EXIT", rc)
    raise SystemExit(rc)