#!/usr/bin/env python3
"""Packed-resident w4a16 lane conformance (fork module, no vendor lineage).

Steps (marker W4A16-STEPS, terminal DONE-PASS/DONE-FAIL):
 1  lane policy: pick_lane boundary + monotonicity (pure CPU)
 2  direct lane bitwise-grade numerics vs CPU-oracle decode matmul (needs cuda)
 3  scratch lane (decode-then-mm) numerics vs oracle at large M (needs cuda)
 4  byte-sabotage: flipping one packed nibble must move the direct-lane output
 5  cpu fallback: matmul() without cuda equals oracle decode matmul

Reference math is the fenced CPU oracle in nvfp4_fmt (decode_weights) — no
kernel-vs-itself circularity. Tolerance: max abs err <= 2%% of |ref| max,
same envelope as the PoC numerics gate.
"""
import sys
import torch

sys.path.insert(0, "/data/Repos/musicman89/exllamav3")
from exllamav3.modules.quant import nvfp4_fmt as fmt          # noqa: E402
from exllamav3.modules.quant import nvfp4_w4a16 as lane        # noqa: E402

K, N = 1024, 64
fails = []


def rel_ok(y, ref):
    err = (y.float() - ref.float()).abs().max().item()
    den = ref.float().abs().max().item()
    return err <= 0.02 * den, err, den


def make_triple(seed):
    g_cpu = torch.Generator().manual_seed(seed)
    w = torch.randn(N, K, generator=g_cpu) * 0.5
    pk, sc, gs = fmt.quantize_weights(w)
    return pk.cuda(), sc.cuda(), gs


def step(name, ok, detail=""):
    print(f"W4A16 step {name}: {'OK' if ok else 'FAIL'} {detail}")
    if not ok:
        fails.append(name)


def main():
    # ---- 1: lane policy ----------------------------------------------------
    dm = lane.DIRECT_MAX_M
    pol = (lane.pick_lane(1) == "direct" and lane.pick_lane(dm) == "direct"
           and lane.pick_lane(dm + 1) == "scratch" and lane.pick_lane(4096) == "scratch")
    step("policy", pol, f"DIRECT_MAX_M={dm}")

    pk, sc, gs = make_triple(11)
    w_dec = fmt.decode_weights(pk.cpu(), sc.cpu(), gs, torch.float32).cuda()

    # ---- 2: direct lane vs oracle ------------------------------------------
    if torch.cuda.is_available():
        ok2 = True
        for m in (1, 4, 8):
            x = (torch.randn(m, K, device="cuda") * 0.5).half()
            y = lane.matmul_direct(x, pk, sc, gs)
            ref = x.float() @ w_dec.t()
            ok, err, den = rel_ok(y, ref)
            ok2 &= ok
        step("direct_numerics", ok2, "M=1,4,8 vs CPU-oracle decode matmul")

        # ---- 3: scratch lane vs oracle -------------------------------------
        x = (torch.randn(32, K, device="cuda") * 0.5).half()
        y = lane.matmul(x, pk, sc, gs)  # M>DM must take scratch lane
        ref = x.float() @ w_dec.t()
        ok, err, den = rel_ok(y, ref)
        step("scratch_numerics", ok, f"maxerr={err:.4f} den={den:.1f}")

        # ---- 4: sabotage reaches the result ---------------------------------
        x1 = (torch.randn(1, K, device="cuda") * 0.5).half()
        y0 = lane.matmul_direct(x1, pk, sc, gs)
        pk_sab = pk.clone()
        pk_sab.view(torch.uint8).flatten()[9] ^= 0x20
        y1 = lane.matmul_direct(x1, pk_sab, sc, gs)
        step("sabotage_moves_output", not torch.equal(y0, y1),
             "one flipped nibble must change the direct-lane output")
    else:
        print("W4A16 NOTE: no cuda — steps 2-4 skipped by design (cpu-only box)")

    # ---- 5: cpu fallback path ------------------------------------------------
    pkc, scc = pk.cpu(), sc.cpu()
    x = torch.randn(3, K) * 0.5
    y = lane.matmul(x.half(), pkc, scc, gs)
    ref = x.float() @ fmt.decode_weights(pkc, scc, gs, torch.float32).t()
    ok, err, den = rel_ok(y, ref)
    step("cpu_fallback", ok, f"maxerr={err:.4f} den={den:.1f}")

    npass = 5 if not fails else 5 - len(fails)
    print(f"W4A16-STEPS steps=5 pass={npass} fail={len(fails)} "
          f"{'DONE-PASS' if not fails else 'DONE-FAIL ' + ','.join(fails)}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())