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

    # ---- inner-class lane (engine wiring surface) -----------------------------
    from exllamav3.modules.quant.nvfp4_linear import LinearNVFP4Packed, pad_triple

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pkd, scd = pk.to(dev), sc.to(dev)
    bias = (torch.randn(N, device=dev) * 0.1).half()
    lin = LinearNVFP4Packed(K, N, pkd, scd, gs, bias, key="blk.mlp.test")

    # ---- 6: inner.forward numerics + shape/dtype handling ---------------------
    ref_dec = fmt.decode_weights(pkd, scd, gs, torch.float32)
    x3 = (torch.randn(2, 3, K, device=dev) * 0.5)
    y = lin.forward(x3.half(), {})
    ref = x3.float() @ ref_dec.t() + bias.float()
    ok, err, den = rel_ok(y, ref)
    ok = ok and y.shape == (2, 3, N) and y.dtype is torch.float16
    y_f = lin.forward(x3.half(), {}, out_dtype=torch.float32)
    ok = ok and y_f.dtype is torch.float32
    step("inner_forward", ok, f"3-D x + bias + out_dtype vs oracle (den={den:.1f})")

    # ---- 7: engine duck-type surface ------------------------------------------
    surf = all(hasattr(lin, a) for a in
               ("quant_type", "forward", "unload", "get_tensors", "get_weight_tensor",
                "get_bias_tensor", "swap_cpu", "unswap_cpu", "bc", "bias",
                "in_features", "out_features"))
    ok = surf and lin.quant_type == "nvfp4"
    wt = lin.get_weight_tensor()
    ok = ok and wt.shape == (K, N) and torch.equal(lin.get_bias_tensor(), bias)
    ts = lin.get_tensors("blk.mlp.test")
    ok = ok and ("blk.mlp.test.weight" in ts
                 and ts["blk.mlp.test.weight"].shape == (N, K))
    lin.unload()
    step("inner_surface", ok, "engine .inner.* consumers find every attribute they call")

    # ---- 8: swap_cpu / unswap_cpu round trip ----------------------------------
    dev_before = pkd.device
    lin.swap_cpu()
    mid_ok = lin.packed.device.type == "cpu"
    lin.unswap_cpu()
    ok = mid_ok and lin.packed.device == dev_before and lin.scales.device == dev_before
    if ok:
        y2 = lin.forward(x3.half(), {})
        ok, err, den = rel_ok(y2, ref)
    step("inner_swap_roundtrip", ok, "cpu park + VRAM restore keeps outputs on-oracle")

    # ---- 9: packed-space padding helper ----------------------------------------
    rp, cp = N + 32, K + 64
    pk_p, sc_p = pad_triple(pkd, scd, rp, cp)
    ok = pk_p.shape == (rp, cp // 2) and sc_p.shape == (rp, cp // 16)
    dec_p = fmt.decode_weights(pk_p, sc_p, gs, torch.float32)
    ok = (ok and torch.equal(dec_p[:N, :K], ref_dec)
          and dec_p[N:, :].abs().sum() == 0 and dec_p[:N, K:].abs().sum() == 0)
    lin_p = LinearNVFP4Packed(cp, rp, pk_p, sc_p, gs, None)
    xp2 = torch.nn.functional.pad(x3.half(), (0, cp - K))
    ypl = lin_p.forward(xp2, {})
    ref_p = torch.nn.functional.pad(x3.float() @ ref_dec.t(), (0, rp - N))
    okr, _, _ = rel_ok(ypl, ref_p)
    ok = ok and okr
    step("pad_triple", ok, f"packed-space pad to ({rp},{cp}): exact head + zeros tail")

    nsteps = 9
    npass = nsteps if not fails else nsteps - len(fails)
    print(f"W4A16-STEPS steps={nsteps} pass={npass} fail={len(fails)} "
          f"{'DONE-PASS' if not fails else 'DONE-FAIL ' + ','.join(fails)}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())