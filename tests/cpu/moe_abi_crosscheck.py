"""
ABI crosscheck fence for the moe_group tier (branch dsh-moe) — three independent definitions of
the same layout must agree, or this test names the drift:

  A) the compiled C++ struct (MoEExl3Ref in exllamav3/exllamav3_ext/moe_group/moe_exl3_table.cuh),
     evidenced by the dot26 probe output committed as tests/cpu/moe_abi_probe_output.txt
     (regeneration command is printed by main() and documented in the kernel README);
  B) the ABI expectation literals (#define MOE_REF_EXPECT_* / MOE_* geometry in the headers);
  C) this file's ctypes mirror + the CPU simulation constants (tests/cpu/moe_grouped_sim.py).

Plus: a pointer-table builder from LinearEXL3-shaped mocks (trellis/suh/svh/mcg/mul1 exactly as
exllamav3/modules/quant/exl3.py exposes them), with the SAME species of positive control as
fence 1 — swap two experts' refs and the validator must go RED naming the pair.

Runs CPU-only: torch here touches no CUDA context; exllamav3 is never imported.
"""

import ctypes
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
HDR_DIR = os.path.join(REPO, "exllamav3", "exllamav3_ext", "moe_group")
FIXTURE = os.path.join(HERE, "moe_abi_probe_output.txt")

sys.path.insert(0, HERE)
from moe_grouped_sim import GATE_SOFTCAP, NUM_EXPERTS, ROUTE_SCALE, TOP_K, UP_SOFTCAP  # noqa: E402

try:
    import pytest
    pytestmark = [pytest.mark.nogpu]
except ImportError:
    pytestmark = []

DEFINE_RE = re.compile(r"^#define\s+(MOE_\w+)\s+([-\d.]+)f?\s*$")


def parse_defines(path):
    out = {}
    with open(path, "r", encoding = "utf-8") as f:
        for line in f:
            m = DEFINE_RE.match(line.strip())
            if m:
                raw = m.group(2)
                out[m.group(1)] = float(raw) if "." in raw else int(raw)
    return out


def parse_fixture(path):
    lines = {}
    with open(path, "r", encoding = "utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "offsetof":
                lines[f"offsetof {parts[1]}"] = (int(parts[2]), int(parts[4]))
            elif parts[0] == "sizeof":
                lines[f"sizeof {parts[1]}"] = (int(parts[2]), int(parts[4]))
            elif parts[0] in ("geom", "layer_bytes"):
                lines[" ".join(parts[:2] if parts[0] == "geom" else parts[:1])] = float(parts[-1])
            elif parts[0] == "capacity":
                # capacity entries <N> padded_rows <R> max_blocks <B>
                lines[f"capacity {parts[1]} {parts[2]}"] = (int(parts[4]), int(parts[6]))
    return lines


class MoEExl3RefC(ctypes.Structure):
    _fields_ = [
        ("trellis", ctypes.c_void_p),
        ("suh", ctypes.c_void_p),
        ("svh", ctypes.c_void_p),
        ("mcg", ctypes.c_void_p),
        ("mul1", ctypes.c_void_p),
        ("in_features", ctypes.c_int32),
        ("out_features", ctypes.c_int32),
        ("k_width", ctypes.c_int32),
        ("has_mcg", ctypes.c_uint8),
        ("has_mul1", ctypes.c_uint8),
        ("pad_", ctypes.c_uint8 * 6),
    ]

REF_FIELDS = ["trellis", "suh", "svh", "mcg", "mul1", "in_features", "out_features", "k_width", "has_mcg"]


class MockExpert:
    """Mirrors the attributes of a served exl3 Linear trio (names per exllamav3/modules/quant/
    exl3.py LinearEXL3): three quantized projections with int16 trellises and fp16 sign/scale
    vectors; mcg/mul1 optional."""

    def __init__(self, gen, e, in_f = 4096, inter = 9216, with_mul1 = False):
        half_rate = with_mul1 and (e % 3 == 0)
        kw_u = 32                                   # int rate K=2 -> k_width 32
        kw_d = 40 if half_rate else 32              # K=2.5 -> k_width 40 (half-integral rate)
        self.gate = self._proj(gen, in_f, inter, kw_u, False)
        self.up   = self._proj(gen, in_f, inter, kw_u, False)
        self.down = self._proj(gen, inter, in_f, kw_d, half_rate)

    @staticmethod
    def _proj(gen, in_f, out_f, k_width, with_mul1):
        # Contents are irrelevant to the table (pointer identity is what is validated); a fresh
        # tensor per field guarantees distinct data_ptr values.
        return dict(
            trellis = torch.zeros((16, out_f // 16, k_width), dtype = torch.int16),
            suh = torch.ones(in_f, dtype = torch.half),
            svh = torch.ones(out_f, dtype = torch.half),
            mcg = torch.zeros(256, dtype = torch.int16),
            mul1 = torch.zeros(512, dtype = torch.int16) if with_mul1 else None,
            in_features = in_f, out_features = out_f,
        )


def ref_from_proj(proj):
    r = MoEExl3RefC()
    r.trellis = proj["trellis"].data_ptr()
    r.suh = proj["suh"].data_ptr()
    r.svh = proj["svh"].data_ptr()
    r.mcg = proj["mcg"].data_ptr() if proj["mcg"] is not None else 0
    r.mul1 = proj["mul1"].data_ptr() if proj["mul1"] is not None else 0
    r.in_features = proj["in_features"]
    r.out_features = proj["out_features"]
    r.k_width = proj["trellis"].shape[-1]
    r.has_mcg = 1 if proj["mcg"] is not None else 0
    r.has_mul1 = 1 if proj["mul1"] is not None else 0
    return r


def clone_ref(r):
    # ctypes array __getitem__ ALIASES the slot, so a tuple swap would write one value twice;
    # materialize by field copy instead.
    n = MoEExl3RefC()
    for f, _ in MoEExl3RefC._fields_:
        setattr(n, f, getattr(r, f))
    return n


def build_table(experts):
    table = (MoEExl3RefC * (NUM_EXPERTS * 3))()
    for e, ex in enumerate(experts):
        for pi, proj in enumerate((ex.gate, ex.up, ex.down)):
            table[e * 3 + pi] = ref_from_proj(proj)
    return table


def validate_table(table, experts):
    """Returns list of (expert, proj, field, want, got) mismatches; naming pairs is the caller's
    job after the swap control."""
    bad = []
    for e in range(NUM_EXPERTS):
        for pi, key in enumerate(("gate", "up", "down")):
            want = ref_from_proj(getattr(experts[e], key))
            got = table[e * 3 + pi]
            for f, _ in MoEExl3RefC._fields_:
                if f == "pad_":           # padding carries no data; ctypes arrays lack value eq
                    continue
                wv, gv = getattr(want, f), getattr(got, f)
                if wv != gv:
                    bad.append((e, key, f, wv, gv))
    return bad


def max_padded_rows(entries):
    return entries + 15 * min(entries, 128)


def max_blocks(entries):
    return (max_padded_rows(entries) + 15) // 16


def run_checks():
    lines = []
    ref_defs = parse_defines(os.path.join(HDR_DIR, "moe_exl3_table.cuh"))
    geom_defs = parse_defines(os.path.join(HDR_DIR, "moe_sm89_common.cuh"))
    fixture = parse_fixture(FIXTURE)

    # C) ctypes mirror vs B) literals
    assert ctypes.sizeof(MoEExl3RefC) == ref_defs["MOE_REF_EXPECT_SIZEOF"], (
        f"ctypes mirror sizeof {ctypes.sizeof(MoEExl3RefC)} != literal {ref_defs['MOE_REF_EXPECT_SIZEOF']}")
    for f in REF_FIELDS:
        off = getattr(MoEExl3RefC, f).offset
        lit = ref_defs[f"MOE_REF_EXPECT_OFF_{f.upper()}"]
        assert off == lit, f"ctypes offset {f}: mirror {off} != literal {lit}"
        compiled, expect = fixture[f"offsetof {f}"]
        assert compiled == expect == lit, (
            f"three-way offset drift {f}: compiled {compiled}, literal {lit}, probe-expect {expect}")
    lines.append(f"ctypes mirror: sizeof {ctypes.sizeof(MoEExl3RefC)}, {len(REF_FIELDS)} offsets "
                 f"match literals AND compiled probe output")

    # A) fixture self-consistency + geometry vs simulation constants
    assert fixture["sizeof MoEExl3Ref"] == (ref_defs["MOE_REF_EXPECT_SIZEOF"],
                                            ref_defs["MOE_REF_EXPECT_SIZEOF"])
    assert fixture["layer_bytes"] == 128 * 3 * ref_defs["MOE_REF_EXPECT_SIZEOF"]
    for name, sim_const in (("num_experts", NUM_EXPERTS), ("top_k", TOP_K),
                            ("gate_cap", GATE_SOFTCAP), ("up_cap", UP_SOFTCAP)):
        hdr = {"num_experts": "MOE_NUM_EXPERTS", "top_k": "MOE_TOP_K",
               "gate_cap": "MOE_GATE_CAP", "up_cap": "MOE_UP_CAP"}[name]
        assert fixture[f"geom {name}"] == float(geom_defs[hdr]) == float(sim_const), (
            f"geometry {name}: probe {fixture[f'geom {name}']}, header {geom_defs[hdr]}, sim {sim_const}")
    lines.append("geometry (128 experts, top_k 8, caps 4/25): probe output == headers == simulation")

    # capacity formula: python expression vs compiled probe (the sizing python will use)
    for entries, (rows_p, blocks_p) in [(k, fixture[f"capacity entries {k}"])
                                         for k in (8, 64, 512, 2048)]:
        assert (max_padded_rows(entries), max_blocks(entries)) == (rows_p, blocks_p), (
            f"arena formula drift at entries={entries}: python "
            f"({max_padded_rows(entries)}, {max_blocks(entries)}) vs compiled ({rows_p}, {blocks_p})")
    lines.append("arena capacity formula matches compiled values at entries {8,64,512,2048}")

    # builder + positive control
    gen = torch.Generator().manual_seed(20261125)
    experts = [MockExpert(gen, e, with_mul1 = True) for e in range(NUM_EXPERTS)]
    table = build_table(experts)
    assert not validate_table(table, experts), f"builder lost refs: {validate_table(table, experts)[:3]}"
    ea, eb = 5, 97
    for pi in range(3):
        ia, ib = ea * 3 + pi, eb * 3 + pi
        tmp_a, tmp_b = clone_ref(table[ia]), clone_ref(table[ib])
        table[ia] = tmp_b
        table[ib] = tmp_a
    bad = validate_table(table, experts)
    assert bad, (f"POSITIVE CONTROL INERT: swapping table refs of experts ({ea}, {eb}) across all "
                 f"3 projections validated clean — the validator never compares stored pointers")
    wrong_experts = sorted({b[0] for b in bad})
    assert wrong_experts == [ea, eb], (
        f"POSITIVE CONTROL MISATTRIBUTED: swapped pair ({ea}, {eb}) flagged experts {wrong_experts}")
    lines.append(f"POSITIVE CONTROL RED AS REQUIRED: expert-ref swap ({ea}, {eb}) flagged exactly "
                 f"those experts across {len(bad)} field mismatches (3 projections)")
    return lines


def main():
    try:
        for ln in run_checks():
            print(ln)
        print("ABI CROSSCHECK GREEN")
    except AssertionError as exc:
        print(f"ABI CROSSCHECK RED: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())