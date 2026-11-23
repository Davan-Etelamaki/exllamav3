"""Whole-forward CUDA-graph eligibility must see graph-managed state nested in a block.

Run standalone (no pytest needed, works in the prod venv):
    python tests/generator/test_fwd_guard_scan.py [<path to exllamav3 package dir>]

The scan lives inside a Generator method that needs a loaded model, so it cannot be called without
weights. Instead this test extracts the ACTUAL expression text from generator.py (balanced-paren slice
starting at 'self._fwd_unsafe = next(') and executes that text against a fake module tree, so the
shipped source is what is judged.

Why a fake tree and the real contract: exllamav3's Module (modules/module.py:14) is an ABC whose
children sit in a plain list `self.modules` (module.py:44) filled by register_submodule, and whose
traversal is __iter__ (module.py:53: yield self, then recurse). FakeModule reproduces that contract
exactly, so the scan exercises the same protocol TransformerBlock uses when it registers block.attn
(modules/transformer.py:68, and qwen3_5 passes GatedDeltaNet there, qwen3_5.py:331).
"""

from __future__ import annotations

import os

try:
    import pytest

    pytestmark = pytest.mark.nogpu
except ImportError:  # the deployed engine venv ships no pytest; the file still runs standalone
    pytestmark = ()


class FakeModule:
    """Reproduces exllamav3 Module's child contract: list of children + __iter__ traversal."""

    def __init__(self, name):
        self.name = name
        self.modules = []

    def __iter__(self):
        yield self
        for module in self.modules:
            yield from module


class FakeGDN(FakeModule):
    pass


class FakeBSMLP(FakeModule):
    pass


class FakePlain(FakeModule):
    pass


class NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def package_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, os.pardir, os.pardir, "exllamav3"))


def extract_expr(pkg: str) -> str:
    """Return the shipped 'next(...)' expression text that computes Generator._fwd_unsafe."""
    with open(os.path.join(pkg, "generator", "generator.py"), encoding = "utf-8") as fh:
        text = fh.read()
    start = text.index("self._fwd_unsafe = next(")
    i = text.index("next(", start) + len("next(")
    depth = 1
    while depth:
        char = text[i]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        i += 1
    return "next(" + text[text.index("next(", start) + len("next("):i - 1] + ")"


def build_tree():
    block_plain = FakeModule("block_plain")
    block_plain.modules = [FakePlain("qkv"), FakePlain("mlp_norm")]
    block_gdn = FakeModule("block_gdn")
    gdn = FakeGDN("linear_attn")
    block_gdn.modules = [FakePlain("norm"), gdn]
    return block_plain, block_gdn, gdn


def run(expr: str, fwd_modules) -> object:
    env = {
        "Module": FakeModule,
        "BlockSparseMLP": FakeBSMLP,
        "BlockSparseMLP_CPU": FakeBSMLP,
        "GatedDeltaNet": FakeGDN,
        "unsafe_types": (FakeBSMLP, FakeGDN),
        "self": NS(model = NS(fwd_modules = fwd_modules)),
    }
    return eval(expr, env)


def test_nested_recurrent_layer_disqualifies_the_block():
    expr = extract_expr(package_dir())
    block_plain, block_gdn, gdn = build_tree()
    assert run(expr, [(block_gdn, 0, 0), (block_plain, 0, 1)]) is gdn


def test_dense_model_stays_eligible():
    expr = extract_expr(package_dir())
    block_plain, _, _ = build_tree()
    assert run(expr, [(block_plain, 0, 0)]) is None


def test_top_level_recurrent_layer_still_disqualifies():
    expr = extract_expr(package_dir())
    top = FakeGDN("toplevel_gdn")
    assert run(expr, [(top, 0, 0)]) is top


def test_flat_pre_fix_scan_missed_the_nested_layer():
    # Positive control: mutate the shipped scan back to the flat isinstance pass it replaced and the
    # nested layer becomes invisible again, which is the defect this commit fixes.
    expr = extract_expr(package_dir())
    flat = expr.replace("for sub in m ", "for sub in (m,) ")
    assert flat != expr, "RED-arm mutation did not land"
    _, block_gdn, gdn = build_tree()
    assert run(flat, [(block_gdn, 0, 0)]) is None


def test_nn_module_style_traversal_is_not_available_on_this_module():
    # The first attempt borrowed torch's API. Module.modules is a plain list, so the nn.Module reflex
    # either misses the tree or raises inside the decode path; both mean it fixes nothing.
    expr = extract_expr(package_dir())
    att1 = expr.replace("for sub in m ",
                        "for sub in (m.modules() if hasattr(m, \"modules\") else (m,)) ")
    assert att1 != expr, "attempt-1 mutation did not land"
    _, block_gdn, gdn = build_tree()
    try:
        found = run(att1, [(block_gdn, 0, 0)])
    except TypeError as exc:
        assert "not callable" in str(exc)
        return
    assert found is not gdn


def _main() -> int:
    import sys
    pkg = sys.argv[1] if len(sys.argv) > 1 else package_dir()
    globals()["package_dir"] = lambda: pkg
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print("ok   %s" % fn.__name__)
        except Exception as exc:  # noqa: BLE001 - standalone runner reports every failure mode
            failed += 1
            print("FAIL %s: %s" % (fn.__name__, exc))
    print("RAN %d tests, %d failed" % (len(tests), failed))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_main())