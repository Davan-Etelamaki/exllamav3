"""
Fence for the VRAM admission gate: the tier table in generator/admission.py and the call
site in Generator.iterate_start_jobs.

The table is calibrated on the dot26 RTX 4090 cliff (owner measurement 2026-11-24): n
concurrent large prefills peak at 21.63 + n * 1.55 GiB and bust the card's 23.59 GiB of
usable memory at n >= 2, while admitting one large prefill at a time holds the worst case
at 23.18 GiB. Costs are counted in the allocator's own unit, a 20 MiB granule.

CPU-only by construction: the decision module is loaded from its file rather than imported,
so this suite runs on an interpreter with neither torch nor a GPU. The wiring checks parse
generator.py as text, because that module pulls torch in at import time. Text-level wiring
checks are also what let the fence run its pristine-base arm: point EXL3_GENERATOR_PATH at
an unpatched generator.py and the wiring tests fail on the gate they cannot find, instead
of the suite dying on an import.

Environment overrides, used by the arms of check_admission_gate.sh:
    EXL3_ADMISSION_PATH   admission.py to price tiers from   (boundary-poison arm)
    EXL3_GENERATOR_PATH   generator.py to check the wiring against   (pristine-base arm)
"""

import ast
import importlib.util
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.nogpu

REPO_DIR = Path(__file__).resolve().parents[2]
ADMISSION_PATH = Path(os.environ.get("EXL3_ADMISSION_PATH") or REPO_DIR / "exllamav3/generator/admission.py")
GENERATOR_PATH = Path(os.environ.get("EXL3_GENERATOR_PATH") or REPO_DIR / "exllamav3/generator/generator.py")

if not ADMISSION_PATH.is_file():
    pytest.exit(f"admission module not found: {ADMISSION_PATH}", returncode = 1)
if not GENERATOR_PATH.is_file():
    pytest.exit(f"generator module not found: {GENERATOR_PATH}", returncode = 1)

_spec = importlib.util.spec_from_file_location("admission_under_test", ADMISSION_PATH)
admission = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(admission)

GRAN = admission.GRANULE_BYTES


def granules(n: int) -> int:
    """n allocator granules, in bytes"""
    return n * GRAN


# --------------------------------------------------------------------------------------
# The tier table, injected free values
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "prompt_tokens, cost",
    [
        (1, 20),      # anything with a prompt to ingest pays the first tier
        (8191, 20),
        (8192, 20),   # top of the first tier
        (8193, 41),   # first token past the ceiling
        (16383, 41),
        (16384, 41),  # top of the second tier
        (16385, 87),  # first token past the ceiling
        (131072, 87),
    ],
)
def test_cost_table_boundaries(prompt_tokens, cost):
    assert admission.prompt_cost_granules(prompt_tokens) == cost


def test_margin_addition():
    """The bar is cost + margin, both in granules"""
    assert admission.admits(granules(28), 8192, 8) is True
    assert admission.admits(granules(27), 8192, 8) is False
    for margin in (0, 1, 8, 32, 128):
        assert admission.admits(granules(20 + margin), 4096, margin) is True
        assert admission.admits(granules(20 + margin) - 1, 4096, margin) is False
    # crossing a ceiling jumps the bar; length inside a tier does not move it
    for tokens, cost in ((1, 20), (8192, 20), (8193, 41), (16384, 41), (16385, 87)):
        assert admission.admits(granules(cost + 8), tokens, 8) is True
        assert admission.admits(granules(cost + 8) - 1, tokens, 8) is False


def test_decode_never_blocked():
    """A job with nothing left to ingest is a decode admission: it already holds its pages"""
    assert admission.prompt_cost_granules(0) == 0
    for margin in (0, 8, 1000):
        for free in (0, 1, GRAN - 1, granules(1)):
            assert admission.admits(free, 0, margin) is True


def test_equality_edge_admits():
    """free_granules >= cost_granules + margin_granules: equality admits, one granule short refuses"""
    assert admission.admits(granules(20 + 8), 8192, 8) is True
    assert admission.admits(granules(20 + 8) - 1, 8192, 8) is False
    assert admission.admits(granules(41 + 8), 16384, 8) is True
    assert admission.admits(granules(41 + 8) - 1, 16384, 8) is False
    assert admission.admits(granules(87 + 8), 16385, 8) is True
    assert admission.admits(granules(87 + 8) - 1, 16385, 8) is False
    # free floors down: a partial granule is not spendable
    partial = granules(49) + GRAN - 1
    assert admission.admits(partial, 8192, 29) is True    # 49 >= 20 + 29
    assert admission.admits(partial, 8192, 30) is False   # 49 <  20 + 30


def test_units_are_allocator_granules():
    """Every figure in the module is counted in the 20 MiB granule the allocator expands by"""
    assert admission.GRANULE_BYTES == 20 * 1024 * 1024
    assert admission.DEFAULT_ADMISSION_MARGIN_GRANULES == 8


def test_decision_module_imports_nothing():
    """Staying import-free is what keeps this fence runnable without torch or a GPU"""
    tree = ast.parse(ADMISSION_PATH.read_text())
    pulled = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            pulled += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            pulled.append(node.module or "")
    assert pulled == [], f"admission.py must not import (torch/CUDA creeps in): {pulled}"


# --------------------------------------------------------------------------------------
# Wiring: the decision must be consulted at the admission point
# --------------------------------------------------------------------------------------

GENERATOR_SRC = GENERATOR_PATH.read_text()
GENERATOR_TREE = ast.parse(GENERATOR_SRC)


def _function(name: str) -> ast.FunctionDef:
    for node in ast.walk(GENERATOR_TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{GENERATOR_PATH} defines no {name}()")


def _called_names(node) -> set:
    out = set()
    for call in ast.walk(node):
        if isinstance(call, ast.Call):
            if isinstance(call.func, ast.Name):
                out.add(call.func.id)
            elif isinstance(call.func, ast.Attribute):
                out.add(call.func.attr)
    return out


def test_wiring_gate_called_at_admission_point():
    fn = _function("iterate_start_jobs")
    calls = _called_names(fn)
    assert "admits" in calls, \
        f"{GENERATOR_PATH.name}:iterate_start_jobs does not consult admits() - gate not wired"
    assert "_admission_prompt_tokens" in calls, \
        f"{GENERATOR_PATH.name}:iterate_start_jobs never prices the job's prompt"
    assert "_admission_free_bytes" in calls, \
        f"{GENERATOR_PATH.name}:iterate_start_jobs never reads free VRAM"


def test_wiring_refusal_takes_the_queue_path():
    """A refusal joins the jobs skipped for capacity and is retried next pass; nothing throws"""
    fn = _function("iterate_start_jobs")
    gated = []
    for node in ast.walk(fn):
        if isinstance(node, ast.If) and "admits" in _called_names(node.test):
            gated.append(node)
    assert gated, f"{GENERATOR_PATH.name}: no branch tests admits() at admission"
    for node in gated:
        body = ast.dump(node)
        assert "skipped_jobs" in body and "Continue" in body, \
            "the admission refusal must reuse the engine's skip/queue path"
    assert not any(isinstance(n, ast.Raise) for n in ast.walk(fn)), \
        "iterate_start_jobs must not raise to refuse a job"


def test_wiring_budget_spent_per_started_job():
    """Jobs admitted in one pass must be priced cumulatively, or two big prefills both fit"""
    fn = _function("iterate_start_jobs")
    spent = [n for n in ast.walk(fn)
             if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name)
             and n.target.id == "free_bytes" and isinstance(n.op, ast.Sub)]
    assert spent, f"{GENERATOR_PATH.name}: the pass budget is never spent, so concurrent prefills double-admit"


def test_wiring_none_margin_takes_the_default():
    """tabbyAPI forwards an Optional[int]: an explicit null must reach the engine default,
    not int(None). Asserted on the source: __init__ coerces None before max()/int()."""
    src = GENERATOR_PATH.read_text()
    assert "if admission_margin_granules is None" in src, \
        "Generator.__init__ does not None-guard admission_margin_granules"
    sig = next(n for n in ast.walk(ast.parse(src))
               if isinstance(n, ast.FunctionDef) and n.name == "__init__"
               and "admission_margin_granules" in [a.arg for a in n.args.args])
    arg = next(a for a in sig.args.args if a.arg == "admission_margin_granules")
    assert arg.annotation is not None and "None" in ast.unparse(arg.annotation), \
        "signature does not admit None"


def test_wiring_margin_is_a_config_value_with_engine_default_8():
    """Generator takes admission_margin_granules and defaults it to the engine default"""
    classes = [n for n in ast.walk(GENERATOR_TREE)
               if isinstance(n, ast.ClassDef) and n.name == "Generator"]
    assert classes, f"{GENERATOR_PATH.name} defines no Generator class"
    init = next((n for n in classes[0].body
                 if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
    assert init is not None, f"{GENERATOR_PATH.name}: Generator defines no __init__"
    positional = [a.arg for a in init.args.args]
    assert "admission_margin_granules" in positional, \
        f"{GENERATOR_PATH.name}: Generator.__init__ does not take admission_margin_granules"
    tail = positional[len(positional) - len(init.args.defaults):]
    default = dict(zip(tail, init.args.defaults))[ "admission_margin_granules"]
    assert isinstance(default, ast.Name) and default.id == "DEFAULT_ADMISSION_MARGIN_GRANULES", \
        "the margin default must come from the shared engine default constant"
    assert admission.DEFAULT_ADMISSION_MARGIN_GRANULES == 8