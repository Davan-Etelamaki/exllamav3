#!/usr/bin/env python3
"""Fence: iterate_start_jobs must never freeze admission behind an elder it cannot serve.

CPU-only (workstation law: never import exllamav3 — importing it initializes CUDA). The real
method body is extracted from generator.py BY AST and EXECUTED against fakes whose attribute
surface is DERIVED FROM THE SERVED SOURCE (law 33/21: a lookalike must not supply a method the
served code never touches, nor lack one it does). `admits` and its helpers are likewise taken
from generator/admission.py by AST, so the price test is the engine's own.

RED repro (dot26 2026-11-26, measured by Performance-Optimizer on payloads 0fe912755ea0e and
060ebbd9368f9): a 22-token job spent 0.07 s computing behind 102.5-489.0 s of `queued`
(job.py:926). Mechanism: skips only advanced when some OTHER job was admitted, and once any
skipped job reached max_skips the pass RETURNED before starting anyone — while skips was never
reset, so the blockage never expired. An elder that cannot fit therefore denied service to
every later job for its entire wait.
"""
import ast, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
GEN = ROOT / "exllamav3/generator/generator.py"
ADM = ROOT / "exllamav3/generator/admission.py"
fails = []


def check(name, ok, detail=""):
    print(("  ok   " if ok else "  FAIL ") + name + (f": {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def literal_consts(tree):
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    try:
                        out[t.id] = ast.literal_eval(n.value)
                    except Exception:
                        pass
    return out


def take_funcs(tree, wanted):
    return [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]


gen_tree = ast.parse(GEN.read_text())
adm_tree = ast.parse(ADM.read_text())
holder = next((c for c in gen_tree.body if isinstance(c, ast.ClassDef)
               and any(getattr(m, "name", "") == "iterate_start_jobs" for m in c.body)), None)
check("iterate_start_jobs found in a served class", holder is not None,
      f"class={holder.name if holder else None}")
if holder is None:
    print("MOE/GEN FAIRNESS FENCE RED:", "no method to fence")
    sys.exit(1)

meth = next(m for m in holder.body if getattr(m, "name", "") == "iterate_start_jobs")
src = ast.unparse(meth)

# ADMITTED VOCABULARY: the method's own uses of the candidate job, read from served source.
touched = sorted({a.attr for a in ast.walk(meth) if isinstance(a, ast.Attribute)
                  and isinstance(a.value, ast.Name) and a.value.id == "job"})
check("the method's job surface is readable from source", bool(touched), f"attrs={touched}")

consts = literal_consts(adm_tree)
ns = dict(consts)
for f in take_funcs(adm_tree, {"admits", "prompt_cost_granules", "granules_for_tokens"}):
    exec(compile(ast.fix_missing_locations(ast.Module(body=[f], type_ignores=[])), "<adm>", "exec"), ns)
check("the engine's own admission price is available to the arm", "admits" in ns,
      f"admission_consts={sorted(k for k in consts if k.isupper())[:4]}")

fn = ast.parse(src).body[0]
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "<gen>", "exec"), ns)
run_pass = ns["iterate_start_jobs"]


class FakePage:
    def __init__(s, free): s.free = free
    def num_unreferenced_pages(s): return s.free


class FakeJob:
    def __init__(s, name, seqs, needs, skips, max_skips):
        s.name, s.sequences, s.needs = name, list(range(seqs)), needs
        s.skips, s.max_skips = skips, max_skips
        s.serial_number = name
        s.identifier = None
        s.active = False
    def current_new_pages_required(s): return s.needs
    def activate(s): s.active = True
    def allocate_pages(s): pass


missing = [a for a in touched if not hasattr(FakeJob("x", 1, 1, 0, 4), a)]
check("the fixture exposes exactly the surface the served method touches", not missing,
      f"missing={missing}")


class FakeGen:
    _fairness_holds = 0
    def __init__(s, pending, active, free_pages, max_batch):
        s.pending_jobs, s.active_jobs = list(pending), list(active)
        s.pagetable, s.max_batch_size = FakePage(free_pages), max_batch
        s.admission_margin_granules = consts.get("DEFAULT_ADMISSION_MARGIN_GRANULES", 8)
    def _admission_free_bytes(s): return -1           # -1 => tier price test skipped (batch regime)
    def _admission_prompt_tokens(s, job): return len(job.sequences)
    def reap_failed_job(s, job, e, results): pass


def passes(n_pending_short_elder, rounds=12):
    elder = FakeJob("elder", 1, 999, n_pending_short_elder, 4)     # never fits: needs 999 pages
    small = FakeJob("small", 1, 1, 0, 4)                           # always fits
    g = FakeGen([elder, small], [], free_pages=5, max_batch=4)
    started = []
    for i in range(rounds):
        res = []
        run_pass(g, res)
        started += [r["job"].name for r in res]
        if "small" in started:
            return i + 1, started, elder.skips
    return None, started, elder.skips


print("[F1] a permanently-unfittable elder must not freeze the queue behind it")
rounds, started, elder_skips = passes(4)   # elder already at max_skips, as production reaches
check("the small job starts within 5 passes despite the blocked elder",
      rounds is not None, f"started={started} elder_skips={elder_skips}")

print("[F2] queue order is kept at the head: an elder that FITS starts before a later job")
elder = FakeJob("elder", 1, 3, 0, 4)          # fits in 5 free pages
small = FakeJob("small", 1, 1, 0, 4)
g = FakeGen([elder, small], [], free_pages=5, max_batch=4)
res = []
run_pass(g, res)
check("the admissible elder at the head starts first", [r["job"].name for r in res][:1] == ["elder"],
      f"started={[r['job'].name for r in res]}")

print("[F3] POSITIVE CONTROL: without the elder the same harness starts at once")
small_only = FakeGen([FakeJob("small", 1, 1, 0, 4)], [], free_pages=5, max_batch=4)
res = []
run_pass(small_only, res)
check("harness is not itself blocking admission",
      [r["job"].name for r in res] == ["small"], f"started={[r['job'].name for r in res]}")

print("[F4] POSITIVE CONTROL: skipping must age the elder (the old code did not)")
elder = FakeJob("elder", 1, 999, 0, 4)
g = FakeGen([elder, FakeJob("small", 1, 1, 0, 4)], [], free_pages=5, max_batch=4)
for _ in range(4):
    run_pass(g, [])
check("four blocked passes raise the elder's skip count to max_skips",
      elder.skips >= 4, f"skips={elder.skips}")

if fails:
    print("GENERATOR FAIRNESS FENCE RED:", ", ".join(fails))
    sys.exit(1)
print("GENERATOR FAIRNESS FENCE GREEN: the queue drains and the elder is honored.")
sys.exit(0)
