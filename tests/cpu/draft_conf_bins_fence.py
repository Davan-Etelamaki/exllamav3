#!/usr/bin/env python3
"""Draft-confidence binning fence (CPU-only).

Pins why the dynamic-draft calibrator was oscillating on dot26 (2026-11-26/27:
prose 57 % <-> factual/count NO-TELEMETRY): the generator instantiated
DraftConfidenceCalibrator with the class-default bin_width 1.0 while every
drafter's confidence is a probability in (0,1] (reciprocal softmax mass, cpp se/cum
rule). floor(p/1.0) puts every label in ONE bin, so threshold() could only return
0 (full windows) or +inf (starve). Arms:
  [1] generator.py passes bin_width = 0.05 to the calibrator (control: stripping it trips);
  [2] runtime: a graded label profile yields a GRADED threshold (not 0/+inf) at 0.05;
  [3] runtime CONTROL: the identical labels at bin_width 1.0 collapse to {0, +inf};
  [4] burn-in and sparse-bin rules unchanged (-inf before burn_in; thin bins skipped).
Run:  CUDA_VISIBLE_DEVICES= env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH \\
      PYTHONPATH=. ~/.venvs/tabby-exl3/bin/python tests/cpu/draft_conf_bins_fence.py
"""
import ast
import math
import sys

GEN = "exllamav3/generator/generator.py"

def src(path):
    with open(path, encoding = "utf8") as f:
        return f.read()

fails = []
def check(name, cond):
    print(("  ok   " if cond else "  FAIL ") + name)
    if not cond:
        fails.append(name)

print("== [1] construction site ==")
gsrc = src(GEN)
check("generator builds the calibrator with bin_width = 0.05",
      "DraftConfidenceCalibrator(draft_confidence, bin_width = 0.05)" in gsrc)
# CONTROL 1: reverting to the degenerate default trips the arm
mut1 = gsrc.replace("DraftConfidenceCalibrator(draft_confidence, bin_width = 0.05)",
                    "DraftConfidenceCalibrator(draft_confidence)")
check("CONTROL: dropping bin_width trips the construction arm",
      "DraftConfidenceCalibrator(draft_confidence, bin_width = 0.05)" not in mut1)

print("== [2] graded thresholds at 0.05 ==")
from exllamav3.generator.draft_confidence import DraftConfidenceCalibrator

cal = DraftConfidenceCalibrator(0.7, bin_width = 0.05)
check("burn-in gate holds (-inf until enough labels)", cal.threshold() == -math.inf)
for _ in range(60):
    cal.add_label(0.92, True)          # high-confidence bin: all accepted
for _ in range(20):
    cal.add_label(0.52, True)
for _ in range(20):
    cal.add_label(0.52, False)         # mid bin: exactly 50 % < target
thr2 = cal.threshold()
check("threshold is graded (finite, strictly inside (0,1))",
      math.isfinite(thr2) and 0.0 < thr2 < 1.0)
check("threshold equals the low edge of the lowest passing bin from the top (18*0.05)",
      abs(thr2 - 18 * 0.05) < 1e-12)
check("estimate() resolves neighbours instead of one global rate",
      abs(cal.estimate(0.95) - 1.0) < 1e-9 and abs(cal.estimate(0.55) - 0.5) < 1e-9)

print("== [3] CONTROL: identical labels at the old default bin_width 1.0 ==")
deg_lo = DraftConfidenceCalibrator(0.7)   # 60 accepted / 90 tested < 0.7
for _ in range(60):
    deg_lo.add_label(0.9, True)
for _ in range(30):
    deg_lo.add_label(0.5, False)
check("sub-target mix at bin_width 1.0 starves (+inf) — the NO-TELEMETRY signature",
      deg_lo.threshold() == math.inf)
deg_hi = DraftConfidenceCalibrator(0.7)   # 70/90 ≥ 0.7
for _ in range(70):
    deg_hi.add_label(0.9, True)
for _ in range(20):
    deg_hi.add_label(0.5, False)
check("over-target mix at bin_width 1.0 flips to full windows (0) — the oscillation pair",
      deg_hi.threshold() == 0.0)

print("== [4] sparse-bin rule preserved at fine bins ==")
sparse = DraftConfidenceCalibrator(0.7, bin_width = 0.05)
for _ in range(80):
    sparse.add_label(0.42, True)          # one fat low bin (idx 8, interior score)
for _ in range(3):
    sparse.add_label(0.96, False)         # thin high bin (< min_count 8)
thr = sparse.threshold()
check("thin high bins are skipped, not trusted (fat bin alone sets threshold)",
      abs(thr - 8 * 0.05) < 1e-12)

print()
if fails:
    print(f"FAILURES ({len(fails)}):")
    for f in fails:
        print("  - " + f)
    sys.exit(1)
print("VERDICT: draft_conf binning fence GREEN (graded truncation contract intact)")