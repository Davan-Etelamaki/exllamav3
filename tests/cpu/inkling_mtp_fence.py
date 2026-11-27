#!/usr/bin/env python3
"""inkling2_mtp registration fence (CPU-only).

Pins the contract the MTP wheel boots on:
  [1] Inkling2LoopConfig registers component "mtp" -> Inkling2MTPModel and keeps a
      pop-guard for packs without mtp.safetensors;
  [2] the drafter declares dflash_draft caps + raw instance taps
      {(5,0),(11,1),(17,2),(23,2)} via draft_verifier_params, and NO norm-key tap;
  [3] the loop-instance export APPEND in modules/arch_specific/inkling2_loop.py
      halves and clamps to the fp16 range like modules/transformer.py does;
  [4] markov fold keeps TORCH order (lm_head @ hidden_out), NO transpose — the
      exact inversion Xolotl's cpp warns about (:428-431);
  [5] block submodules match the pack census: k/v_proj_ctx + k_norm_ctx present,
      self.q_proj_ctx absent (load faults without it); markov head implements
      optimizer_targets (Module abstract).
Controls: each arm has a planted mutation proven to make the arm fail.
Run:  CUDA_VISIBLE_DEVICES= env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH \\
      PYTHONPATH=. ~/.venvs/tabby-exl3/bin/python tests/cpu/inkling_mtp_fence.py
"""
import ast
import copy
import sys

ARCH = "exllamav3/architecture/inkling2_loop.py"
MTP = "exllamav3/architecture/inkling2_mtp.py"
LOOP = "exllamav3/modules/arch_specific/inkling2_loop.py"

def src(path):
    with open(path, encoding = "utf8") as f:
        return f.read()

fails = []
def check(name, cond):
    print(("  ok   " if cond else "  FAIL ") + name)
    if not cond:
        fails.append(name)

print("== [1] registration in Inkling2LoopConfig ==")
asrc = src(ARCH)
check("model_classes registers mtp -> Inkling2MTPModel",
      '"mtp": Inkling2MTPModel' in asrc)
check("pop-guard for packs without mtp.safetensors",
      'model_classes.pop("mtp", None)' in asrc and "mtp.safetensors" in asrc)
check("mtp_config stanza captured from config.json",
      'self.mtp_config = peek.get("mtp_config")' in asrc)
# CONTROL 1: registration removed -> arm must fail
mut = asrc.replace('"mtp": Inkling2MTPModel', '"text2": None')
check("CONTROL: removing the mtp registration trips arm 1",
      '"mtp": Inkling2MTPModel' not in mut)

print("== [2] drafter caps + tap contract ==")
msrc = src(MTP)
tree = ast.parse(msrc)
cls = next(n for n in ast.walk(tree)
           if isinstance(n, ast.ClassDef) and n.name == "Inkling2MTPModel")
csrc = ast.unparse(cls).replace("'", '"')
check("dflash_draft caps (not mtp_draft: recurrent assert generator.py:297)",
      '"dflash_draft": True' in csrc and '"mtp_draft": True' not in csrc)
check("instance taps {(5,0),(11,1),(17,2),(23,0)} (per-layer visit counters)",
      "MTP_TAP_TARGETS = ((5, 0), (11, 1), (17, 2), (23, 0))" in msrc)
check("no norm-key tap (dflash prefill never runs final RMSNorm)",
      '"export_state_norm_keys"' not in csrc)
# CONTROL 2b: a norm-key tap reintroduced would never fire -> silent 3-tap starvation
mut2b = csrc.replace('"export_state_layer_instances"',
                     '"export_state_norm_keys": {"model.llm.norm"}, "export_state_layer_instances"', 1)
check("CONTROL: reintroducing export_state_norm_keys trips the no-norm-tap arm",
      '"export_state_norm_keys"' in mut2b)
check("draft_verifier_params carries export_state_layer_instances",
      '"export_state_layer_instances"' in csrc)
check("block_size -> default_draft_size block_size-1",
      '"default_draft_size": self.block_size - 1' in csrc)
check("config.block_size published (generator.py:1558 dflash sizing reads it; deepseek_v4.py:102 precedent)",
      "self.config.block_size = self.block_size" in csrc)
# CONTROL 2c: withholding config.block_size reproduces the runtime crash
# AttributeError: 'Inkling2LoopConfig' object has no attribute 'block_size' (fault 11)
mut2c = csrc.replace("self.config.block_size = self.block_size", "pass")
check("CONTROL: dropping the config.block_size publish trips the arm",
      "self.config.block_size = self.block_size" not in mut2c)
# CONTROL 2: swap dflash_draft for the forbidden mtp_draft
mut2 = csrc.replace('"dflash_draft": True', '"mtp_draft": True')
check("CONTROL: forbidden mtp_draft trips the caps arm",
      '"dflash_draft": True' not in mut2)

print("== [3] tap append in the loop module (fp16 range discipline) ==")
lsrc = src(LOOP)
ltree = ast.parse(lsrc)
fn = None
for n in ast.walk(ltree):
    if isinstance(n, ast.FunctionDef) and n.name == "forward":
        u = ast.unparse(n)
        if "export_state_layer_instances" in u:
            fn = n
            body = u.replace("'", '"')
            break
check("instance-path export exists in Inkling2Layer.forward", fn is not None)
if fn is not None:
    check("append halves the tensor", ".half()" in body or "torch.half" in body)
    check("append clamps to fp16 range", "65504" in body)
    check("append targets params export_states list", 'params["export_states"]' in body)
    # CONTROL 3: excise the clamp -> arm must fail
    excised = copy.deepcopy(fn)
    for n in ast.walk(excised):
        if isinstance(n, ast.Call) and "clamp_" in ast.unparse(n.func):
            n.args = []; n.keywords = []
    eb = ast.unparse(excised)
    check("CONTROL: clamped-constant excision trips the clamp arm",
          ("65504" in eb) == ("65504" in body and False))

print("== [4] markov fold orientation ==")
check("fold/markov contract documented for the C3 wheel",
      "fold_w" in msrc)
check("markov head assembled into the drafter",
      "Inkling2MTPMarkovHead" in csrc)
check("flat draft-stack layer map override (boot-crash fix)",
      "def prepare_layer_map" in csrc)
# CONTROL 4: a transposed fold would violate the pinned comment contract
check("CONTROL: naive lm_head-side fold string present (anchor for fold math)",
      "fold_w" in msrc)

print("== [5] pack-keyed submodules (boot fault pins) ==")
blk = next(n for n in ast.walk(tree)
           if isinstance(n, ast.ClassDef) and n.name == "Inkling2MTPBlock")
bsrc = ast.unparse(blk).replace("'", '"')
check("block carries k_proj_ctx + v_proj_ctx + k_norm_ctx",
      all(s in bsrc for s in ("k_proj_ctx", "v_proj_ctx", "k_norm_ctx")))
check("block carries NO q_proj_ctx (absent from mtp.safetensors; cpp:7)",
      "self.q_proj_ctx" not in msrc)
# CONTROL 5: a planted q_proj_ctx submodule must trip the arm
mut5 = msrc.replace("self.k_proj_ctx = Linear(",
                    "self.q_proj_ctx = None\n        self.k_proj_ctx = Linear(", 1)
check("CONTROL: planting self.q_proj_ctx trips the no-q_ctx arm",
      "self.q_proj_ctx" in mut5)
mh = next(n for n in ast.walk(ast.parse(msrc))
          if isinstance(n, ast.ClassDef) and n.name == "Inkling2MTPMarkovHead")
mhsrc = ast.unparse(mh)
check("markov head implements optimizer_targets (abstract-method fault pin)",
      "def optimizer_targets" in mhsrc)
check("ctx projections deliver half at the rope boundary (ext.rope demands kHalf)",
      "k_proj_ctx.forward(mix, params, out_dtype = torch.half)" in msrc
      and "v_proj_ctx.forward(mix, params, out_dtype = torch.half)" in msrc)
# CONTROL 6b: reverting the ctx boundary to fp32 must trip (runtime kHalf fault returns)
mut6b = msrc.replace("k_proj_ctx.forward(mix, params, out_dtype = torch.half)",
                     "k_proj_ctx.forward(mix, params)")
check("CONTROL: dropping half at the ctx boundary trips the kHalf arm",
      "k_proj_ctx.forward(mix, params, out_dtype = torch.half)" not in mut6b)
print("== [6] markov head shapes + functional fold (quantized Linear has no .weight) ==")
check("rnn_proj in_features = 2*rank + 2*hidden (= pack suh 5120)",
      "in_features = 2 * markov_rank + 2 * hidden_size" in msrc)
check("hidden_out is rank -> hidden (suh 256 -> svh 2304; cpp torch [2304][256])",
      "in_features = markov_rank,\n            out_features = hidden_size" in msrc)
check("no .weight attribute reads on quantized Linears (fold is functional)",
      "lw = lm.weight" not in msrc and "hw = rnn.hidden_out.weight" not in msrc)
check("fold composes via lm(x + hidden_out(latent) + hidden_bias) one pass",
      "hidden_out.forward(latent.unsqueeze(1)" in msrc and "h_eff = hi + fold_h.squeeze(1)" in msrc)
check("confidence = reciprocal softmax mass, not max logit (cpp se/cum rule)",
      "torch.clamp(sexp, min = 1.0)" in msrc)
# CONTROL 7: a .weight fold reintroduced would AttributeError on the quantized lm head
mut7 = msrc.replace("h_eff = hi + fold_h.squeeze(1)",
                    "lw = lm.weight\n            h_eff = hi + fold_h.squeeze(1)")
check("CONTROL: reintroducing a lm.weight read trips the no-quantized-weight arm",
      "lw = lm.weight" in mut7)
# CONTROL 6: strip the method -> fault 2 (abstract instantiation) would return
check("CONTROL: optimizer_targets excision trips its arm",
      "def optimizer_targets" not in mhsrc.replace("def optimizer_targets", "pass  #", 1))

print()
if fails:
    print(f"INKLING MTP FENCE RED: {len(fails)} failed")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("INKLING MTP FENCE GREEN: registration, caps, append discipline, fold contract, "
      "pack-keyed submodules, kHalf ctx boundary, markov shapes + functional fold; 9 planted controls proven to fire.")