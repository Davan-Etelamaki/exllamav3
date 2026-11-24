"""Fence for the whole-forward graph's eligibility rule (law 12/21).

The served inkling-2-mini path must be declared graph-unsafe: on 2026-11-24 17:38:59 the runner
warmed up and tried to capture it anyway and the boot lost the round at
inkling2_loop.py:756 ("expanded size of the tensor (10880) must match the existing size (256)").
A clean boot-time degrade is the acceptance behaviour; a silent capture attempt is the defect.
"""

import inspect

import exllamav3.generator.generator as generator
from exllamav3.generator import fwd_graph


class _Block(metaclass=type):
    # Module is abstract; a bare stand-in block is enough for the scan, which only walks
    # isinstance and Module.__iter__.
    def __init__(self, subs=()):
        from exllamav3.modules.module import Module
        self.__class__.__mro__  # touch
        self.modules = list(subs)
        self.caps = {}
    def forward(self, x, params = None, *a, **k):
        return x
    def optimizer_targets(self):
        return ()


def _register_block_subclass():
    from exllamav3.modules.module import Module
    class RegisteredBlock(Module):
        def forward(self, x, params = None, *a, **k):
            return x
        def optimizer_targets(self):
            return ()
    return RegisteredBlock


def _stub_block(kind):
    # A registered module is always a Module subclass (BlockSparseMLP_CPU is a plain mixin that
    # only reaches the tree through BlockSparseMLP, which mixes it with Module), so wrap any
    # non-Module kind the way the real module table does.
    from exllamav3.modules.module import Module
    rb = _register_block_subclass()
    if not issubclass(kind, Module):
        kind = type("Stub" + kind.__name__, (kind, rb), {})
    inner = object.__new__(kind)
    inner.modules = []
    block = object.__new__(rb)
    block.modules = [inner]
    return block, inner


def test_the_hybrid_attention_and_moe_are_declared_unsafe():
    from exllamav3.modules.arch_specific.inkling2_loop import InklingAttention, InklingMoE
    for kind in (InklingAttention, InklingMoE):
        block, inner = _stub_block(kind)
        assert fwd_graph.first_unsafe_module([(block, None, None)]) is inner


def test_the_documented_unsafe_types_are_still_declared_unsafe():
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    from exllamav3.modules.block_sparse_mlp_cpu import BlockSparseMLP_CPU
    from exllamav3.modules.gated_delta_net import GatedDeltaNet
    for kind in (BlockSparseMLP, BlockSparseMLP_CPU, GatedDeltaNet):
        block, inner = _stub_block(kind)
        assert fwd_graph.first_unsafe_module([(block, None, None)]) is inner


def test_a_plain_registered_block_stays_eligible():
    rb = _register_block_subclass()
    block = object.__new__(rb)
    block.modules = []
    assert fwd_graph.first_unsafe_module([(block, None, None)]) is None
    assert fwd_graph.first_unsafe_module([]) is None


def test_the_generator_asks_the_shared_helper_not_an_inline_tuple():
    src = inspect.getsource(generator)
    assert "_fg_scan.first_unsafe_module(self.model.fwd_modules)" in src
    assert "unsafe_types = (" not in src
