"""QSAIndexer._select_rows, multi-tile path: the first merge must not read an uninitialized
candidate slot.

Slot 0 of the tiled top-k workspace is the pass's running set and slot 1 is the current tile.
The workspaces come from the bucketed tensor cache, i.e. DIRTY memory, and
dsa_topk_merge_kernel uses ws_cnt[row, g] as the loop bound over ws_idx/ws_scr slot g. On the
first iteration nothing has written slot 0, so a dirty count becomes a garbage loop bound and
the merge reads far outside the slot. In production that is an illegal memory access during
the autosplit load measure (Xid 31 / cudaErrorIllegalAddress), which only appears when
T_slab > SEL_TILE and the slab spans multiple tiles.

This test poisons the counts on purpose, so it fails loudly instead of depending on what the
allocator happened to leave in the bucket.

Run: python tests/kernels/dsa/test_qsa_tiled_slot0.py     (needs one CUDA device)
"""
import math, sys
import torch

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
import exllamav3.modules.qsa_indexer as qi                                    # noqa: E402
from exllamav3.modules.qsa_indexer import QSAIndexer                          # noqa: E402
from exllamav3.util.tensor import g_tensor_cache                              # noqa: E402

# Qwen4Exp prod indexer: compress ratio 4, budget 2048 -> block_topk 512, dk 128, 4 heads
CR, BLOCK_TOPK, HEAD_DIM, N_HEADS = 4, 512, 128, 4
ROWS, POS0 = 8192, 32768
POISON = 0x40000000


def _indexer():
    idx = QSAIndexer.__new__(QSAIndexer)
    idx.compress_ratio, idx.block_topk = CR, BLOCK_TOPK
    idx.n_heads, idx.head_dim = N_HEADS, HEAD_DIM
    idx.scale = 1.0 / math.sqrt(HEAD_DIM)
    idx._sel_w = None
    return idx


def _run(poison):
    idx = _indexer()
    T = (POS0 + ROWS) // CR
    assert T > QSAIndexer.SEL_TILE or poison == 0, \
        "test must exercise the multi-tile path (T_slab > SEL_TILE)"
    torch.manual_seed(0)
    q = torch.randn(ROWS, N_HEADS, HEAD_DIM, device="cuda", dtype=torch.half) * 0.5
    pool = torch.randn(T, HEAD_DIM, device="cuda", dtype=torch.half) * 0.5
    out = torch.empty((ROWS, idx.k_pad()), dtype=torch.int32, device="cuda")

    real = g_tensor_cache.get_bucketed
    seen = []

    def hooked(dev, numel, dtype, tag):
        t = real(dev, numel, dtype, tag)
        if tag.endswith("_cnt"):
            t[:numel].view(-1).fill_(poison)
            seen.append(tag)
        return t

    if poison:
        g_tensor_cache.get_bucketed = hooked
        qi.g_tensor_cache = g_tensor_cache
    try:
        idx._select_rows(q, pool, POS0, T, out, block_table=None, epp=0)
        torch.cuda.synchronize()
    finally:
        g_tensor_cache.get_bucketed = real
        qi.g_tensor_cache = g_tensor_cache
    if poison:
        # positive control: the poison really was handed to the kernel path
        assert seen, "poison never reached a workspace count buffer -- test is inert"
    return int(out.sum()), out


def main():
    clean_sum, _ = _run(0)
    print(f"clean workspace:            checksum={clean_sum}")
    poisoned_sum, out = _run(POISON)
    print(f"poisoned slot counts:       checksum={poisoned_sum}")
    assert poisoned_sum == clean_sum, \
        f"poisoned run differs: {poisoned_sum} != {clean_sum} (accumulator not initialized)"
    assert int((out >= -1).all()), "output contains values below the -1 padding"
    print("PASS: multi-tile merge ignores a stale accumulator count")


if __name__ == "__main__":
    main()