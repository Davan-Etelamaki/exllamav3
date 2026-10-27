# Unit fence for the MTP draft block-index cache (Generator._draft_block_table),
# law 12 companion to the round-30 audit fix. CPU-only: exercises the assembly/
# cache logic directly on a bare instance, no model, no CUDA.
# Semantics: block_index_tensor columns are PAGE indices (one id per page);
# a new column is written when kv_position // PAGE_SIZE increases.
import torch

from exllamav3.generator.generator import Generator, PAGE_SIZE


class FakeSeq:
    def __init__(self, page_ids, kv_position, capacity=16):
        row = list(page_ids) + [0] * (capacity - len(page_ids))
        self.block_index_tensor = torch.tensor([row], dtype=torch.int32)
        self.kv_position = kv_position


class FakeJob:
    def __init__(self, seqs):
        self.sequences = list(seqs)


def make_gen():
    g = Generator.__new__(Generator)
    g.staging_buffers = {}
    return g


def reference_table(ready, batch_size, max_pages_batch):
    # The pre-fix code path, verbatim in structure.
    out = torch.zeros(batch_size, max_pages_batch, dtype=torch.int32)
    batch = 0
    for job in ready:
        for seq in job.sequences:
            s = seq.block_index_tensor[:, :max_pages_batch]
            out[batch:batch + 1, :s.shape[-1]] = s
            batch += 1
    return out


def test_first_call_matches_reference_and_padding_is_zero():
    ready = [FakeJob([FakeSeq(range(100, 103), 3 * PAGE_SIZE)]),
             FakeJob([FakeSeq(range(200, 202), 2 * PAGE_SIZE)])]
    g = make_gen()
    got = g._draft_block_table(ready, 2, 16).clone()
    assert torch.equal(got, reference_table(ready, 2, 16))
    assert int(got[1, 2:16].abs().sum()) == 0  # unused page columns stay zero


def test_hit_skips_reassembly_poison_proves_it():
    seq = FakeSeq([7], PAGE_SIZE - 1)
    ready = [FakeJob([seq])]
    g = make_gen()
    g._draft_block_table(ready, 1, 16)
    buf = g.staging_buffers[("draft_block_index", 16)]
    buf[:1].fill_(-12345)  # poison AFTER assembly, state unchanged
    got = g._draft_block_table(ready, 1, 16)
    # cache hit must return the poisoned buffer untouched: proves reassembly was skipped
    assert int((got[:1] == -12345).sum()) == 16


def test_page_crossing_rebuilds_with_new_page_id():
    seq = FakeSeq([5], PAGE_SIZE - 1)  # still inside page 0
    ready = [FakeJob([seq])]
    g = make_gen()
    g._draft_block_table(ready, 1, 16)
    # crossing: allocator writes the new page id at the new column, kv passes the boundary
    seq.block_index_tensor[0, 1] = 999
    seq.kv_position = PAGE_SIZE + 5
    got = g._draft_block_table(ready, 1, 16).clone()
    assert torch.equal(got, reference_table(ready, 1, 16))
    assert int(got[0, 1]) == 999  # new id present => rebuilt, poison-free


def test_membership_change_rebuilds():
    s0 = FakeSeq(range(4), 4 * PAGE_SIZE)
    s1 = FakeSeq(range(100, 104), 4 * PAGE_SIZE)
    ready = [FakeJob([s0]), FakeJob([s1])]
    g = make_gen()
    g._draft_block_table(ready, 2, 16)
    ready.pop()  # job leaves the batch
    got = g._draft_block_table(ready, 1, 16).clone()
    assert torch.equal(got, reference_table(ready, 1, 16))


def test_row_storage_realloc_rebuilds_then_hits():
    pages = list(range(2))
    seq = FakeSeq(pages, 2 * PAGE_SIZE)
    ready = [FakeJob([seq])]
    g = make_gen()
    g._draft_block_table(ready, 1, 16)
    # same content, new backing tensor (allocator reallocated storage)
    seq.block_index_tensor = torch.tensor([pages + [0] * 14], dtype=torch.int32)
    got = g._draft_block_table(ready, 1, 16).clone()
    assert torch.equal(got, reference_table(ready, 1, 16))
    buf = g.staging_buffers[("draft_block_index", 16)]
    buf[:1].fill_(-99)
    got2 = g._draft_block_table(ready, 1, 16).clone()
    assert int((got2[:1] == -99).sum()) == 16  # same sig now -> hit keeps poison