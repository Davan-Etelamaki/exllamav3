"""
A recurrent checkpoint must record the position its page-boundary key promises.

Checkpoints are filed under the hash of the page that ENDS at a boundary, so a resume derives the state
position it restores from the number of cached pages: Job.allocate_pages calls
Generator.recurrent_state_from_stashed(..., position = cached_pages * PAGE_SIZE), and GDNState.unstash then
asserts that against the position the stash recorded for itself. When the two disagree the request dies with
a bare AssertionError (observed on dot26 2026-11-22 22:04:07 EST, request #1 of a fresh boot).

The generator can leave the recurrent state ahead of the committed K/V position while a speculative window
has not settled, so the writer has to guarantee the invariant, not the reader. commit_recurrent_checkpoint
already does (state.settled_at(boundary)); find_recurrent_stash only masks the symptom by filtering drifted
checkpoints out, which silently costs prompt-cache reuse instead of failing.
"""

from types import SimpleNamespace

import pytest
import torch

from exllamav3.constants import PAGE_SIZE
from exllamav3.generator.job import Job

pytestmark = pytest.mark.nogpu

BOUNDARY_PAGES = 4


class FakeState:
    """The slice of PairedState that the checkpoint writers read."""

    def __init__(self, position, last_history = 0):
        self.position = position
        self.last_history = last_history
        self.checkpoint_size = 1
        self.stash_by_rollback = False

    def settled_at(self, position):
        return self.position == position and self.last_history == 0

    def stash(self):
        return {"position": self.position, "checkpoint_size": self.checkpoint_size}

    def rewind_to(self, position):
        self.position = position


class FakeCache:
    """Records what a writer would have filed under which page hash."""

    def __init__(self):
        self.filed = []

    def put(self, key, state, owner = None):
        self.filed.append((key, state.stash()))


def make_job(kv_position, state_position, last_history = 0):
    job = Job.__new__(Job)
    pages = [
        SimpleNamespace(kv_position = PAGE_SIZE, phash = ("page-%d" % i).encode())
        for i in range(kv_position // PAGE_SIZE)
    ]
    seq = SimpleNamespace(
        kv_position = kv_position,
        sequence_ids = torch.zeros((1, kv_position + 1), dtype = torch.long),
        allocated_pages = pages,
    )
    job.sequences = [seq]
    job.cached_pages = 0
    job.last_recurrent_checkpoint_pos = None
    job.recurrent_state = FakeState(state_position, last_history)
    job.generator = SimpleNamespace(
        max_chunk_size = PAGE_SIZE,
        recurrent_checkpoint_interval = PAGE_SIZE,
        recurrent_checkpoint_interval_pp = 32768,
        rolling_checkpoint = False,
    )
    return job, pages


def file_checkpoint(job):
    cache = FakeCache()
    Job.maybe_stash_recurrent(job, cache)
    return cache.filed


def test_state_ahead_of_committed_tokens_is_not_stashed():
    boundary = BOUNDARY_PAGES * PAGE_SIZE
    job, _ = make_job(boundary, state_position = boundary + 6)
    assert file_checkpoint(job) == [], \
        "an unsettled state was filed as the checkpoint for a boundary it never reached"


def test_unsettled_speculative_history_is_not_stashed():
    boundary = BOUNDARY_PAGES * PAGE_SIZE
    job, _ = make_job(boundary, state_position = boundary, last_history = 5)
    assert file_checkpoint(job) == [], \
        "a state with an unsettled speculative pass was filed as a checkpoint"


def test_settled_state_stashes_one_checkpoint_at_its_boundary():
    boundary = BOUNDARY_PAGES * PAGE_SIZE
    job, pages = make_job(boundary, state_position = boundary)
    filed = file_checkpoint(job)
    assert len(filed) == 1, "a settled state at a checkpoint boundary must still produce a checkpoint"
    key, stashed = filed[0]
    assert key == pages[-1].phash
    assert stashed["position"] == len(pages) * PAGE_SIZE


@pytest.mark.parametrize("drift", [0, 1, 6, PAGE_SIZE])
def test_no_filed_checkpoint_breaks_the_position_a_resume_derives(drift):
    boundary = BOUNDARY_PAGES * PAGE_SIZE
    job, pages = make_job(boundary, state_position = boundary + drift)
    resume_position = len(pages) * PAGE_SIZE
    for _, stashed in file_checkpoint(job):
        assert stashed["position"] == resume_position, \
            "GDNState.unstash would abort a request that resumes from this checkpoint"


# ---------------------------------------------------------------- reader side


class FakePageTable:
    """Hands back the page prefix an allocation would reuse, without touching CUDA."""

    def __init__(self, cached_pages):
        self.cached_pages = cached_pages
        self.metrics = {"alloc_cached_pages": 0, "alloc_kv_only_pages": 0}

    def allocate_pages(self, page_hashes, new_unique_pages, recurrent_pages, protected_hashes, restore_limit):
        cached = min(self.cached_pages, len(recurrent_pages)) if recurrent_pages else 0
        pages = [SimpleNamespace(page_index = i, kv_position = PAGE_SIZE, phash = page_hashes[i])
                 for i in range(cached)]
        return pages, cached * PAGE_SIZE, cached, 0


class FakeCheckpointStore:
    """A recurrent cache whose checkpoints are keyed by page hash, with a drifted one on the last page."""

    def __init__(self, positions):
        self.positions = positions

    def has(self, key):
        return key in self.positions

    def get_stashed(self, key, default = None):
        position = self.positions.get(key)
        return None if position is None else {"position": position, "checkpoint_size": 1}


def make_sequence(n_pages):
    from exllamav3.generator.pagetable import Sequence
    seq = Sequence.__new__(Sequence)
    seq.max_cached_pages = None
    seq.page_hashes = [("hash-%d" % i).encode() for i in range(n_pages)]
    seq.new_unique_pages = 0
    seq.mm_first_page = None
    seq.allocated_pages = []
    seq.kv_position = 0
    return seq


def test_a_resume_never_gets_a_checkpoint_whose_position_it_cannot_derive():
    # Exactly the equality GDNState.unstash asserts after Generator.recurrent_state_from_stashed is handed
    # position = cached_pages * PAGE_SIZE. Here the deepest checkpoint drifted past its page boundary and a
    # lower one is intact: the allocation must fall back to the lower one instead of handing over the drift.
    from exllamav3.generator.pagetable import Sequence
    n_pages = BOUNDARY_PAGES
    seq = make_sequence(n_pages)
    drifted = (n_pages * PAGE_SIZE) + 6
    intact = (n_pages - 2) * PAGE_SIZE
    store = FakeCheckpointStore({
        ("hash-%d" % i).encode(): intact for i in range(n_pages - 1)
    })
    store.positions[("hash-%d" % (n_pages - 1)).encode()] = drifted
    seq.build_block_index_tensor = lambda: None
    cached = Sequence.allocate_pages(seq, FakePageTable(n_pages), store, None)
    _, cached_pages, _, stashed = cached
    assert stashed is not None, "an intact lower checkpoint should still resume the sequence"
    assert stashed["position"] == cached_pages * PAGE_SIZE, \
        "restoring at cached_pages * PAGE_SIZE would abort in GDNState.unstash"
    assert seq.kv_position == cached_pages * PAGE_SIZE, \
        "the sequence kept a K/V position the restored state cannot back"