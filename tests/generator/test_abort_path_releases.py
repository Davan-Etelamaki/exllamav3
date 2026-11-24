"""
Regression tests for the abort path: a job that leaves the generator must give its allocations back and
must hand its finished pages to the persistent tier.

Two seams, both measured on dot26 2026-11-24 (six aborted 40,022-token ingests left the store 5 pages,
journal 18:51:09; the wedge of 17:19-17:38 left VRAM at 24,044 MiB with zero completions):

1. Generator.cancel and Generator.reap_failed_job free a leaving job's VRAM pages through
   Job.deallocate_pages(), which empties seq.allocated_pages, but neither offers those pages to the disk
   tier first. Only an idle transition (on_queue_drained -> flush_cache_to_disk) writes them, and a lane
   under continuous external traffic never idles, so an aborted ingestion's pages are recycled for the next
   request instead of written down. RED on the installed wheel: Generator has no _offer_job_pages_to_disk.
2. When an exception escapes Generator.iterate, AsyncGenerator._run_iteration latches the error, disables
   disk writes and clears its own job map, leaving every job in the sync generator's pending_jobs/active_jobs
   behind. Generator.cancel is the only path that frees a running job, and AsyncGenerator.cancel refuses to
   reach the sync generator once the error is latched, so those pages and recurrent state slots stay
   referenced for the life of the process. The contained-failure path (reap_failed_job) does free them, which
   is the asymmetry. RED on the installed wheel: the stranded jobs are never cancelled.

No GPU or model is required: Generator and AsyncGenerator are built with __new__ and given stubs, so each
test asserts on the calls the release paths make.
"""

import asyncio
import logging
import types

import pytest

from exllamav3.constants import PAGE_SIZE
from exllamav3.generator.async_generator import AsyncGenerator
from exllamav3.generator.generator import Generator
from exllamav3.generator.pagetable import is_content_hash

pytestmark = pytest.mark.nogpu

def content_hash(tag: int, namespace: int = 0) -> bytes:
    """A unique page hash the tier treats as content-addressed (a random placeholder zeroes its top 8 bytes)."""
    h = bytes([0xA1, namespace]) + tag.to_bytes(14, "big")
    assert is_content_hash(h), "fixture must hand out the same kind of hash the tier accepts"
    return h


class StubPage:
    def __init__(self, n: int, complete: bool = True):
        self.phash = content_hash(n)
        self.kv_position = PAGE_SIZE if complete else PAGE_SIZE // 2
        self.persistable = True


class StubSeq:
    def __init__(self, pages, kv_position: int):
        self.allocated_pages = pages
        self.kv_position = kv_position


class StubJob:
    """Records the order of the calls that free it; deallocate empties allocated_pages as the real one does."""

    def __init__(self, pages: int = 6, kv_position: int = 6 * PAGE_SIZE):
        self.sequences = [StubSeq([StubPage(i) for i in range(pages)], kv_position)]
        self.serial_number = 1
        self.calls = []

    def commit_recurrent_checkpoint(self):
        self.calls.append("checkpoint")

    def deallocate_pages(self):
        self.calls.append("deallocate")
        for seq in self.sequences:
            seq.allocated_pages = []


class StubDisk:
    def __init__(self, queued: int = 0, write_disabled: bool = False):
        self.write_disabled = write_disabled
        self.pending_pages = {content_hash(1000 + i) for i in range(queued)}
        self.page_file_size = 2359296
        self.aborted = False

    def room_for(self, nbytes: int) -> bool:
        return True

    def has_page(self, h) -> bool:
        return False

    def abort(self):
        self.aborted = True
        self.write_disabled = True


class StubTier:
    """The CPU tier: what store() receives is what the disk tier will be asked to write."""

    def __init__(self, max_slots: int = 64, holds=(), occupied: int = 0):
        self.max_slots = max_slots
        self.stored = []
        self._holds = set(holds)
        self.entries = {content_hash(500 + i): {} for i in range(occupied)}

    def __contains__(self, h) -> bool:
        return h in self._holds

    def store(self, page, serial, persist=True):
        self.stored.append(page.phash)
        self.entries[page.phash] = {}


class StubPagetable:
    access_serial = 5

    def __init__(self):
        self.referenced_pages = {}
        self.unreferenced_pages = {}


def make_generator(active=(), pending=(), disk=None, tier=None):
    generator = Generator.__new__(Generator)
    generator.pipeline = None
    generator.pagetable = StubPagetable()
    generator.pending_jobs = list(pending)
    generator.active_jobs = list(active)
    generator.disk_page_cache = StubDisk() if disk is None else disk
    generator.cpu_page_cache = StubTier() if tier is None else tier
    generator.recurrent_cache = None
    generator.disk_cache_flush_on_idle = True
    generator.disk_cache_write_interval = 0
    # The idle-transition housekeeping is the pre-existing path; stub it so these tests see only whether a
    # cancellation drained the queue, not what defragmentation and pruning would have done
    generator.drained = []
    generator.on_queue_drained = lambda: generator.drained.append(True)
    return generator


# ---- seam 1: a leaving job offers its complete pages before its VRAM is freed -------------------------

def test_cancel_offers_the_leaving_job_pages_before_freeing_them():
    victim = StubJob(pages=6, kv_position=6 * PAGE_SIZE)
    keeper = StubJob()
    tier = StubTier()
    generator = make_generator(active=[victim, keeper], tier=tier)

    Generator.cancel(generator, victim)

    # The two pages before the sequence position stay out: speculative decoding can still rewind them, so an
    # offer that ran AFTER deallocate_pages (which empties allocated_pages) would also store nothing here.
    assert len(tier.stored) == 4, f"a cancelled job must hand its complete pages to the tier, got {tier.stored}"
    assert victim.calls == ["checkpoint", "deallocate"]
    assert victim not in generator.active_jobs
    assert keeper in generator.active_jobs


def test_reap_offers_the_failed_job_pages():
    victim = StubJob(pages=6, kv_position=6 * PAGE_SIZE)
    tier = StubTier()
    generator = make_generator(active=[victim], tier=tier)
    results = []

    Generator.reap_failed_job(generator, victim, RuntimeError("boom"), results)

    assert len(tier.stored) == 4
    assert victim not in generator.active_jobs
    assert results[-1]["stage"] == "error" and results[-1]["job"] is victim


def test_offer_skips_pages_the_tier_cannot_reuse():
    # 6 pages: [0] already in the CPU tier, [3] not yet complete, [4]/[5] inside the rewind guard
    pages = [StubPage(0), StubPage(1), StubPage(2), StubPage(3, complete=False), StubPage(4), StubPage(5)]
    victim = StubJob()
    victim.sequences = [StubSeq(pages, 6 * PAGE_SIZE)]
    tier = StubTier(holds=[pages[0].phash])
    generator = make_generator(active=[victim], tier=tier)

    Generator.cancel(generator, victim)

    assert tier.stored == [pages[1].phash, pages[2].phash]


def test_offer_is_skipped_while_disk_writes_are_disabled():
    victim = StubJob()
    disk = StubDisk(write_disabled=True)
    tier = StubTier()
    generator = make_generator(active=[victim, StubJob()], disk=disk, tier=tier)

    Generator.cancel(generator, victim)

    assert tier.stored == []
    assert victim.calls == ["checkpoint", "deallocate"]  # the job is still freed


def test_offer_respects_the_writer_budget():
    # Budget mirrors write_behind: a quarter of the CPU tier less the writes still queued. 20 queued against
    # 64 slots leaves nothing, so a cancellation storm cannot make the tier wait on the writer.
    victim = StubJob()
    disk = StubDisk(queued=20)
    tier = StubTier(max_slots=64)
    generator = make_generator(active=[victim, StubJob()], disk=disk, tier=tier)

    Generator.cancel(generator, victim)

    assert tier.stored == []


def test_offer_never_queues_more_than_the_tier_has_room_for():
    # Cancellation runs on the caller's thread, not the engine's: a store that has to steal a busy slot
    # waits for the writer there. 62 of 64 slots are taken, so only the 2 free ones may be queued even
    # though the job has 8 complete pages and the writer budget allows 16.
    victim = StubJob(pages=10, kv_position=10 * PAGE_SIZE)
    tier = StubTier(max_slots=64, occupied=62)
    generator = make_generator(active=[victim, StubJob()], tier=tier)

    Generator.cancel(generator, victim)

    assert len(tier.stored) == 2, f"the offer must fit the tier's free slots, queued {len(tier.stored)}"


def test_offer_without_a_tier_is_inert():
    victim = StubJob()
    generator = make_generator(active=[victim, StubJob()], tier=None)
    generator.cpu_page_cache = None

    assert Generator._offer_job_pages_to_disk(generator, victim) == 0
    assert victim.calls == []


# ---- seam 2: a latched generator error releases the jobs it strands -----------------------------------

class StrandingGenerator:
    """Stand-in for the sync Generator whose iterate() raises: holds jobs, records what it released."""

    def __init__(self, error, pending=(), active=()):
        self.error = error
        self.pending_jobs = list(pending)
        self.active_jobs = list(active)
        self.disk_page_cache = StubDisk()
        self.released = []
        self._jobs = list(pending) + list(active)

    def iterate(self):
        raise self.error

    def cancel(self, job):
        self.released.append(job)
        for where in (self.pending_jobs, self.active_jobs):
            if job in where:
                where.remove(job)

    def num_remaining_jobs(self):
        return len(self.pending_jobs) + len(self.active_jobs)


class StubAsyncJob:
    def __init__(self):
        self.received = []

    def put_result(self, result):
        self.received.append(result)


def make_wrapper(sync_generator):
    wrapper = AsyncGenerator.__new__(AsyncGenerator)
    wrapper.generator = sync_generator
    wrapper.error = None
    wrapper.condition = asyncio.Condition()
    wrapper.iteration_task = types.SimpleNamespace(cancelled = lambda: False)
    return wrapper


def test_latched_error_releases_the_stranded_jobs():
    boom = RuntimeError("escape")
    pending, active = object(), object()
    sync = StrandingGenerator(boom, pending=[pending], active=[active])
    consumer = StubAsyncJob()
    wrapper = make_wrapper(sync)
    wrapper.jobs = {object(): consumer}

    asyncio.run(wrapper._run_iteration())

    assert wrapper.error is boom
    assert sync.released == [pending, active], "the jobs the sync generator holds must be released"
    assert sync.num_remaining_jobs() == 0
    assert sync.disk_page_cache.aborted
    assert consumer.received == [boom]
    assert wrapper.jobs == {}


def test_latched_error_survives_a_failing_release_and_logs_it():
    boom = RuntimeError("escape")

    class BadRelease(StrandingGenerator):
        def cancel(self, job):
            raise RuntimeError("release failed")

    sync = BadRelease(boom, active=[object()])
    consumer = StubAsyncJob()
    wrapper = make_wrapper(sync)
    wrapper.jobs = {object(): consumer}
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    logger = logging.getLogger("exllamav3.generator.async_generator")
    capture = Capture()
    logger.addHandler(capture)
    try:
        asyncio.run(wrapper._run_iteration())
    finally:
        logger.removeHandler(capture)

    assert wrapper.error is boom, "a failed release must not replace the error consumers receive"
    assert consumer.received == [boom]
    assert any("stranded" in r.getMessage() for r in records)


class HealthyGenerator:
    def __init__(self):
        self.active_jobs = [object()]
        self.pending_jobs = []
        self.disk_page_cache = StubDisk()
        self.released = []
        self.rounds = 0

    def has_idle_work(self):
        return False

    def iterate(self):
        self.rounds += 1
        if self.rounds >= 3:
            raise asyncio.CancelledError
        return []

    def cancel(self, job):
        self.released.append(job)


def test_a_healthy_iteration_does_not_release_anything():
    sync = HealthyGenerator()
    wrapper = make_wrapper(sync)
    wrapper.jobs = {object(): StubAsyncJob()}   # a watched job keeps the loop out of the idle wait

    asyncio.run(wrapper._run_iteration())

    assert wrapper.error is None
    assert sync.released == []
    assert sync.rounds == 3