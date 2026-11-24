"""Fence for the disk tier's identity key (laws 12/21/31).

The key must be stable across rebuilds of the same release and must still rotate when the on-disk
byte layout or the release changes. Folding the wheel's local build tag (its git sha) into the key
made every rebuild on dot26 start an empty store, so the persistent tier could never be consulted:
see §G of sm89-kernel-lane-brief.md (four stores, 6.4 GB unreadable, counters pinned at zero).
"""

import pytest

import exllamav3.generator.disk_cache as dc


CANONICAL = [
    {"rows": 256, "canonical_row_bytes": 512, "dtype": "torch.uint8", "image_offset": 0},
    {"rows": 256, "canonical_row_bytes": 512, "dtype": "torch.uint8", "image_offset": 131072},
]
IMAGE_SIZE = 2359296


def _fp(monkeypatch, version=None, fmt=None):
    if version is not None:
        monkeypatch.setattr(dc, "__version__", version)
    if fmt is not None:
        monkeypatch.setattr(dc, "DISK_CACHE_FORMAT_VERSION", fmt)
    return dc.cache_fingerprint([], [dict(c) for c in CANONICAL], IMAGE_SIZE)


def test_rebuilding_the_same_release_keeps_the_store(monkeypatch):
    a, parts_a = _fp(monkeypatch, "1.6.7+0a19785408ce0")
    monkeypatch.undo()
    b, _ = _fp(monkeypatch, "1.6.7+038204d31a713")
    assert a == b, "the local build tag rotated the persistent tier"


def test_a_different_release_still_rotates_the_store(monkeypatch):
    a, _ = _fp(monkeypatch, "1.6.7+0a19785408ce0")
    monkeypatch.undo()
    b, _ = _fp(monkeypatch, "1.6.8+0a19785408ce0")
    assert a != b


def test_a_format_bump_rotates_the_store(monkeypatch):
    a, _ = _fp(monkeypatch, "1.6.7+0a19785408ce0")
    monkeypatch.undo()
    b, _ = _fp(monkeypatch, "1.6.7+0a19785408ce0", fmt=99)
    assert a != b, "the layout duty no longer rotates the key"


def test_the_writing_build_is_recorded_beside_the_key(monkeypatch):
    _, parts = _fp(monkeypatch, "1.6.7+0a19785408ce0")
    assert parts["build"] == "1.6.7+0a19785408ce0"
    assert parts["exllamav3"] == "1.6.7"


def test_the_key_still_tracks_the_page_geometry(monkeypatch):
    a, _ = _fp(monkeypatch, "1.6.7+0a19785408ce0")
    monkeypatch.undo()
    other = [dict(CANONICAL[0]), dict(CANONICAL[1], canonical_row_bytes = 1024)]
    b = dc.cache_fingerprint([], other, IMAGE_SIZE)[0]
    assert a != b
