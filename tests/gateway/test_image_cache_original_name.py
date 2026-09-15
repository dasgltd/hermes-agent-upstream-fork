"""Inbound image attachments keep their original filename in the cache path.

Screenshot filenames are the only chronological signal that survives Discord:
the CDN strips EXIF ``DateTime`` and screenshot XMP carries no date. Dropping the
name made a batch of screenshots unorderable.
"""
import pytest

from gateway.platforms.base import (
    _safe_original_stem,
    cache_image_from_bytes,
    get_image_cache_dir,
)

PNG = (b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)


def _cache(tmp_path, monkeypatch, name):
    monkeypatch.setattr("gateway.platforms.base.get_image_cache_dir", lambda: tmp_path)
    return cache_image_from_bytes(PNG, ".png", name)


def test_screenshot_timestamp_survives_into_the_cache_path(tmp_path, monkeypatch):
    path = _cache(tmp_path, monkeypatch, "Screenshot 2026-09-15 at 11.12.26.png")
    assert "2026-09-15" in path, f"capture date lost: {path}"
    assert path.endswith(".png")


def test_two_screenshots_are_orderable_by_their_preserved_names(tmp_path, monkeypatch):
    first = _cache(tmp_path, monkeypatch, "Screenshot 2026-09-15 at 08.58.50.png")
    second = _cache(tmp_path, monkeypatch, "Screenshot 2026-09-15 at 11.12.26.png")
    stamp = lambda p: p.split("Screenshot-")[1]
    assert stamp(first) < stamp(second)


@pytest.mark.parametrize("hostile", [
    "../../../../etc/passwd.png",
    "a/b/c.png",
    "..",
    "\x00evil.png",
])
def test_hostile_names_never_escape_the_cache_dir(tmp_path, monkeypatch, hostile):
    from pathlib import Path
    path = Path(_cache(tmp_path, monkeypatch, hostile))
    assert path.resolve().parent == tmp_path.resolve()
    assert path.name.startswith("img_")


def test_missing_name_still_caches(tmp_path, monkeypatch):
    for empty in (None, "", "   "):
        path = _cache(tmp_path, monkeypatch, empty)
        assert path.endswith(".png") and "img_" in path


def test_stem_is_bounded_so_the_path_stays_readable():
    assert len(_safe_original_stem("x" * 500)) <= 64
