"""Tests for convert/util.py's shared helpers, especially atomic_write --
the fix for all three output writers previously writing straight onto the
destination path with no crash safety."""

from __future__ import annotations

from pathlib import Path

import pytest

from cerebro.convert.util import atomic_write


def test_atomic_write_creates_file_with_content(tmp_path):
    path = tmp_path / "out.txt"
    atomic_write(path, lambda tmp: tmp.write_text("hello", encoding="utf-8"))
    assert path.read_text(encoding="utf-8") == "hello"


def test_atomic_write_leaves_no_temp_file_behind(tmp_path):
    path = tmp_path / "out.txt"
    atomic_write(path, lambda tmp: tmp.write_text("hello", encoding="utf-8"))
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_write_does_not_touch_existing_file_on_failure(tmp_path):
    path = tmp_path / "out.txt"
    path.write_text("original", encoding="utf-8")

    def _boom(tmp: Path) -> None:
        tmp.write_text("partial", encoding="utf-8")
        raise RuntimeError("simulated crash mid-write")

    with pytest.raises(RuntimeError, match="simulated crash"):
        atomic_write(path, _boom)

    assert path.read_text(encoding="utf-8") == "original"


def test_atomic_write_leaves_no_temp_file_behind_on_failure(tmp_path):
    path = tmp_path / "out.txt"

    def _boom(tmp: Path) -> None:
        tmp.write_text("partial", encoding="utf-8")
        raise RuntimeError("simulated crash mid-write")

    with pytest.raises(RuntimeError):
        atomic_write(path, _boom)

    assert list(tmp_path.iterdir()) == []


def test_atomic_write_overwrites_existing_file(tmp_path):
    path = tmp_path / "out.txt"
    path.write_text("old", encoding="utf-8")
    atomic_write(path, lambda tmp: tmp.write_text("new", encoding="utf-8"))
    assert path.read_text(encoding="utf-8") == "new"
