"""Integration tests against real ffmpeg — this project depends on ffmpeg for
local video ingest, so these exercise the actual subprocess calls rather than
mocking them. Skipped automatically on a machine without ffmpeg on PATH.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

import pytest

from cerebro.cache import Cache
from cerebro.ingest import load_transcript
from cerebro.ingest.video import (
    VideoIngestError,
    find_text_subtitle_stream,
    load_video,
    transcribe_with_whisper,
)

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")

_SRT = (
    "1\n00:00:00,000 --> 00:00:01,500\nHello from embedded subtitles.\n\n"
    "2\n00:00:01,500 --> 00:00:03,000\nThis proves ffmpeg extraction works.\n"
)


def _build_mkv_with_embedded_subtitle(tmp_path):
    srt = tmp_path / "sub.srt"
    srt.write_text(_SRT, encoding="utf-8")
    video = tmp_path / "lesson.mkv"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=64x64:d=3",
            "-i", str(srt),
            "-c:v", "libx264", "-c:s", "srt", "-shortest",
            str(video),
        ],
        capture_output=True, text=True, timeout=60, check=True,
    )
    return video


def _build_mkv_without_subtitle(tmp_path):
    # Needs a real (silent) audio stream, or extract_audio itself fails before
    # ever reaching the Whisper-availability check this test targets.
    video = tmp_path / "no_subs.mkv"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
            "-c:v", "libx264", "-shortest",
            str(video),
        ],
        capture_output=True, text=True, timeout=60, check=True,
    )
    return video


def test_finds_embedded_text_subtitle_stream(tmp_path):
    video = _build_mkv_with_embedded_subtitle(tmp_path)
    assert find_text_subtitle_stream(video) == 1


def test_load_video_extracts_embedded_subtitle_into_transcript(tmp_path):
    video = _build_mkv_with_embedded_subtitle(tmp_path)
    transcript = load_video(video)
    assert transcript.title == "Lesson"
    assert len(transcript.segments) == 2
    assert "Hello from embedded subtitles" in transcript.segments[0].text
    assert transcript.segments[1].start == pytest.approx(1.5, abs=0.05)


def test_ingest_dispatch_routes_video_extension_to_load_video(tmp_path):
    video = _build_mkv_with_embedded_subtitle(tmp_path)
    transcript = load_transcript(str(video))
    assert len(transcript.segments) == 2


def test_no_subtitle_and_no_whisper_raises_actionable_error(tmp_path, monkeypatch):
    # faster-whisper is an optional extra. Force the "not installed" branch via
    # sys.modules rather than relying on the dev environment's actual install
    # state, so this test is deterministic whether or not the extra is present.
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    video = _build_mkv_without_subtitle(tmp_path)
    cache = Cache(root=tmp_path / "cache")
    with pytest.raises(VideoIngestError, match="faster-whisper"):
        load_video(video, cache=cache)


def test_transcribe_with_whisper_not_installed_raises_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    fake_audio = tmp_path / "audio.wav"
    fake_audio.write_bytes(b"\x00")
    cache = Cache(root=tmp_path / "cache")
    with pytest.raises(VideoIngestError, match="pip install cerebro\\[whisper\\]"):
        transcribe_with_whisper(fake_audio, cache=cache)


def test_no_subtitle_uses_real_whisper_when_installed(tmp_path):
    pytest.importorskip("faster_whisper")
    video = _build_mkv_without_subtitle(tmp_path)  # silent audio -> fast, no speech expected
    cache = Cache(root=tmp_path / "cache")
    transcript = load_video(video, whisper_model="tiny", cache=cache)
    assert transcript.title == "No Subs"
    assert isinstance(transcript.segments, list)  # silent audio -> likely empty, must not raise
