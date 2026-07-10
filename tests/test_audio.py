"""Integration tests against real ffmpeg — mirrors test_video.py's own
"real ffmpeg, real dependencies" philosophy. Skipped automatically on a
machine without ffmpeg on PATH.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

import pytest

from cerebro.cache import Cache
from cerebro.ingest import load_transcript
from cerebro.ingest.audio import load_audio
from cerebro.ingest.folder import discover_course_sources
from cerebro.ingest.video import VideoIngestError

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")


def _build_silent_mp3(tmp_path, name="lecture.mp3"):
    audio = tmp_path / name
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "1", str(audio)],
        capture_output=True, text=True, timeout=60, check=True,
    )
    return audio


def test_load_audio_transcribes_a_bare_audio_file(tmp_path):
    pytest.importorskip("faster_whisper")
    audio = _build_silent_mp3(tmp_path)
    cache = Cache(root=tmp_path / "cache")
    transcript = load_audio(audio, whisper_model="tiny", cache=cache)
    assert transcript.title == "Lecture"
    assert isinstance(transcript.segments, list)  # silent audio -> likely empty, must not raise


def test_ingest_dispatch_routes_audio_extensions_to_load_audio(tmp_path):
    pytest.importorskip("faster_whisper")
    audio = _build_silent_mp3(tmp_path, "podcast_ep1.mp3")
    cache = Cache(root=tmp_path / "cache")
    transcript = load_transcript(str(audio), whisper_model="tiny", cache=cache)
    assert transcript.title == "Podcast Ep1"


def test_load_audio_without_whisper_installed_raises_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    audio = _build_silent_mp3(tmp_path)
    cache = Cache(root=tmp_path / "cache")
    with pytest.raises(VideoIngestError, match="faster-whisper"):
        load_audio(audio, cache=cache)


def test_load_audio_raises_when_ffmpeg_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("cerebro.ingest.audio.shutil.which", lambda _: None)
    with pytest.raises(VideoIngestError, match="ffmpeg not found"):
        load_audio(tmp_path / "whatever.mp3")


def test_discover_course_sources_includes_audio_needing_transcription(tmp_path):
    _build_silent_mp3(tmp_path, "lesson1.mp3")
    sources = discover_course_sources(tmp_path)
    assert len(sources) == 1
    assert sources[0].path.name == "lesson1.mp3"
    assert sources[0].needs_transcription is True


def test_discover_course_sources_prefers_a_sidecar_transcript_for_audio(tmp_path):
    _build_silent_mp3(tmp_path, "lesson1.mp3")
    (tmp_path / "lesson1.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n", encoding="utf-8")
    sources = discover_course_sources(tmp_path)
    assert len(sources) == 1
    assert sources[0].path.suffix == ".srt"
    assert sources[0].needs_transcription is False


def test_discover_course_sources_mixes_audio_video_and_pdf(tmp_path):
    _build_silent_mp3(tmp_path, "lesson1.mp3")
    (tmp_path / "lesson2.mp4").write_bytes(b"")
    (tmp_path / "lesson3.pdf").write_bytes(b"%PDF-fake")
    sources = discover_course_sources(tmp_path)
    assert {s.path.name for s in sources} == {"lesson1.mp3", "lesson2.mp4", "lesson3.pdf"}
