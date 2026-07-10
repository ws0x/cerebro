"""Local audio ingest: podcasts, voice memos, lecture recordings -- anything
with no video track at all.

Reuses video.py's own ffmpeg-normalize-then-Whisper path wholesale (ffmpeg's
``-i`` already accepts any audio container it was built with, and
``-vn``/no-video-stream is a no-op either way) -- a bare audio file only
skips the "look for an embedded subtitle track" step video.py tries first,
since an audio-only file was never going to have one.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from ..cache import Cache
from ..transcript import Transcript
from .video import VideoIngestError, extract_audio, transcribe_with_whisper


def load_audio(path: str | Path, whisper_model: str = "base", cache: Cache | None = None) -> Transcript:
    path = Path(path)
    if shutil.which("ffmpeg") is None:
        raise VideoIngestError("ffmpeg not found on PATH; required for local audio ingest.")

    title = path.stem.replace("_", " ").replace("-", " ").strip().title()

    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_path = extract_audio(path, Path(tmp_dir) / "audio.wav")
        segments = transcribe_with_whisper(audio_path, whisper_model, cache=cache)
        return Transcript(source=str(path), title=title, segments=segments)
