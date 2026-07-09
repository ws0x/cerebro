"""Local video ingest: embedded subtitle extraction, with a Whisper fallback.

Two offline paths, tried in order:
  1. The container has a text-based subtitle track (subrip/ass/webvtt/mov_text)
     -> demux it with ffmpeg. Fast, exact, no ML involved.
  2. No usable subtitle track -> transcribe the audio with faster-whisper
     (an optional dependency: ``pip install cerebro[whisper]``).

Image-based subtitle codecs (dvd_subtitle, hdmv_pgs_subtitle — common in some
ripped .mkv files) need OCR and are not supported; they're treated as "no
subtitle" and fall through to Whisper.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..cache import Cache
from ..transcript import Segment, Transcript

_TEXT_SUBTITLE_CODECS = {"subrip", "ass", "ssa", "webvtt", "mov_text"}


class VideoIngestError(RuntimeError):
    pass


def _run(cmd: list[str], timeout: int, error_prefix: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)
    except FileNotFoundError as exc:
        raise VideoIngestError(
            f"{cmd[0]} not found on PATH; ffmpeg is required for local video ingest."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise VideoIngestError(f"{error_prefix}: {exc.stderr[:300]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise VideoIngestError(f"{error_prefix}: timed out after {timeout}s") from exc


def find_text_subtitle_stream(path: Path) -> int | None:
    """Return the ffmpeg stream index of the first text-based subtitle track, if any."""
    proc = _run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(path)],
        timeout=30,
        error_prefix=f"ffprobe failed on {path.name}",
    )
    for stream in json.loads(proc.stdout).get("streams", []):
        if stream.get("codec_type") == "subtitle" and stream.get("codec_name") in _TEXT_SUBTITLE_CODECS:
            return stream["index"]
    return None


def extract_embedded_subtitle(path: Path, stream_index: int, dest: Path) -> Path:
    _run(
        ["ffmpeg", "-y", "-i", str(path), "-map", f"0:{stream_index}", str(dest)],
        timeout=120,
        error_prefix=f"Failed to extract subtitle track from {path.name}",
    )
    return dest


def extract_audio(path: Path, dest: Path) -> Path:
    _run(
        ["ffmpeg", "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", str(dest)],
        timeout=1800,
        error_prefix=f"Failed to extract audio from {path.name}",
    )
    return dest


def _whisper_cache_key(audio_path: Path, model_size: str) -> str:
    stat = audio_path.stat()
    return Cache.key("whisper", model_size, audio_path.name, stat.st_size)


def transcribe_with_whisper(
    audio_path: Path, model_size: str = "base", cache: Cache | None = None
) -> list[Segment]:
    cache = cache or Cache()
    key = _whisper_cache_key(audio_path, model_size)
    cached = cache.get(key)
    if cached is not None:
        return [Segment(**s) for s in cached]

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise VideoIngestError(
            "No subtitle track found and faster-whisper is not installed. "
            "Install it with: pip install cerebro[whisper]"
        ) from exc

    model = WhisperModel(model_size, device="auto", compute_type="auto")
    raw_segments, _info = model.transcribe(str(audio_path))
    segments = [
        Segment(text=s.text.strip(), start=s.start, duration=s.end - s.start)
        for s in raw_segments
        if s.text.strip()
    ]
    cache.set(key, [{"text": s.text, "start": s.start, "duration": s.duration} for s in segments])
    return segments


def load_video(path: str | Path, whisper_model: str = "base", cache: Cache | None = None) -> Transcript:
    path = Path(path)
    if shutil.which("ffmpeg") is None:
        raise VideoIngestError("ffmpeg not found on PATH; required for local video ingest.")

    title = path.stem.replace("_", " ").replace("-", " ").strip().title()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        stream_index = find_text_subtitle_stream(path)
        if stream_index is not None:
            from .subtitles import load_subtitle_file

            srt_path = extract_embedded_subtitle(path, stream_index, tmp / "embedded.srt")
            extracted = load_subtitle_file(srt_path)
            return Transcript(source=str(path), title=title, segments=extracted.segments)

        audio_path = extract_audio(path, tmp / "audio.wav")
        segments = transcribe_with_whisper(audio_path, whisper_model, cache=cache)
        return Transcript(source=str(path), title=title, segments=segments)
