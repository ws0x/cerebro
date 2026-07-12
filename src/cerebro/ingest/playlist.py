"""YouTube playlist listing via yt-dlp's flat extraction (no video downloads).

Flat extraction just reads the playlist page's video list — it's fast and
doesn't touch each video, so listing a 200-video playlist takes seconds, not
minutes.
"""

from __future__ import annotations

from dataclasses import dataclass


class PlaylistIngestError(RuntimeError):
    """Raised when a playlist URL can't be listed (private, deleted, geo-blocked, or offline)."""


def is_playlist_url(url: str) -> bool:
    return "list=" in url or "/playlist" in url


@dataclass
class PlaylistInfo:
    title: str
    items: list[tuple[str, str]]  # (video_title, watch_url)


def load_playlist(url: str) -> PlaylistInfo:
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError

    opts = {"extract_flat": "in_playlist", "quiet": True, "skip_download": True, "no_warnings": True}
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except DownloadError as exc:
        raise PlaylistIngestError(
            f"Could not read playlist {url}: {exc}. It may be private, deleted, "
            "region-locked, or you may be offline."
        ) from exc

    if info is None:
        raise PlaylistIngestError(f"Could not read playlist {url}: no data returned.")

    items: list[tuple[str, str]] = []
    for entry in info.get("entries") or []:
        if not entry:
            continue
        video_id = entry.get("id")
        watch_url = f"https://www.youtube.com/watch?v={video_id}" if video_id else entry.get("url")
        title = entry.get("title") or video_id or "Untitled"
        if watch_url:
            items.append((title, watch_url))

    return PlaylistInfo(title=info.get("title") or "Playlist", items=items)
