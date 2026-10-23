"""Video inspection and deterministic, cacheable frame sampling."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class VideoError(RuntimeError):
    """Raised when ffprobe/ffmpeg cannot process a clip."""


@dataclass(frozen=True)
class FrameSample:
    path: Path
    timestamp_seconds: float


def probe_video(path: str | Path, ffprobe: str = "ffprobe") -> dict[str, Any]:
    path = Path(path)
    command = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames:format=duration",
        "-of", "json", str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise VideoError(f"ffprobe failed for {path}: {detail.strip()}") from exc
    payload = json.loads(result.stdout)
    if not payload.get("streams"):
        raise VideoError(f"No video stream found in {path}")
    stream = payload["streams"][0]
    duration = float(payload.get("format", {}).get("duration") or 0.0)
    if duration <= 0:
        raise VideoError(f"Invalid duration for {path}: {duration}")
    return {**stream, "duration_seconds": duration}


def uniform_timestamps(duration_seconds: float, count: int, edge_margin: float = 0.03) -> list[float]:
    """Return frame-centred timestamps, avoiding often-corrupt first/last frames."""
    if count < 1:
        raise ValueError("count must be positive")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    lo = min(edge_margin, duration_seconds / 4)
    usable = max(duration_seconds - 2 * lo, 0.0)
    return [lo + usable * (i + 0.5) / count for i in range(count)]


def _cache_key(path: Path, count: int) -> str:
    stat = path.stat()
    raw = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{count}".encode()
    return hashlib.sha256(raw).hexdigest()[:20]


def extract_uniform_frames(
    path: str | Path,
    output_root: str | Path,
    count: int = 16,
    *,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    jpeg_quality: int = 2,
) -> list[FrameSample]:
    """Extract frames at exact uniform timestamps and reuse a complete cache."""
    path, output_root = Path(path), Path(output_root)
    info = probe_video(path, ffprobe=ffprobe)
    timestamps = uniform_timestamps(info["duration_seconds"], count)
    target = output_root / _cache_key(path, count)
    manifest_path = target / "frames.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        samples = [FrameSample(target / x["file"], float(x["timestamp_seconds"])) for x in manifest]
        if len(samples) == count and all(x.path.exists() for x in samples):
            return samples

    output_root.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=output_root))
    try:
        samples: list[FrameSample] = []
        for index, timestamp in enumerate(timestamps):
            frame = tmp / f"frame_{index:03d}.jpg"
            command = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{timestamp:.6f}", "-i", str(path), "-frames:v", "1",
                "-q:v", str(jpeg_quality), str(frame),
            ]
            try:
                subprocess.run(command, check=True, capture_output=True)
            except (OSError, subprocess.CalledProcessError) as exc:
                detail = getattr(exc, "stderr", b"")
                if isinstance(detail, bytes):
                    detail = detail.decode(errors="replace")
                raise VideoError(f"ffmpeg failed at {timestamp:.3f}s for {path}: {detail}") from exc
            samples.append(FrameSample(frame, timestamp))
        manifest = [{"file": x.path.name, "timestamp_seconds": round(x.timestamp_seconds, 6)} for x in samples]
        (tmp / "frames.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if target.exists():
            # Another worker may have completed the same cache while we decoded.
            return extract_uniform_frames(path, output_root, count, ffmpeg=ffmpeg, ffprobe=ffprobe)
        os.replace(tmp, target)
        return [FrameSample(target / x["file"], x["timestamp_seconds"]) for x in manifest]
    finally:
        if tmp.exists():
            for child in tmp.iterdir():
                child.unlink()
            tmp.rmdir()
