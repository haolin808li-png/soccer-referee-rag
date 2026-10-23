#!/usr/bin/env python3
"""Audit CAS-FD videos and create reproducible metadata artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "raw" / "cas_fd"
OUTPUT_DIR = ROOT / "data" / "metadata"
MANIFEST = OUTPUT_DIR / "cas_fd_manifest.jsonl"
SUMMARY = OUTPUT_DIR / "cas_fd_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_frames:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {"valid": False, "error": result.stderr.strip()}
    payload = json.loads(result.stdout)
    if not payload.get("streams"):
        return {"valid": False, "error": "No video stream"}
    stream = payload["streams"][0]
    return {
        "valid": True,
        "codec": stream.get("codec_name"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "nb_frames": int(stream["nb_frames"]) if stream.get("nb_frames") else None,
        "duration_seconds": round(float(payload["format"]["duration"]), 3),
    }


def main() -> None:
    paths = sorted(DATASET.glob("*/*/*.mp4"))
    records = []
    for path in paths:
        split, label = path.relative_to(DATASET).parts[:2]
        record = {
            "sample_id": path.stem,
            "split": "validation" if split == "valid" else split,
            "label": label.lower(),
            "path": str(path.relative_to(ROOT)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        record.update(probe(path))
        records.append(record)

    hashes = defaultdict(list)
    for record in records:
        hashes[record["sha256"]].append(record)
    duplicate_groups = [group for group in hashes.values() if len(group) > 1]
    cross_split_duplicates = [
        [item["path"] for item in group]
        for group in duplicate_groups
        if len({item["split"] for item in group}) > 1
    ]

    counts = Counter((record["split"], record["label"]) for record in records)
    resolutions = Counter(
        f'{record.get("width")}x{record.get("height")}'
        for record in records
        if record.get("valid")
    )
    summary = {
        "dataset": "CAS-FD: Single-View Foul vs. Dive Dataset",
        "source": "https://huggingface.co/datasets/jahid114593/foul-vs-dive-dataset",
        "license": "CC BY-NC 4.0",
        "total_videos": len(records),
        "valid_videos": sum(record["valid"] for record in records),
        "invalid_videos": [record["path"] for record in records if not record["valid"]],
        "counts": {
            split: {
                label: counts[(split, label)] for label in ("foul", "dive")
            }
            for split in ("train", "validation", "test")
        },
        "resolutions": dict(sorted(resolutions.items())),
        "duration_seconds": {
            "min": min(record["duration_seconds"] for record in records if record["valid"]),
            "max": max(record["duration_seconds"] for record in records if record["valid"]),
            "mean": round(
                sum(record["duration_seconds"] for record in records if record["valid"])
                / sum(record["valid"] for record in records),
                3,
            ),
        },
        "exact_duplicate_groups": len(duplicate_groups),
        "cross_split_duplicates": cross_split_duplicates,
        "known_limitation": (
            "The official split is clip-level rather than source-match-level; filenames do "
            "not expose match IDs, so match-level leakage cannot be ruled out from metadata."
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    SUMMARY.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
