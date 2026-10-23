#!/usr/bin/env python3
"""Validate the SoccerNet-MVFouls test split and build an event manifest."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "raw" / "mvfouls_2024" / "test"
ANNOTATIONS = DATASET / "annotations.json"
OUTPUT_DIR = ROOT / "data" / "metadata"
MANIFEST = OUTPUT_DIR / "mvfouls_test_manifest.jsonl"
SUMMARY = OUTPUT_DIR / "mvfouls_test_audit.json"

SEVERITY = {
    "1.0": "no_card",
    "2.0": "borderline_no_card_yellow",
    "3.0": "yellow_card",
    "4.0": "borderline_yellow_red",
    "5.0": "red_card",
}


def file_sha256(path: Path) -> str:
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


def optional(value: str):
    return value if value else None


def main() -> None:
    source = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    events = []
    hashes = Counter()
    for action_id, action in sorted(
        source["Actions"].items(), key=lambda item: int(item[0])
    ):
        clips = []
        for index, clip in enumerate(action["Clips"]):
            path = DATASET / f"action_{action_id}" / f"clip_{index}.mp4"
            clip_record = {
                "clip_id": f"clip_{index}",
                "path": str(path.relative_to(ROOT)),
                "exists": path.exists(),
                "camera_type": clip["Camera type"],
                "source_timestamp_ms": clip["Timestamp"],
                "replay_speed": clip["Replay speed"],
            }
            if path.exists():
                clip_record.update(probe(path))
                clip_record["sha256"] = file_sha256(path)
                hashes[clip_record["sha256"]] += 1
            else:
                clip_record.update({"valid": False, "error": "Missing file"})
            clips.append(clip_record)

        labels = {
            "offence": action["Offence"],
            "contact": action["Contact"],
            "body_part": action["Bodypart"],
            "upper_body_part": optional(action["Upper body part"]),
            "action_class": action["Action class"],
            "severity_raw": action["Severity"],
            "disciplinary_action": SEVERITY.get(action["Severity"], "unknown"),
            "multiple_fouls": optional(action["Multiple fouls"]),
            "try_to_play_ball": optional(action["Try to play"]),
            "touch_ball": optional(action["Touch ball"]),
            "handball": action["Handball"],
            "handball_offence": optional(action["Handball offence"]),
        }
        events.append(
            {
                "event_id": f"action_{action_id}",
                "split": "test",
                "match": action["UrlLocal"].replace("\\", "/"),
                "labels": labels,
                "clips": clips,
            }
        )

    all_clips = [clip for event in events for clip in event["clips"]]
    action_counts = Counter(event["labels"]["action_class"] for event in events)
    offence_counts = Counter(event["labels"]["offence"] for event in events)
    sanction_counts = Counter(
        event["labels"]["disciplinary_action"] for event in events
    )
    view_counts = Counter(len(event["clips"]) for event in events)
    summary = {
        "dataset": "SoccerNet-MVFouls 2024",
        "split": "test",
        "event_count_declared": source["Number of actions"],
        "event_count_found": len(events),
        "clip_count": len(all_clips),
        "valid_clip_count": sum(clip["valid"] for clip in all_clips),
        "invalid_clips": [clip["path"] for clip in all_clips if not clip["valid"]],
        "exact_duplicate_clip_groups": sum(count > 1 for count in hashes.values()),
        "action_class_distribution": dict(sorted(action_counts.items())),
        "offence_distribution": dict(sorted(offence_counts.items())),
        "disciplinary_action_distribution": dict(sorted(sanction_counts.items())),
        "views_per_event_distribution": {
            str(key): value for key, value in sorted(view_counts.items())
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events),
        encoding="utf-8",
    )
    SUMMARY.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
