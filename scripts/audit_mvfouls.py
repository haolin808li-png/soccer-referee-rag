#!/usr/bin/env python3
"""Audit any SoccerNet-MVFouls split and emit a reproducible event manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from referee_ai.io import atomic_write_json, atomic_write_jsonl

SEVERITY = {
    "1.0": "no_card",
    "2.0": "borderline_no_card_yellow",
    "3.0": "yellow_card",
    "4.0": "borderline_yellow_red",
    "5.0": "red_card",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(path: Path) -> dict:
    process = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,width,height,avg_frame_rate,nb_frames:format=duration",
         "-of", "json", str(path)], capture_output=True, text=True
    )
    if process.returncode:
        return {"valid": False, "error": process.stderr.strip()}
    payload = json.loads(process.stdout)
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


def pick(action: dict, *names: str, default=None):
    for name in names:
        if name in action:
            value = action[name]
            return value if value != "" else default
    return default


def build(dataset: Path, split: str, root: Path) -> tuple[list[dict], dict]:
    annotation_path = dataset / "annotations.json"
    source = json.loads(annotation_path.read_text(encoding="utf-8"))
    events, hashes = [], Counter()
    for action_id, action in sorted(source["Actions"].items(), key=lambda row: int(row[0])):
        clips = []
        for index, metadata in enumerate(action["Clips"]):
            video = dataset / f"action_{action_id}" / f"clip_{index}.mp4"
            record = {
                "clip_id": f"clip_{index}",
                "path": str(video.relative_to(root) if video.is_relative_to(root) else video),
                "exists": video.exists(),
                "camera_type": pick(metadata, "Camera type"),
                "source_timestamp_ms": pick(metadata, "Timestamp"),
                "replay_speed": pick(metadata, "Replay speed"),
            }
            if video.exists():
                record.update(probe(video))
                record["sha256"] = sha256(video)
                hashes[record["sha256"]] += 1
            else:
                record.update(valid=False, error="Missing file")
            clips.append(record)
        severity = str(pick(action, "Severity", default=""))
        events.append({
            "event_id": f"action_{action_id}",
            "split": split,
            "match": str(pick(action, "UrlLocal", default="")).replace("\\", "/"),
            "labels": {
                "offence": pick(action, "Offence"),
                "contact": pick(action, "Contact"),
                "body_part": pick(action, "Bodypart", "Body part"),
                "upper_body_part": pick(action, "Upper body part"),
                "action_class": pick(action, "Action class"),
                "severity_raw": severity,
                "disciplinary_action": SEVERITY.get(severity, "unknown"),
                "multiple_fouls": pick(action, "Multiple fouls"),
                "try_to_play_ball": pick(action, "Try to play"),
                "touch_ball": pick(action, "Touch ball"),
                "handball": pick(action, "Handball"),
                "handball_offence": pick(action, "Handball offence"),
            },
            "clips": clips,
        })
    all_clips = [clip for event in events for clip in event["clips"]]
    summary = {
        "dataset": "SoccerNet-MVFouls",
        "split": split,
        "event_count_declared": source.get("Number of actions"),
        "event_count_found": len(events),
        "clip_count": len(all_clips),
        "valid_clip_count": sum(bool(clip["valid"]) for clip in all_clips),
        "invalid_clips": [clip["path"] for clip in all_clips if not clip["valid"]],
        "exact_duplicate_clip_groups": sum(count > 1 for count in hashes.values()),
        "action_class_distribution": dict(Counter(e["labels"]["action_class"] for e in events)),
        "offence_distribution": dict(Counter(e["labels"]["offence"] for e in events)),
        "disciplinary_action_distribution": dict(
            Counter(e["labels"]["disciplinary_action"] for e in events)
        ),
        "views_per_event_distribution": dict(Counter(len(e["clips"]) for e in events)),
    }
    return events, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "valid", "test", "challenge"], required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/metadata"))
    args = parser.parse_args()
    root = Path.cwd().resolve()
    events, summary = build(args.dataset.resolve(), args.split, root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(args.output_dir / f"mvfouls_{args.split}_manifest.jsonl", events)
    atomic_write_json(args.output_dir / f"mvfouls_{args.split}_audit.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
