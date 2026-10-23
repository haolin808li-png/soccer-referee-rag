#!/usr/bin/env python3
"""Build a validated manifest for the public SoccerNet MVFouls demo clips."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "external" / "sn-mvfoul" / "VARS interface" / "dataset"
ANNOTATIONS = DATASET / "annotations.json"
OUTPUT = ROOT / "data" / "metadata" / "demo_samples.json"

SEVERITY_MAP = {
    "1.0": "no_card",
    "2.0": "borderline_no_card_yellow",
    "3.0": "yellow_card",
    "4.0": "borderline_yellow_red",
    "5.0": "red_card",
}


def probe_duration(path: Path) -> float | None:
    if not path.exists():
        return None
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nk=1:nw=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return round(float(result.stdout.strip()), 3)


def main() -> None:
    source = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    events = []
    for action_id, action in sorted(
        source["Actions"].items(), key=lambda item: int(item[0])
    ):
        clips = []
        for index, clip in enumerate(action["Clips"]):
            path = DATASET / f"action_{action_id}" / f"clip_{index}.mp4"
            clips.append(
                {
                    "clip_id": f"clip_{index}",
                    "path": str(path.relative_to(ROOT)),
                    "exists": path.exists(),
                    "duration_seconds": probe_duration(path),
                    "camera_type": clip["Camera type"],
                    "replay_speed": clip["Replay speed"],
                    "source_timestamp_ms": clip["Timestamp"],
                }
            )

        events.append(
            {
                "event_id": f"demo_action_{action_id}",
                "split": "public_demo",
                "match": action["UrlLocal"].replace("\\", "/"),
                "labels": {
                    "offence": action["Offence"],
                    "action_class": action["Action class"],
                    "severity_raw": action["Severity"],
                    "disciplinary_action": SEVERITY_MAP.get(
                        action["Severity"], "unknown"
                    ),
                    "contact": action["Contact"],
                    "body_part": action["Bodypart"],
                    "upper_body_part": action["Upper body part"] or None,
                    "try_to_play_ball": action["Try to play"] or None,
                    "touch_ball": action["Touch ball"] or None,
                },
                "clips": clips,
                "usable_clip_count": sum(clip["exists"] for clip in clips),
            }
        )

    manifest = {
        "source": "https://github.com/SoccerNet/sn-mvfoul",
        "selection": "All clips distributed in the official public VARS demo",
        "event_count": len(events),
        "usable_clip_count": sum(event["usable_clip_count"] for event in events),
        "events": events,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
