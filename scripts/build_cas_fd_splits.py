#!/usr/bin/env python3
"""Create deterministic, exact-deduplicated, class-balanced CAS-FD splits."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "metadata" / "cas_fd_manifest.jsonl"
OUTPUT = ROOT / "data" / "metadata" / "cas_fd_splits.csv"

PER_CLASS = {"train": 160, "validation": 42, "test": 42}


def main() -> None:
    records = [json.loads(line) for line in MANIFEST.read_text().splitlines()]

    # Keep one deterministic representative for every byte-identical video.
    by_hash = {}
    for record in sorted(records, key=lambda item: item["path"]):
        by_hash.setdefault(record["sha256"], record)

    by_label = defaultdict(list)
    for record in by_hash.values():
        by_label[record["label"]].append(record)

    target_per_class = sum(PER_CLASS.values())
    if min(map(len, by_label.values())) < target_per_class:
        raise RuntimeError("Not enough unique videos for the requested balanced split")

    rows = []
    for label in ("foul", "dive"):
        # Sorting by content hash makes selection and assignment reproducible.
        selected = sorted(by_label[label], key=lambda item: item["sha256"])[
            :target_per_class
        ]
        offset = 0
        for split, count in PER_CLASS.items():
            for record in selected[offset : offset + count]:
                rows.append(
                    {
                        "sample_id": record["sample_id"],
                        "split": split,
                        "label": label,
                        "path": record["path"],
                        "sha256": record["sha256"],
                    }
                )
            offset += count

    rows.sort(key=lambda item: (item["split"], item["label"], item["sha256"]))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
