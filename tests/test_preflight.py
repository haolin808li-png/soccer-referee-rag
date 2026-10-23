import json
from pathlib import Path

import yaml

from referee_ai.preflight import run_preflight


def _event(split: str, event_id: str, match: str, clip: Path) -> dict:
    return {
        "event_id": event_id,
        "split": split,
        "match": match,
        "labels": {},
        "clips": [{"path": str(clip), "exists": True, "valid": True}],
    }


def test_preflight_detects_wrong_counts(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"video")
    paths = {}
    for split in ("train", "valid", "test"):
        manifest = tmp_path / f"{split}.jsonl"
        manifest.write_text(json.dumps(_event(split, "action_0", f"match-{split}", clip)) + "\n")
        paths[f"{split}_manifest"] = str(manifest)
    rules = tmp_path / "rules.jsonl"
    rules.write_text(json.dumps({
        "chunk_id": "r1", "version": "2026/27", "law": "12", "section": "x",
        "concept": "careless", "sanction": "foul_no_card", "text": "summary",
        "source_url": "https://example.test",
    }) + "\n")
    paths.update({"rules": str(rules), "rule_index": str(tmp_path / "index")})
    config = tmp_path / "pipeline.yaml"
    config.write_text(yaml.safe_dump({"paths": paths}))
    monkeypatch.setattr("referee_ai.preflight.EXPECTED_COUNTS", {
        "train": 2, "valid": 1, "test": 1,
    })
    report = run_preflight(config)
    assert report["status"] == "not_ready"
    assert any(x["name"] == "manifest_train_count" for x in report["failed"])
