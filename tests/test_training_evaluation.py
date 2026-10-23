from __future__ import annotations

import json

import pytest

from referee_ai.evaluation import evaluate_predictions, expected_calibration_error
from referee_ai.training import build_sft_record, prepare_sft_dataset, validate_official_splits


def event(event_id: str, split: str, match: str) -> dict:
    return {
        "event_id": event_id, "split": split, "match": match,
        "labels": {"offence": "Offence", "severity_raw": "3.0", "disciplinary_action": "yellow_card"},
        "clips": [{"path": f"/{event_id}.mp4", "valid": True}],
    }


def test_split_leakage_is_rejected() -> None:
    with pytest.raises(ValueError, match="leakage"):
        validate_official_splits([event("a", "train", "m")], [event("b", "valid", "m")], [event("c", "test", "z")])
    with pytest.raises(ValueError, match="different official split"):
        validate_official_splits([event("a", "test", "m1")], [event("b", "valid", "m2")], [event("c", "test", "m3")])


def test_sft_record_contains_video_semantics_and_target() -> None:
    row = build_sft_record(event("a", "train", "m1"), {"contact": "studs to ankle"})
    assert row["messages"][1]["content"][0] == {"type": "video", "video": "/a.mp4"}
    assert "studs to ankle" in row["messages"][1]["content"][-1]["text"]
    target = json.loads(row["messages"][2]["content"][0]["text"])
    assert target["card"] == "yellow_card"


def test_prepare_writes_all_splits_without_model_download(tmp_path) -> None:
    manifests = {}
    for split, suffix in (("train", "a"), ("valid", "b"), ("test", "c")):
        path = tmp_path / f"{split}.jsonl"
        path.write_text(json.dumps(event(suffix, split, f"match-{suffix}")) + "\n", encoding="utf-8")
        manifests[split] = path
    contract = prepare_sft_dataset(manifests, tmp_path / "out")
    assert contract["counts"] == {"train": 1, "valid": 1, "test": 1}
    assert (tmp_path / "out" / "splits.json").exists()


def test_multitask_metrics_and_bootstrap_are_deterministic() -> None:
    records = [
        {"gold": {"offence": "Offence", "severity": "3.0", "card": "yellow_card"},
         "prediction": {"offence": "Offence", "severity": "3.0", "card": "yellow_card"},
         "confidence": {"offence": .9, "severity": .8, "card": .9}},
        {"gold": {"offence": "No offence", "severity": "1.0", "card": "no_card"},
         "prediction": {"offence": "Offence", "severity": "2.0", "card": "yellow_card"},
         "confidence": {"offence": .8, "severity": .6, "card": .7}},
    ]
    first = evaluate_predictions(records, bootstrap_samples=25, seed=7)
    second = evaluate_predictions(records, bootstrap_samples=25, seed=7)
    assert first == second
    assert first["tasks"]["severity"]["ordinal_distance"] == .5
    assert first["tasks"]["offence"]["macro_f1"] == pytest.approx(1 / 3)


def test_ece() -> None:
    assert expected_calibration_error([True, False], [.8, .8], bins=5) == pytest.approx(.3)
