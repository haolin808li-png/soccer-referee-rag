import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from referee_ai.semantics import FACT_FIELDS, fuse_views, validate_view_semantics
from referee_ai.video import uniform_timestamps


def _view(overrides=None):
    facts = {field: "unclear" for field in FACT_FIELDS}
    facts.update({"contact": "yes", "ball_played_first": "not_visible", "studs_exposed": "not_visible"})
    facts.update(overrides or {})
    return {
        "observable_facts": facts,
        "evidence": [{"timestamp_seconds": 1.0, "supports": ["contact"], "description": "players make contact"}],
        "uncertainty": {"uncertain_fields": [], "occlusion": "none", "reason": ""},
    }


class VideoSemanticsTests(unittest.TestCase):
    def test_uniform_timestamps_are_centred_and_ordered(self):
        values = uniform_timestamps(5.0, 4)
        self.assertEqual(len(values), 4)
        self.assertEqual(values, sorted(values))
        self.assertTrue(0 < values[0] < values[-1] < 5.0)

    def test_validation_rejects_verdict_leakage_and_bad_enum(self):
        value = _view()
        value["verdict"] = "yellow_card"
        with self.assertRaises(ValueError):
            validate_view_semantics(value)
        value = _view({"contact": "probably"})
        with self.assertRaises(ValueError):
            validate_view_semantics(value)

    def test_fusion_exposes_tied_view_disagreement(self):
        views = [
            {"clip_id": "a", "semantics": _view({"force": "low"})},
            {"clip_id": "b", "semantics": _view({"force": "high"})},
        ]
        fused = fuse_views("action_1", views, "mock")
        self.assertEqual(fused["observable_facts"]["force"], "unclear")
        self.assertEqual(fused["uncertainty"]["view_disagreements"]["force"], ["low", "high"])

    def test_cli_mock_is_resumable(self):
        with tempfile.TemporaryDirectory() as folder:
            tmp_path = Path(folder)
            video = tmp_path / "clip.mp4"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=black:s=64x64:d=0.5",
                "-pix_fmt", "yuv420p", "-y", str(video),
            ], check=True)
            manifest = tmp_path / "manifest.jsonl"
            event = {"event_id": "e1", "clips": [{"clip_id": "c1", "path": str(video), "exists": True, "valid": True, "duration_seconds": 0.5}]}
            manifest.write_text(json.dumps(event) + "\n")
            output, failures = tmp_path / "out.jsonl", tmp_path / "fail.jsonl"
            command = [sys.executable, "scripts/extract_semantics.py", "--manifest", str(manifest), "--output", str(output),
                       "--failures", str(failures), "--frame-cache", str(tmp_path / "frames"), "--frames-per-view", "2", "--mock"]
            first = subprocess.run(command, capture_output=True, text=True)
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual((first.returncode, second.returncode), (0, 0), first.stderr + second.stderr)
            rows = [json.loads(x) for x in output.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["event_id"], "e1")
            self.assertTrue(rows[0]["views"][0]["sampled_timestamps_seconds"])


if __name__ == "__main__":
    unittest.main()
