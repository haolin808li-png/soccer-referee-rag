#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

STAGES = ("semantics", "index", "decisions", "prepare", "train", "evaluate")


def stage_command(stage: str, config_path: Path, config: dict, split: str) -> list[str]:
    paths = config["paths"]
    semantics = Path(paths["semantics_dir"]) / f"mvfouls_{split}.jsonl"
    failures = semantics.with_suffix(".failures.jsonl")
    index = Path(paths["rule_index"]) / "index.json"
    decisions = Path(paths["runs"]) / "predictions" / f"{split}.jsonl"
    training_data = Path("artifacts/training_data")
    run = Path(paths["runs"]) / "final"
    commands = {
        "semantics": [
            "scripts/extract_semantics.py", "--manifest", paths[f"{split}_manifest"],
            "--output", str(semantics), "--failures", str(failures),
            "--model", config["video"]["model_id"], "--frames-per-view",
            str(config["video"]["frames_per_view"]), "--load-in-4bit",
        ],
        "index": ["scripts/build_rule_index.py", paths["rules"], "--output", str(index)],
        "decisions": [
            "scripts/run_decisions.py", "--index", str(index), "--facts", str(semantics),
            "--output", str(decisions), "--manifest", paths[f"{split}_manifest"],
            "--model", config["decision"]["model_id"],
        ],
        "prepare": [
            "scripts/prepare_training_data.py", "--train-manifest", paths["train_manifest"],
            "--valid-manifest", paths["valid_manifest"], "--test-manifest", paths["test_manifest"],
            "--semantics", paths["semantics_dir"], "--output-dir", str(training_data),
        ],
        "train": [
            "scripts/train_qlora.py", "--config", str(config_path), "--train",
            str(training_data / "train.jsonl"), "--valid", str(training_data / "valid.jsonl"),
            "--output-dir", str(run),
        ],
        "evaluate": [
            "scripts/evaluate.py", "--predictions", str(decisions), "--run-dir", str(run),
            "--seed", str(config.get("seed", 2026)),
        ],
    }
    return [sys.executable, *commands[stage]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reproducible referee-AI pipeline stages")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--stages", nargs="+", choices=STAGES, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--dry-run", action="store_true")
    args, passthrough = parser.parse_known_args()
    config = Path(args.config)
    with config.open(encoding="utf-8") as handle:
        pipeline = yaml.safe_load(handle)
    if "evaluate" in args.stages and args.split != "test":
        parser.error("evaluate is intentionally restricted to --split test")
    for stage in args.stages:
        command = [*stage_command(stage, config, pipeline, args.split), *passthrough]
        print("+", " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
