#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from referee_ai.evaluation import evaluate_predictions
from referee_ai.io import atomic_write_json, atomic_write_jsonl, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate held-out referee predictions")
    parser.add_argument("--predictions", required=True, help="JSONL with gold/prediction/confidence")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    records = list(read_jsonl(args.predictions))
    if any(record.get("split") != "test" for record in records):
        raise ValueError("Final evaluation accepts only records explicitly marked split=test")
    run = Path(args.run_dir)
    for child in ("config", "splits", "metrics", "predictions", "models", "report"):
        (run / child).mkdir(parents=True, exist_ok=True)
    metrics = evaluate_predictions(records, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    atomic_write_jsonl(run / "predictions" / "test.jsonl", records)
    atomic_write_json(run / "metrics" / "test.json", metrics)
    atomic_write_json(run / "report" / "evaluation.json", {
        "status": "completed", "split": "test", "n_events": len(records),
        "bootstrap_samples": args.bootstrap_samples,
    })
    print(run / "metrics" / "test.json")


if __name__ == "__main__":
    main()
