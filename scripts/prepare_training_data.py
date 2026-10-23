#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from referee_ai.training import prepare_sft_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-checked SoccerNet-MVFouls SFT JSONL")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--valid-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--semantics", help="JSONL file or directory of event JSON files")
    parser.add_argument("--output-dir", default="artifacts/training_data")
    parser.add_argument("--allow-match-overlap", action="store_true")
    args = parser.parse_args()
    contract = prepare_sft_dataset(
        {"train": args.train_manifest, "valid": args.valid_manifest, "test": args.test_manifest},
        Path(args.output_dir), args.semantics, reject_match_overlap=not args.allow_match_overlap,
    )
    print(f"Prepared {contract['counts']} at {args.output_dir}")


if __name__ == "__main__":
    main()
