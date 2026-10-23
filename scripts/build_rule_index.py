#!/usr/bin/env python3
"""Build the dependency-free referee rule index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from referee_ai.rag import RuleIndex, load_rule_documents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Rule .jsonl/.md files")
    parser.add_argument("--output", required=True, help="Output index JSON")
    args = parser.parse_args()
    chunks = load_rule_documents(args.inputs)
    RuleIndex(chunks).save(args.output)
    print(f"indexed {len(chunks)} rule chunks -> {args.output}")


if __name__ == "__main__":
    main()
