#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from referee_ai.io import atomic_write_json
from referee_ai.preflight import run_preflight


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify referee-AI data and runtime readiness")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--output", default="artifacts/preflight/report.json")
    args = parser.parse_args()
    report = run_preflight(args.config, require_cuda=args.require_cuda)
    atomic_write_json(Path(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
