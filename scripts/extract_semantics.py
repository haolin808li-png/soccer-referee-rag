#!/usr/bin/env python3
"""Extract resumable multi-view semantics from a SoccerNet MVFouls manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from referee_ai.semantics import MockVisionBackend, Qwen3VLBackend, process_event


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/mvfouls_test_manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/semantics/mvfouls_test.jsonl"))
    parser.add_argument("--failures", type=Path, default=Path("artifacts/semantics/mvfouls_test.failures.jsonl"))
    parser.add_argument("--frame-cache", type=Path, default=Path("artifacts/frame_cache"))
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--frames-per-view", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--event-id", action="append", help="Repeat to select specific events")
    parser.add_argument("--mock", action="store_true", help="Exercise pipeline without loading a model")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    events = read_jsonl(args.manifest)
    if args.event_id:
        selected = set(args.event_id)
        events = [x for x in events if x["event_id"] in selected]
    if args.limit is not None:
        events = events[: args.limit]
    outputs = read_jsonl(args.output)
    failures = read_jsonl(args.failures)
    complete = {x["event_id"] for x in outputs}
    failed = {x["event_id"] for x in failures}
    backend = MockVisionBackend() if args.mock else Qwen3VLBackend(args.model, load_in_4bit=args.load_in_4bit)

    for number, event in enumerate(events, 1):
        event_id = event["event_id"]
        if event_id in complete or (event_id in failed and not args.retry_failures):
            continue
        try:
            result = process_event(event, backend, args.frame_cache, args.frames_per_view)
            outputs.append(result)
            complete.add(event_id)
            failures = [x for x in failures if x["event_id"] != event_id]
            atomic_write_jsonl(args.output, outputs)
            atomic_write_jsonl(args.failures, failures)
            print(f"[{number}/{len(events)}] completed {event_id}", flush=True)
        except Exception as exc:  # record each event failure and continue the rented-GPU job
            failures = [x for x in failures if x["event_id"] != event_id]
            failures.append({"event_id": event_id, "error_type": type(exc).__name__, "error": str(exc)})
            atomic_write_jsonl(args.failures, failures)
            print(f"[{number}/{len(events)}] failed {event_id}: {exc}", file=sys.stderr, flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
