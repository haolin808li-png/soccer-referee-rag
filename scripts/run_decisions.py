#!/usr/bin/env python3
"""Run grounded decisions from JSONL facts using an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from referee_ai.decision import DecisionEngine
from referee_ai.io import read_jsonl
from referee_ai.rag import RuleIndex


def endpoint_llm(base_url: str, model: str, api_key: str):
    url = base_url.rstrip("/") + "/chat/completions"
    def call(prompt: str) -> str:
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }).encode()
        request = Request(url, data=body, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {api_key}"
        })
        with urlopen(request, timeout=180) as response:
            return json.load(response)["choices"][0]["message"]["content"]
    return call


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--facts", required=True, help="JSONL: each row needs event_id and facts")
    parser.add_argument("--manifest", help="Optional labelled manifest used only for evaluation output")
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "Qwen/Qwen3-8B-Instruct"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "unused"))
    args = parser.parse_args()
    engine = DecisionEngine(RuleIndex.load(args.index), endpoint_llm(args.base_url, args.model, args.api_key))
    manifest = {
        row["event_id"]: row for row in read_jsonl(args.manifest)
    } if args.manifest else {}
    decision_to_tasks = {
        "no_foul": {"offence": "No offence", "severity": "1.0", "card": "no_card"},
        "foul_no_card": {"offence": "Offence", "severity": "1.0", "card": "no_card"},
        "yellow_card": {"offence": "Offence", "severity": "3.0", "card": "yellow_card"},
        "red_card": {"offence": "Offence", "severity": "5.0", "card": "red_card"},
        "cannot_determine": {"offence": "unknown", "severity": "unknown", "card": "unknown"},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out:
        for line in Path(args.facts).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            facts = row.get("facts") or row.get("observable_facts")
            if not isinstance(facts, dict):
                raise ValueError(f"{row.get('event_id')}: missing facts/observable_facts object")
            result = engine.decide(facts)
            output = {"event_id": row["event_id"], **result}
            labelled = manifest.get(row["event_id"])
            if labelled:
                labels = labelled.get("labels", {})
                output.update({
                    "split": labelled.get("split"),
                    "gold": {
                        "offence": labels.get("offence", "unknown"),
                        "severity": str(labels.get("severity_raw", "unknown")),
                        "card": labels.get("disciplinary_action", "unknown"),
                    },
                    "prediction": decision_to_tasks[result["decision"]],
                })
            out.write(json.dumps(output, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
