from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .io import read_jsonl
from .rag import RuleIndex
from .training import validate_official_splits

EXPECTED_COUNTS = {"train": 2916, "valid": 411, "test": 301}


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Pipeline config must be a YAML object")
    return value


def run_preflight(config_path: str | Path, *, require_cuda: bool = False) -> dict[str, Any]:
    config_path = Path(config_path)
    config = _load_config(config_path)
    paths = config["paths"]
    checks: list[Check] = []

    manifests: dict[str, list[dict[str, Any]]] = {}
    for split, expected in EXPECTED_COUNTS.items():
        path = Path(paths[f"{split}_manifest"])
        rows = list(read_jsonl(path)) if path.exists() else []
        manifests[split] = rows
        checks.append(Check(
            f"manifest_{split}_count", len(rows) == expected,
            f"expected={expected}, actual={len(rows)}, path={path}",
        ))
        missing = [
            clip.get("path") for row in rows for clip in row.get("clips", [])
            if not Path(str(clip.get("path", ""))).exists()
        ]
        checks.append(Check(
            f"manifest_{split}_clip_paths", not missing,
            f"events={len(rows)}, missing_clips={len(missing)}",
        ))

    try:
        contract = validate_official_splits(
            manifests["train"], manifests["valid"], manifests["test"]
        )
        checks.append(Check("split_leakage", True, contract["policy"]))
    except (KeyError, ValueError) as exc:
        checks.append(Check("split_leakage", False, str(exc)))

    rules_path = Path(paths["rules"])
    index_path = Path(paths["rule_index"]) / "index.json"
    rule_rows = list(read_jsonl(rules_path)) if rules_path.exists() else []
    versions = sorted({str(row.get("version")) for row in rule_rows})
    checks.append(Check(
        "rules_version_pinned", len(rule_rows) > 0 and len(versions) == 1,
        f"chunks={len(rule_rows)}, versions={versions}",
    ))
    try:
        index = RuleIndex.load(index_path)
        checks.append(Check(
            "rule_index", len(index.chunks) == len(rule_rows) and len(index.chunks) > 0,
            f"indexed={len(index.chunks)}, source_chunks={len(rule_rows)}",
        ))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        checks.append(Check("rule_index", False, str(exc)))

    for executable in ("ffmpeg", "ffprobe"):
        location = shutil.which(executable)
        checks.append(Check(executable, bool(location), location or "not found"))

    cuda_available = False
    cuda_detail = "not required for local preflight"
    if require_cuda:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            cuda_detail = torch.cuda.get_device_name(0) if cuda_available else "CUDA unavailable"
        except ImportError:
            cuda_detail = "PyTorch unavailable"
        checks.append(Check("cuda", cuda_available, cuda_detail))

    failed = [asdict(check) for check in checks if not check.passed]
    return {
        "status": "ready" if not failed else "not_ready",
        "config": str(config_path),
        "require_cuda": require_cuda,
        "checks": [asdict(check) for check in checks],
        "failed": failed,
    }
