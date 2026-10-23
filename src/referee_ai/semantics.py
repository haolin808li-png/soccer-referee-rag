"""Evidence-first multi-view football incident semantics.

The visual stage describes only observable facts. It deliberately does not emit
a foul/card decision; rules and verdicts belong to the downstream RAG stage.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .video import FrameSample, extract_uniform_frames

FACT_FIELDS = (
    "action_type", "challenge_direction", "contact", "contact_point",
    "ball_played_first", "speed", "force", "studs_exposed", "airborne",
    "control", "opponent_endangered",
)
ALLOWED = {
    "action_type": {"tackle", "standing_challenge", "push", "hold", "elbow", "high_foot", "dive", "other", "unclear"},
    "challenge_direction": {"front", "side", "behind", "not_applicable", "unclear"},
    "contact": {"yes", "no", "unclear"},
    "contact_point": {"foot", "ankle", "shin", "knee", "leg_other", "torso", "arm", "head", "none", "unclear"},
    "ball_played_first": {"yes", "no", "simultaneous", "not_visible", "not_applicable", "unclear"},
    "speed": {"low", "medium", "high", "unclear"},
    "force": {"low", "medium", "high", "unclear"},
    "studs_exposed": {"yes", "no", "not_visible", "not_applicable", "unclear"},
    "airborne": {"yes", "no", "unclear"},
    "control": {"controlled", "partially_controlled", "uncontrolled", "unclear"},
    "opponent_endangered": {"visibly_yes", "visibly_no", "unclear"},
}


VIEW_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["observable_facts", "evidence", "uncertainty"],
    "properties": {
        "observable_facts": {
            "type": "object", "additionalProperties": False,
            "required": list(FACT_FIELDS),
            "properties": {key: {"type": "string", "enum": sorted(ALLOWED[key])} for key in FACT_FIELDS},
        },
        "evidence": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["timestamp_seconds", "supports", "description"],
                "properties": {
                    "timestamp_seconds": {"type": "number", "minimum": 0},
                    "supports": {"type": "array", "items": {"type": "string", "enum": list(FACT_FIELDS)}},
                    "description": {"type": "string"},
                },
            },
        },
        "uncertainty": {
            "type": "object", "additionalProperties": False,
            "required": ["uncertain_fields", "occlusion", "reason"],
            "properties": {
                "uncertain_fields": {"type": "array", "items": {"type": "string", "enum": list(FACT_FIELDS)}},
                "occlusion": {"type": "string", "enum": ["none", "partial", "severe"]},
                "reason": {"type": "string"},
            },
        },
    },
}


def validate_view_semantics(value: dict[str, Any], duration: float | None = None) -> dict[str, Any]:
    if set(value) != {"observable_facts", "evidence", "uncertainty"}:
        raise ValueError("View output must contain exactly observable_facts, evidence, uncertainty")
    facts = value["observable_facts"]
    if set(facts) != set(FACT_FIELDS):
        raise ValueError(f"observable_facts must contain exactly: {', '.join(FACT_FIELDS)}")
    for field, answer in facts.items():
        if answer not in ALLOWED[field]:
            raise ValueError(f"Invalid {field}: {answer!r}")
    uncertainty = value["uncertainty"]
    if set(uncertainty) != {"uncertain_fields", "occlusion", "reason"}:
        raise ValueError("Invalid uncertainty object")
    if uncertainty["occlusion"] not in {"none", "partial", "severe"}:
        raise ValueError("Invalid occlusion")
    if any(x not in FACT_FIELDS for x in uncertainty["uncertain_fields"]):
        raise ValueError("Invalid uncertain_fields")
    for item in value["evidence"]:
        if set(item) != {"timestamp_seconds", "supports", "description"}:
            raise ValueError("Invalid evidence object")
        timestamp = float(item["timestamp_seconds"])
        if timestamp < 0 or (duration is not None and timestamp > duration + 0.1):
            raise ValueError(f"Evidence timestamp outside clip: {timestamp}")
        if any(x not in FACT_FIELDS for x in item["supports"]):
            raise ValueError("Evidence supports unknown field")
    return value


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("Model output contains no JSON object")
        text = text[start:end + 1]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Model output must be a JSON object")
    return value


class VisionBackend(Protocol):
    model_id: str
    def infer_view(self, frames: list[FrameSample], prompt: str) -> dict[str, Any]: ...


@dataclass
class MockVisionBackend:
    model_id: str = "mock/evidence-first-v1"

    def infer_view(self, frames: list[FrameSample], prompt: str) -> dict[str, Any]:
        facts = {field: "unclear" for field in FACT_FIELDS}
        facts.update({"contact": "unclear", "ball_played_first": "not_visible", "studs_exposed": "not_visible"})
        return {
            "observable_facts": facts,
            "evidence": [],
            "uncertainty": {"uncertain_fields": list(FACT_FIELDS), "occlusion": "severe", "reason": "Mock backend; no visual inference performed."},
        }


class Qwen3VLBackend:
    """Lazily imported Hugging Face Qwen3-VL backend for rented CUDA GPUs."""

    def __init__(self, model_id: str, *, load_in_4bit: bool = False, device_map: str = "auto", max_new_tokens: int = 768):
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("Install torch, transformers, accelerate and qwen-vl-utils on the GPU host") from exc
        kwargs: dict[str, Any] = {"device_map": device_map, "torch_dtype": torch.bfloat16}
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model_id, self.max_new_tokens = model_id, max_new_tokens

    def infer_view(self, frames: list[FrameSample], prompt: str) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for frame in frames:
            content.append({"type": "image", "image": str(frame.path)})
            content.append({"type": "text", "text": f"Frame timestamp: {frame.timestamp_seconds:.3f}s"})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[rendered], images=[str(x.path) for x in frames], padding=True, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        generated = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens)
        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated)]
        text = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return validate_view_semantics(parse_json_object(text), duration=max(x.timestamp_seconds for x in frames))


def semantics_prompt(camera_type: str, timestamps: list[float]) -> str:
    return (
        "Describe only directly observable visual facts about the football incident. "
        "Do not decide whether it is a foul and do not recommend a card. "
        "Use 'unclear'/'not_visible' instead of guessing. Evidence timestamps must be selected "
        f"from these sampled times: {timestamps}. Camera: {camera_type}. "
        "Return JSON only, conforming exactly to this schema: " + json.dumps(VIEW_JSON_SCHEMA, separators=(",", ":"))
    )


def fuse_views(event_id: str, views: list[dict[str, Any]], model_id: str) -> dict[str, Any]:
    """Fuse facts by agreement while preserving dissent and timestamp evidence.

    A fact is accepted only when it has a unique plurality among informative views;
    ties become ``unclear``. This makes disagreements visible to downstream RAG.
    """
    fused: dict[str, str] = {}
    disagreement: dict[str, list[str]] = {}
    uncertain: list[str] = []
    for field in FACT_FIELDS:
        values = [v["semantics"]["observable_facts"][field] for v in views]
        informative = [x for x in values if x not in {"unclear", "not_visible"}]
        counts = Counter(informative)
        if not counts:
            fused[field] = "not_visible" if "not_visible" in ALLOWED[field] else "unclear"
            uncertain.append(field)
        else:
            ranked = counts.most_common()
            if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
                fused[field] = "unclear"
                uncertain.append(field)
            else:
                fused[field] = ranked[0][0]
        if len(set(informative)) > 1:
            disagreement[field] = values
            if field not in uncertain:
                uncertain.append(field)
    return {
        "schema_version": "1.0.0",
        "event_id": event_id,
        "model_id": model_id,
        "observable_facts": fused,
        "uncertainty": {"uncertain_fields": uncertain, "view_disagreements": disagreement},
        "views": views,
    }


def process_event(event: dict[str, Any], backend: VisionBackend, frame_cache: str | Path, frames_per_view: int = 16) -> dict[str, Any]:
    views: list[dict[str, Any]] = []
    for clip in event["clips"]:
        if not clip.get("valid", True) or not clip.get("exists", True):
            continue
        frames = extract_uniform_frames(clip["path"], frame_cache, frames_per_view)
        prompt = semantics_prompt(clip.get("camera_type", "unknown"), [round(x.timestamp_seconds, 3) for x in frames])
        semantics = validate_view_semantics(backend.infer_view(frames, prompt), duration=float(clip.get("duration_seconds") or frames[-1].timestamp_seconds + 0.1))
        views.append({
            "clip_id": clip["clip_id"], "camera_type": clip.get("camera_type", "unknown"),
            "source_timestamp_ms": clip.get("source_timestamp_ms"), "replay_speed": clip.get("replay_speed"),
            "sampled_timestamps_seconds": [round(x.timestamp_seconds, 6) for x in frames], "semantics": semantics,
        })
    if not views:
        raise ValueError(f"Event {event['event_id']} has no valid clips")
    fused = fuse_views(event["event_id"], views, backend.model_id)
    # Preserve the official identity so action_N values that restart across
    # splits can never overwrite one another in downstream training caches.
    fused["split"] = event.get("split")
    fused["match"] = event.get("match")
    return fused
