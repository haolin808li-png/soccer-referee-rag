from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .io import atomic_write_json, atomic_write_jsonl, read_jsonl

SYSTEM_PROMPT = """You are a football referee assistant. Use only the supplied observable facts.
Return one JSON object with keys offence, severity, card, explanation, and uncertainty.
Do not invent contact, force, intent, or a rule citation."""


@dataclass(frozen=True)
class QLoRAConfig:
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    output_dir: str = "artifacts/runs/qlora"
    seed: int = 2026
    epochs: float = 3.0
    learning_rate: float = 1e-4
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_seq_length: int = 2048
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    )
    gradient_checkpointing: bool = True
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 10
    warmup_ratio: float = 0.03
    resume_from_checkpoint: str | None = None
    report_to: str = "none"


def load_manifest(path: str | Path, expected_split: str | None = None) -> list[dict[str, Any]]:
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    ids: set[str] = set()
    for row in rows:
        event_id = str(row.get("event_id", ""))
        if not event_id or event_id in ids:
            raise ValueError(f"Missing or duplicate event_id {event_id!r} in {path}")
        ids.add(event_id)
        if expected_split and row.get("split") != expected_split:
            raise ValueError(
                f"Split mismatch for {event_id}: expected {expected_split}, got {row.get('split')}"
            )
    return rows


def validate_official_splits(
    train: Sequence[Mapping[str, Any]],
    valid: Sequence[Mapping[str, Any]],
    test: Sequence[Mapping[str, Any]],
    *,
    reject_match_overlap: bool = True,
) -> dict[str, Any]:
    """Fail closed on leakage. Split manifests must retain their official split field."""
    named = {"train": train, "valid": valid, "test": test}
    event_sets: dict[str, set[str]] = {}
    match_sets: dict[str, set[str]] = {}
    for split, rows in named.items():
        bad = [str(r.get("event_id")) for r in rows if r.get("split") != split]
        if bad:
            raise ValueError(f"{split} contains rows with a different official split: {bad[:5]}")
        # MVFouls action_N identifiers restart in each official split, so the
        # stable identity is its source match plus local event ID.
        event_sets[split] = {
            f"{r.get('match', '<unknown>')}::{r['event_id']}" for r in rows
        }
        match_sets[split] = {str(r["match"]) for r in rows if r.get("match")}
        if len(event_sets[split]) != len(rows):
            raise ValueError(f"Duplicate event IDs inside {split}")
    leakage: dict[str, list[str]] = {}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap = event_sets[left] & event_sets[right]
        if overlap:
            leakage[f"{left}_{right}_events"] = sorted(overlap)
        match_overlap = match_sets[left] & match_sets[right]
        if reject_match_overlap and match_overlap:
            leakage[f"{left}_{right}_matches"] = sorted(match_overlap)
    if leakage:
        raise ValueError(f"Official split leakage detected: {leakage}")
    return {
        "policy": "official_split_only; no resplitting; event and match disjoint",
        "counts": {name: len(rows) for name, rows in named.items()},
        "event_ids": {name: sorted(values) for name, values in event_sets.items()},
        "matches": {name: sorted(values) for name, values in match_sets.items()},
    }


def _normalise_semantics(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        return {"description": value}
    if not isinstance(value, Mapping):
        raise TypeError("Semantics must be a JSON object or string")
    # Permit an extraction envelope without coupling this module to the extractor schema.
    for key in ("human_verified", "verified", "fused_semantics", "semantics"):
        if key in value and isinstance(value[key], (Mapping, str)):
            return _normalise_semantics(value[key])
    return dict(value)


def load_semantics(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    source = Path(path)
    result: dict[str, dict[str, Any]] = {}
    files = (
        sorted([*source.glob("*.json"), *source.glob("*.jsonl")])
        if source.is_dir()
        else [source]
    )
    for file in files:
        if file.suffix == ".jsonl":
            values = list(read_jsonl(file))
        else:
            raw = json.loads(file.read_text(encoding="utf-8"))
            values = raw if isinstance(raw, list) else [raw]
        for value in values:
            event_id = str(value.get("event_id") or file.stem)
            key = f"{value.get('match')}::{event_id}" if value.get("match") else event_id
            result[key] = _normalise_semantics(value)
    return result


def _target(labels: Mapping[str, Any]) -> dict[str, Any]:
    severity = labels.get("severity_raw")
    return {
        "offence": labels.get("offence", "unknown"),
        "severity": str(severity) if severity not in (None, "") else "unknown",
        "card": labels.get("disciplinary_action", "unknown"),
        "explanation": "Supervised target: explain the decision from the observable evidence.",
        "uncertainty": [],
    }


def _last_subsequence_start(sequence: Sequence[int], subsequence: Sequence[int]) -> int | None:
    if not subsequence or len(subsequence) > len(sequence):
        return None
    for start in range(len(sequence) - len(subsequence), -1, -1):
        if list(sequence[start : start + len(subsequence)]) == list(subsequence):
            return start
    return None


def build_sft_record(event: Mapping[str, Any], semantics: Mapping[str, Any] | None = None) -> dict[str, Any]:
    clips = [str(c["path"]) for c in event.get("clips", []) if c.get("valid", c.get("exists", True))]
    facts = _normalise_semantics(semantics)
    user_text = (
        "Observed structured facts (may contain uncertainty):\n"
        + json.dumps(facts, ensure_ascii=False, sort_keys=True)
        + "\nAssess offence, severity and card. Return strict JSON."
    )
    content: list[dict[str, Any]] = [{"type": "video", "video": path} for path in clips]
    content.append({"type": "text", "text": user_text})
    return {
        "event_id": str(event["event_id"]),
        "event_key": f"{event.get('match', '<unknown>')}::{event['event_id']}",
        "split": str(event["split"]),
        "match": event.get("match"),
        "videos": clips,
        "semantics": facts,
        "labels": dict(event.get("labels", {})),
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": json.dumps(_target(event.get("labels", {})), ensure_ascii=False)}]},
        ],
    }


def prepare_sft_dataset(
    manifests: Mapping[str, str | Path],
    output_dir: str | Path,
    semantics_path: str | Path | None = None,
    *,
    reject_match_overlap: bool = True,
) -> dict[str, Any]:
    rows = {name: load_manifest(path, name) for name, path in manifests.items()}
    missing = {"train", "valid", "test"} - rows.keys()
    if missing:
        raise ValueError(f"All official splits are required; missing {sorted(missing)}")
    split_contract = validate_official_splits(
        rows["train"], rows["valid"], rows["test"], reject_match_overlap=reject_match_overlap
    )
    semantics = load_semantics(semantics_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split, events in rows.items():
        atomic_write_jsonl(
            output / f"{split}.jsonl",
            (
                build_sft_record(
                    event,
                    semantics.get(f"{event.get('match')}::{event['event_id']}")
                    or semantics.get(str(event["event_id"])),
                )
                for event in events
            ),
        )
    split_contract["semantics_coverage"] = {
        split: sum(
            f"{event.get('match')}::{event['event_id']}" in semantics
            or str(event["event_id"]) in semantics
            for event in events
        )
        for split, events in rows.items()
    }
    atomic_write_json(output / "splits.json", split_contract)
    return split_contract


def run_qlora_training(config: QLoRAConfig, train_path: str | Path, valid_path: str | Path) -> Path:
    """Train on CUDA. Heavy ML imports are deliberately delayed for local tooling/tests."""
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            BitsAndBytesConfig,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:  # pragma: no cover - depends on rental image
        raise RuntimeError("Install referee-ai[train] on the CUDA rental instance") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA requires an NVIDIA CUDA GPU; use the configured rental instance")

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    output = Path(config.output_dir)
    for child in ("config", "splits", "metrics", "predictions", "models", "report"):
        (output / child).mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "config" / "training.json", asdict(config))

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(config.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        config.model_id, quantization_config=quant, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=config.gradient_checkpointing
    )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_targets),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.add_adapter(peft_config)
    dataset = load_dataset(
        "json", data_files={"train": str(train_path), "validation": str(valid_path)}
    )

    def collate(examples: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("qwen-vl-utils is required for video collation") from exc
        messages = [example["messages"] for example in examples]
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in messages]
        images, videos = zip(*(process_vision_info(m) for m in messages))
        batch = processor(
            text=texts, images=list(images), videos=list(videos), padding=True,
            truncation=True, max_length=config.max_seq_length, return_tensors="pt"
        )
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        # Mask vision placeholders as recommended for Qwen VL supervised tuning.
        for token_id in (151652, 151653, 151655):
            labels[labels == token_id] = -100
        # Supervise the decision only, not reproduction of the prompt. Finding the
        # target subsequence remains robust to the variable number of vision tokens.
        for row_index, example in enumerate(examples):
            target_text = example["messages"][-1]["content"][0]["text"]
            target_ids = processor.tokenizer(target_text, add_special_tokens=False)["input_ids"]
            input_ids = batch["input_ids"][row_index].tolist()
            start = _last_subsequence_start(input_ids, target_ids)
            if start is None:
                raise ValueError(f"Could not locate assistant target for {example['event_id']}")
            labels[row_index, :start] = -100
        batch["labels"] = labels
        return batch

    args = TrainingArguments(
        output_dir=str(output / "models"),
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=True,
        fp16=False,
        optim="paged_adamw_8bit",
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        report_to=[] if config.report_to == "none" else [config.report_to],
        seed=config.seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=dataset["train"], eval_dataset=dataset["validation"],
        data_collator=collate, callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_model(output / "models" / "final_adapter")
    processor.save_pretrained(output / "models" / "final_adapter")
    atomic_write_json(output / "metrics" / "train.json", result.metrics)
    atomic_write_json(output / "report" / "run.json", {
        "status": "completed", "model": config.model_id, "method": "QLoRA NF4",
        "test_used_for_training": False,
    })
    return output
