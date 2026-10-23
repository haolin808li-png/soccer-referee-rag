#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import fields

import yaml

from referee_ai.training import QLoRAConfig, run_qlora_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Qwen3-VL with CUDA NF4 QLoRA")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--train", default="artifacts/training_data/train.jsonl")
    parser.add_argument("--valid", default="artifacts/training_data/valid.jsonl")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--report-to", choices=("none", "wandb"))
    args = parser.parse_args()
    raw = yaml.safe_load(open(args.config, encoding="utf-8"))
    supplied = dict(raw.get("training", raw))
    aliases = {"per_device_batch_size": supplied.get("per_device_batch_size")}
    allowed = {field.name for field in fields(QLoRAConfig)}
    values = {key: value for key, value in supplied.items() if key in allowed and value is not None}
    values.update({key: value for key, value in aliases.items() if value is not None})
    if "lora_targets" in values:
        values["lora_targets"] = tuple(values["lora_targets"])
    if args.output_dir:
        values["output_dir"] = args.output_dir
    if args.resume_from_checkpoint:
        values["resume_from_checkpoint"] = args.resume_from_checkpoint
    if args.report_to:
        values["report_to"] = args.report_to
    run_qlora_training(QLoRAConfig(**values), args.train, args.valid)


if __name__ == "__main__":
    main()
