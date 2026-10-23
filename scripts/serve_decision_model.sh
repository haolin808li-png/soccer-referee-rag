#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B-Instruct}"
PORT="${PORT:-8000}"
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --served-model-name "$MODEL_NAME" \
  --dtype bfloat16 \
  --port "$PORT"
