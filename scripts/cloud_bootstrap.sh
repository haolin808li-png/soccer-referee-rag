#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[video,train,rag,dev]'
nvidia-smi
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
mkdir -p artifacts/runs artifacts/semantics artifacts/rule_index
echo "Environment ready. Transfer NDA-protected data through a private encrypted channel."
