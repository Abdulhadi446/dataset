#!/usr/bin/env bash
set -euo pipefail

echo "=== Installing dependencies ==="
pip install -U -q pip >/dev/null 2>&1
pip install -U -q \
    unsloth \
    "transformers<5" \
    trl \
    peft \
    accelerate \
    bitsandbytes \
    huggingface_hub \
    torch torchvision torchaudio \
    xformers \
    triton \
    sentencepiece \
    protobuf >/dev/null 2>&1

echo ""
echo "=== Starting fine-tuning ==="
python kimi_vl_finetune.py
