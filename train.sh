#!/usr/bin/env bash
set -euo pipefail

echo "=== Installing dependencies ==="
pip install -U pip
pip install -U \
    unsloth \
    transformers \
    trl \
    peft \
    accelerate \
    bitsandbytes \
    huggingface_hub \
    torch torchvision torchaudio \
    xformers \
    triton \
    sentencepiece \
    protobuf

echo ""
echo "=== Starting fine-tuning ==="
python kimi_vl_finetune.py
