import os
import sys

import unsloth
import torch
import torch.nn as nn
import json
import gc
from huggingface_hub import HfApi
from datasets import Dataset
import transformers
from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    activations,
)
from trl import SFTTrainer
from accelerate import Accelerator

import transformers.utils.import_utils as _iu
if not hasattr(_iu, "is_torch_fx_available"):
    def is_torch_fx_available():
        try:
            import torch.fx
            return True
        except ImportError:
            return False
    _iu.is_torch_fx_available = is_torch_fx_available

if not hasattr(activations, "PytorchGELUTanh"):
    class PytorchGELUTanh(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, input):
            return torch.nn.functional.gelu(input)
    activations.PytorchGELUTanh = PytorchGELUTanh

accelerator = Accelerator()

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
print(f"Total VRAM: {sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) / 1e9:.2f} GB")

HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN)

print("\n--- Downloading dataset ---")
dataset_path = "/kaggle/working/opt1.jsonl"
if not os.path.exists(dataset_path):
    url = ("https://huggingface.co/datasets/thetrillioniar/"
           "claude-sonnet-4.6-opus-4.8-mythos-5-fable-5-openai-finetuning-dataset/"
           "resolve/main/opts/opt1.jsonl?download=true")
    import requests
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    with open(dataset_path, "wb") as f:
        f.write(r.content)
    print(f"Downloaded {len(r.content)} bytes")
else:
    print("Dataset already exists")

print("\n--- Loading tokenizer ---")
model_id = "moonshotai/Kimi-VL-A3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
    token=HF_TOKEN,
)

print("\n--- Loading model with QLoRA (4-bit) ---")
model_config = AutoConfig.from_pretrained(
    model_id,
    trust_remote_code=True,
    token=HF_TOKEN,
)
def _fix_rope_scaling(cfg):
    if not hasattr(cfg, "rope_scaling") or not isinstance(cfg.rope_scaling, dict):
        return
    rs = cfg.rope_scaling
    if "type" in rs and "factor" in rs:
        return
    cfg.rope_scaling = {"type": "linear", "factor": 1.0}

_fix_rope_scaling(model_config)
if hasattr(model_config, "text_config"):
    _fix_rope_scaling(model_config.text_config)
if hasattr(model_config, "vision_config"):
    _fix_rope_scaling(model_config.vision_config)
model, tokenizer = unsloth.FastModel.from_pretrained(
    model_name=model_id,
    config=model_config,
    max_seq_length=4096,
    dtype=torch.float16,
    load_in_4bit=True,
    trust_remote_code=True,
    token=HF_TOKEN,
)

def print_gpu_mem(label=""):
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"  [GPU] {label} — allocated: {allocated:.2f} GB, reserved: {reserved:.2f} GB")

print_gpu_mem("after model load")

print("\n--- Applying QLoRA ---")
model = unsloth.FastModel.get_peft_model(
    model,
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
)
print("QLoRA applied successfully")

print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

print("\n--- Preparing dataset ---")
def convert_thinking(text):
    text = text.replace("<think>", "\u25c1think\u25b7")
    text = text.replace("</think>", "\u25c1/think\u25b7")
    return text

def format_conversation(messages):
    formatted = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        content = convert_thinking(content)
        formatted.append({"role": role, "content": content})

    try:
        text = tokenizer.apply_chat_template(
            formatted,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        parts = []
        for msg in formatted:
            if msg["role"] == "system":
                parts.append(f"<|system|>\n{msg['content']}")
            elif msg["role"] == "user":
                parts.append(f"<|user|>\n{msg['content']}")
            elif msg["role"] == "assistant":
                parts.append(f"<|assistant|>\n{msg['content']}")
            else:
                parts.append(f"<|{msg['role']}|>\n{msg['content']}")
        text = "\n".join(parts) + "\n"
    return text

records = []
with open(dataset_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        messages = data.get("messages", [])
        if not messages:
            continue
        text = format_conversation(messages)
        records.append({"text": text})

print(f"Loaded {len(records)} examples")
if len(records) > 0:
    print("First example preview (first 300 chars):")
    print(records[0]["text"][:300])

dataset = Dataset.from_list(records)

print("\n--- Setting up training ---")
training_args = TrainingArguments(
    output_dir="/kaggle/working/checkpoints",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    fp16=True,
    bf16=False,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    remove_unused_columns=False,
    report_to="none",
    dataloader_num_workers=2,
    ddp_find_unused_parameters=False,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
    max_seq_length=4096,
    dataset_text_field="text",
)

print("\n--- Starting QLoRA training ---")
try:
    trainer.train()
except torch.cuda.OutOfMemoryError:
    print("OOM! Retrying with smaller batch...")
    gc.collect()
    torch.cuda.empty_cache()
    training_args.per_device_train_batch_size = 1
    training_args.gradient_accumulation_steps = 8
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        max_seq_length=4096,
        dataset_text_field="text",
    )
    trainer.train()

print("\n--- Saving model ---")
save_path = "/kaggle/working/kimi-vl-finetuned"
trainer.model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Adapter saved to {save_path}")

print("\n--- Pushing to HuggingFace Hub ---")
if HF_TOKEN:
    try:
        repo_id = "thetrillioniar/Kimi-VL-A3B-TriMind-v1"
        api = HfApi()
        api.create_repo(repo_id=repo_id, exist_ok=True, private=False, token=HF_TOKEN)
        trainer.model.push_to_hub(repo_id, token=HF_TOKEN, private=False)
        tokenizer.push_to_hub(repo_id, token=HF_TOKEN, private=False)
        print(f"Model pushed to https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Failed to push to hub: {e}")
else:
    print("HF_TOKEN not found in environment — skipping hub push")

print("\n--- Done ---")
print_gpu_mem("final")
