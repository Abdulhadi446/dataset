#!/usr/bin/env python3
"""Download HF datasets and convert to OpenAI-compatible JSONL format."""

import json
import os
import sys
import traceback
from datasets import load_dataset

OUTPUT_DIR = "/home/abdulhadi/Desktop/dataset/openai"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def make_messages(instruction, input_text="", output=""):
    """Convert instruction/input/output to OpenAI messages format."""
    user_content = instruction
    if input_text and input_text.strip():
        user_content = user_content + "\n\n" + input_text
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output}
    ]

def make_messages_from_prompt(prompt, output=""):
    """Convert prompt/output to OpenAI messages format."""
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": output}
    ]

def save_jsonl(rows, filepath):
    """Save list of messages dicts to JSONL."""
    with open(filepath, "w") as f:
        for msg in rows:
            f.write(json.dumps({"messages": msg}, ensure_ascii=False) + "\n")
    return filepath

def try_configs(ds_id, configs, split="train"):
    """Try loading a dataset with given configs, return first success."""
    for config in configs:
        try:
            ds = load_dataset(ds_id, config, split=split, streaming=True)
            sample = next(iter(ds))
            return ds, config, sample
        except Exception:
            continue
    return None, None, None

def convert_instruct_format(ds, output_path, limit=None):
    """Convert dataset with instruction/input/output columns."""
    rows = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        msg = make_messages(
            row.get("instruction", ""),
            row.get("input", ""),
            row.get("output", "")
        )
        rows.append(msg)
        if len(rows) % 1000 == 0:
            print(f"    processed {i+1} rows...", end="\r")
    print(f"    processed {len(rows)} total")
    return save_jsonl(rows, output_path)

def convert_conversation_format(ds, output_path, conv_col="conversations", limit=None):
    """Convert dataset with conversations column (list of {role, content})."""
    rows = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        conv = row.get(conv_col, [])
        if not conv and "messages" in row:
            conv = row["messages"]
        # already in OpenAI format?
        if all(k in c for c in conv for k in ("role", "content")):
            rows.append(conv)
        else:
            # try converting from {'from': ..., 'value': ...} format
            converted = []
            for turn in conv:
                role_map = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}
                role = role_map.get(turn.get("from", "").lower(), turn.get("from", "user"))
                converted.append({"role": role, "content": turn.get("value", "")})
            if converted:
                rows.append(converted)
        if len(rows) % 1000 == 0:
            print(f"    processed {i+1} rows...", end="\r")
    print(f"    processed {len(rows)} total")
    return save_jsonl(rows, output_path)

def convert_qa_format(ds, output_path, question_col="question", answer_col="answer", limit=None):
    """Convert dataset with question/answer columns."""
    rows = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        q = row.get(question_col, "")
        a = row.get(answer_col, "")
        if not a:
            a = row.get("completion", row.get("output", row.get("target", "")))
        if not q:
            q = row.get("prompt", row.get("input", row.get("instruction", "")))
        msg = make_messages_from_prompt(str(q), str(a))
        rows.append(msg)
        if len(rows) % 1000 == 0:
            print(f"    processed {i+1} rows...", end="\r")
    print(f"    processed {len(rows)} total")
    return save_jsonl(rows, output_path)

def convert_trajectory_format(ds, output_path, trajectory_col="TRAJECTORY", prompt_col="PROMPT", limit=None):
    """Convert dataset with trajectory (list of chat turns) + prompt."""
    rows = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        traj = row.get(trajectory_col, [])
        if traj and isinstance(traj, list):
            # Check if trajectory is already messages
            if all(isinstance(t, dict) and "role" in t for t in traj):
                rows.append(traj)
            else:
                # Try to convert
                prompt = row.get(prompt_col, "")
                messages = [{"role": "user", "content": prompt}]
                for turn in traj:
                    if isinstance(turn, dict):
                        role = "assistant" if turn.get("role") == "assistant" else "user"
                        content = turn.get("content", turn.get("value", str(turn)))
                        messages.append({"role": role, "content": content})
                if len(messages) > 1:
                    rows.append(messages)
        if len(rows) % 1000 == 0:
            print(f"    processed {i+1} rows...", end="\r")
    print(f"    processed {len(rows)} total")
    return save_jsonl(rows, output_path)

def auto_convert(ds_id, config=None, ds_name=None, limit=None):
    """Auto-detect dataset format and convert."""
    print(f"\n{'='*60}")
    print(f"Processing: {ds_id}" + (f" ({config})" if config else ""))
    print(f"{'='*60}")

    if ds_name is None:
        ds_name = ds_id.replace("/", "_").replace("-", "_").lower()

    output_path = os.path.join(OUTPUT_DIR, f"{ds_name}.jsonl")
    if os.path.exists(output_path):
        print(f"  SKIP: already exists at {output_path}")
        sz = os.path.getsize(output_path)
        print(f"  size: {sz} bytes")
        return output_path

    try:
        kwargs = {"streaming": True, "split": "train"}
        if config:
            kwargs["name"] = config

        print(f"  Loading dataset...")
        ds = load_dataset(ds_id, **kwargs)
        sample = next(iter(ds))
        cols = list(sample.keys())
        print(f"  Columns: {cols}")
        
        # Check for gated
        if "gated" in str(sample).lower() or "private" in str(sample).lower():
            pass  # will fail naturally if gated

        # Detect format
        if "conversations" in cols or "messages" in cols:
            print(f"  Detected: conversation format")
            return convert_conversation_format(ds, output_path, limit=limit)
        elif "TRAJECTORY" in cols:
            print(f"  Detected: trajectory format")
            return convert_trajectory_format(ds, output_path, limit=limit)
        elif "instruction" in cols:
            print(f"  Detected: instruct format")
            return convert_instruct_format(ds, output_path, limit=limit)
        elif "question" in cols:
            print(f"  Detected: QA format (question)")
            return convert_qa_format(ds, output_path, limit=limit)
        elif "prompt" in cols and "output" in cols:
            print(f"  Detected: prompt/output format")
            return convert_qa_format(ds, output_path, question_col="prompt", answer_col="output", limit=limit)
        elif "prompt" in cols:
            print(f"  Detected: prompt format")
            return convert_qa_format(ds, output_path, question_col="prompt", answer_col="output", limit=limit)
        elif "code" in cols or "solution" in cols:
            print(f"  Detected: code format")
            return convert_qa_format(ds, output_path, question_col="prompt", answer_col="code" if "code" in cols else "solution", limit=limit)
        else:
            # Generic: use first 2 columns
            text_cols = [k for k in cols if isinstance(sample[k], str)]
            print(f"  Text columns: {text_cols}")
            if len(text_cols) >= 2:
                print(f"  Using columns: {text_cols[0]} / {text_cols[1]}")
                return convert_qa_format(ds, output_path, question_col=text_cols[0], answer_col=text_cols[1], limit=limit)
            else:
                print(f"  WARNING: Unknown format, saving raw")
                save_jsonl([sample], output_path.replace(".jsonl", ".raw.jsonl"))
                return None

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Define all datasets
    datasets = [
        # Already done
        # ("lazarus19/Vibe-Coding-Claude-Fable-5", None, "vibe-coding-fable-5"),
        
        # Accessible datasets (gated=False)
        ("victor/claude-fable-worldcup-2026-session", None, "claude-fable-worldcup-2026"),
        ("Glint-Research/Fable-5-traces", None, "fable-5-traces"),
        ("armand0e/claude-fable-5-claude-code", None, "fable-5-claude-code"),
        ("Roman1111111/claude-sonnet-4.6-100000X-filtered", None, "sonnet-4dot6-100k-filtered"),
        ("angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k", None, "opus-4dot6-4dot7-reasoning"),
        ("lordx64/agentic-distill-fable-5-sft", None, "agentic-distill-fable-5"),
        ("Norquinal/claude_evol_instruct_210k", None, "claude-evol-instruct-210k"),
        ("openai/gsm8k", "main", "gsm8k"),
        ("allenai/ai2_arc", "ARC-Challenge", "ai2-arc"),
        ("google/IFEval", None, "ifeval"),
        ("ScaleAI/SWE-bench_Pro", None, "swe-bench-pro"),
        ("ScaleAI/MCP-Atlas", None, "mcp-atlas"),
        ("penfever/terminal-bench-2", None, "terminal-bench-2"),
        ("zai-org/terminal-bench-2-verified", None, "terminal-bench-2-verified"),
        
        # Gated=auto - may need token, try anyway
        ("VINAY-UMRETHE/Sonnet-Opus-4.5-4.6-Gemini-3.0-3.1-Pro-GPT-5-5.1-5.2-GLM-4.7-MiniMax-M2.1-DeepSeek-V3.2-High",
         None, "sonnet-opus-mixed-reasoning"),
        ("cais/hle", None, "hle"),
        ("datacurve/deep-swe", None, "deep-swe"),
    ]

    results = {}
    for ds_id, cfg, name in datasets:
        result = auto_convert(ds_id, cfg, name)
        results[name] = result

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    success = 0
    fail = 0
    for name, path in results.items():
        if path and os.path.exists(path):
            sz = os.path.getsize(path)
            import subprocess
            line_count = int(subprocess.getoutput(f"wc -l < {path}").strip())
            print(f"  [OK] {name}: {line_count} rows, {sz/1024/1024:.1f}MB -> {path}")
            success += 1
        else:
            print(f"  [FAIL] {name}")
            fail += 1
    print(f"\nSuccess: {success}, Failed: {fail}")
