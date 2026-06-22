import json
import os
import sys
import hashlib

INPUT_DIR = "/home/abdulhadi/Desktop/dataset/thetrillioniar_claude-sonnet-4.6-opus-4.8-mythos-5-fable-5-openai-finetuning-dataset"

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
REPLACE_OPEN = "\u25c1think\u25b7"
REPLACE_CLOSE = "\u25c1/think\u25b7"


def has_null_bytes_or_broken_unicode(s):
    if not isinstance(s, str):
        return True
    if "\x00" in s:
        return True
    try:
        s.encode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return True
    return False


ALLOWED_ROLES = {"system", "user", "assistant"}


def clean_row(messages):
    reasons = []

    if not messages or not isinstance(messages, list):
        return None, ["no_messages"]

    has_user = False
    has_assistant = False

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return None, ["invalid_message_type"]
        if "role" not in msg or "content" not in msg:
            return None, [f"missing_role_or_content"]
        role = msg["role"]
        content = msg["content"]
        if not isinstance(role, str):
            return None, [f"invalid_role_type"]
        role_lower = role.strip().lower()
        if role_lower not in ALLOWED_ROLES:
            return None, [f"unknown_role: {role_lower}"]
        if content is None or (isinstance(content, str) and content.strip() == ""):
            return None, [f"empty_content"]
        if not isinstance(content, str):
            return None, [f"non_string_content"]
        if has_null_bytes_or_broken_unicode(content):
            return None, [f"null_byte_or_broken_unicode"]
        if role_lower == "user":
            has_user = True
            if len(content.strip()) < 3:
                return None, [f"user_content_too_short"]
        elif role_lower == "assistant":
            has_assistant = True
            if len(content.strip()) < 10:
                return None, [f"assistant_content_too_short"]

    if not has_user:
        return None, ["no_user_turn"]
    if not has_assistant:
        return None, ["no_assistant_turn"]

    return messages, []


def process_file(filepath):
    filename = os.path.basename(filepath)
    base, ext = os.path.splitext(filepath)
    outpath = f"{base}_cleaned{ext}"

    total = 0
    kept = 0
    reasons = {}
    seen_hashes = set()

    with open(filepath, "r", encoding="utf-8", errors="strict") as fin, \
         open(outpath, "w", encoding="utf-8") as fout:

        for line in fin:
            total += 1
            line = line.strip()
            if not line:
                reasons["empty_line"] = reasons.get("empty_line", 0) + 1
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                reasons["json_decode_error"] = reasons.get("json_decode_error", 0) + 1
                continue

            if not isinstance(obj, dict) or "messages" not in obj:
                reasons["missing_messages_key"] = reasons.get("missing_messages_key", 0) + 1
                continue

            messages = obj["messages"]
            cleaned_msgs, row_reasons = clean_row(messages)

            if cleaned_msgs is None:
                for r in row_reasons:
                    reasons[r] = reasons.get(r, 0) + 1
                continue

            # Normalize
            normalized_msgs = []
            for msg in cleaned_msgs:
                role = msg["role"].strip().lower()
                content = msg["content"].strip()

                if role == "assistant":
                    content = content.replace(THINK_OPEN, REPLACE_OPEN)
                    content = content.replace(THINK_CLOSE, REPLACE_CLOSE)

                # Skip empty system messages
                if role == "system" and (not content or content.strip() == ""):
                    continue

                normalized_msgs.append({"role": role, "content": content})

            # Deduplicate
            msg_json = json.dumps(normalized_msgs, ensure_ascii=False, sort_keys=True)
            msg_hash = hashlib.sha256(msg_json.encode("utf-8")).hexdigest()
            if msg_hash in seen_hashes:
                reasons["duplicate"] = reasons.get("duplicate", 0) + 1
                continue
            seen_hashes.add(msg_hash)

            output = {"messages": normalized_msgs}
            fout.write(json.dumps(output, ensure_ascii=False) + "\n")
            kept += 1

    return {
        "file": filename,
        "total": total,
        "kept": kept,
        "removed": total - kept,
        "reasons": reasons,
    }


def main():
    files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.endswith(".jsonl") and not f.endswith("_cleaned.jsonl")
    ])

    print(f"Found {len(files)} JSONL files to process\n")

    grand_total = 0
    grand_kept = 0
    grand_reasons = {}

    for fname in files:
        fpath = os.path.join(INPUT_DIR, fname)
        stats = process_file(fpath)

        grand_total += stats["total"]
        grand_kept += stats["kept"]
        for r, c in stats["reasons"].items():
            grand_reasons[r] = grand_reasons.get(r, 0) + c

        print(f"  {fname}:")
        print(f"    total={stats['total']}  kept={stats['kept']}  removed={stats['removed']}")
        if stats["reasons"]:
            detail = ", ".join(f"{k}={v}" for k, v in sorted(stats["reasons"].items()))
            print(f"    reasons: {detail}")
        print()

    print("=" * 60)
    print(f"GRAND TOTAL: {grand_total}")
    print(f"GRAND KEPT:  {grand_kept}")
    print(f"GRAND REMOVED: {grand_total - grand_kept}")
    print()
    print("REMOVAL REASONS (all files):")
    for r, c in sorted(grand_reasons.items(), key=lambda x: -x[1]):
        print(f"  {r}: {c}")


if __name__ == "__main__":
    main()
