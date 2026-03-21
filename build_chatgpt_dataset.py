#!/usr/bin/env python3
"""
Parse ChatGPT conversation export and extract training examples.

Two outputs:
  chatgpt_dataset.jsonl   — training examples (Omar's messages as assistant turns)
  chatgpt_facts.txt       — extracted facts about Omar for omar_context.txt review

Usage:
    python build_chatgpt_dataset.py [--stats]
"""

import json
import glob
import os
import argparse
import re
from pathlib import Path
from constitution_loader import load_system_prompt

CHATGPT_DIR  = os.path.join(os.path.dirname(__file__), "ChatgptHistory")
OUTPUT_DIR   = Path(__file__).parent
YOUR_NAME    = "Your Name"   # ← set to your name as it appears in ChatGPT exports

MIN_RESPONSE_LEN = 40    # chars — skip one-liners and short commands
MAX_RESPONSE_LEN = 3000  # chars — skip massive code/document dumps
MIN_TEXT_RATIO   = 0.4   # fraction of non-code, non-URL content required

# Lines that indicate a message is mostly a command or paste, not voice
CODE_PATTERNS = re.compile(
    r"^(```|\s{4}|\t|import |from |def |class |const |let |var |<[a-z]+[ />]|SELECT |INSERT |UPDATE |http[s]?://\S{30,})",
    re.MULTILINE | re.IGNORECASE,
)

SYSTEM_PROMPT = load_system_prompt()


def extract_text(message: dict) -> str:
    """Pull plain text from a ChatGPT message node, skipping media/code blocks."""
    if not message:
        return ""
    content = message.get("content", {})
    ctype = content.get("content_type", "")
    parts = content.get("parts", [])

    if ctype == "text":
        return " ".join(p for p in parts if isinstance(p, str)).strip()
    elif ctype == "multimodal_text":
        # Extract only the string parts (skip image/audio asset pointers)
        return " ".join(p for p in parts if isinstance(p, str)).strip()
    else:
        # code, execution_output, thoughts, tether_*, etc. — not useful as voice
        return ""


def is_quality_message(text: str) -> bool:
    """Return True if the message is substantive natural language, not a command/paste."""
    if len(text) < MIN_RESPONSE_LEN or len(text) > MAX_RESPONSE_LEN:
        return False
    # Count code-like lines
    code_lines = len(CODE_PATTERNS.findall(text))
    total_lines = max(text.count("\n") + 1, 1)
    if code_lines / total_lines > (1 - MIN_TEXT_RATIO):
        return False
    return True


def get_linear_path(mapping: dict, current_node_id: str) -> list[dict]:
    """
    Follow parent pointers from current_node back to root,
    reverse to get chronological order. Returns list of message dicts.
    """
    path = []
    node_id = current_node_id
    while node_id and node_id in mapping:
        node = mapping[node_id]
        msg = node.get("message")
        if msg:
            path.append(msg)
        node_id = node.get("parent")
    path.reverse()
    return path


def extract_examples(conversations: list[dict]) -> list[dict]:
    """
    Extract training examples from a list of conversations.
    Each of Omar's (user-role) substantive text messages becomes an assistant turn,
    with the preceding GPT response as the user turn.
    """
    examples = []

    for convo in conversations:
        mapping = convo.get("mapping", {})
        current_node = convo.get("current_node")
        if not mapping or not current_node:
            continue

        messages = get_linear_path(mapping, current_node)

        # Build list of (role, text) turns, skipping system/tool/empty
        turns = []
        for msg in messages:
            role = msg.get("author", {}).get("role", "")
            if role not in ("user", "assistant"):
                continue
            text = extract_text(msg)
            if not text:
                continue
            turns.append({"role": role, "content": text})

        # Merge consecutive same-role turns
        merged = []
        for turn in turns:
            if merged and merged[-1]["role"] == turn["role"]:
                merged[-1]["content"] += "\n" + turn["content"]
            else:
                merged.append(dict(turn))

        # Extract Omar's (user) messages as assistant responses
        # Context: the preceding assistant (GPT) turn becomes the "user" prompt
        for i, turn in enumerate(merged):
            if turn["role"] != "user":
                continue
            text = turn["content"]
            if not is_quality_message(text):
                continue

            # Find the most recent assistant turn before this
            prior_gpt = None
            for j in range(i - 1, -1, -1):
                if merged[j]["role"] == "assistant":
                    prior_gpt = merged[j]["content"]
                    break

            if prior_gpt:
                # Prior GPT message as user context, Omar's message as assistant response
                examples.append({
                    "messages": [
                        {"role": "system",    "content": SYSTEM_PROMPT},
                        {"role": "user",      "content": prior_gpt[:1500]},  # cap long GPT turns
                        {"role": "assistant", "content": text},
                    ]
                })
            else:
                # First message in convo — Omar initiating a topic, no prior context
                # Use the conversation title as a light framing cue if available
                title = convo.get("title", "")
                user_turn = f"[Starting a new conversation about: {title}]" if title else "Hey"
                examples.append({
                    "messages": [
                        {"role": "system",    "content": SYSTEM_PROMPT},
                        {"role": "user",      "content": user_turn},
                        {"role": "assistant", "content": text},
                    ]
                })

    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(CHATGPT_DIR, "conversations-*.json")))
    print(f"Found {len(files)} conversation files")

    all_conversations = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            all_conversations.extend(json.load(f))
    print(f"Total conversations: {len(all_conversations):,}")

    examples = extract_examples(all_conversations)

    lengths = [len(e["messages"][-1]["content"]) for e in examples]
    avg = sum(lengths) / len(lengths) if lengths else 0
    print(f"\n--- ChatGPT Dataset Stats ---")
    print(f"Training examples extracted: {len(examples):,}")
    print(f"Avg response length: {avg:.0f} chars")
    print(f"Min: {min(lengths) if lengths else 0}  Max: {max(lengths) if lengths else 0}")

    if args.stats:
        return

    out_path = OUTPUT_DIR / "chatgpt_dataset.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(examples):,} examples to {out_path}")


if __name__ == "__main__":
    main()
