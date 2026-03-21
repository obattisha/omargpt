#!/usr/bin/env python3
"""
Build fine-tuning dataset from Meta/Facebook message exports.

Extracts (context, response) pairs where Omar is the author, formats them
as ChatML instruction-tuning examples, and writes to JSONL.

Usage:
    python build_dataset.py [--output dataset.jsonl] [--stats]
"""

import json
import os
import glob
import argparse
import random
from pathlib import Path
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────

YOUR_NAME = "Omar Battisha"
META_DIR = os.path.expanduser("~/Downloads/meta/")
CONTEXT_WINDOW = 5          # messages before Omar's reply to include as context
MIN_RESPONSE_LEN = 10       # drop Omar's replies shorter than this (chars)
MIN_CONTEXT_LEN = 2         # need at least this many prior messages for context
MAX_RESPONSE_LEN = 2000     # drop unusually long messages (likely pastes)
SYSTEM_PROMPT = (
    "Your name is Omar Mohamed Battisha. You are Egyptian-American, based in the Bay Area, "
    "and a practicing Muslim. You studied Political Science and Economics at the University "
    "of Chicago, with a minor in Data Science. You were heavily involved in CHOMUN (UChicago's "
    "Model UN conference) as part of CHOSEC, the secretariat, and you competed in Arabic debate, "
    "including traveling to Qatar for tournaments.\n\n"
    "You're a Senior PM in AI and tech, with experience building 0-to-1 AI products — you led "
    "the launch of QuickBooks' first conversational AI agent at Intuit. You have a data science "
    "background and started your career in ML. You eat halal and are mostly pescatarian/vegetarian. "
    "You fence epee (aggressive style), summited Kilimanjaro, and have traveled to 35+ countries. "
    "You do woodworking. You avoid riba and engage seriously with Islamic philosophy and jurisprudence.\n\n"
    "You are progressive and politically engaged — you support Palestinian rights, believe in police "
    "reform, and care about working-class issues and climate action. You are not Zionist and you "
    "believe in gender equality. You speak Egyptian Arabic and naturally mix it into conversations.\n\n"
    "You have a dry, sardonic sense of humor. You're direct and warm with people you trust, but not "
    "performatively enthusiastic. You value effectiveness over hard work and ownership over "
    "effort-signaling. You're skeptical of received wisdom — you think common beliefs often contain "
    "mistruths.\n\n"
    "Respond naturally as yourself — in your own voice, with your own personality, vocabulary, and "
    "style. You're funny, direct, and warm. You use casual language (lol, lmao, fr, tbh, inshallah). "
    "If you don't know something about yourself, say so rather than guessing."
)
SYNTHETIC_EXAMPLES_PATH = os.path.join(os.path.dirname(__file__), "constitution_synthetic_examples.jsonl")
CHATGPT_EXAMPLES_PATH   = os.path.join(os.path.dirname(__file__), "chatgpt_dataset.jsonl")
WHATSAPP_EXAMPLES_PATH  = os.path.join(os.path.dirname(__file__), "whatsapp_dataset.jsonl")


# ── Encoding fix ──────────────────────────────────────────────────────────────

def fix_encoding(text: str) -> str:
    """
    Meta exports are UTF-8 text that was decoded as latin-1,
    resulting in mojibake. Re-encode as latin-1 and decode as UTF-8.
    Falls back to original string if the round-trip fails.
    """
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


# ── Parsing ───────────────────────────────────────────────────────────────────

def load_thread(path: str) -> dict | None:
    """Load a single message_N.json file. Returns None on parse error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def collect_threads(meta_dir: str) -> dict[str, list[dict]]:
    """
    Collect all message JSON files, grouped by thread directory.
    Threads can be split across message_1.json, message_2.json, etc.
    Returns {thread_dir: [sorted messages oldest→newest]}.
    """
    pattern = os.path.join(meta_dir, "**/messages/inbox/*/message_*.json")
    files = glob.glob(pattern, recursive=True)

    threads: dict[str, list] = defaultdict(list)
    for path in files:
        thread_dir = os.path.dirname(path)
        data = load_thread(path)
        if data:
            threads[thread_dir].append(data)

    # Merge multi-part threads and sort messages oldest → newest
    merged: dict[str, list[dict]] = {}
    for thread_dir, parts in threads.items():
        all_messages = []
        participants = set()
        for part in parts:
            all_messages.extend(part.get("messages", []))
            for p in part.get("participants", []):
                participants.add(p["name"])
        # Sort oldest first (timestamps are ms)
        all_messages.sort(key=lambda m: m.get("timestamp_ms", 0))
        merged[thread_dir] = {
            "messages": all_messages,
            "participants": list(participants),
            "is_dm": len(participants) == 2,
        }

    return merged


# ── Extraction ────────────────────────────────────────────────────────────────

def build_group_turns(messages: list[dict]) -> list[dict]:
    """
    Merge consecutive same-sender messages in a group chat into one entry.
    Omar's name is replaced with '[you]'. Media-only messages are skipped.
    Returns list of {"sender": ..., "content": ...}.
    """
    turns = []
    for msg in messages:
        content = msg.get("content", "")
        if not content:
            continue
        content = fix_encoding(content)
        sender = fix_encoding(msg.get("sender_name", ""))
        label = "[you]" if sender == YOUR_NAME else sender

        if turns and turns[-1]["sender"] == label:
            turns[-1]["content"] += "\n" + content
        else:
            turns.append({"sender": label, "content": content})
    return turns


def format_group_context(turns: list[dict]) -> str:
    """Render merged group chat turns as a labelled context string."""
    return "\n".join(f"{t['sender']}: {t['content']}" for t in turns)


def build_dm_turns(messages: list[dict]) -> list[dict]:
    """
    Convert a DM message list into alternating user/assistant turns.
    Consecutive messages from the same sender are merged into one turn.
    Media-only messages are skipped.
    Returns list of {"role": ..., "content": ...}.
    """
    turns = []
    for msg in messages:
        content = msg.get("content", "")
        if not content:
            continue
        content = fix_encoding(content)
        role = "assistant" if msg.get("sender_name") == YOUR_NAME else "user"

        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += "\n" + content
        else:
            turns.append({"role": role, "content": content})
    return turns


def extract_dm_examples(thread: dict) -> list[dict]:
    """
    Extract multi-turn examples from a DM thread.
    Each assistant turn becomes one training example with preceding turns as context.
    The context always starts with a user turn (leading assistant turns are dropped).
    """
    turns = build_dm_turns(thread["messages"])
    examples = []

    for i, turn in enumerate(turns):
        if turn["role"] != "assistant":
            continue

        content = turn["content"]
        if len(content) < MIN_RESPONSE_LEN or len(content) > MAX_RESPONSE_LEN:
            continue

        # Grab the preceding window of turns
        prior = turns[max(0, i - CONTEXT_WINDOW):i]

        # Drop any leading assistant turns so the context starts with a user turn
        while prior and prior[0]["role"] == "assistant":
            prior = prior[1:]

        # For DMs, turns are already merged so 1 prior turn is sufficient context
        if len(prior) < 1:
            continue

        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                *prior,
                {"role": "assistant", "content": content},
            ],
            "_meta": {
                "is_dm": True,
                "participants": thread["participants"],
            },
        })

    return examples


def extract_group_examples(thread: dict) -> list[dict]:
    """
    Extract flat (context, response) pairs from a group chat thread.
    Consecutive same-sender messages are merged. Omar's name is replaced
    with '[you]' in the context. Each of Omar's turns becomes one example.
    """
    # Build the full merged turn list for the thread
    all_turns = build_group_turns(thread["messages"])
    examples = []

    for i, turn in enumerate(all_turns):
        if turn["sender"] != "[you]":
            continue

        content = turn["content"]
        if len(content) < MIN_RESPONSE_LEN or len(content) > MAX_RESPONSE_LEN:
            continue

        prior = all_turns[max(0, i - CONTEXT_WINDOW):i]
        if len(prior) < MIN_CONTEXT_LEN:
            continue

        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": format_group_context(prior)},
                {"role": "assistant", "content": content},
            ],
            "_meta": {
                "is_dm": False,
                "participants": thread["participants"],
            },
        })

    return examples


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build fine-tuning dataset from Meta exports")
    parser.add_argument("--output", default="dataset.jsonl", help="Output JSONL file")
    parser.add_argument("--stats", action="store_true", help="Print stats and exit without writing")
    parser.add_argument("--dm-only", action="store_true", help="Only include DM threads (no group chats)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffle")
    args = parser.parse_args()

    print(f"Scanning {META_DIR} ...")
    threads = collect_threads(META_DIR)
    print(f"Found {len(threads)} threads")

    all_examples = []
    dm_count = 0
    group_count = 0

    for thread_dir, thread in threads.items():
        thread["_path"] = thread_dir
        if args.dm_only and not thread["is_dm"]:
            continue

        if thread["is_dm"]:
            examples = extract_dm_examples(thread)
        else:
            examples = extract_group_examples(thread)
        all_examples.extend(examples)

        if thread["is_dm"]:
            dm_count += 1
        else:
            group_count += 1

    # Shuffle
    random.seed(args.seed)
    random.shuffle(all_examples)

    # Stats
    dm_examples = sum(1 for e in all_examples if e["_meta"]["is_dm"])
    group_examples = len(all_examples) - dm_examples
    response_lengths = [len(e["messages"][-1]["content"]) for e in all_examples]
    avg_len = sum(response_lengths) / len(response_lengths) if response_lengths else 0

    print(f"\n--- Dataset Stats ---")
    print(f"Threads processed:  {dm_count} DMs + {group_count} group chats")
    print(f"Total examples:     {len(all_examples):,}")
    print(f"  DM examples:      {dm_examples:,}")
    print(f"  Group examples:   {group_examples:,}")
    print(f"Avg response len:   {avg_len:.0f} chars")
    print(f"Min response len:   {min(response_lengths) if response_lengths else 0} chars")
    print(f"Max response len:   {max(response_lengths) if response_lengths else 0} chars")

    if args.stats:
        return

    # Load synthetic constitution examples (prepend so they're always seen early)
    synthetic = []
    if os.path.exists(SYNTHETIC_EXAMPLES_PATH):
        with open(SYNTHETIC_EXAMPLES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    synthetic.append(json.loads(line))
        print(f"Loaded {len(synthetic)} synthetic examples from constitution")

    # Load ChatGPT examples
    chatgpt = []
    if os.path.exists(CHATGPT_EXAMPLES_PATH):
        with open(CHATGPT_EXAMPLES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chatgpt.append(json.loads(line))
        print(f"Loaded {len(chatgpt):,} examples from ChatGPT history")

    # Load WhatsApp examples
    whatsapp = []
    if os.path.exists(WHATSAPP_EXAMPLES_PATH):
        with open(WHATSAPP_EXAMPLES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    whatsapp.append(json.loads(line))
        print(f"Loaded {len(whatsapp):,} examples from WhatsApp")

    # Merge all real sources, then shuffle together
    real_examples = [{"messages": e["messages"]} for e in all_examples]
    combined = real_examples + chatgpt + whatsapp
    random.seed(args.seed)
    random.shuffle(combined)

    # Write JSONL — synthetic first (identity anchors), then shuffled combined
    output_path = os.path.join(os.path.dirname(__file__), args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        for example in synthetic:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
        for example in combined:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    total = len(synthetic) + len(combined)
    print(f"\nWrote {total:,} examples to {output_path}")
    print(f"  {len(synthetic)} synthetic  +  {len(real_examples):,} Messenger  +  {len(chatgpt):,} ChatGPT  +  {len(whatsapp):,} WhatsApp")

    # Also write a small sample for manual review
    sample_path = output_path.replace(".jsonl", "_sample.jsonl")
    with open(sample_path, "w", encoding="utf-8") as f:
        for example in all_examples[:50]:
            out = {"messages": example["messages"]}
            f.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n---\n")
    print(f"Wrote 50-example sample to {sample_path}")


if __name__ == "__main__":
    main()
