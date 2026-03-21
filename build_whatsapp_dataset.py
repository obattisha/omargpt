#!/usr/bin/env python3
"""
Parse WhatsApp chat exports (.txt) and extract training examples.

WhatsApp exports each chat as a .txt file with format:
    [M/D/YY, H:MM:SS AM] Name: message
    [M/D/YY, H:MM:SS AM] Name: message

Or (locale variant):
    M/D/YY, H:MM AM - Name: message

Drop them all in:  ~/meta-finetune/WhatsApp conversations/

Usage:
    python build_whatsapp_dataset.py [--stats]
"""

import re
import os
import json
import glob
import argparse
from pathlib import Path

YOUR_NAME      = "Omar Battisha"
WHATSAPP_DIR   = os.path.expanduser("~/Downloads/Whatsapp Conversations")
OUTPUT_DIR     = Path(__file__).parent

CONTEXT_WINDOW   = 5
MIN_RESPONSE_LEN = 10
MAX_RESPONSE_LEN = 2000
MIN_CONTEXT_LEN  = 1

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

# Matches both common WhatsApp export timestamp formats:
# [1/15/23, 9:42:01\u202fAM] Name: text   (iOS — uses narrow no-break space before AM/PM)
# 1/15/23, 9:42 AM - Name: text           (Android/older — uses dash separator)
MSG_PATTERN = re.compile(
    r"^\u200e?\[?(\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:[\s\u202f]*[AP]M)?)\]?\s*(?:[-–]\s*)?(.+?):\s*(.+)$",
    re.MULTILINE,
)

SKIP_CONTENT = {
    "<Media omitted>", "image omitted", "video omitted", "audio omitted",
    "sticker omitted", "document omitted", "GIF omitted", "Contact card omitted",
    "This message was deleted", "You deleted this message",
    "Messages and calls are end-to-end encrypted.",
    "null",
}


def parse_whatsapp_file(path: str) -> list[dict]:
    """Parse a WhatsApp .txt export into list of {sender, content} dicts."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Strip Unicode directional/invisible marks
    text = text.replace('\u200e', '').replace('\u200f', '').replace('\u2068', '').replace('\u2069', '')

    messages = []
    for match in MSG_PATTERN.finditer(text):
        sender  = match.group(2).strip()
        content = match.group(3).strip()

        # Skip system messages and media placeholders
        if any(skip.lower() in content.lower() for skip in SKIP_CONTENT):
            continue
        if len(content) < 2:
            continue

        messages.append({"sender": sender, "content": content})

    return messages


def is_dm(messages: list[dict]) -> bool:
    """Return True if the chat has exactly 2 unique senders (a DM)."""
    senders = {m["sender"] for m in messages}
    return len(senders) == 2


def merge_consecutive(messages: list[dict]) -> list[dict]:
    """Merge consecutive messages from the same sender."""
    merged = []
    for msg in messages:
        if merged and merged[-1]["sender"] == msg["sender"]:
            merged[-1]["content"] += "\n" + msg["content"]
        else:
            merged.append(dict(msg))
    return merged


def extract_dm_examples(messages: list[dict]) -> list[dict]:
    """Extract alternating user/assistant examples from a DM thread."""
    turns = merge_consecutive(messages)
    # Convert to role-based turns
    role_turns = []
    for t in turns:
        role = "assistant" if t["sender"] == YOUR_NAME else "user"
        role_turns.append({"role": role, "content": t["content"]})

    examples = []
    for i, turn in enumerate(role_turns):
        if turn["role"] != "assistant":
            continue
        content = turn["content"]
        if len(content) < MIN_RESPONSE_LEN or len(content) > MAX_RESPONSE_LEN:
            continue

        prior = role_turns[max(0, i - CONTEXT_WINDOW):i]
        while prior and prior[0]["role"] == "assistant":
            prior = prior[1:]
        if len(prior) < MIN_CONTEXT_LEN:
            continue

        examples.append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            *prior,
            {"role": "assistant", "content": content},
        ]})

    return examples


def extract_group_examples(messages: list[dict]) -> list[dict]:
    """Extract examples from a group chat, using [you] for Omar."""
    turns = merge_consecutive(messages)
    labeled = []
    for t in turns:
        label = "[you]" if t["sender"] == YOUR_NAME else t["sender"]
        labeled.append({"sender": label, "content": t["content"]})

    examples = []
    for i, turn in enumerate(labeled):
        if turn["sender"] != "[you]":
            continue
        content = turn["content"]
        if len(content) < MIN_RESPONSE_LEN or len(content) > MAX_RESPONSE_LEN:
            continue

        prior = labeled[max(0, i - CONTEXT_WINDOW):i]
        if len(prior) < MIN_CONTEXT_LEN:
            continue

        context_str = "\n".join(f"{t['sender']}: {t['content']}" for t in prior)
        examples.append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": context_str},
            {"role": "assistant", "content": content},
        ]})

    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    files = glob.glob(os.path.join(WHATSAPP_DIR, "*.txt"))
    if not files:
        print(f"No .txt files found in {WHATSAPP_DIR}")
        print("Export chats from WhatsApp (chat → ⋮ → More → Export chat → Without media)")
        return

    print(f"Found {len(files)} WhatsApp export file(s)")

    all_examples = []
    dm_count = group_count = 0

    for path in files:
        messages = parse_whatsapp_file(path)
        if not messages:
            continue

        # Only process chats that include Omar
        senders = {m["sender"] for m in messages}
        if YOUR_NAME not in senders:
            print(f"  Skipping {os.path.basename(path)} — '{YOUR_NAME}' not found (senders: {senders})")
            continue

        if is_dm(messages):
            examples = extract_dm_examples(messages)
            dm_count += 1
        else:
            examples = extract_group_examples(messages)
            group_count += 1

        all_examples.extend(examples)
        print(f"  {os.path.basename(path)}: {len(messages)} messages → {len(examples)} examples ({'DM' if is_dm(messages) else 'group'})")

    if not all_examples:
        print("No examples extracted — check that YOUR_NAME matches exactly what appears in the exports")
        return

    lengths = [len(e["messages"][-1]["content"]) for e in all_examples]
    print(f"\n--- WhatsApp Dataset Stats ---")
    print(f"Files: {dm_count} DMs + {group_count} group chats")
    print(f"Total examples: {len(all_examples):,}")
    print(f"Avg response length: {sum(lengths)/len(lengths):.0f} chars")

    if args.stats:
        return

    out_path = OUTPUT_DIR / "whatsapp_dataset.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(all_examples):,} examples to {out_path}")


if __name__ == "__main__":
    main()
