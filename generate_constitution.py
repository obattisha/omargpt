#!/usr/bin/env python3
"""
Mine Omar's messages and use Claude to draft a constitution for the chatbot.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python generate_constitution.py

Outputs:
    constitution.md  — human-readable draft for review
    constitution_system_prompt.txt  — ready to paste into build_dataset.py
    constitution_synthetic_examples.jsonl  — Q&A pairs to add to training data
"""

import json
import os
import glob
import random
from pathlib import Path
import anthropic

META_DIR    = os.path.expanduser("~/Downloads/meta/")
YOUR_NAME   = "Omar Battisha"
SAMPLE_SIZE = 400   # messages to send to Claude — enough signal, stays well within context
SEED        = 42
OUTPUT_DIR  = Path(__file__).parent


def fix_encoding(text: str) -> str:
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def collect_your_messages() -> list[dict]:
    """Collect all of Omar's text messages with thread context."""
    pattern = os.path.join(META_DIR, "**/messages/inbox/*/message_*.json")
    files = glob.glob(pattern, recursive=True)

    messages = []
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        participants = [p["name"] for p in data.get("participants", [])]
        is_dm = len(participants) == 2

        thread_msgs = sorted(data.get("messages", []), key=lambda m: m.get("timestamp_ms", 0))

        for i, msg in enumerate(thread_msgs):
            if msg.get("sender_name") != YOUR_NAME:
                continue
            content = msg.get("content", "")
            if not content or len(content) < 15:
                continue
            content = fix_encoding(content)

            # Grab a little context (what triggered this message)
            prior = []
            for j in range(max(0, i - 2), i):
                m = thread_msgs[j]
                if m.get("content"):
                    sender = fix_encoding(m.get("sender_name", ""))
                    label = "you" if sender == YOUR_NAME else sender.split()[0]
                    prior.append(f"{label}: {fix_encoding(m['content'])}")

            messages.append({
                "content": content,
                "context": " | ".join(prior) if prior else "",
                "is_dm": is_dm,
            })

    return messages


def build_sample(messages: list[dict], n: int) -> str:
    """Build a readable sample string from a random selection of messages."""
    random.seed(SEED)
    sample = random.sample(messages, min(n, len(messages)))

    # Interleave DMs and group chats for variety
    dms     = [m for m in sample if m["is_dm"]]
    groups  = [m for m in sample if not m["is_dm"]]
    mixed   = []
    for i in range(max(len(dms), len(groups))):
        if i < len(dms):    mixed.append(dms[i])
        if i < len(groups): mixed.append(groups[i])

    lines = []
    for m in mixed[:n]:
        if m["context"]:
            lines.append(f"[context: {m['context']}]")
        lines.append(f"Omar: {m['content']}")
        lines.append("")
    return "\n".join(lines)


def call_claude(sample_text: str) -> str:
    client = anthropic.Anthropic()

    prompt = f"""Below are {SAMPLE_SIZE} real messages written by Omar Battisha, sampled from years of Facebook Messenger conversations. Some include a little context showing what was said just before Omar's message.

Read them carefully, then produce a CONSTITUTION for an AI chatbot that will impersonate Omar on his personal website. The constitution has three parts:

---

## PART 1 — SYSTEM PROMPT
A concise paragraph (max 150 words) that will be prepended to every conversation. It should:
- State Omar's full name and key identity facts you can confidently infer
- State his religion, values, and any strong positions that are clearly evident
- Use first-person ("Your name is Omar Battisha. You are...")
- End with: "Respond naturally as yourself — in your own voice, with your own personality, vocabulary, and style. If you don't know something about yourself, say so rather than guessing."
- Only include facts you are confident about from the messages — do NOT invent things

## PART 2 — THINGS TO NEVER SAY
A bullet list of specific claims or stances the chatbot must never assert, based on what would clearly contradict Omar's evident identity or values.

## PART 3 — SYNTHETIC TRAINING EXAMPLES
20 realistic Q&A pairs in this exact JSON format (one per line):
{{"messages": [{{"role": "system", "content": "<system prompt>"}}, {{"role": "user", "content": "<question>"}}, {{"role": "assistant", "content": "<Omar-like answer>"}}]}}

Cover: religion, background, values, identity questions ("who are you?", "are you Muslim?", "where are you from?", "what do you do?", "are you a feminist?", etc.). Make the answers sound like Omar based on his actual voice in the messages.

---

Here are Omar's messages:

{sample_text}"""

    print("Sending to Claude... (this may take a minute)")

    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=4000,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
        final = stream.get_final_message()

    print("\n")
    return next(b.text for b in final.content if b.type == "text")


def parse_output(raw: str) -> tuple[str, list[str], list[dict]]:
    """Split Claude's response into the three parts."""
    system_prompt = ""
    never_say     = []
    examples      = []

    lines = raw.split("\n")
    current_section = None

    for line in lines:
        l = line.strip()
        if "PART 1" in l or "SYSTEM PROMPT" in l.upper():
            current_section = "system"
        elif "PART 2" in l or "NEVER SAY" in l.upper():
            current_section = "never"
        elif "PART 3" in l or "SYNTHETIC" in l.upper():
            current_section = "examples"
        elif current_section == "system" and l and not l.startswith("#"):
            system_prompt += line + "\n"
        elif current_section == "never" and l.startswith("-"):
            never_say.append(l[1:].strip())
        elif current_section == "examples" and l.startswith("{"):
            try:
                examples.append(json.loads(l))
            except json.JSONDecodeError:
                pass

    return system_prompt.strip(), never_say, examples


def main():
    print("Collecting your messages...")
    messages = collect_your_messages()
    print(f"Found {len(messages):,} of your messages across all threads")

    sample = build_sample(messages, SAMPLE_SIZE)

    raw = call_claude(sample)

    system_prompt, never_say, examples = parse_output(raw)

    # Write full output for review
    output_path = OUTPUT_DIR / "constitution.md"
    with open(output_path, "w") as f:
        f.write("# Omar Chatbot Constitution\n\n")
        f.write("*Generated by Claude from your messages. Review and edit before using.*\n\n")
        f.write("---\n\n")
        f.write(raw)
    print(f"Full constitution written to {output_path}")

    # Write system prompt snippet
    if system_prompt:
        sp_path = OUTPUT_DIR / "constitution_system_prompt.txt"
        with open(sp_path, "w") as f:
            f.write(system_prompt)
        print(f"System prompt written to {sp_path}")
        print(f"\n--- SYSTEM PROMPT DRAFT ---\n{system_prompt}\n---")

    # Write synthetic examples
    if examples:
        ex_path = OUTPUT_DIR / "constitution_synthetic_examples.jsonl"
        with open(ex_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"\n{len(examples)} synthetic examples written to {ex_path}")

    if never_say:
        print("\n--- THINGS TO NEVER SAY ---")
        for item in never_say:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
