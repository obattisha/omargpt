#!/usr/bin/env python3
"""
Test the base model before fine-tuning via ollama.
Useful for comparing base model behaviour vs fine-tuned model.

Requires ollama running locally:
  brew install ollama
  ollama pull llama3.1:8b-instruct  (or your chosen model)
  ollama serve   (in a separate terminal, or it auto-starts)

Usage:
  python test_base_model.py              # run default prompts
  python test_base_model.py --chat       # interactive chat mode
"""

import argparse
import json
import urllib.request
import urllib.error
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from constitution_loader import load_system_prompt

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL      = "llama3.1:8b-instruct"   # ← update to match your fine-tuned model
SYSTEM     = load_system_prompt()

DEFAULT_PROMPTS = [
    "hey Omar, what's up?",
    "are you religious at all?",
    "what's your take on the Middle East situation?",
    "are you a feminist?",
    "where are you from originally?",
    "what do you do for work?",
    "what kind of music are you into?",
    "do you have any controversial opinions?",
]


def ask(message: str, history: list[dict] = None) -> str:
    messages = [{"role": "system", "content": SYSTEM}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": message})

    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.8,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        },
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            return data["message"]["content"].strip()
    except urllib.error.URLError:
        return "[ERROR] Could not reach ollama. Is it running? Try: ollama serve"


def run_default_prompts():
    print(f"Model: {MODEL}")
    print(f"System prompt: {SYSTEM}\n")
    print("=" * 60)
    for prompt in DEFAULT_PROMPTS:
        print(f"User:  {prompt}")
        response = ask(prompt)
        print(f"Model: {response}")
        print("-" * 60)


def run_chat():
    print(f"Chatting with base {MODEL} (system prompt applied)")
    print("Type 'quit' to exit, 'reset' to clear history\n")
    history = []
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "reset":
            history = []
            print("[History cleared]")
            continue

        response = ask(user_input, history)
        print(f"Omar: {response}\n")

        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat", action="store_true", help="Interactive chat mode")
    args = parser.parse_args()

    if args.chat:
        run_chat()
    else:
        run_default_prompts()
