#!/usr/bin/env python3
"""
Chatbot inference server.

Sits between the web frontend and ollama. Applies safety checks on every
request (PromptGuard on input, LlamaGuard on output).

Setup:
  1. Install ollama (brew install ollama) and pull the model:
       ollama pull hf.co/obattisha/omar-llama3.1-8b-GGUF
  2. pip install flask
  3. python serve.py

The server listens on http://localhost:5000.
For public hosting, put Cloudflare Tunnel or nginx in front of it.
"""

import json
import os
import urllib.error
import urllib.request

from flask import Flask, jsonify, request

from constitution_loader import load_system_prompt
from safety import SAFETY_ADDENDUM, check_input, check_output, input_refusal, output_refusal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL      = os.environ.get("CHATBOT_MODEL", "hf.co/obattisha/omar-llama3.1-8b-GGUF")
MAX_HISTORY_TURNS = 10   # cap context to avoid bloat

# Load system prompt once at startup; append safety hardening
SYSTEM = load_system_prompt() + "\n" + SAFETY_ADDENDUM

print(f"Model: {MODEL}")
print(f"System prompt: {len(SYSTEM)} chars")

# ---------------------------------------------------------------------------
# Ollama helper
# ---------------------------------------------------------------------------

def _call_ollama(messages: list[dict]) -> str | None:
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
            return json.loads(resp.read())["message"]["content"].strip()
    except urllib.error.URLError:
        return None


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/chat", methods=["POST"])
def chat():
    """
    POST /chat
    Body:  { "message": "hey what's up", "history": [...] }
    Reply: { "response": "..." }

    history is a list of { "role": "user"|"assistant", "content": "..." }
    The client is responsible for maintaining and sending history each turn.
    """
    body = request.get_json(force=True, silent=True) or {}
    user_msg = (body.get("message") or "").strip()
    history   = body.get("history", [])

    if not user_msg:
        return jsonify({"error": "empty message"}), 400

    # 1. Input safety check
    input_check = check_input(user_msg)
    if not input_check.is_safe:
        return jsonify({"response": input_refusal(input_check.reason)})

    # 2. Build message list (system + capped history + new user turn)
    messages = [{"role": "system", "content": SYSTEM}]
    messages.extend(history[-MAX_HISTORY_TURNS:])
    messages.append({"role": "user", "content": user_msg})

    # 3. Call model
    response = _call_ollama(messages)
    if response is None:
        return jsonify({"error": "model unavailable — is ollama running?"}), 503

    # 4. Output safety check
    output_check = check_output(user_msg, response)
    if not output_check.is_safe:
        return jsonify({"response": output_refusal()})

    return jsonify({"response": response})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL})


if __name__ == "__main__":
    # Load safety models at startup so the first request isn't slow
    print("Pre-loading safety models…")
    from safety import _load_prompt_guard, _load_llama_guard
    _load_prompt_guard()
    _load_llama_guard()
    print("Ready.")
    app.run(host="0.0.0.0", port=5000, debug=False)
