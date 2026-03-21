"""
Loads the system prompt from constitution_system_prompt.txt (gitignored, personal).
Falls back to a generic placeholder if the file doesn't exist.

To use: create constitution_system_prompt.txt with your own identity description.
See constitution_system_prompt.example.txt for the expected format.
"""

import os

_DIR = os.path.dirname(__file__)
_PROMPT_FILE = os.path.join(_DIR, "constitution_system_prompt.txt")

GENERIC_SYSTEM_PROMPT = (
    "Your name is [YOUR NAME]. Respond naturally as yourself — in your own voice, "
    "with your own personality, vocabulary, and style. If you don't know something "
    "about yourself, say so rather than guessing."
)


def load_system_prompt() -> str:
    if os.path.exists(_PROMPT_FILE):
        with open(_PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    print(f"Warning: {_PROMPT_FILE} not found — using generic placeholder.")
    print("Create it with your own identity description to personalise the model.")
    return GENERIC_SYSTEM_PROMPT
