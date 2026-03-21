#!/usr/bin/env python3
"""Tests for build_dataset.py"""

import sys
sys.path.insert(0, ".")

from build_dataset import (
    fix_encoding,
    build_dm_turns,
    build_group_turns,
    extract_dm_examples,
    extract_group_examples,
    YOUR_NAME,
)

OTHER = "Alice"

def make_msg(sender, content, ts=0):
    return {"sender_name": sender, "timestamp_ms": ts, "content": content}

def make_thread(messages, is_dm=True):
    participants = list({m["sender_name"] for m in messages if m.get("sender_name")})
    return {"messages": messages, "participants": participants, "is_dm": is_dm}


# ── fix_encoding ──────────────────────────────────────────────────────────────

def test_fix_encoding_mojibake():
    # "😂" encoded as latin-1 mojibake
    mojibake = "ð\x9f\x98\x82"
    assert fix_encoding(mojibake) == "😂", f"got {fix_encoding(mojibake)!r}"

def test_fix_encoding_plain():
    assert fix_encoding("hello") == "hello"

def test_fix_encoding_arabic():
    # Generate real mojibake by round-tripping through the wrong codec
    original = "عربي"
    mojibake = original.encode("utf-8").decode("latin-1")
    assert fix_encoding(mojibake) == original, f"got {fix_encoding(mojibake)!r}"


# ── build_dm_turns ────────────────────────────────────────────────────────────

def test_turns_basic_alternating():
    msgs = [
        make_msg(OTHER, "hey"),
        make_msg(YOUR_NAME, "hi!"),
        make_msg(OTHER, "what's up"),
        make_msg(YOUR_NAME, "not much"),
    ]
    turns = build_dm_turns(msgs)
    assert len(turns) == 4
    assert turns[0] == {"role": "user", "content": "hey"}
    assert turns[1] == {"role": "assistant", "content": "hi!"}
    assert turns[2] == {"role": "user", "content": "what's up"}
    assert turns[3] == {"role": "assistant", "content": "not much"}

def test_turns_merges_consecutive():
    msgs = [
        make_msg(OTHER, "line 1"),
        make_msg(OTHER, "line 2"),
        make_msg(YOUR_NAME, "ok"),
    ]
    turns = build_dm_turns(msgs)
    assert len(turns) == 2
    assert turns[0]["content"] == "line 1\nline 2"
    assert turns[0]["role"] == "user"

def test_turns_merges_consecutive_assistant():
    msgs = [
        make_msg(OTHER, "hey"),
        make_msg(YOUR_NAME, "part 1"),
        make_msg(YOUR_NAME, "part 2"),
    ]
    turns = build_dm_turns(msgs)
    assert len(turns) == 2
    assert turns[1]["content"] == "part 1\npart 2"

def test_turns_skips_media_only():
    msgs = [
        make_msg(OTHER, "check this out"),
        {"sender_name": OTHER, "timestamp_ms": 0},  # no content key
        make_msg(YOUR_NAME, "cool"),
    ]
    turns = build_dm_turns(msgs)
    assert len(turns) == 2

def test_turns_empty():
    assert build_dm_turns([]) == []


# ── extract_dm_examples ───────────────────────────────────────────────────────

def test_dm_examples_basic():
    msgs = [
        make_msg(OTHER, "how are you?"),
        make_msg(OTHER, "long time no talk"),
        make_msg(YOUR_NAME, "doing well, been busy with work"),
    ]
    examples = extract_dm_examples(make_thread(msgs))
    assert len(examples) == 1
    ex = examples[0]
    roles = [m["role"] for m in ex["messages"]]
    assert roles[0] == "system"
    assert roles[-1] == "assistant"
    assert ex["messages"][-1]["content"] == "doing well, been busy with work"

def test_dm_examples_context_starts_with_user():
    # If context window captures leading assistant turns, they should be dropped
    msgs = [
        make_msg(OTHER, "hey"),
        make_msg(YOUR_NAME, "hey!"),           # assistant turn
        make_msg(YOUR_NAME, "what's up"),      # merged → one assistant turn
        make_msg(OTHER, "not much, you?"),
        make_msg(YOUR_NAME, "same honestly"),  # the example we care about
    ]
    examples = extract_dm_examples(make_thread(msgs))
    # Find the example for "same honestly"
    target = [e for e in examples if e["messages"][-1]["content"] == "same honestly"]
    assert target, "expected an example for 'same honestly'"
    # First non-system turn must be user
    non_system = [m for m in target[0]["messages"] if m["role"] != "system"]
    assert non_system[0]["role"] == "user", f"first turn was {non_system[0]['role']}"

def test_dm_examples_short_response_filtered():
    msgs = [
        make_msg(OTHER, "check this out"),
        make_msg(OTHER, "pretty cool right"),
        make_msg(YOUR_NAME, "lol"),  # too short (< MIN_RESPONSE_LEN=10)
    ]
    assert extract_dm_examples(make_thread(msgs)) == []

def test_dm_examples_no_prior_context_filtered():
    # Omar sends the very first message — no prior turns at all
    msgs = [
        make_msg(YOUR_NAME, "hey, how's it going?"),
    ]
    assert extract_dm_examples(make_thread(msgs)) == []

def test_dm_examples_proper_multi_turn_structure():
    msgs = [
        make_msg(OTHER, "are you free Saturday?"),
        make_msg(YOUR_NAME, "yeah should be"),
        make_msg(OTHER, "want to grab lunch?"),
        make_msg(YOUR_NAME, "definitely, where were you thinking?"),
    ]
    examples = extract_dm_examples(make_thread(msgs))
    # Second example should have 3 non-system turns: user, assistant, user → assistant
    second = [e for e in examples if "definitely" in e["messages"][-1]["content"]]
    assert second
    non_sys = [m for m in second[0]["messages"] if m["role"] != "system"]
    assert non_sys[0]["role"] == "user"
    assert non_sys[-1]["role"] == "assistant"
    # Roles must strictly alternate user/assistant
    for i in range(len(non_sys) - 1):
        assert non_sys[i]["role"] != non_sys[i+1]["role"], \
            f"consecutive same roles at positions {i},{i+1}: {[m['role'] for m in non_sys]}"


# ── build_group_turns ─────────────────────────────────────────────────────────

def test_group_turns_omar_becomes_you():
    msgs = [make_msg(YOUR_NAME, "hey all")]
    turns = build_group_turns(msgs)
    assert turns[0]["sender"] == "[you]"

def test_group_turns_others_keep_names():
    msgs = [make_msg("Alice", "hi"), make_msg("Bob", "hey")]
    turns = build_group_turns(msgs)
    assert turns[0]["sender"] == "Alice"
    assert turns[1]["sender"] == "Bob"

def test_group_turns_merges_consecutive_same_sender():
    msgs = [
        make_msg(YOUR_NAME, "line 1"),
        make_msg(YOUR_NAME, "line 2"),
        make_msg("Alice", "ok"),
    ]
    turns = build_group_turns(msgs)
    assert len(turns) == 2
    assert turns[0]["sender"] == "[you]"
    assert turns[0]["content"] == "line 1\nline 2"

def test_group_turns_does_not_merge_different_senders():
    msgs = [
        make_msg(YOUR_NAME, "hey"),
        make_msg("Alice", "hi"),
        make_msg(YOUR_NAME, "how are you"),
    ]
    turns = build_group_turns(msgs)
    assert len(turns) == 3


# ── extract_group_examples ────────────────────────────────────────────────────

def test_group_examples_basic():
    msgs = [
        make_msg("Alice", "anyone seen the game?"),
        make_msg("Bob", "yes insane ending"),
        make_msg(YOUR_NAME, "I watched it live, incredible"),
    ]
    thread = make_thread(msgs, is_dm=False)
    examples = extract_group_examples(thread)
    assert len(examples) == 1
    ex = examples[0]
    non_sys = [m for m in ex["messages"] if m["role"] != "system"]
    assert non_sys[0]["role"] == "user"
    assert non_sys[-1]["role"] == "assistant"
    assert "Alice" in non_sys[0]["content"]
    assert "Bob" in non_sys[0]["content"]

def test_group_examples_no_omar_name_in_context():
    msgs = [
        make_msg("Alice", "what do you think?"),
        make_msg(YOUR_NAME, "honestly not sure"),
        make_msg("Alice", "fair enough"),
        make_msg(YOUR_NAME, "I'll look into it"),
    ]
    thread = make_thread(msgs, is_dm=False)
    examples = extract_group_examples(thread)
    # Omar's name must not appear in any context turn
    for ex in examples:
        ctx = ex["messages"][1]["content"]
        assert "Omar Battisha" not in ctx, f"real name leaked into context: {ctx}"
    # But [you] should appear when Omar had a prior message
    last_ex = examples[-1]
    assert "[you]" in last_ex["messages"][1]["content"]

def test_group_examples_consecutive_omar_merged():
    msgs = [
        make_msg("Bob", "so what do we do today?"),   # extra prior so context >= 2
        make_msg("Alice", "what's the plan?"),
        make_msg(YOUR_NAME, "first we go"),
        make_msg(YOUR_NAME, "then we eat"),            # consecutive — should merge with above
        make_msg("Alice", "sounds good"),
        make_msg(YOUR_NAME, "great, settled"),
    ]
    thread = make_thread(msgs, is_dm=False)
    examples = extract_group_examples(thread)
    # The two consecutive Omar messages should be merged into a single assistant turn
    first = examples[0]
    assert first["messages"][-1]["content"] == "first we go\nthen we eat"


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
