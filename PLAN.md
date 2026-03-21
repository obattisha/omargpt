# Omar Chatbot Fine-tune Plan

Goal: fine-tune an LLM on Meta/Facebook message exports to sound like Omar,
then host it as a chatbot on his website.

## Data
- Source: `~/Downloads/meta/` — 3 Facebook export archives
- 1,229 conversation threads (inbox), ~61,615 messages from Omar, ~600K tokens
- Format: JSON files with `sender_name`, `timestamp_ms`, `content`

## Model
**Qwen2.5-7B-Instruct** — beats Llama 3.1 8B on most benchmarks at same size,
excellent instruction following, well-supported by Unsloth.

## Fine-tuning Method
QLoRA via Unsloth on Google Colab (free T4 for prototyping, RunPod for final run).

## Training Format
ChatML instruction-tuning pairs:
```json
{"messages": [
  {"role": "system", "content": "You are Omar. Reply naturally in his voice."},
  {"role": "user", "content": "<last 3-5 messages before Omar's reply>"},
  {"role": "assistant", "content": "<Omar's actual reply>"}
]}
```

## Steps
- [x] 1. Data pipeline — parse JSONs, clean encoding, extract (context, response) pairs, filter, export to JSONL
- [x] 2. Review sample — looks good, arabizi fine, issues fixed (merging, [you] label, system prompt)
- [ ] 3. Training run on Colab with Unsloth (Qwen2.5-7B-Instruct, 1-2 epochs)
- [ ] 4. Evaluate — chat with the model, check if it sounds right
- [ ] 5. Iterate on filtering/hyperparams if needed
- [ ] 6. Quantize (GGUF 4-bit) and deploy to VPS or HF Inference Endpoint
- [ ] 7. Wire up a chat widget on website

## Hosting options
- Cheap: Hetzner VPS + llama.cpp or ollama (~$8/mo)
- Managed: HuggingFace Inference Endpoints (~$0.06/hr, pay-per-use)

## Key notes
- Meta exports have mojibake: bytes are UTF-8 encoded as latin-1 — fix with `.encode('latin-1').decode('utf-8')`
- Filter: drop media-only, very short messages (<10 chars), keep DMs and group chats separately
- Context window: last 5 messages before Omar's reply as the "user" turn
- Project dir: `~/meta-finetune/`
- Data pipeline script: `~/meta-finetune/build_dataset.py`
