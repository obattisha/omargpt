# Personal Chatbot Fine-Tune Pipeline

Fine-tune an open-source LLM to sound like a specific person, using their real message history as training data. The result is a deployable chatbot that mimics their voice, personality, and communication style.

This repo contains the full pipeline. The actual training data is private and not included — see [Data](#data) below for the expected formats.

---

## What This Does

1. **Parses** Facebook Messenger, WhatsApp, and ChatGPT exports into a unified training dataset
2. **Generates** a "constitution" — a system prompt + identity anchors that prevent the model from hallucinating wrong values or beliefs
3. **Fine-tunes** LLaMA 3.1 8B Instruct using QLoRA via [Unsloth](https://github.com/unslothai/unsloth) on Google Colab
4. **Pushes** the trained adapter + GGUF-quantized model to HuggingFace Hub
5. **Serves** via [ollama](https://ollama.ai) — locally, on a VPS, or behind a Cloudflare Tunnel

---

## Repo Structure

```
├── build_dataset.py            # Main pipeline — merges all sources into dataset.jsonl
├── build_chatgpt_dataset.py    # Parses ChatGPT conversation export
├── build_whatsapp_dataset.py   # Parses WhatsApp .txt exports
├── generate_constitution.py    # Generates system prompt + synthetic examples via Claude API
├── omar_finetune.ipynb         # Google Colab training notebook (LLaMA 3.1 8B + Unsloth)
├── test_base_model.py          # Test base model locally via ollama before fine-tuning
├── test_build_dataset.py       # Unit tests for the data pipeline
├── constitution.md             # Human-readable identity constitution (review before training)
├── constitution_system_prompt.txt  # System prompt baked into every training example
├── data/
│   ├── dataset_example.jsonl         # Example of final training format
│   ├── meta_export_example.json      # Example Meta/Facebook export structure
│   ├── whatsapp_export_example.txt   # Example WhatsApp export structure
│   └── chatgpt_export_example.json   # Example ChatGPT export structure
```

---

## Data

The actual training data is not included in this repo for privacy reasons. The pipeline expects three sources:

### 1. Facebook Messenger Export
Request your data at **Facebook → Settings → Your Facebook Information → Download Your Information**. Select Messages, JSON format.

Expected location: `~/Downloads/meta/`

Structure: see `data/meta_export_example.json`

### 2. WhatsApp Exports
Export individual chats from **WhatsApp → Chat → ⋮ → More → Export Chat → Without Media**.

Expected location: `WhatsApp conversations/*.txt` (one file per chat)

Structure: see `data/whatsapp_export_example.txt`

### 3. ChatGPT History
Request your data at **ChatGPT → Settings → Data Controls → Export Data**.

Expected location: `ChatgptHistory/conversations-*.json`

Structure: see `data/chatgpt_export_example.json`

### 4. Constitution Synthetic Examples
A set of hand-crafted Q&A pairs covering identity, values, and beliefs — used to anchor the model and prevent hallucination. Not included (personal), but the format matches `data/dataset_example.jsonl`.

---

## Quickstart

```bash
# 1. Install dependencies
pip install anthropic

# 2. Build the dataset (requires data sources above)
python build_dataset.py

# 3. (Optional) Generate a constitution from your messages
export ANTHROPIC_API_KEY=sk-ant-...
python generate_constitution.py

# 4. Upload dataset.jsonl + omar_finetune.ipynb to Google Colab
#    Runtime → A100 GPU + High RAM → Run all cells

# 5. After training, test locally
ollama pull YOUR-HF-USERNAME/your-model-GGUF
python test_base_model.py --chat
```

---

## Dataset Format

All training examples use the standard ChatML messages format:

```json
{
  "messages": [
    {"role": "system",    "content": "Your name is ... [identity constitution]"},
    {"role": "user",      "content": "hey what's up?"},
    {"role": "assistant", "content": "not much lol, just got back from the gym"}
  ]
}
```

See `data/dataset_example.jsonl` for more examples.

---

## Training

The notebook (`omar_finetune.ipynb`) uses:
- **Model:** LLaMA 3.1 8B Instruct via Unsloth
- **Method:** QLoRA (r=16, all projection layers)
- **Hardware:** A100 40GB on Google Colab Pro
- **Epochs:** 2
- **Dataset:** ~33K examples across Messenger, WhatsApp, and ChatGPT history

After training, the notebook pushes:
- LoRA adapter to HuggingFace Hub (~100MB)
- GGUF Q4_K_M quantized model for ollama hosting (~4GB)

---

## Serving

```bash
# Pull and run via ollama
ollama pull hf.co/YOUR-HF-USERNAME/your-model-GGUF:Q4_K_M
ollama run hf.co/YOUR-HF-USERNAME/your-model-GGUF:Q4_K_M
```

For always-on hosting, deploy to a Hetzner or any ARM VPS and expose via Cloudflare Tunnel.

---

## Constitution

The "constitution" is a two-layer identity anchor:

1. **System prompt** — baked into every training example and every inference call. States the person's name, background, values, communication style.
2. **Synthetic Q&A examples** — hand-crafted examples covering identity questions the base model would otherwise hallucinate wrong answers to (religion, politics, background, etc.).

See `constitution.md` and `constitution_system_prompt.txt` for the structure.

---

## Adapting for Yourself

1. Replace the system prompt in `constitution_system_prompt.txt` with your own identity
2. Point the data paths to your own exports
3. Run `build_dataset.py` to generate your training data
4. Upload to Colab and train
