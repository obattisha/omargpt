# Oracle Cloud Setup

## Step 1 — Create your Oracle Cloud account

1. Go to cloud.oracle.com → Start for Free
2. Sign up — you'll need a credit card but **Always Free resources never charge you**
3. Choose your home region (pick one close to you — can't change later)

---

## Step 2 — Set a $0 budget alert (do this first)

Billing & Cost Management → Budgets → Create Budget
- Name: `safety-cap`
- Amount: `$1`
- Alert threshold: 1% ($0.01)
- Email: your email

This emails you the moment anything charges. Oracle has no hard spending cap,
but you won't be charged if you only use Always Free resources.

---

## Step 3 — Create the VM

Compute → Instances → Create Instance

| Setting | Value |
|---|---|
| Name | `omar-chatbot` |
| Image | Ubuntu 22.04 |
| Shape | VM.Standard.A1.Flex |
| OCPUs | 4 |
| Memory | 24 GB |
| Boot volume | 50 GB (default) |

Under "Add SSH keys": generate a key pair and download the private key.

---

## Step 4 — Open port 5000 in Oracle's firewall

This is separate from the OS firewall — Oracle blocks everything by default.

Networking → Virtual Cloud Networks → your VCN → Security Lists → Default Security List → Add Ingress Rule

| Field | Value |
|---|---|
| Source CIDR | `0.0.0.0/0` |
| IP Protocol | TCP |
| Destination Port | `5000` |

---

## Step 5 — SSH in and run setup

```bash
chmod 400 ~/.ssh/oracle_key.pem
ssh -i ~/.ssh/oracle_key.pem ubuntu@<your-public-ip>

# On the server:
curl -O https://raw.githubusercontent.com/obattisha/omargpt/main/deploy/setup.sh
bash setup.sh
```

The script:
- Opens OS firewall port 5000
- Installs ollama and pulls the model (~4.7 GB, takes a few minutes)
- Clones the repo and installs Python deps
- Prompts for HuggingFace login (for safety models)
- Installs the chatbot as a systemd service

---

## Step 6 — Copy your system prompt

From your local machine:
```bash
scp -i ~/.ssh/oracle_key.pem \
  ~/meta-finetune/constitution_system_prompt.txt \
  ubuntu@<your-public-ip>:~/omargpt/
```

---

## Step 7 — Start the server

```bash
sudo systemctl start chatbot
sudo systemctl status chatbot  # should say "active (running)"
```

Test it:
```bash
curl http://<your-public-ip>:5000/health
# {"status": "ok", "model": "hf.co/obattisha/omar-llama3.1-8b-GGUF"}

curl -X POST http://<your-public-ip>:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "hey what'\''s up"}'
```

---

## Useful commands

```bash
# View live logs
journalctl -u chatbot -f

# Restart after code changes
cd ~/omargpt && git pull
sudo systemctl restart chatbot

# Check ollama
systemctl status ollama
ollama list
```

---

## RAM usage at steady state

| Component | RAM |
|---|---|
| Model (Q4_K_M GGUF in ollama) | ~5.5 GB |
| LlamaGuard-3-1B | ~2.2 GB |
| PromptGuard-86M | ~0.2 GB |
| Python + Flask + OS | ~1.5 GB |
| **Total** | **~9.4 GB** |

24 GB free tier gives you comfortable headroom.
