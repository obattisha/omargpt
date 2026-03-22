#!/usr/bin/env bash
# Oracle Cloud ARM A1 — one-time server setup
# Run this after SSH'ing in: bash setup.sh
set -euo pipefail

echo "=== 1. System packages ==="
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv git curl netfilter-persistent iptables-persistent

echo "=== 2. Open port 5000 in OS firewall ==="
# Oracle puts iptables rules in front of ufw — must do this or port is blocked
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 5000 -j ACCEPT
sudo netfilter-persistent save

echo "=== 3. Install ollama ==="
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable ollama
sudo systemctl start ollama
sleep 5   # give it a moment to start

echo "=== 4. Pull model from HuggingFace ==="
# This downloads the Q4_K_M GGUF (~4.7 GB) — takes a few minutes
ollama pull hf.co/obattisha/omar-llama3.1-8b-GGUF

echo "=== 5. Clone repo ==="
cd ~
git clone https://github.com/obattisha/omargpt
cd omargpt

echo "=== 6. Python dependencies ==="
# CPU-only torch (smaller, sufficient for safety models running on CPU)
pip3 install flask
pip3 install transformers
pip3 install torch --index-url https://download.pytorch.org/whl/cpu

echo "=== 7. HuggingFace login (needed to download safety models) ==="
echo "    Before running this, accept the Meta license at:"
echo "    https://huggingface.co/meta-llama/Prompt-Guard-86M"
echo "    https://huggingface.co/meta-llama/Llama-Guard-3-1B"
echo ""
huggingface-cli login

echo "=== 8. Install chatbot as a system service ==="
sudo cp deploy/chatbot.service /etc/systemd/system/chatbot.service
sudo systemctl daemon-reload
sudo systemctl enable chatbot

echo ""
echo "=== DONE — almost. ==="
echo ""
echo "One manual step remaining:"
echo "  Copy your constitution_system_prompt.txt to this machine:"
echo ""
echo "  scp -i ~/.ssh/oracle_key.pem constitution_system_prompt.txt ubuntu@$(curl -s ifconfig.me):~/omargpt/"
echo ""
echo "Then start the server:"
echo "  sudo systemctl start chatbot"
echo "  sudo systemctl status chatbot"
echo ""
echo "Your chatbot will be at: http://$(curl -s ifconfig.me):5000/health"
