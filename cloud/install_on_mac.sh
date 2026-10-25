#!/usr/bin/env bash
# Install the downloaded GGUF into Ollama as "forge-8b".
# Usage: bash cloud/install_on_mac.sh path/to/file.gguf
set -euo pipefail
GGUF="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
cd "$(dirname "$0")/.."

# Reuse the official Qwen3 chat template from the qwen3:4b model already installed.
ollama show qwen3:4b >/dev/null 2>&1 || ollama pull qwen3:4b
{
  echo "FROM $GGUF"
  echo 'TEMPLATE """'"$(ollama show qwen3:4b --template)"'"""'
  grep '^SYSTEM' Modelfile
  echo 'PARAMETER stop "<|im_start|>"'
  echo 'PARAMETER stop "<|im_end|>"'
  echo 'PARAMETER temperature 0.6'
  echo 'PARAMETER top_p 0.95'
  echo 'PARAMETER top_k 20'
  echo 'PARAMETER num_ctx 8192'
} > Modelfile.8b
ollama create forge-8b -f Modelfile.8b
echo 'Done. Chat with: uv run chat.py --model forge-8b'
