#!/usr/bin/env bash
# Install training tools on the rented NVIDIA GPU machine (Ubuntu + CUDA image).
# Versions pinned to match Unsloth's official Qwen3 notebook.
set -euo pipefail
pip install --upgrade pip
pip install unsloth
pip install "transformers==4.57.6"
pip install --no-deps "trl==0.22.2"
python -c "import torch; assert torch.cuda.is_available(), 'No NVIDIA GPU found'; print('GPU:', torch.cuda.get_device_name(0))"
