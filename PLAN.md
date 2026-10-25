# Forge — plan and status

Last updated: 2026-10-25

## Done

- [x] Ollama installed; `qwen3:4b` and `nomic-embed-text` downloaded
- [x] `forge` model with custom rules (`Modelfile`)
- [x] Terminal chat (`chat.py`) — tested
- [x] RAG over `knowledge/` (`rag.py`) — tested: answered "8883" from the IoT note and cited it
- [x] 84 starter training examples (6 topics)
- [x] Mac fine-tuning (`train.sh`) with auto-resume — ran to step 140; best checkpoint was step 50
      (validation loss 1.516). Later steps overfitted.
- [x] 405 more training examples for the 8B model (all valid JSON, no duplicates, C programs compile)
- [x] Cloud 8B training script (`cloud/train_8b.py`) and Mac install script (`cloud/install_on_mac.sh`)
      — install script tested with a stand-in GGUF

## Next steps

Each step is ≤15 minutes of your time (training itself runs unattended).

1. **Finish the missing training data** (135 examples). Add to: `training/teaching_python_js.jsonl` (+15),
   `training/security_appsec.jsonl` (+15), `training/python_ai_pytorch.jsonl` (+15). Create:
   `training/teaching_c_bash_algorithms.jsonl` (45) and `training/reasoning_practical.jsonl` (45).
   Done when: about 600 new examples in total, all passing the format and compile checks.
2. **Dry-run the 8B data formatting** — run `uv run --with datasets cloud/train_8b.py --dry-run` on a machine
   with ≥16 GB free memory (it was killed for lack of memory while other jobs ran).
   Done when: it prints the example count and one formatted example.
3. **Rent a GPU** — RunPod or Vast.ai, 24 GB+ NVIDIA GPU (e.g. RTX 4090, L4, A5000), PyTorch/CUDA image.
   Done when: you can open a terminal on it.
4. **Train in the cloud** — `git clone` this repo on the machine, `bash cloud/setup.sh`,
   `python cloud/train_8b.py`. Done when: it prints the path of a `.gguf` file (~5 GB).
5. **Download and stop the GPU** — copy the `.gguf` to your Mac, then stop/delete the rented machine.
   Done when: the file is on your Mac and the provider shows no running machines.
6. **Install on the Mac** — `bash cloud/install_on_mac.sh path/to/file.gguf`.
   Done when: `uv run chat.py --model forge-8b` answers as Forge.
7. **Compare** — ask `forge` and `forge-8b` the same 5 questions from your topics.
   Done when: you've noted which one is better for each topic.

## Decisions log

- **Cloudflare Workers can't be used for training.** As far as I know, Cloudflare doesn't rent GPU machines
  you can train on; Workers AI runs hosted models for inference. Use RunPod / Vast.ai / Lambda for training.
- **Hosting as a website** (later): start with Open WebUI on the Mac + Tailscale (free, private).
  Move to a serverless GPU only if you want to share it.
- **Buying hardware**: wait until after step 7. If needed: Mac with ≥64 GB for MLX, or
  Framework Desktop 128 GB for running big models.
