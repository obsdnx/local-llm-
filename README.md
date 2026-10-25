# Forge — a local, open-source, customisable LLM

Runs fully on your Mac. Base model: Qwen3 4B (Apache 2.0). Fine-tuned with Apple MLX.
Looks things up in your own documents (RAG).

Forge is trained for: C programming, teaching code, step-by-step reasoning,
cybersecurity, Linux, and Python AI model training.

## Daily use

| I want to… | Run |
|---|---|
| Chat (base model + my rules) | `uv run chat.py` |
| Chat with my fine-tuned model | Terminal 1: `uv run mlx_lm.server --model forge-fused --port 8080`<br>Terminal 2: `uv run chat.py --tuned` |
| Chat without document lookup | add `--no-rag` |

In chat: `/reset` clears the conversation, `/exit` quits.

## Add knowledge (RAG) — IoT docs, notes, writeups, PDFs

1. Put `.md`, `.txt`, `.pdf` or code files in `knowledge/` (subfolders are fine).
2. Run `uv run rag.py ingest`.
3. Chat as normal. Forge prints `(looked up: …)` when it used your documents.

Test a lookup: `uv run rag.py search "mqtt tls port"`.

## Train it further (fine-tuning)

1. Add examples to any file in `training/` (one per line, same format as the existing lines),
   or create a new file like `training/iot.jsonl`.
2. Run `./train.sh` (about 45 minutes on an M2). More examples: raise `ITERS`, e.g. `ITERS=400 ./train.sh`.
3. Restart the mlx server and chat with `--tuned`.

Rule of thumb: fine-tuning changes *how* it answers (style, focus).
RAG changes *what it knows*. For new facts, use RAG.

## Change the rules / personality

Edit `Modelfile` (Ollama) and `SYSTEM_PROMPT` in `chat.py`, then run `ollama create forge -f Modelfile`.

## Files

- `Modelfile` — Ollama model "forge": system prompt + settings
- `chat.py` — terminal chat, streams replies, adds RAG context
- `rag.py` — indexes `knowledge/` with the `nomic-embed-text` embedding model
- `training/*.jsonl` — fine-tuning examples, one file per topic
- `prepare_data.py` — builds `data/train.jsonl` + `data/valid.jsonl`
- `train.sh` — fine-tune (LoRA) + fuse into `forge-fused/`
