# Forge — architecture

Forge is a local, fully open-source AI assistant. Everything runs on your Mac
except optional training on a rented cloud GPU.

## The pieces

```
                         ┌──────────────────────────────┐
  you ──> chat.py ──────>│ Ollama (localhost:11434)     │  runs the model
            │            │   forge     = qwen3:4b + rules│
            │            │   forge-8b  = your 8B GGUF    │
            │            └──────────────────────────────┘
            │
            │  before each question
            ▼
         rag.py ──> rag_index/  <── uv run rag.py ingest <── knowledge/  (your docs: IoT, notes, PDFs)
            (nomic-embed-text embeddings via Ollama)


  Training (changes HOW it answers)

  training/*.jsonl ──┬─> train.sh (Mac, MLX, 4B)  ──> adapters/ ──> forge-fused/ ──> mlx_lm.server :8080
                     │
                     └─> cloud/train_8b.py (rented NVIDIA GPU, Unsloth, 8B)
                              ──> forge-8b-gguf/*.gguf ──download──> cloud/install_on_mac.sh ──> Ollama "forge-8b"
```

## Components

| File / folder | What it does |
|---|---|
| `Modelfile` | Defines Ollama model `forge`: base model, system prompt (rules), settings |
| `chat.py` | Terminal chat. Streams replies, hides `<think>` text, adds RAG context |
| `rag.py` | Splits `knowledge/` files into chunks, embeds them, searches by meaning |
| `knowledge/` | Your own documents. Private: git ignores everything except the starter note |
| `training/*.jsonl` | Fine-tuning examples, one file per topic |
| `prepare_data.py` | Builds `data/train.jsonl` + `data/valid.jsonl` for MLX |
| `train.sh` | Mac fine-tuning (LoRA, 4B). Saves every 10 steps, auto-resumes after macOS GPU stops |
| `cloud/setup.sh` | Installs Unsloth on a rented GPU machine |
| `cloud/train_8b.py` | Cloud fine-tuning of Qwen3-8B. Keeps the best checkpoint, exports a 4-bit GGUF |
| `cloud/install_on_mac.sh` | Loads the downloaded GGUF into Ollama as `forge-8b` |

## Key decisions

| Decision | Why |
|---|---|
| Base model: Qwen3 (Apache 2.0) | Fully open licence, strong for its size, has a thinking mode |
| 4B on the Mac, 8B from the cloud | 16 GB Mac can run an 8B 4-bit model (~5 GB) but can't train it reliably |
| LoRA, not full training | Trains ~1% of weights: fast, cheap, fits in memory |
| RAG for facts, fine-tuning for style | Fine-tuning doesn't reliably add new facts; RAG does and updates instantly |
| 25% open maths reasoning data in 8B training | Unsloth's Qwen3 guide: training only on direct answers can weaken reasoning |
| Best checkpoint chosen by validation loss | The Mac run overfitted after ~50 steps; picking the best avoids that |
| No Hack The Box training examples | Use RAG with your own HTB notes instead |

## Hardware limits found (M2, 16 GB)

- macOS stops long GPU jobs ("Impacting Interactivity") when memory is tight. `train.sh` works around it
  with smaller GPU batches, gradient checkpointing, a 1024-token limit and auto-resume.
- 4B training runs about 10–15 seconds per step. 8B training is not practical on this Mac.
