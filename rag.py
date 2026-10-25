"""Local RAG: index your own documents so Forge can look things up.

Usage:
    uv run rag.py ingest              # (re)build the index from knowledge/
    uv run rag.py search "mqtt auth"  # show the best-matching passages
"""

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
KNOWLEDGE_DIR = ROOT / "knowledge"
INDEX_DIR = ROOT / "rag_index"
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHUNK_CHARS = 1200
OVERLAP_CHARS = 200
TEXT_SUFFIXES = {".md", ".txt", ".rst", ".c", ".h", ".py", ".sh", ".json", ".yaml", ".yml", ".html", ".csv"}


def embed(texts: list[str]) -> np.ndarray:
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embed",
        data=json.dumps({"model": EMBED_MODEL, "input": texts}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        vectors = np.array(json.load(resp)["embeddings"], dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(errors="ignore")


def chunk(text: str) -> list[str]:
    text = text.strip()
    step = CHUNK_CHARS - OVERLAP_CHARS
    return [text[i : i + CHUNK_CHARS] for i in range(0, max(len(text), 1), step) if text[i : i + CHUNK_CHARS].strip()]


def ingest() -> None:
    files = [p for p in sorted(KNOWLEDGE_DIR.rglob("*")) if p.is_file() and (p.suffix.lower() in TEXT_SUFFIXES or p.suffix.lower() == ".pdf")]
    if not files:
        sys.exit(f"No documents found. Put .md/.txt/.pdf/code files in {KNOWLEDGE_DIR}/ and run again.")

    chunks = []
    for path in files:
        for i, piece in enumerate(chunk(read_file(path))):
            chunks.append({"source": str(path.relative_to(KNOWLEDGE_DIR)), "part": i, "text": piece})
        print(f"read {path.relative_to(KNOWLEDGE_DIR)}")

    vectors = []
    for start in range(0, len(chunks), 32):
        batch = chunks[start : start + 32]
        vectors.append(embed(["search_document: " + c["text"] for c in batch]))
        print(f"embedded {min(start + 32, len(chunks))}/{len(chunks)} chunks")

    INDEX_DIR.mkdir(exist_ok=True)
    np.save(INDEX_DIR / "vectors.npy", np.vstack(vectors))
    (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks))
    print(f"Done: {len(files)} files, {len(chunks)} chunks -> {INDEX_DIR}/")


def index_exists() -> bool:
    return (INDEX_DIR / "vectors.npy").exists()


def search(query: str, k: int = 4, min_score: float = 0.55) -> list[dict]:
    if not index_exists():
        return []
    vectors = np.load(INDEX_DIR / "vectors.npy")
    chunks = json.loads((INDEX_DIR / "chunks.json").read_text())
    scores = vectors @ embed(["search_query: " + query])[0]
    best = np.argsort(scores)[::-1][:k]
    return [{**chunks[i], "score": float(scores[i])} for i in best if scores[i] >= min_score]


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "ingest":
        ingest()
    elif len(sys.argv) >= 3 and sys.argv[1] == "search":
        for hit in search(" ".join(sys.argv[2:]), min_score=0.0):
            print(f"\n[{hit['score']:.2f}] {hit['source']} (part {hit['part']})\n{hit['text'][:400]}")
    else:
        print(__doc__)
