"""Terminal chat with Forge.

Usage:
    uv run chat.py               # chat with the Ollama model "forge" (Modelfile only)
    uv run chat.py --tuned       # chat with your fine-tuned model (start mlx server first, see README)
    uv run chat.py --no-rag      # don't look anything up in knowledge/

In chat: /reset clears the conversation, /exit quits.
"""

import argparse
import json
import re
import urllib.request

import rag

SYSTEM_PROMPT = (
    "You are Forge, a local AI assistant. You specialise in C programming, teaching code, "
    "step-by-step reasoning, cybersecurity, Linux, Hack The Box lab practice, and Python AI model training. "
    "Be direct and literal. Use short numbered steps, correct code, and plain-language definitions. "
    "Say when you are unsure."
)
BACKENDS = {
    "ollama": {"url": "http://localhost:11434/v1/chat/completions", "model": "forge"},
    "tuned": {"url": "http://localhost:8080/v1/chat/completions", "model": "forge-fused"},
}


def with_context(question: str) -> str:
    hits = rag.search(question)
    if not hits:
        return question
    sources = "\n\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    print(f"\033[2m(looked up: {', '.join(sorted({h['source'] for h in hits}))})\033[0m")
    return (
        "Use these excerpts from my documents if they are relevant. Cite the [source] you used. "
        "If they don't answer the question, say so and answer from your own knowledge.\n\n"
        f"{sources}\n\nQuestion: {question}"
    )


def stream_reply(backend: dict, messages: list[dict]) -> str:
    req = urllib.request.Request(
        backend["url"],
        data=json.dumps({"model": backend["model"], "messages": messages, "stream": True, "max_tokens": 4096}).encode(),
        headers={"Content-Type": "application/json"},
    )
    reply, thinking = "", False
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            piece = json.loads(line[6:])["choices"][0]["delta"].get("content") or ""
            reply += piece
            # Hide the model's <think>...</think> scratchpad, print only the answer.
            if "<think>" in piece:
                thinking = True
                print("\033[2m(thinking...)\033[0m", flush=True)
            if thinking:
                if "</think>" in reply:
                    thinking = False
                    print(reply.split("</think>", 1)[1].lstrip(), end="", flush=True)
                continue
            print(piece, end="", flush=True)
    print()
    return re.sub(r"<think>.*?</think>", "", reply, flags=re.S).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuned", action="store_true", help="use the fine-tuned model served by mlx_lm.server")
    parser.add_argument("--no-rag", action="store_true", help="don't search knowledge/")
    args = parser.parse_args()

    backend = BACKENDS["tuned" if args.tuned else "ollama"]
    use_rag = not args.no_rag and rag.index_exists()
    print(f"Forge ({backend['model']}, RAG {'on' if use_rag else 'off'}). /reset to clear, /exit to quit.")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question == "/exit":
            break
        if question == "/reset":
            messages = messages[:1]
            print("(conversation cleared)")
            continue
        if not question:
            continue
        messages.append({"role": "user", "content": with_context(question) if use_rag else question})
        messages.append({"role": "assistant", "content": stream_reply(backend, messages)})


if __name__ == "__main__":
    main()
