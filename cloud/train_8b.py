"""Fine-tune Forge on Qwen3-8B with Unsloth on a rented NVIDIA GPU, then export a GGUF for Ollama.

Run on the cloud machine (see cloud/README.md):
    bash cloud/setup.sh
    python cloud/train_8b.py

Check the data formatting on any machine without a GPU:
    uv run --with datasets cloud/train_8b.py --dry-run

Output: forge-8b-gguf/*.gguf (about 5 GB). Download it to your Mac and run cloud/install_on_mac.sh.
"""

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = "unsloth/Qwen3-8B"
MAX_SEQ_LENGTH = 4096
VALID_PER_TOPIC = 4
# Share of open-source maths reasoning examples mixed in, so Qwen3 keeps its <think> ability.
# Unsloth's Qwen3 guide warns that training only on direct answers can weaken reasoning.
REASONING_SHARE = 0.25
SEED = 3407


def load_own_data() -> tuple[list, list]:
    """Read training/*.jsonl. Hold out VALID_PER_TOPIC examples per file for validation."""
    rng = random.Random(SEED)
    train, valid = [], []
    for path in sorted((ROOT / "training").glob("*.jsonl")):
        rows = [json.loads(line)["messages"] for line in path.read_text().splitlines() if line.strip()]
        rng.shuffle(rows)
        valid += rows[:VALID_PER_TOPIC]
        train += rows[VALID_PER_TOPIC:]
        print(f"{path.name}: {len(rows)} examples")
    return train, valid


def no_think(messages: list) -> list:
    # Same trick as Unsloth's Qwen3 notebook: tag direct-answer examples with /no_think.
    return [{**m, "content": m["content"] + " /no_think"} if m["role"] == "user" else m for m in messages]


def build_texts(tokenizer, n_own: int, own: list) -> list[str]:
    from datasets import load_dataset

    texts = [tokenizer.apply_chat_template(no_think(m), tokenize=False) for m in own]
    n_reasoning = int(n_own * REASONING_SHARE / (1 - REASONING_SHARE))
    if n_reasoning:
        maths = load_dataset("unsloth/OpenMathReasoning-mini", split="cot").shuffle(seed=SEED)
        added = 0
        for row in maths:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["problem"]}, {"role": "assistant", "content": row["generated_solution"]}],
                tokenize=False,
            )
            if len(tokenizer(text)["input_ids"]) <= MAX_SEQ_LENGTH:
                texts.append(text)
                added += 1
            if added == n_reasoning:
                break
        print(f"Added {added} open-source reasoning examples")
    random.Random(SEED).shuffle(texts)
    return texts


def dry_run() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    train, valid = load_own_data()
    texts = build_texts(tokenizer, len(train), train)
    lengths = sorted(len(tokenizer(t)["input_ids"]) for t in texts)
    print(f"\ntrain texts: {len(texts)}, valid: {len(valid)}, tokens max {lengths[-1]}, median {lengths[len(lengths) // 2]}")
    print("\n--- first own example as the model will see it ---\n")
    print(tokenizer.apply_chat_template(no_think(train[0]), tokenize=False)[:1500])


def train() -> None:
    from unsloth import FastLanguageModel  # must be imported before transformers/trl
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    own_train, own_valid = load_own_data()
    train_ds = Dataset.from_dict({"text": build_texts(tokenizer, len(own_train), own_train)})
    valid_ds = Dataset.from_dict({"text": [tokenizer.apply_chat_template(no_think(m), tokenize=False) for m in own_valid]})

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            learning_rate=1e-4,
            warmup_steps=10,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            weight_decay=0.01,
            logging_steps=5,
            # Test on held-out examples every 20 steps and keep the best checkpoint (stops overfitting).
            eval_strategy="steps",
            eval_steps=20,
            save_strategy="steps",
            save_steps=20,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            output_dir="checkpoints",
            seed=SEED,
            report_to="none",
        ),
    )
    # Only learn from the assistant's replies, not from the questions.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    trainer.train()
    print(f"Best checkpoint: {trainer.state.best_model_checkpoint} (eval loss {trainer.state.best_metric})")

    model.save_pretrained("forge-8b-lora")
    tokenizer.save_pretrained("forge-8b-lora")
    model.save_pretrained_gguf("forge-8b-gguf", tokenizer, quantization_method="q4_k_m")
    ggufs = sorted(Path(".").rglob("*.gguf"))
    print("\nDone. GGUF file(s) to download:\n" + "\n".join(f"  {p.resolve()}  ({p.stat().st_size / 1e9:.1f} GB)" for p in ggufs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="check data formatting only, no GPU needed")
    dry_run() if parser.parse_args().dry_run else train()
