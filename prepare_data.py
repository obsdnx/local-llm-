"""Build data/train.jsonl and data/valid.jsonl from training/*.jsonl.

Each file in training/ is one topic. Add your own examples to any file (or add a new file),
then run:  uv run prepare_data.py
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).parent
VALID_PER_TOPIC = 2

random.seed(0)
train, valid = [], []
for path in sorted((ROOT / "training").glob("*.jsonl")):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    random.shuffle(rows)
    valid += rows[:VALID_PER_TOPIC]
    train += rows[VALID_PER_TOPIC:]
    print(f"{path.name}: {len(rows)} examples")

random.shuffle(train)
(ROOT / "data").mkdir(exist_ok=True)
for name, rows in (("train", train), ("valid", valid)):
    (ROOT / "data" / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
print(f"Wrote data/train.jsonl ({len(train)}) and data/valid.jsonl ({len(valid)})")
