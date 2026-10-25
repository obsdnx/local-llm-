#!/usr/bin/env bash
# Fine-tune Forge on training/*.jsonl, then fuse the result into forge-fused/.
# Close other apps while training. If you run out of memory, change MODEL to mlx-community/Qwen3-1.7B-4bit.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="mlx-community/Qwen3-4B-4bit"
ITERS="${ITERS:-220}"   # about 3 passes over 74 examples; raise it when you add more data

# Split GPU work into small chunks so macOS doesn't kill training for "impacting interactivity".
export MLX_MAX_OPS_PER_BUFFER="${MLX_MAX_OPS_PER_BUFFER:-1}"
export MLX_MAX_MB_PER_BUFFER="${MLX_MAX_MB_PER_BUFFER:-256}"

uv run prepare_data.py
# macOS sometimes stops long GPU jobs anyway. Save every 10 steps and resume after a crash.
rm -rf adapters
DONE=0
ATTEMPT=0
while [ "$DONE" -lt "$ITERS" ]; do
  ATTEMPT=$((ATTEMPT + 1))
  [ "$ATTEMPT" -gt 30 ] && { echo "Gave up after 30 attempts."; exit 1; }
  RESUME=()
  [ -f adapters/resume.safetensors ] && RESUME=(--resume-adapter-file adapters/resume.safetensors)
  if uv run mlx_lm.lora --model "$MODEL" --train --data data \
    --iters "$((ITERS - DONE))" --batch-size 1 --num-layers 8 --learning-rate 1e-4 --mask-prompt \
    --max-seq-length 1024 --grad-checkpoint \
    --steps-per-report 10 --steps-per-eval 50 --val-batches 5 --save-every 10 \
    --adapter-path adapters ${RESUME[@]+"${RESUME[@]}"}; then
    DONE=$ITERS
  else
    LAST=$(ls adapters/[0-9]*_adapters.safetensors 2>/dev/null | sort | tail -1 || true)
    if [ -n "$LAST" ]; then
      DONE=$((DONE + 10#$(basename "$LAST" | cut -d_ -f1)))
      mv "$LAST" adapters/resume.safetensors
      rm -f adapters/[0-9]*_adapters.safetensors
    fi
    echo "GPU job stopped by macOS. Resuming at step $DONE of $ITERS (attempt $ATTEMPT)."
    sleep 5
  fi
done
uv run mlx_lm.fuse --model "$MODEL" --adapter-path adapters --save-path forge-fused
echo "Done. Serve it with: uv run mlx_lm.server --model forge-fused --port 8080"
