#!/usr/bin/env bash
# Matched Adam/Muon runs. Invoke from the repository root.
set -euo pipefail

track=${1:-fixed}
case "$track" in
  fixed) stop=() ;;
  speedrun) stop=(--stop-at-target) ;;
  *) echo "Usage: bash train/dust_muon_runs.sh [fixed|speedrun]" >&2; exit 1 ;;
esac

output=${OUTPUT_ROOT:-logs/dust-muon/$track}
for optimizer in adam muon; do
  uv run torchrun --standalone --nproc_per_node="${GPUS:-8}" train/val/dust.py \
    --optimizer "$optimizer" --muon-lr "${MUON_LR:-0.01}" \
    --steps "${STEPS:-1831}" --scale "${SCALE:-1.5}" \
    --eval-every "${EVAL_EVERY:-100}" --target-loss "${TARGET_LOSS:-3.5}" \
    --data-dir "${DATA_DIR:-data}" --output "$output/$optimizer" "${stop[@]}"
done
