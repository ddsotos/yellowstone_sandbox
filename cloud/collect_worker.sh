#!/usr/bin/env bash
set -euo pipefail

worker_index="${1:?worker index 0..3 required}"
max_games="${MAX_GAMES:-100000}"
seed=$((20260730 + worker_index))
game_id_offset=$((2000000 + worker_index * 100000))
root="$(cd "$(dirname "$0")/.." && pwd)"
out="$root/cloud/artifacts/worker_${worker_index}/replay"
mkdir -p "$out"
cd "$root"
stop_file="$root/cloud/artifacts/worker_${worker_index}/STOP"
status_file="$root/cloud/artifacts/worker_${worker_index}/status.json"
PYTHONPATH=src python -m yellowstone.exploratory_collection \
  --checkpoint models/win_value_v1_original_generation0_197800_epoch002.pt \
  --seed "$seed" --game-id-offset "$game_id_offset" \
  --output "$out" --stop-file "$stop_file" --status-file "$status_file" \
  --shard-games 100 --max-games "$max_games" \
  --lazy-single-pass
PYTHONPATH=src python scripts/validate_cloud_collection.py "$out" \
  --expected-games "$max_games" \
  --output "$root/cloud/artifacts/worker_${worker_index}/validation.json"
tar -czf "$root/cloud/artifacts/worker_${worker_index}.tar.gz" \
  -C "$root/cloud/artifacts" "worker_${worker_index}"
