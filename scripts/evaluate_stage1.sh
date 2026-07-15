#!/usr/bin/env bash
set -euo pipefail

root=/nvme-data/ligm
repo="$root/repo"
results="$root/runs/stage1-evaluation"

while [[ ! -f "$root/runs/stage1-ligm-seed11/final/model.safetensors" ]]; do
  sleep 60
done

mkdir -p "$results"
cd "$repo"
export HF_HUB_OFFLINE=1
uv="$HOME/.local/bin/uv"

"$uv" run ligm-evaluate "$root/models/ModernBERT-base" \
  --output "$results/base-synthetic.json"
"$uv" run ligm-evaluate "$root/runs/stage1-random-seed11/final" \
  --output "$results/random-synthetic.json"
"$uv" run ligm-evaluate "$root/runs/stage1-ligm-seed11/final" \
  --output "$results/ligm-synthetic.json"
"$uv" run ligm-evaluate "$root/models/ModernBERT-base" \
  --natural-config configs/stage1-random.yaml --output "$results/base-natural.json"
"$uv" run ligm-evaluate "$root/runs/stage1-random-seed11/final" \
  --natural-config configs/stage1-random.yaml --output "$results/random-natural.json"
"$uv" run ligm-evaluate "$root/runs/stage1-ligm-seed11/final" \
  --natural-config configs/stage1-random.yaml --output "$results/ligm-natural.json"
"$uv" run ligm-report "$root/runs" --output "$results/training-summary.json"
"$uv" run ligm-gate "$results" --output "$results/mechanism-gate.json"
