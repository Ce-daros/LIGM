#!/usr/bin/env bash
set -euo pipefail

root=/nvme-data/ligm
repo="$root/repo"
results="$root/runs/stage1-evaluation"
uv="$HOME/.local/bin/uv"

cd "$repo"
"$uv" run --no-sync python - <<'PY'
import json
from pathlib import Path

report = json.loads(
    Path("/nvme-data/ligm/runs/stage1-evaluation/mechanism-gate.json").read_text()
)
if not report["passed"]:
    raise SystemExit("Mechanism gate did not pass; conditional experiments are not authorized")
PY

export HF_HUB_OFFLINE=1
"$uv" run --no-sync ligm-train configs/stage1-entropy.yaml \
  > "$root/runs/stage1-entropy.log" 2>&1
"$uv" run --no-sync ligm-train configs/stage1-ligm-gain.yaml \
  > "$root/runs/stage1-ligm-gain.log" 2>&1

for method in entropy ligm-gain; do
  model="$root/runs/stage1-${method}-seed11/final"
  "$uv" run --no-sync ligm-evaluate "$model" \
    --output "$results/${method}-synthetic.json"
  "$uv" run --no-sync ligm-evaluate "$model" \
    --natural-config configs/stage1-random.yaml \
    --output "$results/${method}-natural.json"
done

unset HF_HUB_OFFLINE
bash scripts/download_retrieval.sh
export HF_HUB_OFFLINE=1

declare -A checkpoints=(
  [base]="$root/models/ModernBERT-base"
  [random]="$root/runs/stage1-random-seed11/final"
  [ligm]="$root/runs/stage1-ligm-seed11/final"
)

for method in base random ligm; do
  retriever="$root/runs/retrieval-${method}"
  evaluation="$results/${method}-mldr-work"
  "$uv" run --no-sync ligm-retrieval train configs/retrieval-probe.yaml \
    "${checkpoints[$method]}" --output "$retriever"
  "$uv" run --no-sync ligm-retrieval evaluate configs/retrieval-probe.yaml \
    "$retriever/final" --output "$evaluation"
  install -m 0644 "$evaluation/mldr-dev.json" "$results/${method}-mldr.json"
done

"$uv" run --no-sync ligm-gate "$results" \
  --full --output "$results/full-gate.json"
