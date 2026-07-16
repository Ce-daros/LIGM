#!/usr/bin/env bash
set -euo pipefail

root=/nvme-data/ligm
repo="$root/repo"
run="$root/runs/stage1-ligm-seed11"
results="$root/runs/stage1-evaluation"
curve="$root/runs/stage2-online-curve.json"
output="${1:-$root/release/ModernBERT-base-LIGM}"

test -f "$run/online-evaluation/selection.json"
test -f "$curve"
test ! -e "$output"

cd "$repo"
"$HOME/.local/bin/uv" run --no-sync ligm-release \
  --model "$run/final" \
  --run "$run" \
  --results "$results" \
  --manifests "$repo/manifests" \
  --license "$repo/LICENSE" \
  --online-curve "$curve" \
  --online-evaluation "$run/online-evaluation" \
  --output "$output" \
  --repo-id raincandy-u/ModernBERT-base-LIGM

find "$output" -type f -printf '%P\t%s\n' | sort > "$output/release-files.tsv"
