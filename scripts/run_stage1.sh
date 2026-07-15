#!/usr/bin/env bash
set -euo pipefail

random_pid="$1"
root=/nvme-data/ligm
repo="$root/repo"

while kill -0 "$random_pid" 2>/dev/null; do
  sleep 60
done

test -f "$root/runs/stage1-random-seed11/final/model.safetensors"
cd "$repo"
HF_HUB_OFFLINE=1 "$HOME/.local/bin/uv" run ligm-train configs/stage1-ligm.yaml \
  > "$root/runs/stage1-ligm.log" 2>&1
