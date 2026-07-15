#!/usr/bin/env bash
set -euo pipefail

root=/nvme-data/ligm
repo="$root/repo"

cd "$repo"
HF_HUB_OFFLINE=1 "$HOME/.local/bin/uv" run ligm-train configs/stage1-random.yaml \
  > "$root/runs/stage1-random.log" 2>&1
HF_HUB_OFFLINE=1 "$HOME/.local/bin/uv" run ligm-train configs/stage1-ligm.yaml \
  > "$root/runs/stage1-ligm.log" 2>&1
