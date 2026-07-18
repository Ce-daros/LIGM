#!/usr/bin/env bash
set -euo pipefail

repo=/nvme-data/ligm/repo
root=/nvme-data/ligm

while pgrep -f 'bash scripts/continue_stage2_ligm.sh' >/dev/null; do
  sleep 60
done

cd "$repo"
export HF_HUB_OFFLINE=1
uv_bin="$HOME/.local/bin/uv"

"$uv_bin" run --no-sync ligm-train configs/smoke-stage3-weighted.yaml
"$uv_bin" run --no-sync ligm-train configs/stage3-random-long.yaml
"$uv_bin" run --no-sync ligm-train configs/stage3-ligm-weighted4.yaml
"$uv_bin" run --no-sync ligm-train configs/stage3-ligm-weighted8.yaml

for variant in weighted4 weighted8; do
  "$uv_bin" run --no-sync python -m ligm.online_report \
    "$root/runs/stage3-ligm-${variant}-seed11/online-evaluation" \
    "$root/runs/stage3-random-long-seed11/online-evaluation" \
    --output "$root/runs/stage3-${variant}-curve.json"
done
