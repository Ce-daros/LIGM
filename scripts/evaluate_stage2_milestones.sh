#!/usr/bin/env bash
set -euo pipefail

root=/nvme-data/ligm
repo="$root/repo"
base="$root/models/ModernBERT-base"
results="$root/runs/stage2-milestones"
uv="$HOME/.local/bin/uv"

mkdir -p "$results"
cd "$repo"
export HF_HUB_OFFLINE=1

for method in random ligm; do
  run="$root/runs/stage1-${method}-seed11"
  for milestone in 100000000 250000000 500000000 750000000 1000000000; do
    checkpoint="$($uv run --no-sync python - "$run" "$milestone" <<'PY'
import sys
from pathlib import Path

run = Path(sys.argv[1])
milestone = int(sys.argv[2])
checkpoints = sorted(
    run.joinpath("checkpoints").glob("tokens-*.pt"),
    key=lambda path: int(path.stem.removeprefix("tokens-")),
)
match = next(
    (
        path
        for path in checkpoints
        if int(path.stem.removeprefix("tokens-")) >= milestone
    ),
    None,
)
if match is not None:
    print(match)
PY
)"
    if [[ -z "$checkpoint" ]]; then
      continue
    fi
    actual="${checkpoint##*tokens-}"
    actual="${actual%.pt}"
    "$uv" run --no-sync ligm-evaluate "$base" \
      --checkpoint "$checkpoint" \
      --output "$results/${method}-tokens-${actual}-synthetic.json"
  done
done
