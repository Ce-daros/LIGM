#!/usr/bin/env bash
set -euo pipefail

repo_id="${1:-raincandy-u/ModernBERT-base-LIGM}"
target="${2:-/nvme-data/ligm/verification/ModernBERT-base-LIGM}"

test ! -e "$target"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

"$HOME/.local/bin/uv" run --no-sync hf download "$repo_id" \
  --local-dir "$target"
"$HOME/.local/bin/uv" run --no-sync python - "$target" <<'PY'
import sys
from pathlib import Path

from transformers import AutoModelForMaskedLM, AutoTokenizer

target = Path(sys.argv[1])
required = {
    "README.md",
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "research/stage2-online-curve.json",
    "research/online-evaluation/selection.json",
}
missing = sorted(path for path in required if not target.joinpath(path).is_file())
if missing:
    raise SystemExit(f"Missing release files: {missing}")
AutoTokenizer.from_pretrained(target, local_files_only=True)
AutoModelForMaskedLM.from_pretrained(target, local_files_only=True)
print("Hugging Face release verified")
PY
