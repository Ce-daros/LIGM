#!/usr/bin/env bash
set -euo pipefail

root=/nvme-data/ligm
export HF_ENDPOINT=https://hf-mirror.com
uv="$HOME/.local/bin/uv"

"$uv" run --no-sync ligm-download dataset \
  sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1 \
  --revision 84ed2d35626f617d890bd493b4d6db69a741e0e2 \
  --include 'triplet/train-00000-of-00001.parquet' \
  --output "$root/data/msmarco"

"$uv" run --no-sync ligm-download dataset Shitao/MLDR \
  --revision d67138e705d963e346253a80e59676ddb418810a \
  --include 'mldr-v1.0-en/corpus.jsonl.gz' \
  --include 'mldr-v1.0-en/dev.jsonl.gz' \
  --output "$root/data/mldr"
