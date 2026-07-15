#!/usr/bin/env bash
set -euo pipefail

repo_id="$1"
release_dir="$2"

hf() {
  "$HOME/.local/bin/uv" run --no-sync hf "$@"
}

HF_ENDPOINT=https://huggingface.co hf repo create "$repo_id" --exist-ok
HF_ENDPOINT=https://huggingface.co hf upload-large-folder \
  "$repo_id" "$release_dir" --repo-type model --num-workers 8
