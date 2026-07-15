#!/usr/bin/env bash
set -euo pipefail

repo_id="$1"
release_dir="$2"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

hf() {
  "$HOME/.local/bin/uv" run --no-sync hf "$@"
}

hf repo create "$repo_id" --exist-ok
hf upload-large-folder \
  "$repo_id" "$release_dir" --repo-type model --num-workers 8
