import argparse
import fnmatch
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoSibling


@dataclass(frozen=True)
class Artifact:
    url: str
    relative_path: str
    sha256: str | None
    size: int | None


def resolve_artifacts(
    repo_id: str,
    repo_type: str,
    endpoint: str,
    revision: str,
    includes: tuple[str, ...],
) -> tuple[str, list[Artifact]]:
    api = HfApi(endpoint=endpoint)
    info = api.repo_info(repo_id, repo_type=repo_type, revision=revision, files_metadata=True)
    commit = info.sha
    prefix = "datasets/" if repo_type == "dataset" else ""
    artifacts: list[Artifact] = []
    for sibling in info.siblings:
        if not isinstance(sibling, RepoSibling):
            continue
        if includes and not any(
            fnmatch.fnmatch(sibling.rfilename, pattern) for pattern in includes
        ):
            continue
        lfs = sibling.lfs or {}
        artifacts.append(
            Artifact(
                url=(
                    f"{endpoint.rstrip('/')}/{prefix}{repo_id}/resolve/{commit}/"
                    f"{quote(sibling.rfilename, safe='/')}"
                ),
                relative_path=sibling.rfilename,
                sha256=lfs.get("sha256"),
                size=sibling.size,
            )
        )
    if not artifacts:
        raise ValueError("No repository files matched the include patterns")
    return commit, artifacts


def write_aria_manifest(path: Path, output: Path, artifacts: list[Artifact]) -> None:
    lines: list[str] = []
    for artifact in artifacts:
        relative = Path(artifact.relative_path)
        lines.extend(
            [
                artifact.url,
                f"  dir={output / relative.parent}",
                f"  out={relative.name}",
            ]
        )
        if artifact.sha256:
            lines.append(f"  checksum=sha-256={artifact.sha256}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_aria(manifest: Path, dataset: bool) -> None:
    concurrency = "16" if dataset else "4"
    splits = "4" if dataset else "16"
    subprocess.run(
        [
            "aria2c",
            f"--input-file={manifest}",
            f"--max-concurrent-downloads={concurrency}",
            f"--split={splits}",
            f"--max-connection-per-server={splits}",
            "--min-split-size=1M",
            "--continue=true",
            "--auto-file-renaming=false",
            "--file-allocation=none",
            "--check-integrity=true",
        ],
        check=True,
    )


def verify_artifacts(output: Path, artifacts: list[Artifact]) -> None:
    for artifact in artifacts:
        path = output / artifact.relative_path
        if artifact.size is not None and path.stat().st_size != artifact.size:
            raise RuntimeError(f"Size mismatch: {path}")
        if artifact.sha256:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != artifact.sha256:
                raise RuntimeError(f"SHA256 mismatch: {path}")


def pending_artifacts(output: Path, artifacts: list[Artifact]) -> list[Artifact]:
    pending: list[Artifact] = []
    for artifact in artifacts:
        path = output / artifact.relative_path
        if not path.exists() or artifact.size is None or path.stat().st_size != artifact.size:
            pending.append(artifact)
            continue
        if artifact.sha256:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != artifact.sha256:
                pending.append(artifact)
    return pending


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_type", choices=["model", "dataset"])
    parser.add_argument("repo_id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--resolve-only", action="store_true")
    args = parser.parse_args()

    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    includes = tuple(args.include)
    if args.repo_type == "model" and not includes:
        includes = (
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        )
    commit, artifacts = resolve_artifacts(
        args.repo_id,
        args.repo_type,
        endpoint,
        args.revision,
        includes,
    )
    manifest = args.manifest or args.output.with_suffix(".aria2.txt")
    write_aria_manifest(manifest, args.output, artifacts)
    metadata = {
        "repo_id": args.repo_id,
        "repo_type": args.repo_type,
        "endpoint": endpoint,
        "commit": commit,
        "files": [artifact.__dict__ for artifact in artifacts],
    }
    manifest.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not args.resolve_only:
        pending = pending_artifacts(args.output, artifacts)
        if pending:
            write_aria_manifest(manifest, args.output, pending)
            run_aria(manifest, args.repo_type == "dataset")
        verify_artifacts(args.output, artifacts)


if __name__ == "__main__":
    main()
