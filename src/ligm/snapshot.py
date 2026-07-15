import argparse
import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import requests
import yaml
from huggingface_hub import HfApi

from ligm.download import Artifact, run_aria, verify_artifacts, write_aria_manifest


@dataclass(frozen=True)
class StreamSpec:
    name: str
    prefixes: tuple[str, ...]
    proportion: float


def _read_index(endpoint: str, repo_id: str, revision: str, path: str) -> dict:
    url = (
        f"{endpoint.rstrip('/')}/datasets/{repo_id}/resolve/{revision}/"
        f"{quote(path, safe='/')}"
    )
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def _leaf_indexes(paths: set[str], prefix: str) -> list[str]:
    mds_directories = {path.rsplit("/", 1)[0] for path in paths if path.endswith(".mds")}
    return sorted(
        path
        for path in paths
        if path.endswith("/index.json")
        and path.startswith(prefix)
        and path.rsplit("/", 1)[0] in mds_directories
    )


def _interleaved_indexes(paths: set[str], prefixes: tuple[str, ...]) -> list[str]:
    groups = [_leaf_indexes(paths, prefix) for prefix in prefixes]
    return [
        group[index]
        for index in range(max(map(len, groups)))
        for group in groups
        if index < len(group)
    ]


def build_snapshot(config_path: Path) -> None:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    repo_id = raw["repo_id"]
    revision = raw["revision"]
    output = Path(raw["output"])
    target_tokens = int(raw["target_tokens"])
    specs = [
        StreamSpec(item["name"], tuple(item["prefixes"]), float(item["proportion"]))
        for item in raw["streams"]
    ]

    info = HfApi(endpoint=endpoint).repo_info(
        repo_id, repo_type="dataset", revision=revision, files_metadata=True
    )
    if info.sha != revision:
        raise RuntimeError(f"Resolved revision {info.sha} does not match {revision}")
    siblings = {item.rfilename: item for item in info.siblings}
    paths = set(siblings)
    manifest_artifacts: list[Artifact] = []
    snapshot_metadata: dict = {"repo_id": repo_id, "revision": revision, "streams": {}}

    for spec in specs:
        token_quota = round(target_tokens * spec.proportion)
        sample_quota = (token_quota + 8191) // 8192
        selected_shards: list[dict] = []
        selected_samples = 0
        stream_artifacts: list[Artifact] = []
        for index_path in _interleaved_indexes(paths, spec.prefixes):
            index = _read_index(endpoint, repo_id, revision, index_path)
            source_dir = index_path.removesuffix("index.json")
            directory_tag = source_dir.rstrip("/").replace("/", "__")
            for shard in index["shards"]:
                if selected_samples >= sample_quota:
                    break
                raw_data = shard["raw_data"]
                source_path = f"{source_dir}{raw_data['basename']}"
                sibling = siblings[source_path]
                output_name = f"{directory_tag}__{raw_data['basename']}"
                copied = copy.deepcopy(shard)
                copied["compression"] = None
                copied["raw_data"]["basename"] = output_name
                copied["zip_data"] = None
                selected_shards.append(copied)
                selected_samples += int(shard["samples"])
                lfs = sibling.lfs or {}
                stream_artifacts.append(
                    Artifact(
                        url=(
                            f"{endpoint.rstrip('/')}/datasets/{repo_id}/resolve/{revision}/"
                            f"{quote(source_path, safe='/')}"
                        ),
                        relative_path=f"{spec.name}/{output_name}",
                        sha256=lfs.get("sha256"),
                        size=sibling.size,
                    )
                )
            if selected_samples >= sample_quota:
                break
        if selected_samples < sample_quota:
            raise RuntimeError(f"Insufficient samples for stream {spec.name}")

        stream_dir = output / spec.name
        stream_dir.mkdir(parents=True, exist_ok=True)
        (stream_dir / "index.json").write_text(
            json.dumps({"version": 2, "shards": selected_shards}, indent=2) + "\n",
            encoding="utf-8",
        )
        snapshot_metadata["streams"][spec.name] = {
            "proportion": spec.proportion,
            "samples": selected_samples,
            "tokens": selected_samples * 8192,
            "shards": len(selected_shards),
        }
        manifest_artifacts.extend(stream_artifacts)

    manifest = output / "snapshot.aria2.txt"
    write_aria_manifest(manifest, output, manifest_artifacts)
    (output / "snapshot.json").write_text(
        json.dumps(snapshot_metadata, indent=2) + "\n", encoding="utf-8"
    )
    run_aria(manifest, dataset=True)
    verify_artifacts(output, manifest_artifacts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    build_snapshot(args.config)


if __name__ == "__main__":
    main()
