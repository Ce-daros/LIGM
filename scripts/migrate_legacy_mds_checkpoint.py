import argparse
from pathlib import Path

import torch


def migrate_source_state(state: dict) -> dict:
    source = state["source"]
    if source.get("format_version") == 2:
        raise ValueError("Checkpoint already uses MDS source format version 2")
    old_offset = int(source["dataset"]["sample_in_epoch"])
    corrected_offset = int(source["samples_consumed"])
    if corrected_offset >= old_offset:
        raise ValueError("Legacy checkpoint does not contain a duplicated resume offset")
    source["dataset"]["sample_in_epoch"] = corrected_offset
    source["format_version"] = 2
    return {
        "old_sample_in_epoch": old_offset,
        "corrected_sample_in_epoch": corrected_offset,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    state = torch.load(args.source, map_location="cpu", weights_only=False)
    migration = migrate_source_state(state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(args.output)
    print(
        f"old_sample_in_epoch={migration['old_sample_in_epoch']} "
        f"corrected_sample_in_epoch={migration['corrected_sample_in_epoch']}"
    )


if __name__ == "__main__":
    main()
