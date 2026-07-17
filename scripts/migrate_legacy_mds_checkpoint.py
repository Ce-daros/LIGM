import argparse
from pathlib import Path

import torch

from ligm.migrate import migrate_source_state


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
