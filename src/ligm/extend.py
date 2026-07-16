import argparse
from pathlib import Path

import yaml


def build_extension_config(
    base_config: Path,
    run_dir: Path,
    add_tokens: int,
    output: Path,
) -> dict:
    checkpoints = sorted(
        (run_dir / "checkpoints").glob("tokens-*.pt"),
        key=lambda path: int(path.stem.removeprefix("tokens-")),
    )
    latest = checkpoints[-1]
    current_tokens = int(latest.stem.removeprefix("tokens-"))
    config = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    config["output_dir"] = str(run_dir)
    config["resume_from"] = str(latest)
    config["training"]["max_tokens"] = current_tokens + add_tokens
    config["training"]["checkpoint_every_tokens"] = 25_000_000
    config["training"]["keep_recent_checkpoints"] = 2
    config["training"]["keep_every_checkpoints"] = 4
    config["training"]["keep_milestone_tokens"] = [
        100_000_000,
        250_000_000,
        500_000_000,
        750_000_000,
        1_000_000_000,
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_config", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--add-tokens", type=int, default=500_000_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = build_extension_config(
        args.base_config,
        args.run_dir,
        args.add_tokens,
        args.output,
    )
    print(
        f"resume_from={config['resume_from']} "
        f"max_tokens={config['training']['max_tokens']}"
    )


if __name__ == "__main__":
    main()
