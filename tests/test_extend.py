import yaml

from ligm.extend import build_extension_config


def test_extension_uses_latest_actual_checkpoint_tokens(tmp_path):
    base = tmp_path / "base.yaml"
    run = tmp_path / "run"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "tokens-100.pt").touch()
    (checkpoints / "tokens-125.pt").touch()
    base.write_text(
        yaml.safe_dump(
            {
                "output_dir": "old",
                "training": {
                    "max_tokens": 100,
                    "checkpoint_every_tokens": 25,
                },
            }
        )
    )

    config = build_extension_config(base, run, 500, tmp_path / "extended.yaml")

    assert config["resume_from"].endswith("tokens-125.pt")
    assert config["training"]["max_tokens"] == 625
    assert config["output_dir"] == str(run)
