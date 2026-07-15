from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ligm.checkpoint import prune_checkpoints, save_checkpoint


def test_prune_checkpoints_keeps_recent_and_milestones(tmp_path):
    for tokens in (250, 500, 750, 1000, 1250, 1500):
        (tmp_path / f"tokens-{tokens}.pt").touch()

    prune_checkpoints(tmp_path, keep_recent=2, keep_every=4)

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "tokens-1000.pt",
        "tokens-1250.pt",
        "tokens-1500.pt",
    ]


def test_save_checkpoint_uses_no_partial_target(tmp_path, monkeypatch):
    target = tmp_path / "tokens-1.pt"

    def fail_save(state, path):
        Path(path).write_bytes(b"partial")
        raise RuntimeError("interrupted")

    monkeypatch.setattr(torch, "save", fail_save)
    stateful = SimpleNamespace(state_dict=lambda: {})
    with pytest.raises(RuntimeError, match="interrupted"):
        save_checkpoint(
            target,
            model=stateful,
            teacher=None,
            optimizer=stateful,
            scheduler=stateful,
            source=stateful,
            step=1,
            micro_step=1,
            tokens_seen=1,
            generators={},
        )
    assert not target.exists()
