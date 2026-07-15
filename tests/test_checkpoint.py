from ligm.checkpoint import prune_checkpoints


def test_prune_checkpoints_keeps_recent_and_milestones(tmp_path):
    for tokens in (250, 500, 750, 1000, 1250, 1500):
        (tmp_path / f"tokens-{tokens}.pt").touch()

    prune_checkpoints(tmp_path, keep_recent=2, keep_every=4)

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "tokens-1000.pt",
        "tokens-1250.pt",
        "tokens-1500.pt",
    ]
