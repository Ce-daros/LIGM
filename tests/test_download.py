from pathlib import Path

from ligm.download import Artifact, pending_artifacts


def test_pending_artifacts_skips_complete_file(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("{}", encoding="utf-8")
    artifact = Artifact("https://example/config.json", "config.json", None, 2)

    assert pending_artifacts(tmp_path, [artifact]) == []
