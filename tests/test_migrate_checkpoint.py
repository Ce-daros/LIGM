import pytest

from ligm.migrate import migrate_source_state


def test_migrate_legacy_source_removes_duplicated_resume_offset() -> None:
    state = {
        "source": {
            "samples_consumed": 149_174,
            "dataset": {"epoch": 0, "sample_in_epoch": 193_997},
        }
    }

    result = migrate_source_state(state)

    assert result == {
        "old_sample_in_epoch": 193_997,
        "corrected_sample_in_epoch": 149_174,
    }
    assert state["source"]["dataset"]["sample_in_epoch"] == 149_174
    assert state["source"]["format_version"] == 2


def test_migrate_source_refuses_version_two_checkpoint() -> None:
    state = {
        "source": {
            "format_version": 2,
            "samples_consumed": 1,
            "dataset": {"sample_in_epoch": 1},
        }
    }

    with pytest.raises(ValueError, match="already uses"):
        migrate_source_state(state)
