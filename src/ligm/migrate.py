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
