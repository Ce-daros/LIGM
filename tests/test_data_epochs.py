from ligm import data


class EpochDataset:
    def __init__(self, epochs: list[list[dict]]) -> None:
        self.epochs = epochs
        self.next_epoch = 0
        self.saved_samples = None
        self.loaded_state = None

    def __iter__(self):
        epoch = self.epochs[self.next_epoch]
        self.next_epoch += 1
        return iter(epoch)

    def state_dict(self, samples: int, from_beginning: bool) -> dict:
        self.saved_samples = (samples, from_beginning)
        return {"epoch": self.next_epoch - 1, "sample_in_epoch": samples}

    def load_state_dict(self, state: dict) -> None:
        self.loaded_state = state


def source_with(dataset: EpochDataset) -> data.MDSDocumentSource:
    source = data.MDSDocumentSource.__new__(data.MDSDocumentSource)
    source.dataset = dataset
    source.iterator = iter(dataset)
    source.samples_seen = 0
    source.samples_consumed = 0
    source.split = "train"
    return source


def test_mds_source_continues_at_next_epoch(monkeypatch) -> None:
    monkeypatch.setattr(data, "document_split", lambda _: "train")
    source = source_with(
        EpochDataset(
            [
                [{"id": "a"}, {"id": "b"}],
                [{"id": "c"}, {"id": "d"}],
            ]
        )
    )

    assert [next(source)["id"] for _ in range(4)] == ["a", "b", "c", "d"]
    assert source.samples_consumed == 2
    assert source.samples_seen == 4


def test_mds_resume_counts_only_samples_after_loaded_offset(monkeypatch) -> None:
    monkeypatch.setattr(data, "document_split", lambda _: "train")
    dataset = EpochDataset([[{"id": "a"}], [{"id": "b"}]])
    source = source_with(dataset)
    state = {
        "samples_seen": 10,
        "samples_consumed": 20,
        "dataset": {"epoch": 2, "sample_in_epoch": 20},
    }

    source.load_state_dict(state)
    assert source.samples_consumed == 0
    assert dataset.loaded_state == state["dataset"]
    next(source)
    source.state_dict()
    assert dataset.saved_samples == (1, False)
