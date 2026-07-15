from ligm.snapshot import _interleaved_indexes, _leaf_indexes


def test_leaf_indexes_exclude_aggregate_indexes() -> None:
    paths = {
        "train/books/index.json",
        "train/books/books_0001/index.json",
        "train/books/books_0001/shard.00000.mds",
    }

    assert _leaf_indexes(paths, "train/books/") == ["train/books/books_0001/index.json"]


def test_indexes_are_interleaved_across_prefixes() -> None:
    paths = {
        "train/a/a1/index.json",
        "train/a/a1/shard.mds",
        "train/a/a2/index.json",
        "train/a/a2/shard.mds",
        "train/b/b1/index.json",
        "train/b/b1/shard.mds",
    }

    assert _interleaved_indexes(paths, ("train/a/", "train/b/")) == [
        "train/a/a1/index.json",
        "train/b/b1/index.json",
        "train/a/a2/index.json",
    ]
