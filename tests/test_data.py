from types import SimpleNamespace

from ligm.data import _pack_modernbert, document_split


def test_pack_modernbert_adds_boundaries_and_padding():
    tokenizer = SimpleNamespace(cls_token_id=101, sep_token_id=102, pad_token_id=0)
    packed = _pack_modernbert([5, 6], [0, 1], tokenizer, 6)

    assert packed == {
        "input_ids": [101, 5, 6, 102, 0, 0],
        "attention_mask": [1, 1, 1, 1, 0, 0],
        "word_ids": [-1, 0, 1, -1, -1, -1],
    }


def test_document_split_is_stable_and_has_all_partitions():
    assignments = [document_split(f"document-{index}") for index in range(1000)]

    assert assignments == [document_split(f"document-{index}") for index in range(1000)]
    assert set(assignments) == {"train", "validation", "test"}
