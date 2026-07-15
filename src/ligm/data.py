from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from ligm.config import DataConfig, SequenceBucket


@dataclass(frozen=True)
class EncodedBatch:
    input_ids: Tensor
    attention_mask: Tensor
    word_ids: Tensor


class SequenceSchedule:
    def __init__(self, buckets: tuple[SequenceBucket, ...]) -> None:
        self._cycle = [bucket for bucket in buckets for _ in range(bucket.slots)]

    def at(self, micro_step: int) -> SequenceBucket:
        return self._cycle[micro_step % len(self._cycle)]


class SyntheticDocumentSource:
    def __init__(self) -> None:
        self.index = 0

    def __iter__(self) -> "SyntheticDocumentSource":
        return self

    def __next__(self) -> str:
        self.index += 1
        prefix = f"Document {self.index}. The marker is cobalt. "
        body = "Long context requires evidence from an earlier paragraph. " * 900
        return prefix + body + "The marker remains cobalt."

    def state_dict(self) -> dict:
        return {"index": self.index}

    def load_state_dict(self, state: dict) -> None:
        self.index = int(state["index"])


class MDSDocumentSource:
    def __init__(self, config: DataConfig, seed: int, batch_size: int) -> None:
        from streaming import Stream, StreamingDataset

        streams = [
            Stream(
                remote=stream.remote,
                local=stream.local,
                proportion=stream.proportion,
            )
            for stream in config.streams
        ]
        self.dataset = StreamingDataset(
            streams=streams,
            shuffle=True,
            shuffle_seed=seed,
            batch_size=batch_size,
            cache_limit=config.cache_limit,
            num_canonical_nodes=1,
        )
        self.iterator: Iterator = iter(self.dataset)
        self.samples_seen = 0

    def __iter__(self) -> "MDSDocumentSource":
        return self

    def __next__(self) -> Mapping:
        sample = next(self.iterator)
        self.samples_seen += 1
        return sample

    def state_dict(self) -> dict:
        return {
            "samples_seen": self.samples_seen,
            "dataset": self.dataset.state_dict(self.samples_seen, False),
        }

    def load_state_dict(self, state: dict) -> None:
        self.samples_seen = int(state["samples_seen"])
        self.dataset.load_state_dict(state["dataset"])
        self.iterator = iter(self.dataset)


def create_document_source(config: DataConfig, seed: int, batch_size: int):
    if config.kind == "synthetic":
        return SyntheticDocumentSource()
    if config.kind == "mds":
        return MDSDocumentSource(config, seed, batch_size)
    raise ValueError(f"Unsupported data kind: {config.kind}")


def _word_ids(text: str, offsets: list[tuple[int, int]]) -> list[int]:
    result: list[int] = []
    current = -1
    previous_end = 0
    for start, end in offsets:
        if start == end:
            result.append(-1)
            continue
        if current < 0 or start > previous_end or (start > 0 and text[start - 1].isspace()):
            current += 1
        result.append(current)
        previous_end = end
    return result


def _word_ids_from_token_ids(token_ids: list[int], tokenizer) -> list[int]:
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    word_ids: list[int] = []
    current = -1
    for token in tokens:
        piece = token.lstrip("Ġ▁")
        is_punctuation = piece and not any(character.isalnum() for character in piece)
        if token.startswith(("Ġ", "▁")) or is_punctuation or current < 0:
            current += 1
        word_ids.append(current)
    return word_ids


def _encode_token_ids(
    token_ids: list[int], tokenizer, length: int, generator: torch.Generator
) -> dict:
    while token_ids and token_ids[0] in tokenizer.all_special_ids:
        token_ids = token_ids[1:]
    while token_ids and token_ids[-1] in tokenizer.all_special_ids:
        token_ids = token_ids[:-1]
    content_length = length - tokenizer.num_special_tokens_to_add(pair=False)
    if len(token_ids) > content_length:
        start = int(torch.randint(len(token_ids) - content_length + 1, (), generator=generator))
        token_ids = token_ids[start : start + content_length]
    grouped = _word_ids_from_token_ids(token_ids, tokenizer)
    prepared = tokenizer.prepare_for_model(
        token_ids,
        add_special_tokens=True,
        max_length=length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_special_tokens_mask=True,
    )
    grouped_iter = iter(grouped)
    word_ids = [
        -1 if is_special else next(grouped_iter) for is_special in prepared["special_tokens_mask"]
    ]
    return {
        "input_ids": prepared["input_ids"],
        "attention_mask": prepared["attention_mask"],
        "word_ids": word_ids,
    }


def _encode_document(sample, tokenizer, length: int, generator: torch.Generator) -> dict:
    if isinstance(sample, Mapping):
        return _encode_token_ids(sample["input_ids"].tolist(), tokenizer, length, generator)
    raw = tokenizer(sample, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = raw["input_ids"]
    offsets = raw["offset_mapping"]
    if len(token_ids) > length:
        start = int(torch.randint(len(token_ids) - length + 1, (), generator=generator))
        token_ids = token_ids[start : start + length]
        offsets = offsets[start : start + length]
    grouped = _word_ids(sample, offsets)
    prepared = tokenizer.prepare_for_model(
        token_ids,
        add_special_tokens=True,
        max_length=length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_special_tokens_mask=True,
    )
    grouped_iter = iter(grouped)
    word_ids = [
        -1 if is_special else next(grouped_iter) for is_special in prepared["special_tokens_mask"]
    ]
    return {
        "input_ids": prepared["input_ids"],
        "attention_mask": prepared["attention_mask"],
        "word_ids": word_ids,
    }


def next_encoded_batch(
    source,
    tokenizer,
    length: int,
    batch_size: int,
    generator: torch.Generator,
) -> EncodedBatch:
    encoded = [
        _encode_document(next(source), tokenizer, length, generator) for _ in range(batch_size)
    ]
    return EncodedBatch(
        input_ids=torch.tensor([item["input_ids"] for item in encoded], dtype=torch.long),
        attention_mask=torch.tensor([item["attention_mask"] for item in encoded], dtype=torch.long),
        word_ids=torch.tensor([item["word_ids"] for item in encoded], dtype=torch.long),
    )
