import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.modules import Normalize, Pooling, Transformer
from sentence_transformers.sentence_transformer.training_args import BatchSamplers

from ligm.rotary import use_torch_rotary


def build_retriever(model_path: str, max_seq_length: int) -> SentenceTransformer:
    use_torch_rotary()
    transformer = Transformer(
        model_path,
        max_seq_length=max_seq_length,
        model_kwargs={"attn_implementation": "flash_attention_2", "dtype": torch.bfloat16},
        config_kwargs={"reference_compile": False},
    )
    pooling = Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
    return SentenceTransformer(modules=[transformer, pooling, Normalize()], device="cuda")


def train_probe(config: dict, model_path: str, output: Path) -> None:
    model = build_retriever(model_path, config["train_max_seq_length"])
    dataset = Dataset.from_parquet(config["triplet_parquet"])
    dataset = dataset.shuffle(seed=config["seed"]).select(range(config["train_examples"]))
    loss = MultipleNegativesRankingLoss(model)
    arguments = SentenceTransformerTrainingArguments(
        output_dir=str(output / "trainer"),
        num_train_epochs=1,
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=1,
        learning_rate=config["learning_rate"],
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no",
        logging_steps=100,
        report_to="none",
        seed=config["seed"],
        data_seed=config["seed"],
        dataloader_num_workers=4,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        loss=loss,
    )
    trainer.train()
    model.save_pretrained(str(output / "final"))


def _jsonl_gzip(path: str | Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def encode_corpus(
    model: SentenceTransformer,
    corpus_path: str,
    output: Path,
    corpus_size: int,
    batch_size: int,
) -> tuple[np.memmap, list[str]]:
    dimension = model.get_sentence_embedding_dimension()
    embeddings = np.memmap(
        output / "corpus-embeddings.f32",
        dtype=np.float32,
        mode="w+",
        shape=(corpus_size, dimension),
    )
    document_ids: list[str] = []
    texts: list[str] = []
    offset = 0
    for record in _jsonl_gzip(corpus_path):
        document_ids.append(record["docid"])
        texts.append(record["text"])
        if len(texts) == batch_size:
            encoded = model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            embeddings[offset : offset + len(texts)] = encoded
            offset += len(texts)
            texts.clear()
    if texts:
        encoded = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        embeddings[offset : offset + len(texts)] = encoded
        offset += len(texts)
    if offset != corpus_size:
        raise RuntimeError(f"Expected {corpus_size} corpus records, encoded {offset}")
    embeddings.flush()
    (output / "corpus-ids.json").write_text(json.dumps(document_ids) + "\n", encoding="utf-8")
    return embeddings, document_ids


def _top_k(query_embeddings: np.ndarray, corpus: np.ndarray, k: int) -> np.ndarray:
    best_scores = np.full((len(query_embeddings), k), -np.inf, dtype=np.float32)
    best_indices = np.full((len(query_embeddings), k), -1, dtype=np.int64)
    for start in range(0, len(corpus), 10_000):
        scores = query_embeddings @ np.asarray(corpus[start : start + 10_000]).T
        local = np.argpartition(scores, -k, axis=1)[:, -k:]
        local_scores = np.take_along_axis(scores, local, axis=1)
        candidates = np.concatenate((best_indices, local + start), axis=1)
        candidate_scores = np.concatenate((best_scores, local_scores), axis=1)
        selected = np.argpartition(candidate_scores, -k, axis=1)[:, -k:]
        best_indices = np.take_along_axis(candidates, selected, axis=1)
        best_scores = np.take_along_axis(candidate_scores, selected, axis=1)
    order = np.argsort(-best_scores, axis=1)
    return np.take_along_axis(best_indices, order, axis=1)


def evaluate_mldr(config: dict, model_path: str, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    model = build_retriever(model_path, config["eval_max_seq_length"])
    queries = list(_jsonl_gzip(config["mldr_dev"]))
    query_embeddings = model.encode(
        [record["query"] for record in queries],
        batch_size=16,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    corpus, document_ids = encode_corpus(
        model,
        config["mldr_corpus"],
        output,
        config["mldr_corpus_size"],
        config["eval_batch_size"],
    )
    top_indices = _top_k(query_embeddings, corpus, 10)
    scores = []
    per_query = []
    for query, indices in zip(queries, top_indices, strict=True):
        relevant = {passage["docid"] for passage in query["positive_passages"]}
        retrieved = [document_ids[index] for index in indices]
        gains = [
            1 / math.log2(rank + 2)
            for rank, docid in enumerate(retrieved)
            if docid in relevant
        ]
        ideal = sum(1 / math.log2(rank + 2) for rank in range(min(10, len(relevant))))
        score = sum(gains) / ideal
        scores.append(score)
        per_query.append({"query_id": query["query_id"], "ndcg_at_10": score})
    report = {
        "model": model_path,
        "queries": len(queries),
        "corpus": len(document_ids),
        "ndcg_at_10": sum(scores) / len(scores),
        "per_query": per_query,
    }
    (output / "mldr-dev.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "evaluate"])
    parser.add_argument("config", type=Path)
    parser.add_argument("model_path")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.command == "train":
        train_probe(config, args.model_path, args.output)
    else:
        evaluate_mldr(config, args.model_path, args.output)


if __name__ == "__main__":
    main()
