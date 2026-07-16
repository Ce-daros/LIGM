import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from scipy.stats import spearmanr
from transformers import AutoModelForMaskedLM, AutoTokenizer

from ligm.config import DataConfig, load_config
from ligm.data import create_document_source, next_encoded_batch
from ligm.masking import IGNORE_INDEX, random_word_mask
from ligm.rotary import use_torch_rotary
from ligm.scoring import information_gain_statistics

DISTANCE_BUCKETS = ((128, 512), (512, 2048), (2048, 4096), (4096, 8192))


def _load_model(model_path: str):
    model = (
        AutoModelForMaskedLM.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        .to("cuda")
        .eval()
    )
    model.config.reference_compile = False
    return model


def _single_token_markers(tokenizer, count: int) -> list[tuple[str, int]]:
    markers = []
    for word in ("cobalt", "amber", "violet", "silver", "crimson", "indigo", "coral"):
        token_ids = tokenizer(f" {word}", add_special_tokens=False).input_ids
        if len(token_ids) == 1:
            markers.append((word, token_ids[0]))
        if len(markers) == count:
            break
    if len(markers) < count:
        raise RuntimeError("Tokenizer does not provide enough single-token marker words")
    return markers


@torch.no_grad()
def synthetic_long_range(model_path: str, output: Path, samples_per_bucket: int = 32) -> dict:
    device = torch.device("cuda")
    use_torch_rotary()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = _load_model(model_path)
    markers = _single_token_markers(tokenizer, 4)
    filler_ids = tokenizer(
        "The intervening document contains unrelated explanatory material. ",
        add_special_tokens=False,
    ).input_ids
    results = []
    for bucket_index, (low, high) in enumerate(DISTANCE_BUCKETS):
        correct = 0
        observed_distances = []
        confidences = []
        information_gains = []
        marker, marker_id = markers[bucket_index]
        prefix = tokenizer(f"The access marker is {marker}. ", add_special_tokens=False).input_ids
        if prefix.count(marker_id) != 1:
            raise RuntimeError("Marker token does not round-trip in the synthetic prefix")
        suffix = tokenizer(" The access marker is", add_special_tokens=False).input_ids + [
            tokenizer.mask_token_id
        ]
        for sample_index in range(samples_per_bucket):
            target_distance = low + (sample_index * (high - low) // samples_per_bucket)
            repeats = max(1, (target_distance - len(prefix)) // len(filler_ids))
            content = prefix + filler_ids * repeats + suffix
            content = content[:8190]
            encoded = {
                "input_ids": torch.tensor(
                    [[tokenizer.cls_token_id, *content, tokenizer.sep_token_id]]
                ),
                "attention_mask": torch.ones((1, len(content) + 2), dtype=torch.long),
            }
            input_ids = encoded["input_ids"].to(device)
            mask_index = torch.where(input_ids[0] == tokenizer.mask_token_id)[0][-1]
            labels = torch.full_like(input_ids, IGNORE_INDEX)
            labels[0, mask_index] = marker_id
            scores, global_logits = information_gain_statistics(
                model,
                input_ids,
                encoded["attention_mask"].to(device),
                labels,
            )
            prediction = int(global_logits[0].argmax())
            probability = float(global_logits[0].float().softmax(-1)[marker_id])
            correct += prediction == marker_id
            observed_distances.append(int(mask_index) - len(prefix))
            confidences.append(probability)
            information_gains.append(float(scores[0, mask_index]))
        results.append(
            {
                "bucket": f"{low}-{high}",
                "accuracy": correct / samples_per_bucket,
                "mean_confidence": sum(confidences) / len(confidences),
                "mean_information_gain": sum(information_gains) / len(information_gains),
                "mean_distance": sum(observed_distances) / len(observed_distances),
            }
        )
    confidence_correlation = spearmanr(
        [item["mean_distance"] for item in results],
        [item["mean_confidence"] for item in results],
    ).statistic
    score_correlation = spearmanr(
        [item["mean_distance"] for item in results],
        [item["mean_information_gain"] for item in results],
    ).statistic
    report = {
        "model": model_path,
        "distance_buckets": results,
        "distance_confidence_spearman": confidence_correlation,
        "distance_information_gain_spearman": score_correlation,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _repetition_buckets(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    word_ids: torch.Tensor,
    selected: torch.Tensor,
) -> list[tuple[int, str]]:
    available: dict[int, list[int]] = {}
    available_positions = torch.where(
        (attention_mask == 1) & (word_ids >= 0) & ~selected
    )[0].tolist()
    for position in available_positions:
        available.setdefault(int(input_ids[position]), []).append(position)

    buckets: list[tuple[int, str]] = []
    for position in torch.where(selected)[0].tolist():
        occurrences = available.get(int(input_ids[position]), [])
        if not occurrences:
            continue
        distance = min(abs(position - other) for other in occurrences)
        if distance <= 128:
            buckets.append((position, "local"))
        elif distance >= 512:
            buckets.append((position, "long"))
    return buckets


@torch.no_grad()
def natural_repetition_recovery_model(
    model,
    tokenizer,
    data_config: DataConfig,
    output: Path,
    documents: int = 32,
    model_name: str | None = None,
) -> dict:
    was_training = model.training
    model.eval()
    data_config = replace(data_config, split="validation")
    source = create_document_source(data_config, seed=20260716, batch_size=1)
    data_generator = torch.Generator().manual_seed(20260716)
    mask_generator = torch.Generator().manual_seed(20260717)
    totals = {"local": 0, "long": 0}
    correct = {"local": 0, "long": 0}
    document_results = []

    for document_index in range(documents):
        batch = next_encoded_batch(source, tokenizer, 8000, 1, data_generator)
        original_ids = batch.input_ids[0]
        masked = random_word_mask(
            batch.input_ids.cuda(),
            batch.word_ids,
            tokenizer.mask_token_id,
            tokenizer.vocab_size,
            mask_generator,
        )
        logits = model(
            input_ids=masked.input_ids,
            attention_mask=batch.attention_mask.cuda(),
        ).logits
        selected = masked.selected[0].cpu()
        predictions = logits[0, selected.cuda()].argmax(-1).cpu()
        predicted_by_position = dict(
            zip(torch.where(selected)[0].tolist(), predictions.tolist(), strict=True)
        )
        document_totals = {"local": 0, "long": 0}
        document_correct = {"local": 0, "long": 0}
        for position, bucket in _repetition_buckets(
            original_ids,
            batch.attention_mask[0],
            batch.word_ids[0],
            selected,
        ):
            totals[bucket] += 1
            document_totals[bucket] += 1
            is_correct = predicted_by_position[position] == int(original_ids[position])
            correct[bucket] += is_correct
            document_correct[bucket] += is_correct
        document_results.append(
            {
                "document_index": document_index,
                "buckets": {
                    bucket: {
                        "count": document_totals[bucket],
                        "correct": document_correct[bucket],
                        "accuracy": (
                            document_correct[bucket] / document_totals[bucket]
                            if document_totals[bucket]
                            else None
                        ),
                    }
                    for bucket in ("local", "long")
                },
            }
        )

    report = {
        "model": model_name,
        "documents": documents,
        "buckets": {
            bucket: {
                "count": totals[bucket],
                "correct": correct[bucket],
                "accuracy": correct[bucket] / totals[bucket] if totals[bucket] else None,
            }
            for bucket in ("local", "long")
        },
        "document_results": document_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    model.train(was_training)
    return report


@torch.no_grad()
def natural_repetition_recovery(
    model_path: str,
    config_path: str,
    output: Path,
    documents: int = 32,
) -> dict:
    use_torch_rotary()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = _load_model(model_path)
    config = load_config(config_path)
    return natural_repetition_recovery_model(
        model,
        tokenizer,
        config.data,
        output,
        documents,
        model_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-bucket", type=int, default=32)
    parser.add_argument("--natural-config")
    parser.add_argument("--documents", type=int, default=32)
    args = parser.parse_args()
    if args.natural_config:
        natural_repetition_recovery(
            args.model_path,
            args.natural_config,
            args.output,
            args.documents,
        )
    else:
        synthetic_long_range(args.model_path, args.output, args.samples_per_bucket)


if __name__ == "__main__":
    main()
