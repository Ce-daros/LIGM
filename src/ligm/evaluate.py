import argparse
import json
from pathlib import Path

import torch
from scipy.stats import spearmanr
from transformers import AutoModelForMaskedLM, AutoTokenizer

from ligm.masking import IGNORE_INDEX
from ligm.scoring import information_gain_scores

DISTANCE_BUCKETS = ((128, 512), (512, 2048), (2048, 4096), (4096, 8192))


def _single_token_words(tokenizer, count: int) -> list[str]:
    words = []
    for word in ("cobalt", "amber", "violet", "silver", "crimson", "indigo", "coral"):
        if len(tokenizer(word, add_special_tokens=False).input_ids) == 1:
            words.append(word)
        if len(words) == count:
            break
    if len(words) < count:
        raise RuntimeError("Tokenizer does not provide enough single-token marker words")
    return words


@torch.no_grad()
def synthetic_long_range(model_path: str, output: Path, samples_per_bucket: int = 32) -> dict:
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = (
        AutoModelForMaskedLM.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        .to(device)
        .eval()
    )
    model.config.reference_compile = False
    markers = _single_token_words(tokenizer, 4)
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
        marker = markers[bucket_index]
        marker_id = tokenizer(marker, add_special_tokens=False).input_ids[0]
        prefix = tokenizer(f"The access marker is {marker}. ", add_special_tokens=False).input_ids
        suffix = tokenizer(" The access marker is ", add_special_tokens=False).input_ids + [
            tokenizer.mask_token_id
        ]
        for sample_index in range(samples_per_bucket):
            target_distance = low + (sample_index * (high - low) // samples_per_bucket)
            repeats = max(1, (target_distance - len(prefix)) // len(filler_ids))
            content = prefix + filler_ids * repeats + suffix
            content = content[:8190]
            encoded = tokenizer.prepare_for_model(
                content,
                add_special_tokens=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            mask_index = torch.where(input_ids[0] == tokenizer.mask_token_id)[0][-1]
            logits = model(**{key: value.to(device) for key, value in encoded.items()}).logits
            prediction = int(logits[0, mask_index].argmax())
            probability = float(logits[0, mask_index].float().softmax(-1)[marker_id])
            labels = torch.full_like(input_ids, IGNORE_INDEX)
            labels[0, mask_index] = marker_id
            scores = information_gain_scores(
                model,
                input_ids,
                encoded["attention_mask"].to(device),
                labels,
            )
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-bucket", type=int, default=32)
    args = parser.parse_args()
    synthetic_long_range(args.model_path, args.output, args.samples_per_bucket)


if __name__ == "__main__":
    main()
