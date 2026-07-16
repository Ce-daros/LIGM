import argparse
import json
import shutil
from pathlib import Path

MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _model_card(run: Path, results: Path, repo_id: str) -> str:
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    config = _read(run / "resolved-config.json")
    synthetic = _read(results / "ligm-synthetic.json")
    natural = _read(results / "ligm-natural.json")
    random_natural_path = results / "random-natural.json"
    random_natural = _read(random_natural_path) if random_natural_path.exists() else None
    gate = _read(results / "mechanism-gate.json")
    mldr_path = results / "ligm-mldr.json"
    mldr = _read(mldr_path) if mldr_path.exists() else None
    gate_label = "passed" if gate["passed"] else "did not pass"
    mean_throughput = sum(item["tokens_per_second"] for item in metrics) / len(metrics)
    distance_rows = "\n".join(
        f"| {item['bucket']} | {item['accuracy']:.4f} | "
        f"{item['mean_information_gain']:.6f} |"
        for item in synthetic["distance_buckets"]
    )
    mldr_row = (
        f"\n| MLDR-English dev nDCG@10 | {mldr['ndcg_at_10']:.4f} |" if mldr else ""
    )
    comparison = ""
    if random_natural:
        random_local = random_natural["buckets"]["local"]["accuracy"]
        random_long = random_natural["buckets"]["long"]["accuracy"]
        ligm_local = natural["buckets"]["local"]["accuracy"]
        ligm_long = natural["buckets"]["long"]["accuracy"]
        comparison = f"""

Compared with the token-matched random-MLM baseline, long-distance recovery
changed from {random_long:.4%} to {ligm_long:.4%} ({(ligm_long - random_long) * 100:+.3f}
percentage points) and local recovery changed from {random_local:.4%} to
{ligm_local:.4%} ({(ligm_local - random_local) * 100:+.3f} percentage points).
"""
    return f"""---
library_name: transformers
license: apache-2.0
language:
- en
base_model: answerdotai/ModernBERT-base
base_model_relation: finetune
pipeline_tag: fill-mask
datasets:
- jhu-clsp/ettin-extension-data
tags:
- modernbert
- masked-lm
- long-context
- continued-pretraining
---

# ModernBERT-base-LIGM

`{repo_id}` is a continued-pretraining checkpoint of
[`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base).
It was trained with Long-range Information-Gain Masking (LIGM), which selects MLM
targets using the prediction difference between normal ModernBERT attention and
an all-local attention counterfactual. The architecture and 8,192-token context
limit are unchanged.

The pre-registered first-stage mechanism gate **{gate_label}**. This statement is
reported regardless of outcome; additional training is not treated as evidence
when the gate fails.

## Usage

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id, attn_implementation="sdpa")

text = "Long documents can use evidence from an earlier [MASK]."
inputs = tokenizer(text, return_tensors="pt")
mask = inputs.input_ids[0] == tokenizer.mask_token_id
prediction = model(**inputs).logits[0, mask].argmax(-1)
print(tokenizer.decode(prediction))
```

This is a masked-language model, not a chat or generative model. Fine-tune the
encoder for retrieval or classification before using it for those tasks.

## Training

- Base revision: `8949b909ec900327062f0ebf497f51aef5e6f0c8`
- Ettin extension-data revision: `996ec10f55ee16739389f4afc0993bbc28716fe5`
- Tokens processed by this run: `{metrics[-1]['tokens_seen']:,}`
- Precision: BF16
- Optimizer: StableAdamW
- Peak allocated GPU memory: `{max(item['peak_memory_gib'] for item in metrics):.2f}` GiB
- Mean measured throughput: `{mean_throughput:.0f}` token/s
- Seed: `{config['training']['seed']}`
- Hardware: one NVIDIA RTX 3090 24GB

The fixed mixture contains books, arXiv/PeS2o, DCLM, Wikipedia, StackExchange,
and code. Documents are assigned to train/validation/test by stable document-ID
hash at 98/1/1. LIGM uses 20% information-gain-selected whole-word spans plus 10%
random replay spans. An EMA teacher supplies scores and receives no gradients.

## Evaluation

| Distance bucket (tokens) | Recovery accuracy | Mean information gain |
|---|---:|---:|
{distance_rows}

| Held-out natural repetition bucket | Accuracy |
|---|---:|
| Local (≤128 tokens) | {natural['buckets']['local']['accuracy']:.4f} |
| Long (≥512 tokens) | {natural['buckets']['long']['accuracy']:.4f} |{mldr_row}
{comparison}

The complete JSON reports, per-query results when available, resolved training
configuration, and download manifests are included in `research/`. The source
code is available at [Ce-daros/LIGM](https://github.com/Ce-daros/LIGM).

## Limitations

The base model and continued-pretraining mixture are primarily English and code.
The checkpoint has not been instruction-tuned and should not be interpreted as a
general-purpose assistant. Synthetic recovery measures mechanism behavior rather
than broad language understanding. A failed gate is a negative experimental
result, not evidence of improved long-document retrieval.

The reference FlashAttention environment required a PyTorch implementation of
ModernBERT's unpadded rotary operation because Torch 2.6/Triton 3.2 could not
compile the bundled Triton rotary kernel. The released weights remain compatible
with the standard Transformers architecture; the usage example selects SDPA.

## License and citations

Released under Apache-2.0, matching ModernBERT. Ettin data retains its MIT
license. See the source repository for the ModernBERT and Ettin BibTeX entries.
"""


def build_release(
    model: Path,
    run: Path,
    results: Path,
    manifests: Path,
    license_path: Path,
    output: Path,
    repo_id: str,
) -> None:
    output.mkdir(parents=True)
    for filename in MODEL_FILES:
        shutil.copy2(model / filename, output / filename)
    shutil.copy2(license_path, output / "LICENSE")
    research = output / "research"
    research.mkdir()
    shutil.copy2(run / "metrics.jsonl", research / "metrics.jsonl")
    shutil.copy2(run / "resolved-config.json", research / "resolved-config.json")
    for path in sorted(results.iterdir()):
        if path.is_file():
            shutil.copy2(path, research / path.name)
    for path in sorted(manifests.iterdir()):
        if path.is_file():
            shutil.copy2(path, research / path.name)
    (output / "README.md").write_text(
        _model_card(run, results, repo_id),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args()
    build_release(
        args.model,
        args.run,
        args.results,
        args.manifests,
        args.license,
        args.output,
        args.repo_id,
    )


if __name__ == "__main__":
    main()
