import argparse

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from ligm.attention import all_local_attention
from ligm.rotary import use_torch_rotary


@torch.no_grad()
def verify_attention_counterfactual(model_path: str) -> float:
    use_torch_rotary()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).cuda()
    model.config.reference_compile = False
    encoded = tokenizer(
        "The short sequence must fit entirely inside the local attention window.",
        return_tensors="pt",
        padding="max_length",
        max_length=64,
    ).to("cuda")
    global_logits = model(**encoded).logits
    with all_local_attention(model):
        local_logits = model(**encoded).logits
    difference = float((global_logits - local_logits).abs().max())
    if difference > 0.02:
        raise RuntimeError(f"Short-context G/L mismatch: {difference}")
    return difference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    args = parser.parse_args()
    difference = verify_attention_counterfactual(args.model_path)
    print(f"short_context_max_difference={difference:.8f}")


if __name__ == "__main__":
    main()
