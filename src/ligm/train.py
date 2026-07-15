import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from optimi import StableAdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForMaskedLM, AutoTokenizer

from ligm.checkpoint import load_checkpoint, save_checkpoint
from ligm.config import RunConfig, load_config
from ligm.data import SequenceSchedule, create_document_source, next_encoded_batch
from ligm.ema import EMATeacher
from ligm.masking import (
    candidate_word_mask,
    random_word_mask,
    select_ligm_targets,
)
from ligm.rotary import use_torch_rotary
from ligm.scoring import information_gain_scores


def set_seed(seed: int) -> dict[str, torch.Generator]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return {
        "data": torch.Generator().manual_seed(seed),
        "mask": torch.Generator().manual_seed(seed + 1),
    }


def create_scheduler(optimizer, total_steps: int, config) -> LambdaLR:
    warmup_steps = max(1, round(total_steps * config.warmup_ratio))
    stable_end = round(total_steps * (config.warmup_ratio + config.stable_ratio))

    def scale(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        if step < stable_end:
            return 1.0
        decay_steps = max(1, total_steps - stable_end)
        return max(0.0, 1.0 - (step - stable_end) / decay_steps)

    return LambdaLR(optimizer, scale)


def build_masked_batch(config: RunConfig, teacher, batch, tokenizer, generator):
    if config.training.method == "random":
        return random_word_mask(
            batch.input_ids,
            batch.word_ids,
            tokenizer.mask_token_id,
            tokenizer.vocab_size,
            generator,
        )
    candidates = candidate_word_mask(
        batch.input_ids,
        batch.word_ids,
        tokenizer.mask_token_id,
        generator,
    )
    scores = information_gain_scores(
        teacher.model,
        candidates.input_ids,
        batch.attention_mask,
        candidates.labels,
    )
    return select_ligm_targets(
        batch.input_ids,
        batch.word_ids,
        candidates.candidates,
        scores,
        tokenizer.mask_token_id,
        tokenizer.vocab_size,
        generator,
    )


def _append_metric(path: Path, metric: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metric, ensure_ascii=False) + "\n")


def train(config: RunConfig) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("LIGM training requires a CUDA GPU")
    device = torch.device("cuda")
    use_torch_rotary()
    generators = set_seed(config.training.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved-config.json").write_text(
        json.dumps(config, default=lambda value: value.__dict__, indent=2) + "\n",
        encoding="utf-8",
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(
        config.model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=config.attention_implementation,
    ).to(device)
    model.config.reference_compile = False
    model.config.sparse_prediction = True
    model.gradient_checkpointing_enable()
    model.train()

    teacher = (
        EMATeacher(model, config.training.ema_decay) if config.training.method == "ligm" else None
    )
    optimizer = StableAdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        foreach=True,
        triton=False,
    )
    tokens_per_cycle = sum(
        bucket.length * bucket.micro_batch_size * bucket.slots for bucket in config.training.buckets
    )
    slots_per_cycle = sum(bucket.slots for bucket in config.training.buckets)
    tokens_per_micro_step = tokens_per_cycle / slots_per_cycle
    tokens_per_update = tokens_per_micro_step * config.training.gradient_accumulation
    total_updates = math.ceil(config.training.max_tokens / tokens_per_update)
    scheduler = create_scheduler(optimizer, total_updates, config.training)
    schedule = SequenceSchedule(config.training.buckets)
    largest_batch = max(bucket.micro_batch_size for bucket in config.training.buckets)
    source = create_document_source(config.data, config.training.seed, largest_batch)

    step = 0
    micro_step = 0
    tokens_seen = 0
    next_checkpoint = config.training.checkpoint_every_tokens
    if config.resume_from:
        restored = load_checkpoint(
            config.resume_from,
            model=model,
            teacher=teacher,
            optimizer=optimizer,
            scheduler=scheduler,
            source=source,
            generators=generators,
        )
        step = restored["step"]
        micro_step = restored["micro_step"]
        tokens_seen = restored["tokens_seen"]
        next_checkpoint = (
            tokens_seen // config.training.checkpoint_every_tokens + 1
        ) * config.training.checkpoint_every_tokens

    metric_path = output_dir / "metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    interval_start = time.perf_counter()
    interval_tokens = 0
    while (
        tokens_seen < config.training.max_tokens
        or micro_step % config.training.gradient_accumulation != 0
    ):
        bucket = schedule.at(micro_step)
        batch = next_encoded_batch(
            source,
            tokenizer,
            bucket.length,
            bucket.micro_batch_size,
            generators["data"],
        )
        batch = type(batch)(
            input_ids=batch.input_ids.to(device),
            attention_mask=batch.attention_mask.to(device),
            word_ids=batch.word_ids,
        )
        masked = build_masked_batch(config, teacher, batch, tokenizer, generators["mask"])
        output = model(
            input_ids=masked.input_ids,
            attention_mask=batch.attention_mask,
            labels=masked.labels,
        )
        loss = output.loss / config.training.gradient_accumulation
        loss.backward()

        batch_tokens = int(batch.attention_mask.sum())
        tokens_seen += batch_tokens
        interval_tokens += batch_tokens
        micro_step += 1
        if micro_step % config.training.gradient_accumulation == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if teacher is not None:
                teacher.update(model)
            step += 1

            if step % config.training.log_every_steps == 0:
                elapsed = time.perf_counter() - interval_start
                metric = {
                    "step": step,
                    "micro_step": micro_step,
                    "tokens_seen": tokens_seen,
                    "loss": float(loss.detach()) * config.training.gradient_accumulation,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "tokens_per_second": interval_tokens / elapsed,
                    "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
                _append_metric(metric_path, metric)
                interval_start = time.perf_counter()
                interval_tokens = 0
                torch.cuda.reset_peak_memory_stats()

        if (
            tokens_seen >= next_checkpoint
            and micro_step % config.training.gradient_accumulation == 0
        ):
            save_checkpoint(
                output_dir / "checkpoints" / f"tokens-{tokens_seen}.pt",
                model=model,
                teacher=teacher,
                optimizer=optimizer,
                scheduler=scheduler,
                source=source,
                step=step,
                micro_step=micro_step,
                tokens_seen=tokens_seen,
                generators=generators,
            )
            next_checkpoint += config.training.checkpoint_every_tokens

    final_dir = output_dir / "final"
    model.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)
    return final_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
