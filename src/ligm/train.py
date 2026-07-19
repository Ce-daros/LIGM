import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from optimi import StableAdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForMaskedLM, AutoTokenizer

from ligm.checkpoint import load_checkpoint, prune_checkpoints, save_checkpoint
from ligm.config import RunConfig, load_config
from ligm.data import SequenceSchedule, create_document_source, next_encoded_batch
from ligm.ema import EMATeacher
from ligm.evaluate import natural_repetition_recovery_model
from ligm.masking import (
    candidate_word_mask,
    random_word_mask,
    remote_evidence_ablation,
    select_ligm_targets,
    weight_ligm_targets,
)
from ligm.online import compare_local_recovery, read_report
from ligm.rotary import use_torch_rotary
from ligm.scoring import (
    entropy_scores,
    information_gain_scores,
    remote_evidence_scores,
)


def set_seed(seed: int) -> dict[str, torch.Generator]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return {
        "data": torch.Generator().manual_seed(seed),
        "mask": torch.Generator().manual_seed(seed + 1),
    }


def create_scheduler(optimizer, total_tokens: int, config) -> LambdaLR:
    phase_tokens = total_tokens - config.scheduler_origin_tokens
    if phase_tokens <= 0:
        raise ValueError("Scheduler origin must be below max tokens")
    warmup_tokens = max(1, round(phase_tokens * config.warmup_ratio))
    stable_end = round(phase_tokens * (config.warmup_ratio + config.stable_ratio))

    def scale(tokens_seen: int) -> float:
        elapsed_tokens = max(0, tokens_seen - config.scheduler_origin_tokens)
        if elapsed_tokens < warmup_tokens:
            return elapsed_tokens / warmup_tokens
        if elapsed_tokens < stable_end:
            return 1.0
        decay_tokens = max(1, phase_tokens - stable_end)
        return max(0.0, 1.0 - (elapsed_tokens - stable_end) / decay_tokens)

    return LambdaLR(optimizer, scale)


def rebase_scheduler(scheduler: LambdaLR, tokens_seen: int) -> None:
    learning_rates = [
        base_lr * schedule(tokens_seen)
        for base_lr, schedule in zip(scheduler.base_lrs, scheduler.lr_lambdas, strict=True)
    ]
    for group, learning_rate in zip(
        scheduler.optimizer.param_groups, learning_rates, strict=True
    ):
        group["lr"] = learning_rate
    scheduler.last_epoch = tokens_seen
    scheduler._last_lr = learning_rates


def build_masked_batch(config: RunConfig, teacher, batch, tokenizer, generator):
    if config.training.method == "random":
        return random_word_mask(
            batch.input_ids,
            batch.word_ids,
            tokenizer.mask_token_id,
            tokenizer.vocab_size,
            generator,
            config.training.mask_ratio,
        )
    if config.training.method == "ligm_weighted":
        masked = random_word_mask(
            batch.input_ids,
            batch.word_ids,
            tokenizer.mask_token_id,
            tokenizer.vocab_size,
            generator,
            config.training.mask_ratio,
        )
        scores = information_gain_scores(
            teacher.model,
            masked.input_ids,
            batch.attention_mask,
            masked.labels,
        )
        return weight_ligm_targets(
            masked,
            batch.word_ids,
            scores,
            config.training.target_ratio,
            config.training.target_loss_weight,
        )
    if config.training.method in {"red", "red_route"}:
        masked = random_word_mask(
            batch.input_ids,
            batch.word_ids,
            tokenizer.mask_token_id,
            tokenizer.vocab_size,
            generator,
            config.training.mask_ratio,
        )
        ablated, eligibility = remote_evidence_ablation(
            batch.input_ids,
            masked,
            batch.attention_mask,
            tokenizer.mask_token_id,
        )
        scores = remote_evidence_scores(
            teacher.model,
            masked.input_ids,
            ablated,
            batch.attention_mask,
            masked.labels,
            eligibility,
        )
        return weight_ligm_targets(
            masked,
            batch.word_ids,
            scores,
            config.training.target_ratio,
            config.training.target_loss_weight,
            eligibility,
        )
    candidates = candidate_word_mask(
        batch.input_ids,
        batch.word_ids,
        tokenizer.mask_token_id,
        generator,
    )
    if config.training.method == "entropy":
        scores = entropy_scores(
            teacher.model,
            candidates.input_ids,
            batch.attention_mask,
            candidates.labels,
        )
    else:
        scores = information_gain_scores(
            teacher.model,
            candidates.input_ids,
            batch.attention_mask,
            candidates.labels,
            learnability=config.training.method == "ligm",
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


def _run_online_evaluation(
    config: RunConfig,
    model,
    tokenizer,
    tokens_seen: int,
) -> dict:
    evaluation_dir = Path(config.output_dir) / "online-evaluation"
    report_path = evaluation_dir / f"tokens-{tokens_seen}-natural.json"
    torch.cuda.empty_cache()
    report = natural_repetition_recovery_model(
        model,
        tokenizer,
        config.data,
        report_path,
        config.online_evaluation.documents,
        config.output_dir,
    )
    report["tokens_seen"] = tokens_seen
    reference_dir = config.online_evaluation.reference_dir
    if reference_dir:
        reference_path = Path(reference_dir) / report_path.name
        comparison = compare_local_recovery(
            report,
            read_report(reference_path),
            config.online_evaluation.max_local_drop,
        )
        comparison["reference"] = str(reference_path)
    else:
        comparison = {"mode": "reference", "passed": True}
    report["local_guard"] = comparison
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _append_metric(
        evaluation_dir / "events.jsonl",
        {
            "tokens_seen": tokens_seen,
            "report": str(report_path),
            "local_guard": comparison,
        },
    )
    return comparison


def train(config: RunConfig) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("LIGM training requires a CUDA GPU")
    device = torch.device("cuda")
    use_torch_rotary()
    generators = set_seed(config.training.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = json.dumps(config, default=lambda value: value.__dict__, indent=2) + "\n"
    (output_dir / "resolved-config.json").write_text(resolved, encoding="utf-8")
    history = output_dir / "config-history"
    history.mkdir(exist_ok=True)
    (history / f"tokens-{config.training.max_tokens}.json").write_text(
        resolved,
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
    model.config.repad_logits_with_grad = config.training.method in {
        "ligm_weighted",
        "red",
        "red_route",
    }
    if config.training.method == "red_route":
        model.requires_grad_(False)
        interval = model.config.global_attn_every_n_layers
        for layer_index, layer in enumerate(model.model.layers):
            if layer_index % interval == 0:
                layer.requires_grad_(True)
    model.gradient_checkpointing_enable()
    model.train()

    teacher = (
        EMATeacher(model, config.training.ema_decay)
        if config.training.method != "random"
        else None
    )
    optimizer = StableAdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        foreach=True,
        triton=False,
    )
    scheduler = create_scheduler(optimizer, config.training.max_tokens, config.training)
    schedule = SequenceSchedule(config.training.buckets)
    largest_batch = max(bucket.micro_batch_size for bucket in config.training.buckets)
    source = create_document_source(config.data, config.training.seed, largest_batch)

    step = 0
    micro_step = 0
    tokens_seen = 0
    next_checkpoint = config.training.checkpoint_every_tokens
    last_checkpoint_tokens = 0
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
        last_checkpoint_tokens = tokens_seen
        if config.training.restart_optimizer:
            optimizer.state.clear()
            for group in optimizer.param_groups:
                group["lr"] = config.training.learning_rate
                group.pop("initial_lr", None)
            scheduler = create_scheduler(optimizer, config.training.max_tokens, config.training)
        rebase_scheduler(scheduler, tokens_seen)

    metric_path = output_dir / "metrics.jsonl"
    stop_requested = False
    last_safe_checkpoint: Path | None = None
    selected_checkpoint: Path | None = None
    if config.online_evaluation.enabled:
        if not config.resume_from:
            raise ValueError("Online evaluation requires a resume checkpoint")
        initial_checkpoint = Path(config.resume_from)
        initial_guard = _run_online_evaluation(config, model, tokenizer, tokens_seen)
        if initial_guard["passed"]:
            last_safe_checkpoint = initial_checkpoint
        else:
            stop_requested = True
            selected_checkpoint = initial_checkpoint

    optimizer.zero_grad(set_to_none=True)
    interval_start = time.perf_counter()
    interval_tokens = 0
    while (
        not stop_requested
        and (tokens_seen < config.training.max_tokens
        or micro_step % config.training.gradient_accumulation != 0
        )
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
        if masked.loss_weights is None:
            batch_loss = output.loss
        else:
            selected_logits = (
                output.logits[masked.selected]
                if output.logits.ndim == 3
                else output.logits
            )
            selected_labels = masked.labels[masked.selected]
            token_losses = F.cross_entropy(
                selected_logits.float(), selected_labels, reduction="none"
            )
            weights = masked.loss_weights[masked.selected].to(token_losses)
            batch_loss = (token_losses * weights).sum() / weights.sum()
        loss = batch_loss / config.training.gradient_accumulation
        loss.backward()

        batch_tokens = int(batch.attention_mask.sum())
        tokens_seen += batch_tokens
        interval_tokens += batch_tokens
        micro_step += 1
        if micro_step % config.training.gradient_accumulation == 0:
            optimizer.step()
            rebase_scheduler(scheduler, tokens_seen)
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
                if masked.loss_weights is not None:
                    targeted = masked.loss_weights > 1
                    metric["targeted_tokens"] = int(targeted.sum())
                    metric["target_fraction"] = float(
                        targeted.sum() / masked.selected.sum()
                    )
                    metric["mean_target_score"] = float(
                        masked.scores[targeted].mean() if targeted.any() else 0.0
                    )
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
            prune_checkpoints(
                output_dir / "checkpoints",
                config.training.keep_recent_checkpoints,
                config.training.keep_every_checkpoints,
                config.training.keep_milestone_tokens,
            )
            last_checkpoint_tokens = tokens_seen
            next_checkpoint += config.training.checkpoint_every_tokens
            current_checkpoint = (
                output_dir / "checkpoints" / f"tokens-{tokens_seen}.pt"
            )
            if config.online_evaluation.enabled:
                del output, loss, batch, masked
                guard = _run_online_evaluation(config, model, tokenizer, tokens_seen)
                if guard["passed"]:
                    last_safe_checkpoint = current_checkpoint
                else:
                    stop_requested = True
                    selected_checkpoint = last_safe_checkpoint

    if not stop_requested and last_checkpoint_tokens != tokens_seen:
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
        prune_checkpoints(
            output_dir / "checkpoints",
            config.training.keep_recent_checkpoints,
            config.training.keep_every_checkpoints,
            config.training.keep_milestone_tokens,
        )

    if config.online_evaluation.enabled:
        selected_checkpoint = selected_checkpoint or last_safe_checkpoint
        selection = {
            "early_stopped": stop_requested,
            "stopped_at_tokens": tokens_seen,
            "selected_checkpoint": str(selected_checkpoint),
        }
        (output_dir / "online-evaluation" / "selection.json").write_text(
            json.dumps(selection, indent=2) + "\n",
            encoding="utf-8",
        )
        if selected_checkpoint is None:
            raise RuntimeError("Online evaluation did not produce a safe checkpoint")
        selected_state = torch.load(selected_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(selected_state["model"])

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
