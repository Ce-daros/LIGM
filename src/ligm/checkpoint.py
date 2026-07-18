from pathlib import Path

import numpy as np
import torch


def prune_checkpoints(
    path: str | Path,
    keep_recent: int,
    keep_every: int,
    milestone_tokens: tuple[int, ...] = (),
) -> None:
    directory = Path(path)
    checkpoints = sorted(
        directory.glob("tokens-*.pt"),
        key=lambda item: int(item.stem.removeprefix("tokens-")),
    )
    if milestone_tokens:
        milestones = {
            next(
                (
                    checkpoint
                    for checkpoint in checkpoints
                    if int(checkpoint.stem.removeprefix("tokens-")) >= milestone
                ),
                None,
            )
            for milestone in milestone_tokens
        }
        milestones.discard(None)
    else:
        milestones = {
            checkpoint
            for index, checkpoint in enumerate(checkpoints, start=1)
            if index % keep_every == 0
        }
    retained = milestones | set(checkpoints[-keep_recent:])
    for checkpoint in checkpoints:
        if checkpoint not in retained:
            checkpoint.unlink()


def save_checkpoint(
    path: str | Path,
    *,
    model,
    teacher,
    optimizer,
    scheduler,
    source,
    step: int,
    micro_step: int,
    tokens_seen: int,
    generators: dict[str, torch.Generator],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "teacher": teacher.state_dict() if teacher is not None else None,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "source": source.state_dict(),
            "step": step,
            "micro_step": micro_step,
            "tokens_seen": tokens_seen,
            "generators": {
                name: generator.get_state() for name, generator in generators.items()
            },
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng": np.random.get_state(),
        },
        temporary,
    )
    temporary.replace(target)


def load_checkpoint(
    path: str | Path,
    *,
    model,
    teacher,
    optimizer,
    scheduler,
    source,
    generators: dict[str, torch.Generator],
) -> dict[str, int]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if teacher is not None:
        if state["teacher"] is None:
            teacher.model.load_state_dict(state["model"])
        else:
            teacher.load_state_dict(state["teacher"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    source.load_state_dict(state["source"])
    for name, generator in generators.items():
        generator.set_state(state["generators"][name])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    np.random.set_state(state["numpy_rng"])
    return {
        "step": int(state["step"]),
        "micro_step": int(state["micro_step"]),
        "tokens_seen": int(state["tokens_seen"]),
    }
