from pathlib import Path

import numpy as np
import torch


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
    generator: torch.Generator,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
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
            "generator": generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng": np.random.get_state(),
        },
        target,
    )


def load_checkpoint(
    path: str | Path,
    *,
    model,
    teacher,
    optimizer,
    scheduler,
    source,
    generator: torch.Generator,
) -> dict[str, int]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if teacher is not None:
        teacher.load_state_dict(state["teacher"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    source.load_state_dict(state["source"])
    generator.set_state(state["generator"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    np.random.set_state(state["numpy_rng"])
    return {
        "step": int(state["step"]),
        "micro_step": int(state["micro_step"]),
        "tokens_seen": int(state["tokens_seen"]),
    }
