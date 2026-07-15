from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SequenceBucket:
    length: int
    micro_batch_size: int
    slots: int


@dataclass(frozen=True)
class DataStreamConfig:
    remote: str
    local: str
    proportion: float


@dataclass(frozen=True)
class DataConfig:
    kind: str = "synthetic"
    split: str = "train"
    cache_limit: str = "220gb"
    streams: tuple[DataStreamConfig, ...] = ()


@dataclass(frozen=True)
class TrainingConfig:
    method: str = "ligm"
    seed: int = 11
    max_tokens: int = 100_000_000
    gradient_accumulation: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    ema_decay: float = 0.999
    warmup_ratio: float = 0.02
    stable_ratio: float = 0.83
    checkpoint_every_tokens: int = 25_000_000
    log_every_steps: int = 10
    buckets: tuple[SequenceBucket, ...] = field(
        default_factory=lambda: (
            SequenceBucket(2048, 4, 5),
            SequenceBucket(4096, 2, 3),
            SequenceBucket(8192, 1, 2),
        )
    )


@dataclass(frozen=True)
class RunConfig:
    model_path: str
    output_dir: str
    attention_implementation: str
    data: DataConfig
    training: TrainingConfig
    resume_from: str | None = None


def _training_config(raw: dict) -> TrainingConfig:
    buckets = tuple(SequenceBucket(**bucket) for bucket in raw.pop("buckets", []))
    return TrainingConfig(**raw, **({"buckets": buckets} if buckets else {}))


def load_config(path: str | Path) -> RunConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    data_raw = raw.pop("data")
    streams = tuple(DataStreamConfig(**stream) for stream in data_raw.pop("streams", []))
    data = DataConfig(**data_raw, streams=streams)
    training = _training_config(raw.pop("training"))
    config = RunConfig(data=data, training=training, **raw)
    if training.method not in {"ligm", "random"}:
        raise ValueError(f"Unsupported method: {training.method}")
    if data.split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported data split: {data.split}")
    if abs(training.warmup_ratio + training.stable_ratio - 0.85) > 1e-9:
        raise ValueError("Warmup and stable ratios must sum to 0.85")
    return config
