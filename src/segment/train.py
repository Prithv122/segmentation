"""Training loop: Dice + BCE on non-ignored pixels, best checkpoint chosen on val IoU only."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    chips_dir: Path
    manifest: Path
    out_dir: Path
    encoder: str = "resnet34"
    encoder_weights: str | None = "imagenet"
    epochs: int = 40
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    crop: int | None = 256  # random crop size for training; None = full chip
    num_workers: int = 2
    seed: int = 0
    device: str = "cuda"
    patience: int = 8  # early stop when val IoU hasn't improved for this many epochs
    extra: dict = field(default_factory=dict)


def train(cfg: TrainConfig) -> dict:
    """Train, write ``best.pt`` and ``run_record.json`` into ``cfg.out_dir``, return the record.

    The record holds the config, the environment (torch/smp versions, device, git commit if
    known), per-epoch train loss and val IoU, the best epoch and its val IoU. ``test`` chips are
    never loaded here.
    """
    raise NotImplementedError
