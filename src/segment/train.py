"""Training loop: Dice + BCE on non-ignored pixels, best checkpoint chosen on val IoU only."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import segment.data as data
from segment.bands import IGNORE
from segment.metrics import Counts, confusion, iou, pooled
from segment.model import build_unet, pad_to_multiple

log = logging.getLogger(__name__)

#: Splits the training code may open. ``test`` is deliberately absent.
TRAIN_SPLITS = ("train", "val")


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


def _jsonable(value: object) -> object:
    """Recursively turn paths and containers into JSON-safe values (paths as posix strings)."""
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


def augment(
    image: np.ndarray, mask: np.ndarray, rng: np.random.Generator, crop: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Random crop, flips and 90-degree rotations, applied identically to image (C,H,W) and mask.

    Rotations by 90/270 degrees are only drawn for square windows.
    """
    h, w = mask.shape
    if crop is not None and h >= crop and w >= crop:
        top = int(rng.integers(0, h - crop + 1))
        left = int(rng.integers(0, w - crop + 1))
        image = image[:, top : top + crop, left : left + crop]
        mask = mask[top : top + crop, left : left + crop]
    if rng.random() < 0.5:
        image, mask = image[:, :, ::-1], mask[:, ::-1]
    if rng.random() < 0.5:
        image, mask = image[:, ::-1, :], mask[::-1, :]
    square = mask.shape[0] == mask.shape[1]
    k = int(rng.integers(0, 4)) if square else 2 * int(rng.integers(0, 2))
    if k:
        image, mask = np.rot90(image, k, axes=(1, 2)), np.rot90(mask, k)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


class ChipDataset(Dataset):
    """Chips of one split: (float32 (4,H,W) model input, uint8 (H,W) mask), padded to 32.

    The mask is padded with IGNORE, so padding never counts in the loss or in a metric.
    Augmentation (train only) is seeded per sample from (seed, epoch, index), so it does not
    depend on the number of workers; set ``epoch`` before each pass.
    """

    def __init__(
        self,
        records: list[data.ChipRecord],
        root: Path,
        *,
        augment_data: bool = False,
        crop: int | None = None,
        seed: int = 0,
    ) -> None:
        banned = {r.split for r in records} - set(TRAIN_SPLITS)
        if banned:
            raise ValueError(f"training code must not load splits {sorted(banned)}")
        self.records = records
        self.root = Path(root)
        self.augment_data = augment_data
        self.crop = crop
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = data.load_chip(self.root / self.records[idx].image_path_rel)
        x = data.to_model_input(image)
        if self.augment_data:
            rng = np.random.default_rng([self.seed, self.epoch, idx])
            x, mask = augment(x, mask, rng, self.crop)
        x = pad_to_multiple(x, mode="reflect")
        mask = pad_to_multiple(mask, mode="constant", value=IGNORE)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(
            np.ascontiguousarray(mask)
        )


def masked_bce_dice_loss(
    logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0
) -> torch.Tensor:
    """BCE (masked mean) + soft Dice over the batch, both only on pixels where target != IGNORE.

    ``logits`` is (B,1,H,W) or (B,H,W); ``target`` is (B,H,W) holding 0, 1 or IGNORE. A batch
    with no valid pixel gives a loss of exactly 0.
    """
    logits = logits.float().reshape(target.shape)
    valid = (target != IGNORE).float()
    truth = (target == 1).float()
    bce = (F.binary_cross_entropy_with_logits(logits, truth, reduction="none") * valid).sum()
    bce = bce / valid.sum().clamp(min=1.0)
    prob = torch.sigmoid(logits) * valid
    inter = (prob * truth).sum()
    soft_dice = 1.0 - (2.0 * inter + eps) / (prob.sum() + truth.sum() + eps)
    return bce + soft_dice


@torch.no_grad()
def evaluate_val(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Pooled IoU over every val chip: sigmoid > 0.5, full chips, no augmentation."""
    model.eval()
    counts: list[Counts] = []
    use_amp = device.type == "cuda"
    for x, y in loader:
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x.to(device))
        pred = (torch.sigmoid(logits.float())[:, 0] > 0.5).cpu().numpy().astype(np.uint8)
        counts.extend(confusion(p, t) for p, t in zip(pred, y.numpy(), strict=True))
    value = iou(pooled(counts))
    return 0.0 if value != value else value


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _records(cfg: TrainConfig, split: str) -> list[data.ChipRecord]:
    if split not in TRAIN_SPLITS:
        raise ValueError(f"training code must not load split {split!r}")
    rows = [r for r in data.read_manifest(cfg.manifest) if r.split == split]
    if not rows:
        raise ValueError(f"{cfg.manifest}: no chips in split {split!r}")
    return rows


def train(cfg: TrainConfig) -> dict:
    """Train, write ``best.pt`` and ``run_record.json`` into ``cfg.out_dir``, return the record.

    The record holds the config, the environment (torch/smp versions, device, git commit if
    known), per-epoch train loss and val IoU, the best epoch and its val IoU. ``test`` chips are
    never loaded here. Epochs are numbered from 1. A val IoU of NaN (no water in val and none
    predicted) is scored 0.0.
    """
    import segmentation_models_pytorch as smp

    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cfg.device is 'cuda' but CUDA is not available")
    use_amp = device.type == "cuda"
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(cfg.chips_dir) if cfg.chips_dir is not None else Path(cfg.manifest).parent

    _seed_everything(cfg.seed)
    train_ds = ChipDataset(
        _records(cfg, "train"), root, augment_data=True, crop=cfg.crop, seed=cfg.seed
    )
    val_ds = ChipDataset(_records(cfg, "val"), root)
    loader_args = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": use_amp,
    }
    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed),
        drop_last=len(train_ds) > 1 and len(train_ds) % cfg.batch_size == 1,
        **loader_args,
    )
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)

    model = build_unet(cfg.encoder, cfg.encoder_weights, 4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config = _jsonable(asdict(cfg))
    history: list[dict] = []
    best_iou, best_epoch, stale = -1.0, 0, 0
    for epoch in range(1, cfg.epochs + 1):
        start = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        train_ds.epoch = epoch
        model.train()
        loss_sum, n_seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
            loss = masked_bce_dice_loss(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * x.shape[0]
            n_seen += x.shape[0]
        scheduler.step()
        val_iou = evaluate_val(model, val_loader, device)
        entry = {
            "epoch": epoch,
            "train_loss": loss_sum / max(n_seen, 1),
            "val_iou": val_iou,
            "lr": lr,
            "seconds": time.perf_counter() - start,
        }
        history.append(entry)
        log.info(
            "epoch %d/%d loss %.4f val_iou %.4f lr %.2e (%.1fs)",
            epoch,
            cfg.epochs,
            entry["train_loss"],
            val_iou,
            lr,
            entry["seconds"],
        )
        if val_iou > best_iou:
            best_iou, best_epoch, stale = val_iou, epoch, 0
            torch.save(
                {
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "encoder": cfg.encoder,
                    "in_channels": 4,
                    "epoch": epoch,
                    "val_iou": val_iou,
                    "config": config,
                },
                out_dir / "best.pt",
            )
        else:
            stale += 1
            if stale >= cfg.patience:
                log.info("early stop at epoch %d (best epoch %d)", epoch, best_epoch)
                break

    record = {
        "config": config,
        "versions": {
            "torch": torch.__version__,
            "segmentation_models_pytorch": smp.__version__,
            "numpy": np.__version__,
        },
        "device": str(device),
        "git_commit": os.environ.get("SEGMENT_GIT_COMMIT"),
        "epochs": history,
        "best_epoch": best_epoch,
        "best_val_iou": best_iou if best_epoch else None,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
    }
    (out_dir / "run_record.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return record
