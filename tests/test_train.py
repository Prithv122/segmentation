import json
from pathlib import Path

import numpy as np
import pytest
import torch

import segment.data as data
from segment.bands import IGNORE
from segment.predict import unet_predict_fn
from segment.train import ChipDataset, TrainConfig, augment, masked_bce_dice_loss, train


def _target() -> torch.Tensor:
    t = torch.zeros(2, 8, 8, dtype=torch.uint8)
    t[:, 2:5, 2:5] = 1
    t[:, :2, :2] = IGNORE
    return t


def test_loss_ignores_ignored_pixels():
    target = _target()
    logits = torch.randn(2, 1, 8, 8)
    changed = logits.clone()
    changed[:, 0][target == IGNORE] += 50.0
    assert torch.isclose(
        masked_bce_dice_loss(logits, target), masked_bce_dice_loss(changed, target)
    )


def test_loss_all_ignore_is_zero_and_finite():
    target = torch.full((2, 8, 8), IGNORE, dtype=torch.uint8)
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    loss = masked_bce_dice_loss(logits, target)
    loss.backward()
    assert loss.item() == 0.0
    assert torch.isfinite(logits.grad).all() and (logits.grad == 0).all()


def test_loss_prefers_correct_logits():
    target = _target()
    good = torch.where(target == 1, 8.0, -8.0).unsqueeze(1)
    assert masked_bce_dice_loss(good, target) < masked_bce_dice_loss(-good, target)


@pytest.mark.parametrize("crop", [None, 16])
def test_augment_keeps_image_and_mask_aligned(crop):
    ids = np.arange(32 * 32, dtype=np.int64).reshape(32, 32)
    image = np.stack([ids + c * 10_000 for c in range(4)])
    for seed in range(30):
        img, msk = augment(image, ids, np.random.default_rng(seed), crop)
        size = crop or 32
        assert img.shape == (4, size, size) and msk.shape == (size, size)
        for c in range(4):
            assert np.array_equal(img[c], msk + c * 10_000)


def test_augment_is_deterministic_and_varied():
    ids = np.arange(32 * 32, dtype=np.int64).reshape(32, 32)
    image = np.stack([ids] * 4)
    a = augment(image, ids, np.random.default_rng(3), 16)[1]
    b = augment(image, ids, np.random.default_rng(3), 16)[1]
    assert np.array_equal(a, b)
    outs = {augment(image, ids, np.random.default_rng(s), 16)[1].tobytes() for s in range(10)}
    assert len(outs) > 1


def test_dataset_pads_to_32_and_rejects_test_split(fixture_manifest):
    records = data.read_manifest(fixture_manifest)
    root = fixture_manifest.parent
    train_rows = [r for r in records if r.split == "train"]
    ds = ChipDataset(train_rows, root, augment_data=True, crop=20, seed=1)
    x, y = ds[0]
    assert x.shape == (4, 32, 32) and y.shape == (32, 32)
    assert (y[20:] == IGNORE).all()
    with pytest.raises(ValueError, match="test"):
        ChipDataset([r for r in records if r.split == "test"], root)


@pytest.mark.slow
def test_train_end_to_end_on_fixture(fixture_manifest, tmp_path, monkeypatch):
    opened: list[str] = []
    real_load = data.load_chip

    def spy(path):
        opened.append(Path(path).name)
        return real_load(path)

    monkeypatch.setattr(data, "load_chip", spy)
    threads = torch.get_num_threads()
    torch.set_num_threads(2)  # tiny tensors: more threads only add contention
    cfg = TrainConfig(
        chips_dir=fixture_manifest.parent,
        manifest=fixture_manifest,
        out_dir=tmp_path / "run",
        encoder="resnet18",
        encoder_weights=None,
        epochs=3,
        batch_size=2,
        lr=3e-3,
        crop=None,
        num_workers=0,
        device="cpu",
        patience=5,
    )
    try:
        record = train(cfg)
    finally:
        torch.set_num_threads(threads)

    epochs = record["epochs"]
    assert len(epochs) == 3
    assert epochs[-1]["train_loss"] < epochs[0]["train_loss"]
    assert record["n_train"] == 4 and record["n_val"] == 2
    assert opened and not any(name.startswith("D_") for name in opened)

    saved = json.loads((tmp_path / "run" / "run_record.json").read_text(encoding="utf-8"))
    assert set(saved) == {
        "config",
        "versions",
        "device",
        "git_commit",
        "epochs",
        "best_epoch",
        "best_val_iou",
        "n_train",
        "n_val",
    }
    assert set(saved["epochs"][0]) == {"epoch", "train_loss", "val_iou", "lr", "seconds"}
    assert saved["config"]["out_dir"] == (tmp_path / "run").as_posix()
    assert saved["best_val_iou"] == record["best_val_iou"]

    ckpt = torch.load(tmp_path / "run" / "best.pt", weights_only=True)
    assert set(ckpt) == {"state_dict", "encoder", "in_channels", "epoch", "val_iou", "config"}
    assert ckpt["epoch"] == saved["best_epoch"]

    predict = unet_predict_fn(tmp_path / "run" / "best.pt", "cpu")
    image, _ = real_load(fixture_manifest.parent / "chips" / "C_0.npz")
    pred = predict(image)
    assert pred.shape == (32, 32) and pred.dtype == np.uint8
    assert set(np.unique(pred)) <= {0, 1}
