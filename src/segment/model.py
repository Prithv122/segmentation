"""U-Net factory (segmentation-models-pytorch) and the size-padding helpers it needs."""

from __future__ import annotations

import numpy as np

from segment.bands import IGNORE

#: A U-Net with a 5-stage encoder halves the input five times, so H and W must divide by 32.
DIVISOR = 32


def build_unet(
    encoder: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 4,
):
    """smp.Unet with one output logit per pixel.

    With ImageNet weights and 4 input channels, smp copies the RGB filters into the extra
    channel; that adaptation is smp's documented behaviour, recorded in NOTES.
    """
    import segmentation_models_pytorch as smp

    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=1,
    )


def pad_to_multiple(
    array: np.ndarray,
    multiple: int = DIVISOR,
    mode: str = "reflect",
    value: int = IGNORE,
) -> np.ndarray:
    """Pad the last two axes (bottom and right only) up to the next multiple of ``multiple``.

    ``mode="reflect"`` suits images; ``mode="constant"`` fills with ``value`` and suits masks
    (the default fill is IGNORE, so padding never counts in a metric).
    """
    h, w = array.shape[-2:]
    pad_h, pad_w = -h % multiple, -w % multiple
    if pad_h == 0 and pad_w == 0:
        return array
    widths = [(0, 0)] * (array.ndim - 2) + [(0, pad_h), (0, pad_w)]
    if mode == "constant":
        return np.pad(array, widths, mode="constant", constant_values=value)
    return np.pad(array, widths, mode=mode)


def crop_to(array: np.ndarray, height: int, width: int) -> np.ndarray:
    """Undo :func:`pad_to_multiple`: keep the top-left ``height`` x ``width`` window."""
    return array[..., :height, :width]
