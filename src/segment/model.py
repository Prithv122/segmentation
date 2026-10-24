"""U-Net factory (segmentation-models-pytorch)."""

from __future__ import annotations


def build_unet(
    encoder: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 4,
):
    """smp.Unet with one output logit per pixel.

    With ImageNet weights and 4 input channels, smp copies the RGB filters into the extra
    channel; that adaptation is smp's documented behaviour, recorded in NOTES.
    """
    raise NotImplementedError
