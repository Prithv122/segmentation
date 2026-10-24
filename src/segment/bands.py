"""Chip band layout and mask codes, shared by every module.

A chip is a ``.npz`` file with two arrays:

* ``image``: ``uint16`` of shape ``(5, H, W)``, Sentinel-2 L1C top-of-atmosphere reflectance
  scaled by 10000, bands in :data:`BANDS` order.
* ``mask``: ``uint8`` of shape ``(H, W)`` with :data:`NO_WATER`, :data:`WATER` or
  :data:`IGNORE` (cloud, cloud shadow or nodata in the source's valid-pixel mask).
"""

BANDS: tuple[str, ...] = ("blue", "green", "red", "nir", "swir1")
BAND_INDEX: dict[str, int] = {name: i for i, name in enumerate(BANDS)}

#: The model sees blue, green, red and NIR only (the same information NDWI uses, plus RGB).
MODEL_BANDS: tuple[str, ...] = ("blue", "green", "red", "nir")

REFLECTANCE_SCALE = 10000.0

NO_WATER = 0
WATER = 1
IGNORE = 255
