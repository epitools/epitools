"""
This module is an example of a barebones writer plugin for napari.

It implements the Writer specification.
see: https://napari.org/plugins/guides.html?#writers

Replace code below according to your needs.
"""
from __future__ import annotations

import pathlib
from typing import Any

import PartSegImage

SCALE_MASKS = {
    "XY": [0, 1],  # 2D grayscale
    "TYX": [1, 2],  # 2D grayscale timeseries
    "TYXC": [1, 2],  # 2D multichannel timeseries
    "ZYX": [0, 1, 2],  # 3D grayscale
    "TZYX": [1, 2, 3],  # 3D grayscale timeseries
    "TZYXC": [1, 2, 3],  # 3D multichannel timeseries
}


def write_single_image(path: str | pathlib.Path, data: Any, meta: dict):
    """Writes a single image layer"""

    path = pathlib.Path(path)
    if not path.suffix:
        path = path.with_suffix(".tif")

    if path.suffix not in [".tif", ".tiff"]:
        return []

    # Undo scaling before saving
    scale_factor = PartSegImage.Image.DEFAULT_SCALE_FACTOR
    scale_shift = min(data.ndim, 3)

    # TODO: launch dialogue for user to select axes order
    axes = "TZYX"
    channel_names = [meta["name"]]
    if data.shape[-1] < 6:  # noqa: PLR2004
        axes += "C"
        scale_shift -= 1
        channel_names = [f'{meta["name"]} {i}' for i in range(1, data.shape[-1] + 1)]

    image = PartSegImage.Image(
        data=data,
        image_spacing=(meta["scale"] / scale_factor)[-scale_shift:],
        axes_order="TZXY"[-data.ndim :],
        channel_names=channel_names,
        shift=(meta["translate"] / scale_factor)[-scale_shift:],
        name="ROI",
    )
    PartSegImage.ImageWriter.save(
        image=image,
        path=path.as_posix(),
    )

    return [path]


def write_single_labels(path: str | pathlib.Path, data: Any, meta: dict):
    """Writes a single labels layer"""

    path = pathlib.Path(path)
    if not path.suffix:
        path = path.with_suffix(".tif")

    if path.suffix not in [".tif", ".tiff"]:
        return []

    # Undo scaling before saving
    scale_factor = PartSegImage.Image.DEFAULT_SCALE_FACTOR
    scale_shift = min(data.ndim, 3)

    image = PartSegImage.Image(
        data=data,
        image_spacing=(meta["scale"] / scale_factor)[-scale_shift:],
        axes_order="TZXY"[-data.ndim :],
        channel_names=[meta["name"]],
        shift=(meta["translate"] / scale_factor)[-scale_shift:],
        name="ROI",
    )
    PartSegImage.ImageWriter.save(
        image=image,
        path=path.as_posix(),
    )

    return [path]
