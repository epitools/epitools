"""
This module is an example of a barebones writer plugin for napari.

It implements the Writer specification.
see: https://napari.org/plugins/guides.html?#writers

Replace code below according to your needs.
"""
from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import PartSegImage
from PartSegImage.image import DEFAULT_SCALE_FACTOR

import napari

import epitools.widgets.dialogue

TWO_DIMENSIONAL = 2
THREE_DIMENSIONAL = 3
FOUR_DIMENSIONAL = 4
FIVE_DIMENSIONAL = 5
AXES_ORDER_MASKS = {
    "YX": [0, 1],  # 2D grayscale
    "TYX": [1, 2],  # 2D grayscale timeseries
    "TYXC": [1, 2],  # 2D multichannel timeseries
    "ZYX": [0, 1, 2],  # 3D grayscale
    "TZYX": [1, 2, 3],  # 3D grayscale timeseries
    "ZYXC": [0, 1, 2],  # 3D multichannel
    "TZYXC": [1, 2, 3],  # 3D multichannel timeseries
}


def _get_axes_dimensions(ndim: int, name: str) -> str | None:
    """Determine the dimensions the axes in an image correspond to.

    If there are multiple possibilities, the user will be asked to select
    the correct dimensions.
    """

    if ndim == TWO_DIMENSIONAL:
        return "YX"
    elif ndim == FIVE_DIMENSIONAL:
        return "TZYXC"

    title = "Select axes dimensions"
    prompt = (
        f"{name} has {ndim} dimensions.\n"
        "Please select the dimensions the axes correspond to."
    )
    parent = napari.current_viewer().window.qt_viewer

    if ndim == THREE_DIMENSIONAL:
        options = {
            "2D timeseries (TYX)": "TYX",
            "3D (ZYX)": "ZYX",
        }
    elif ndim == FOUR_DIMENSIONAL:
        options = {
            "2D multichannel timeseries (TYXC)": "TYXC",
            "3D timeseries (TZYX)": "TZYX",
            "3D multichannel (ZYXC)": "ZYXC",
        }
    else:
        # TODO: show notification that we cannot save 1D images
        # or images with >5 dimensions
        return None

    selected = epitools.widgets.dialogue.select_option(
        options=options.keys(),
        title=title,
        prompt=prompt,
        parent=parent,
    )

    return selected if selected is None else options[selected]


def write_single_image(path: str | pathlib.Path, data: Any, meta: dict):
    """Writes a single image layer"""

    path = pathlib.Path(path)
    if not path.suffix:
        path = path.with_suffix(".tif")

    if path.suffix not in [".tif", ".tiff"]:
        return []

    axes_order = _get_axes_dimensions(ndim=data.ndim, name=meta["name"])
    if axes_order is None:
        return []

    axes_order_mask = AXES_ORDER_MASKS[axes_order]

    # Set channel names
    n_channels = data.shape[axes_order.index("C")] if "C" in axes_order else 1
    channel_names = (
        [f'{meta["name"]} {i}' for i in range(1, n_channels + 1)]
        if "C" in axes_order
        else [meta["name"]]
    )

    image = PartSegImage.Image(
        data=data,
        image_spacing=np.divide(meta["scale"], DEFAULT_SCALE_FACTOR)[axes_order_mask],
        axes_order=axes_order,
        channel_names=channel_names,
        shift=np.divide(meta["translate"], DEFAULT_SCALE_FACTOR)[axes_order_mask],
        name="Image",  # PartSeg identifier for correct loading later on
    )
    PartSegImage.ImageWriter.save(
        image=image,
        save_path=path.as_posix(),
        compression=False,
    )

    return [path]


def write_single_labels(path: str | pathlib.Path, data: Any, meta: dict):
    """Writes a single labels layer"""

    path = pathlib.Path(path)
    if not path.suffix:
        path = path.with_suffix(".tif")

    if path.suffix not in [".tif", ".tiff"]:
        return []

    axes_order = _get_axes_dimensions(ndim=data.ndim, name=meta["name"])
    if axes_order is None:
        return []

    axes_order_mask = AXES_ORDER_MASKS[axes_order]

    image = PartSegImage.Image(
        data=data,
        image_spacing=np.divide(meta["scale"], DEFAULT_SCALE_FACTOR)[axes_order_mask],
        axes_order=axes_order,
        channel_names=[meta["name"]],
        shift=np.divide(meta["translate"], DEFAULT_SCALE_FACTOR)[axes_order_mask],
        name="ROI",  # PartSeg identifier for correct loading later on
    )
    PartSegImage.ImageWriter.save(
        image=image,
        save_path=path.as_posix(),
        compression=False,
    )

    return [path]
