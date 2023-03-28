"""
This module contains functions for calculating region-based properties
of labelled images using skimage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari.types
    import numpy.typing as npt

import napari
import skimage.measure


def calculate_regionprops(
    image: napari.types.ImageData,
    labels: napari.types.LabelsData,
) -> npt.ArrayLike:
    """Calculate the region based properties of a segmented image"""

    properties = ["label", "area", "perimeter", "orientation"]

    # Calculate regionprops for each frame
    regionprops = [
        skimage.measure.regionprops_table(
            label_image=frame_labels,
            intensity_image=frame_image,
            properties=properties,
        )
        for frame_labels, frame_image in zip(labels, image)
    ]

    # skimage uses 'label' for what napari calls 'index'
    for frame_regionprops in regionprops:
        frame_regionprops["index"] = frame_regionprops.pop("label")

    return regionprops
