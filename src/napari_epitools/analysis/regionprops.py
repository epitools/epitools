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
    regionprops = skimage.measure.regionprops_table(
        label_image=labels[0],
        intensity_image=image[0],
        properties=properties,
    )

    return regionprops
