"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/plugins/guides.html?#sample-data

Replace code below according to your needs.
"""
from __future__ import annotations

from pathlib import Path

import napari

import epitools._reader


def load_sample_data() -> napari.layers.Image:
    """Load a sample dataset"""

    img_path = Path("sample_data") / "8bitDataset" / "test_image.tif"
    return epitools._reader.reader_function(path=img_path.as_posix())


def load_projected_data() -> napari.layers.Image:
    """Load a smaple projected dataset"""

    img_path = Path("sample_data") / "8bitDataset" / "test_image-projected.tif"
    return epitools._reader.reader_function(path=img_path.as_posix())


def load_segmented_data() -> napari.layers.Label:
    """Load a sample segmented dataset"""

    img_path = (
        Path("sample_data") / "8bitDataset" / "test_image-projected-segmented.tif"
    )
    return epitools._reader.reader_function(path=img_path.as_posix())
