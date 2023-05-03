"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/plugins/guides.html?#sample-data

Replace code below according to your needs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import epitools._reader


def load_sample_data():
    """Load a sample dataset"""

    img_path = Path("sample_data") / "8bitDataset" / "test_image.tif"
    return epitools._reader.reader_function(path=img_path.as_posix())


def load_projected_data():
    """Load a smaple projected dataset"""

    img_path = Path("sample_data") / "8bitDataset" / "test_image-projected.tif"
    return epitools._reader.reader_function(path=img_path.as_posix())


def load_segmented_data():
    """Load a sample segmented dataset"""

    # Load labels
    labels_path = (
        Path("sample_data") / "8bitDataset" / "test_image-projected-segmented.tif"
    )
    labels_data, labels_metadata, labels_layer_type = epitools._reader.reader_function(
        path=labels_path.as_posix(),
    )[0]
    labels_metadata["name"] = "Cells"

    # Load seeds
    seeds_path = (
        Path("sample_data") / "8bitDataset" / "test_image-projected-segmented-seeds.npy"
    )
    seeds_data = np.load(seeds_path)
    seeds_metadata = {"name": "Seeds", "translate": labels_metadata["translate"]}
    seeds_layer_type = "points"

    seeds_metadata["name"] = "Seeds"
    seeds_metadata["translate"] = labels_metadata["translate"]

    # arrange data in format required by napari
    labels_layer_data = (labels_data, labels_metadata, labels_layer_type)
    seeds_layer_data = (seeds_data, seeds_metadata, seeds_layer_type)

    return [seeds_layer_data, labels_layer_data]
