"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/plugins/guides.html?#sample-data

Replace code below according to your needs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

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
    labels_data, labels_kwargs, labels_layer_type = epitools._reader.reader_function(
        path=labels_path.as_posix(),
    )[0]

    labels_kwargs["name"] = "Cells"
    labels_kwargs["metadata"] = _load_cell_statistics()

    # Load seeds
    seeds_path = (
        Path("sample_data") / "8bitDataset" / "test_image-projected-segmented-seeds.npy"
    )
    seeds_data = np.load(seeds_path)
    seeds_kwargs = {"name": "Seeds", "translate": labels_kwargs["translate"]}
    seeds_layer_type = "points"

    seeds_kwargs["name"] = "Seeds"
    seeds_kwargs["translate"] = labels_kwargs["translate"]

    # arrange data in format required by napari
    labels_layer_data = (labels_data, labels_kwargs, labels_layer_type)
    seeds_layer_data = (seeds_data, seeds_kwargs, seeds_layer_type)

    return [seeds_layer_data, labels_layer_data]


def _load_cell_statistics():
    """Load cell staistics associated with sample segmentation"""

    stats_path = (
        Path("sample_data") / "8bitDataset" / "test_image-projected-segmented-stats.csv"
    )
    stats = pd.read_csv(stats_path)

    return stats
