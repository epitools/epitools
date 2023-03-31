"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/plugins/guides.html?#sample-data

Replace code below according to your needs.
"""
from __future__ import annotations

from pathlib import Path

from skimage.io import imread


def load_sample_data():
    """Generates an image"""
    img_path = (
        Path("sample_data") / "4d" / "181210_DM_CellMOr_subsub_reg_decon-1_cropped.tif"
    )
    data = imread(img_path)
    return [(data, {"name": "Epitools test data"})]
