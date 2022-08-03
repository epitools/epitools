"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/plugins/guides.html?#sample-data

Replace code below according to your needs.
"""
from __future__ import annotations

from skimage.io import imread


def load_sample_data():
    """Generates an image"""
    img_path = "sample_data/8bitDataset/test_image.tif"
    data = imread(img_path)
    return [(data, {"name": "Epitools test data"})]
