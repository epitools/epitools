"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/plugins/guides.html?#sample-data

Replace code below according to your needs.
"""
from __future__ import annotations

from pathlib import Path

import napari_epitools._reader


def load_sample_data():
    """Load a sample dataset"""

    img_path = Path("sample_data") / "8bitDataset" / "test_image.tif"
    return napari_epitools._reader.reader_function(path=img_path.as_posix())
