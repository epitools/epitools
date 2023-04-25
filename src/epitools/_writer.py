"""
This module is an example of a barebones writer plugin for napari.

It implements the Writer specification.
see: https://napari.org/plugins/guides.html?#writers

Replace code below according to your needs.
"""
from __future__ import annotations

import pathlib
from typing import Any

import PartSegCore.napari_plugins.save_tiff_layer


def write_single_image(path: str | pathlib.Path, data: Any, meta: dict):
    """Writes a single image layer"""

    path = pathlib.Path(path)
    if not path.suffix:
        path = path.with_suffix(".tif")

    if path.suffix not in [".tif", ".tiff"]:
        return

    path = PartSegCore.napari_plugins.save_tiff_layer.napari_write_image(
        path=path.as_posix(),
        data=data,
        meta=meta,
    )

    return path


def write_single_labels(path: str, data: Any, meta: dict):
    """Writes a single labels layer"""

    path = PartSegCore.napari_plugins.save_tiff_layer.napari_write_labels(
        path=path,
        data=data,
        meta=meta,
    )

    return path
