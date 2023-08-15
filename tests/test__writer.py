from __future__ import annotations

import tempfile
from pathlib import Path

import napari

from epitools._writer import write_single_image, write_single_labels

_EXTENSION_MAP = {
    "image": ".tif",
    "labels": ".tif",
}


def test_write_single_image(
    viewer_with_image: napari.Viewer,
):
    viewer_with_image.layers[0].metadata["name"] = "test"
    viewer_with_image.layers[0].metadata["scale"] = (1, 1, 1, 1)
    viewer_with_image.layers[0].metadata["translate"] = (0, 0, 0, 0)

    with tempfile.TemporaryDirectory() as tmpdirname:
        output_file = Path(tmpdirname) / "test.tif"
        written = write_single_image(
            path=output_file.as_posix(),
            data=viewer_with_image.layers[0].data,
            meta=viewer_with_image.layers[0].metadata,
            axes_order="TZYX",
        )

        # check expected files were written
        assert written[0] == output_file
        assert output_file.is_file()


def test_write_single_labels(
    viewer_with_labels: napari.Viewer,
):
    viewer_with_labels.layers[0].metadata["name"] = "test"
    viewer_with_labels.layers[0].metadata["scale"] = (1, 1, 1, 1)
    viewer_with_labels.layers[0].metadata["translate"] = (0, 0, 0, 0)

    with tempfile.TemporaryDirectory() as tmpdirname:
        output_file = Path(tmpdirname) / "test.tif"
        written = write_single_labels(
            path=output_file.as_posix(),
            data=viewer_with_labels.layers[0].data,
            meta=viewer_with_labels.layers[0].metadata,
            axes_order="TZYX",
        )

        # check expected files were written
        assert written[0] == output_file
        assert output_file.is_file()
