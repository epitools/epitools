from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose

import napari

import epitools._sample_data
from epitools.analysis import calculate_segmentation

SPOT_SIGMA = 4.0
OUTLINE_SIGMA = 3.0
THRESHOLD = 3.0


@pytest.fixture(scope="function")
def projected_image() -> napari.layers.Image:
    data, kwargs, layer_type = epitools._sample_data.load_projected_data()[0]
    return napari.layers.Image(data, **kwargs)


@pytest.fixture(scope="function")
def seeds_and_labels(
    make_napari_viewer,
) -> tuple[napari.layers.Points, napari.layers.Labels]:
    seeds_layer_data, labels_layer_data = epitools._sample_data.load_segmented_data()
    seeds_data, seeds_kwargs, _ = seeds_layer_data
    labels_data, labels_kwargs, _ = labels_layer_data
    return (
        napari.layers.Points(seeds_data, **seeds_kwargs),
        napari.layers.Labels(labels_data, **labels_kwargs),
    )


@pytest.fixture(scope="function")
def viewer_with_image(make_napari_viewer, projected_image) -> napari.Viewer:
    viewer = make_napari_viewer()
    viewer.add_layer(projected_image)
    return viewer


def test_add_segmentation_widget(make_napari_viewer):
    """Checks that the segmentation widget can be added inside a dock widget."""

    viewer = make_napari_viewer()
    num_dw = len(list(viewer.window._dock_widgets))
    viewer.window.add_plugin_dock_widget(
        plugin_name="epitools",
        widget_name="Segmentation (local minima seeded watershed)",
    )

    assert len(list(viewer.window._dock_widgets)) == num_dw + 1


def test_segmentation_widget_run_button(
    viewer_with_image,
    seeds_and_labels,
):
    """
    Check that pressing the 'Run' button performs segmentation of the selected
    image and adds two new layers (cells and seeds) to the viewer
    """

    dock_widget, container = viewer_with_image.window.add_plugin_dock_widget(
        plugin_name="epitools",
        widget_name="Segmentation (local minima seeded watershed)",
    )

    seeds, labels = seeds_and_labels

    # use saved image data so we don't run the segmentation analysis
    # when the button is pressed
    with patch("epitools.analysis.calculate_segmentation") as calculate_segmentation:
        calculate_segmentation.return_value = (seeds.data, labels.data)
        container.run.clicked()

    assert len(viewer_with_image.layers) == 3  # noqa: PLR2004

    original_layer, cells_layer, seeds_layer = viewer_with_image.layers

    assert isinstance(cells_layer, napari.layers.Labels)
    assert cells_layer.name == "Cells"
    assert (
        cells_layer.data.shape[-2:] == original_layer.data.shape[-2:]
    )  # yx dimensions

    assert isinstance(seeds_layer, napari.layers.Points)
    assert seeds_layer.name == "Seeds"
    n_cells = np.unique(cells_layer.data).size
    assert seeds_layer.data.shape == (n_cells, cells_layer.ndim)


def test_calculate_segmentation(
    projected_image,
    seeds_and_labels,
):
    seeds_layer, labels_layer = seeds_and_labels

    seeds_data, labels_data = calculate_segmentation(
        projection=projected_image.data,
        spot_sigma=SPOT_SIGMA,
        outline_sigma=OUTLINE_SIGMA,
        threshold=THRESHOLD,
    )
    assert labels_data.ndim == projected_image.ndim
    assert labels_data.shape == projected_image.data.shape
    assert_allclose(labels_data, labels_layer.data)
    assert_allclose(seeds_data, seeds_layer.data)
