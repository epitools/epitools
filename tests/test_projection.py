from unittest.mock import patch

import numpy as np
import pytest

import napari

import epitools._sample_data
from epitools.analysis import calculate_projection

SMOOTHING_RADIUS = 0.2
SURFACE_SMOOTHNESS_1 = 50
SURFACE_SMOOTHNESS_2 = 50
CUT_OFF_DISTANCE = 20
PROJECTION_NDIM = 3


@pytest.fixture(scope="function")
def test_image() -> napari.layers.Image:
    data, metadata, layer_type = epitools._sample_data.load_sample_data()[0]
    metadata["name"] = "Test Image"
    return napari.layers.Image(data, **metadata)


@pytest.fixture(scope="function")
def viewer_with_test_image(make_napari_viewer, test_image) -> napari.Viewer:
    viewer = make_napari_viewer()
    viewer.add_layer(test_image)

    return viewer


def test_add_projection_widget(make_napari_viewer):
    """Checks that the projection widget can be added inside a dock widget."""

    viewer = make_napari_viewer()
    num_dw = len(list(viewer.window._dock_widgets))
    viewer.window.add_plugin_dock_widget(
        plugin_name="epitools",
        widget_name="Projection (selective plane)",
    )

    assert len(list(viewer.window._dock_widgets)) == num_dw + 1


def test_projection_widget_run_button(
    viewer_with_test_image,
):
    dock_widget, container = viewer_with_test_image.window.add_plugin_dock_widget(
        plugin_name="epitools",
        widget_name="Projection (selective plane)",
    )
    container.run.clicked()

    assert len(viewer_with_test_image.layers) == 2  # noqa: PLR2004

    original_layer, new_layer = viewer_with_test_image.layers

    assert isinstance(new_layer, napari.layers.Image)
    assert new_layer.name == "Projection"
    assert new_layer.data.shape[-2:] == original_layer.data.shape[-2:]  # yx dimensions


def test_calculate_projection(test_image):
    with patch("epitools.analysis.projection._interpolate") as interp:
        mock_interpolation = np.zeros(
            (test_image.data.shape[2], test_image.data.shape[3])
        )
        interp.return_value = mock_interpolation
        projection = calculate_projection(
            test_image.data,
            SMOOTHING_RADIUS,
            SURFACE_SMOOTHNESS_1,
            SURFACE_SMOOTHNESS_2,
            CUT_OFF_DISTANCE,
        )
        assert projection.ndim == PROJECTION_NDIM
        assert projection.shape == (
            1,  # single frame in the timeseries
            test_image.data.shape[2],
            test_image.data.shape[3],
        )
