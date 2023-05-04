from __future__ import annotations

from typing import Callable
from unittest.mock import patch

import napari

from epitools.analysis import calculate_projection

SMOOTHING_RADIUS = 0.2
SURFACE_SMOOTHNESS_1 = 50
SURFACE_SMOOTHNESS_2 = 50
CUT_OFF_DISTANCE = 20
PROJECTION_NDIM = 3


def test_add_projection_widget(
    make_napari_viewer: Callable,
):
    """Checks that the projection widget can be added inside a dock widget."""

    viewer = make_napari_viewer()
    num_dw = len(viewer.window._dock_widgets)
    viewer.window.add_plugin_dock_widget(
        plugin_name="epitools",
        widget_name="Projection (selective plane)",
    )

    assert len(viewer.window._dock_widgets) == num_dw + 1


def test_projection_widget_run_button(
    viewer_with_image: napari.Viewer,
    projected_image: napari.layers.Image,
):
    """
    Check that pressing the 'Run' button performs projection of the selected
    image and adds a new layer to the viewer
    """

    dock_widget, container = viewer_with_image.window.add_plugin_dock_widget(
        plugin_name="epitools",
        widget_name="Projection (selective plane)",
    )

    # use saved image data so we don't run the projection analysis
    # when the button is pressed
    with patch("epitools.analysis.calculate_projection") as calculate_projection:
        calculate_projection.return_value = projected_image.data
        container.run.clicked()

    assert len(viewer_with_image.layers) == 2  # noqa: PLR2004

    original_layer, new_layer = viewer_with_image.layers

    assert isinstance(new_layer, napari.layers.Image)
    assert new_layer.name == "Projection"
    assert new_layer.data.shape[-2:] == original_layer.data.shape[-2:]  # yx dimensions


def test_calculate_projection(
    image: napari.layers.Image,
):
    projection = calculate_projection(
        image.data,
        SMOOTHING_RADIUS,
        SURFACE_SMOOTHNESS_1,
        SURFACE_SMOOTHNESS_2,
        CUT_OFF_DISTANCE,
    )

    assert projection.ndim == PROJECTION_NDIM
    assert projection.shape == (
        1,  # single frame in the timeseries
        image.data.shape[2],
        image.data.shape[3],
    )
