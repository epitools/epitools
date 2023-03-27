import gc
from pathlib import Path
from unittest.mock import patch

import magicgui
import numpy as np
import pytest
from skimage.io import imread

from napari_epitools._widget import projection_widget
from napari_epitools.analysis import calculate_projection

SMOOTHING_RADIUS = 0.2
SURFACE_SMOOTHNESS_1 = 50
SURFACE_SMOOTHNESS_2 = 50
CUT_OFF_DISTANCE = 20
PROJECTION_NDIM = 3


@pytest.fixture
def sample_data():
    img_path = img_path = Path("sample_data") / "8bitDataset" / "test_image.tif"
    return imread(img_path)


@pytest.fixture
def projection_widget_fixture(make_napari_viewer):
    make_napari_viewer()
    return projection_widget()


def test_add_projection_widget(make_napari_viewer):
    """Checks that the projection widget can be added inside a dock widget."""

    viewer = make_napari_viewer()
    num_dw = len(list(viewer.window._dock_widgets))
    viewer.window.add_plugin_dock_widget(
        plugin_name="napari-epitools",
        widget_name="Projection (selective plane)",
    )

    assert len(list(viewer.window._dock_widgets)) == num_dw + 1  # noqa: S101

    viewer.close()
    # Run a full garbage collection here so that any remaining viewer
    # and related instances are removed for future tests that may use
    # make_napari_viewer.
    gc.collect()


@pytest.mark.skip(reason="unfinished")
def test_projection_widget_run_button(projection_widget_fixture, sample_data):
    pbar = magicgui.widgets.ProgressBar()
    with patch(
        "napari_epitools.projection.analysis.calculate_projection"
    ) as calculate_projection:
        mock_projection = np.zeros((sample_data.shape[1], sample_data.shape[1]))
        calculate_projection.return_value = mock_projection
        projection_widget_fixture.viewer.add_image(sample_data)
        projection_widget_fixture(pbar, sample_data)
        projection_widget_fixture.call_button.clicked()


def test_calculate_projection(sample_data):
    with patch("napari_epitools.analysis.projection._interpolate") as interp:
        mock_interpolation = np.zeros((sample_data.shape[1], sample_data.shape[2]))
        interp.return_value = mock_interpolation
        projection = calculate_projection(
            sample_data,
            SMOOTHING_RADIUS,
            SURFACE_SMOOTHNESS_1,
            SURFACE_SMOOTHNESS_2,
            CUT_OFF_DISTANCE,
        )
        assert projection.ndim == PROJECTION_NDIM  # noqa: S101
        assert projection.shape == (  # noqa: S101
            1,  # single frame in the timeseries
            sample_data.shape[1],
            sample_data.shape[2],
        )
