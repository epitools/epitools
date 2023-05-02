from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from skimage.io import imread

from epitools.analysis import calculate_projection
from epitools.main import create_projection_widget

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
def projection_widget_fixture():
    return create_projection_widget()


def test_add_projection_widget(make_napari_viewer):
    """Checks that the projection widget can be added inside a dock widget."""

    viewer = make_napari_viewer()
    num_dw = len(list(viewer.window._dock_widgets))
    viewer.window.add_plugin_dock_widget(
        plugin_name="epitools",
        widget_name="Projection (selective plane)",
    )

    assert len(list(viewer.window._dock_widgets)) == num_dw + 1


@pytest.mark.skip(reason="unfinished")
def test_projection_widget_run_button(
    make_napari_viewer, projection_widget_fixture, sample_data
):
    with patch("epitools.analysis.calculate_projection") as calculate_projection:
        mock_projection = np.zeros((sample_data.shape[1], sample_data.shape[1]))
        calculate_projection.return_value = mock_projection
        viewer = make_napari_viewer()
        viewer.add_image(sample_data)
        projection_widget_fixture.run.clicked()


def test_calculate_projection(sample_data):
    with patch("epitools.analysis.projection._interpolate") as interp:
        mock_interpolation = np.zeros((sample_data.shape[1], sample_data.shape[2]))
        interp.return_value = mock_interpolation
        projection = calculate_projection(
            sample_data,
            SMOOTHING_RADIUS,
            SURFACE_SMOOTHNESS_1,
            SURFACE_SMOOTHNESS_2,
            CUT_OFF_DISTANCE,
        )
        assert projection.ndim == PROJECTION_NDIM
        assert projection.shape == (
            1,  # single frame in the timeseries
            sample_data.shape[1],
            sample_data.shape[2],
        )
