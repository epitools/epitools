from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

from epitools.analysis.cell_statistics import calculate_cell_statistics


def test_add_cell_statistics_widget(
    make_napari_viewer: Callable,
):
    """Checks that the cell statistics widget can be added inside a dock widget."""

    viewer = make_napari_viewer()
    num_dw = len(viewer.window._dock_widgets)
    viewer.window.add_plugin_dock_widget(
        plugin_name="epitools",
        widget_name="Cell statistics",
    )

    assert len(viewer.window._dock_widgets) == num_dw + 1


@pytest.mark.parametrize(
    "pixel_spacing",
    [
        np.array([1.0e-6]),  # 1-element (bug case: isotropic from PartSeg metadata)
        np.array([1.0e-6, 1.0e-6]),  # 2-element (standard YX)
        (1.0, 1.0e-6, 1.0e-6),  # 3-element (e.g. from image.scale with T dim)
    ],
)
def test_calculate_cell_statistics_spacing_shapes(pixel_spacing):
    """Regression test: cell statistics must not raise when pixel_spacing has
    1, 2, or more elements for a 2-D timeseries (TYX).

    Previously a 1-element spacing caused::

        ValueError: spacing isn't a scalar nor a sequence of shape (2,), got [1.e-06].
    """
    np.random.seed(0)
    image = np.random.randint(0, 255, (1, 20, 20), dtype=np.uint8)
    labels = np.zeros((1, 20, 20), dtype=int)
    labels[0, 2:8, 2:8] = 1
    labels[0, 12:18, 12:18] = 2

    # Should not raise
    cell_statistics, graphs = calculate_cell_statistics(image, labels, pixel_spacing)

    assert len(cell_statistics) == 1
    assert len(graphs) == 1
