from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from epitools.analysis import calculate_cell_statistics

if TYPE_CHECKING:
    from typing import Callable

    import napari


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


def test_segmentation_widget_run_button(
    viewer_with_segmentation: napari.Viewer,
    seeds_and_labels: tuple[napari.layers.Points, napari.layers.Labels],
):
    """
    Check that pressing the 'Run' button calculates cell statistics
    using the selected image and labels layers, and that the labels layer
    has it's features set to the calculated cell statistics.
    """

    dock_widget, container = viewer_with_segmentation.window.add_plugin_dock_widget(
        plugin_name="epitools",
        widget_name="Cell statistics",
    )

    reference_seeds, reference_labels = seeds_and_labels
    reference_stats = reference_labels.metadata["cell_statistics"]

    # Check cells layer has no features (i.e. no cell stats)
    # before pressing the 'Run' button
    image_layer, cells_layer, seeds_layer = viewer_with_segmentation.layers
    assert viewer_with_segmentation.layers["Cells"].features.size == 0

    # use saved image data so we don't run the segmentation analysis
    # when the button is pressed
    with patch(
        "epitools.analysis.calculate_cell_statistics"
    ) as calculate_cell_statistics:
        # return the cell stats, but ignore graph data as we're not using it
        calculate_cell_statistics.return_value = (reference_stats, None)
        container.run.clicked()

    assert cells_layer.features.size == reference_stats[0].size
    assert np.allclose(
        cells_layer.features.to_numpy(),
        reference_stats[0].to_numpy(),
    )


@pytest.mark.xfail(
    raises=NotImplementedError,
    reason="This test should be fixed by the changes in PR #67",
    stict=True,
)
def test_calculate_cell_statistics(
    projected_image: napari.layers.Image,
    seeds_and_labels: tuple[napari.layers.Points, napari.layers.Labels],
):
    reference_seeds, reference_labels = seeds_and_labels
    reference_stats: pd.DataFrame = reference_labels.metadata["cell_statistics"]

    cell_statistics, graphs = calculate_cell_statistics(
        image=projected_image.data,
        labels=reference_labels.data,
        pixel_spacing=reference_labels.metadata["spacing"],
    )

    assert np.allclose(
        pd.DataFrame.from_dict(cell_statistics[0]).set_index("index").to_numpy(),
        reference_stats.to_numpy(),
    )
