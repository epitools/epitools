from unittest.mock import patch

import numpy as np
import pytest

import napari

import epitools._sample_data
from epitools.analysis import calculate_cell_statistics


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
def viewer_with_image(
    make_napari_viewer,
    projected_image,
    seeds_and_labels,
) -> napari.Viewer:
    seeds, labels = seeds_and_labels
    viewer = make_napari_viewer()

    viewer.add_layer(projected_image)
    viewer.add_layer(labels)
    viewer.add_layer(seeds)

    return viewer


def test_add_cell_statistics_widget(make_napari_viewer):
    """Checks that the cell statistics widget can be added inside a dock widget."""

    viewer = make_napari_viewer()
    num_dw = len(list(viewer.window._dock_widgets))
    viewer.window.add_plugin_dock_widget(
        plugin_name="epitools",
        widget_name="Cell statistics",
    )

    assert len(list(viewer.window._dock_widgets)) == num_dw + 1


def test_segmentation_widget_run_button(
    viewer_with_image,
    seeds_and_labels,
):
    dock_widget, container = viewer_with_image.window.add_plugin_dock_widget(
        plugin_name="epitools",
        widget_name="Cell statistics",
    )

    reference_seeds, reference_labels = seeds_and_labels
    reference_stats = reference_labels.metadata["cell_statistics"]

    # Check cells layer has no features (i.e. no cell stats)
    # before pressing the 'Run' button
    image_layer, cells_layer, seeds_layer = viewer_with_image.layers
    assert not viewer_with_image.layers["Cells"].features.size

    # use saved image data so we don't run the segmentation analysis
    # when the button is pressed
    with patch(
        "epitools.analysis.calculate_cell_statistics"
    ) as calculate_cell_statistics:
        calculate_cell_statistics.return_value = (
            reference_stats,
            None,
        )  # ignore graph data
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
def test_calculate_cell_statistics(projected_image, seeds_and_labels):
    reference_seeds, reference_labels = seeds_and_labels
    reference_stats = reference_labels.metadata["cell_statistics"]

    cell_statistics, graphs = calculate_cell_statistics(
        image=projected_image.data,
        labels=reference_labels.data,
        pixel_spacing=reference_labels.metadata["spacing"],
    )

    assert np.allclose(
        cell_statistics,
        reference_stats.to_numpy(),
    )
