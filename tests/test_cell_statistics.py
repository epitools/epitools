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
    1, 2, or more elements for a 2-D timeseries (TYX, T=1, single-frame 2D branch).

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


def test_id_neighbours_contains_plain_ints():
    """Regression test: id_neighbours must contain plain Python ints, not np.int64.

    Previously neighbour IDs were stored as np.int64, causing CSV output like
    '[np.int64(31), np.int64(552), ...]' instead of '[31, 552, ...]'.
    """
    np.random.seed(0)
    image = np.random.randint(0, 255, (1, 20, 20), dtype=np.uint8)
    labels = np.zeros((1, 20, 20), dtype=int)
    # Place two adjacent cells so they are neighbours
    labels[0, 2:12, 2:10] = 1
    labels[0, 2:12, 10:18] = 2

    cell_statistics, _ = calculate_cell_statistics(image, labels, np.array([1.0, 1.0]))

    for frame_stats in cell_statistics:
        neighbours = frame_stats["id_neighbours"]
        for neighbour_list in neighbours:
            if neighbour_list is not None:
                for n in neighbour_list:
                    assert type(n) is int, (
                        f"Expected plain int, got {type(n).__name__!r} ({n!r})"
                    )


@pytest.mark.parametrize(
    "image_shape, pixel_spacing",
    [
        # TYX with T>1 (3D branch, 2-D frames) - yx_spacing (2-elem)
        ((3, 20, 20), np.array([1.0e-6, 1.0e-6])),
        # TYX with T>1 (3D branch, 2-D frames) - 1-elem spacing
        ((3, 20, 20), np.array([1.0e-6])),
        # TYX with T>1 (3D branch, 2-D frames) - image.scale with T dim (3-elem)
        ((3, 20, 20), (1.0, 1.0e-6, 1.0e-6)),
        # TZYX with Z>1 (3D branch, 3-D frames) - yx_spacing (2-elem)
        ((2, 3, 20, 20), np.array([1.0e-6, 1.0e-6])),
        # TZYX with Z>1 (3D branch, 3-D frames) - 1-elem spacing
        ((2, 3, 20, 20), np.array([1.0e-6])),
        # TZYX with Z>1 (3D branch, 3-D frames) - image.scale with T+Z+Y+X (4-elem)
        ((2, 3, 20, 20), (1.0, 1.0e-6, 1.0e-6, 1.0e-6)),
    ],
)
def test_calculate_cell_statistics_3d_branch_spacing(image_shape, pixel_spacing):
    """Regression test: cell statistics must not raise for multi-frame/multi-Z
    images (3D branch) regardless of how many elements pixel_spacing has.

    Previously ``pixel_spacing = pixel_spacing[1:]`` produced a 1-element array
    when yx_spacing (2 elements) was passed, causing::

        ValueError: spacing isn't a scalar nor a sequence of shape (N,), got [1.e-06].
    """
    np.random.seed(0)
    image = np.random.randint(0, 255, image_shape, dtype=np.uint8)
    labels = np.zeros(image_shape, dtype=int)
    labels[..., 2:8, 2:8] = 1
    labels[..., 12:18, 12:18] = 2

    # Should not raise
    cell_statistics, graphs = calculate_cell_statistics(image, labels, pixel_spacing)

    n_frames = image_shape[0]
    assert len(cell_statistics) == n_frames
    assert len(graphs) == n_frames
