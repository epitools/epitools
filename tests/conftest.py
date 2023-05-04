from __future__ import annotations

from typing import Callable

import pytest

import napari

import epitools._sample_data


@pytest.fixture(scope="function")
def image() -> napari.layers.Image:
    """Load an sample 3D image from a tif file and convert to a Napari Image layer.

    Note, the Napari Image will be 4D (TZYX) with a single frame in the
    time dimension.
    """
    data, metadata, layer_type = epitools._sample_data.load_sample_data()[0]
    metadata["name"] = "Test Image"
    return napari.layers.Image(data, **metadata)


@pytest.fixture(scope="function")
def projected_image() -> napari.layers.Image:
    """Load a sample 2D image from a tif file and convert to a Napari Image layer.

    Note, the Napari Image will be 4D (TZYX) with single frame in the time dimension
    and a single slice in Z.
    """
    data, metadata, layer_type = epitools._sample_data.load_projected_data()[0]
    return napari.layers.Image(data, **metadata)


@pytest.fixture(scope="function")
def seeds_and_labels(
    make_napari_viewer: Callable,
) -> tuple[napari.layers.Points, napari.layers.Labels]:
    """Load a sample segmentaiton and the seeds use in generating the segmentation.

    Load sample cells and seeds and convert to Napari Points and Labels layers,
    respectively.

    Note, the Napari Labels will be 4D (TZYX) with a single frame in the time dimension
    and a single slice in Z.
    """

    seeds_layer_data, labels_layer_data = epitools._sample_data.load_segmented_data()
    seeds_data, seeds_kwargs, _ = seeds_layer_data
    labels_data, labels_kwargs, _ = labels_layer_data
    return (
        napari.layers.Points(seeds_data, **seeds_kwargs),
        napari.layers.Labels(labels_data, **labels_kwargs),
    )


@pytest.fixture(scope="function")
def viewer_with_image(
    make_napari_viewer: Callable,
    image: napari.layers.Image,
) -> napari.Viewer:
    """Create a Napari Viewer with a sample Image layer added to it."""

    viewer = make_napari_viewer()
    viewer.add_layer(image)

    return viewer


@pytest.fixture(scope="function")
def viewer_with_projected_image(
    make_napari_viewer: Callable,
    projected_image: napari.layers.Image,
) -> napari.Viewer:
    """Create a Napari Viewer with a sample Image layer added to it.

    The Image layer is a 2D projection of a 3D image.
    """

    viewer = make_napari_viewer()
    viewer.add_layer(projected_image)
    return viewer


@pytest.fixture(scope="function")
def viewer_with_segmentation(
    make_napari_viewer: Callable,
    projected_image: napari.layers.Image,
    seeds_and_labels: tuple[napari.layers.Points, napari.layers.Labels],
) -> napari.Viewer:
    """Create a Napari Viewer with a sample segmented dataset added to it.

    A 2D image is added to the viewer, along with the segmentation and seeds
    used in generating the segmentation.
    """

    seeds, labels = seeds_and_labels
    viewer = make_napari_viewer()

    viewer.add_layer(projected_image)
    viewer.add_layer(labels)
    viewer.add_layer(seeds)

    return viewer
