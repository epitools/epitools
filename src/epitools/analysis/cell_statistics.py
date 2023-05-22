"""Calculate cell statistics --- :mod:`epitools.analysis.cell_statistics`
=========================================================================

This module contains functions for calculating region-based properties
of labelled images using skimage.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt

    import napari.types

import networkx.exception
import numpy as np
import skimage.graph
import skimage.measure

import napari

logger = logging.getLogger(__name__)


FOUR_DIMENSIONAL = 4


def calculate_cell_statistics(
    image: napari.types.ImageData,
    labels: napari.types.LabelsData,
    pixel_spacing: tuple[float],
) -> tuple[list[dict[str, npt.NDArray]], list[skimage.graph.RAG]]:
    """Calculate the region based properties of a timeseries of segmented images.

    Currently the following statistics are calculated for each frame of the timeseries:
        - area
        - perimeter
        - number of neighbours

    ``skimage.measure.regionprops_table`` is used to calculated the area
    and perimeter.

    ``skimage.graph.RAG`` is used to create a graph of neighbouring cells
    at each frame, from which the number of neighbours of each cell is
    calculated.

    Args:
        image :
            Timeseries of images (TYX or TZYX) for which to calculate the cell
            statistics.
        labels :
            Labelled input image, must be the same shape as ``image``.
            Labels with value 0 are ignored.
        pixel_spacing :


    Note:
        It is assumed that the first dimension of both ``image`` and ``labels``
        corresponds to time.

    Returns:
        list[dict[str, np.NDArray]]
            List of dictionaries, where each dictionary contains the cell statistics
            for a single frame. The dictionary keys are: area; perimeter; neighbours.
        list[skimage.graph.RAG]
            List of the network graphs constructed for each frame of the timeseries

    """

    # Calculate cell statistics for each frame
    cell_statistics = _calculate_cell_statistics(image, labels, pixel_spacing)

    # Create graph of neighbouring cells at each frame
    graphs = _create_graphs(labels)

    # Calculate additional statistics from the graph
    # Update 'cell_statistics' in place
    _calculate_graph_statistics(
        cell_statistics,
        graphs,
    )

    return cell_statistics, graphs


def _calculate_cell_statistics(
    image: napari.types.ImageData,
    labels: napari.types.LabelsData,
    pixel_spacing: tuple[float],
) -> list[dict[str, npt.NDArray]]:
    """Calculate cell properties using skimage regionprops"""

    # TODO: fix the commented out properties
    # https://github.com/epitools/epitools/issues/98
    # contents of `skimage.measure._regionprops.PROP_VALS`
    properties = [
        "area_bbox",
        "area_convex",
        "area_filled",
        "area",
        "axis_major_length",
        "axis_minor_length",
        "bbox",
        "centroid_local",
        "centroid_weighted_local",
        "centroid_weighted",
        "centroid",
        # 'coords',
        "eccentricity",
        "equivalent_diameter_area",
        "euler_number",
        "extent",
        "feret_diameter_max",
        # 'image_convex',
        # 'image_filled',
        # 'image_intensity',
        # 'image',
        "inertia_tensor_eigvals",
        "inertia_tensor",
        "intensity_max",
        "intensity_mean",
        "intensity_min",
        "label",
        "moments_central",
        "moments_normalized",
        "moments_weighted_central",
        # 'moments_weighted_hu',
        "moments_weighted_normalized",
        "moments_weighted",
        "moments",
        "orientation",
        "perimeter_crofton",
        "perimeter",
        # 'slice',
        "solidity",
    ]

    # remove z axis if necessary
    image = image[:, 0] if image.ndim == FOUR_DIMENSIONAL else image
    labels = labels[:, 0] if labels.ndim == FOUR_DIMENSIONAL else labels

    cell_statistics = [
        skimage.measure.regionprops_table(
            label_image=frame_labels,
            intensity_image=frame_image,
            properties=properties,
            spacing=pixel_spacing,
        )
        for frame_labels, frame_image in zip(labels, image)
    ]

    # skimage uses 'label' for what napari calls 'index'
    for frame_stats in cell_statistics:
        frame_stats["index"] = frame_stats.pop("label")
        # TODO: required for these to work as maps, otherwise all the same colour
        # however, this means that the CSV has that value also
        # https://github.com/epitools/epitools/issues/97
        """
        frame_stats["perimeter"] *= 1e6  # convert to um
        frame_stats["area"] *= 1e12  # convert to um2
        """

    return cell_statistics


def _create_graphs(
    labels: napari.types.LabelsData,
) -> list[skimage.graph.RAG]:
    """Create graph of neighbouring cells"""

    graphs = [skimage.graph.RAG(frame_labels) for frame_labels in labels]

    # remove the background if it exists
    for index, graph in enumerate(graphs):
        try:
            graph.remove_node(0)
            message = f"Removing background node for graph at frame {index}"
            logger.log(level=2, msg=message)
        except networkx.exception.NetworkXError:
            message = f"No background node to remove for graph at frame {index}"
            logger.log(level=2, msg=message)

    return graphs


def _calculate_graph_statistics(
    cell_statistics: list[dict[str, npt.NDArray]],
    graphs: skimage.graph._rag.RAG,
) -> None:
    """Calculate additional cell statistics from graphs.

    Adds results directly to 'cell_statistics' dictionaries.
    """

    for frame, (stats, graph) in enumerate(zip(cell_statistics, graphs)):
        indices = stats["index"]

        num_neighbours = np.asarray([len(graph[index]) for index in indices])
        cell_statistics[frame]["num_neighbours"] = num_neighbours

        id_neighbours = [list(graph.neighbors(index)) for index in indices]
        cell_statistics[frame]["id_neighbours"] = id_neighbours


