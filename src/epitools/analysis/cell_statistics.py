"""Calculate cell statistics --- :mod:`epitools.analysis.cell_statistics`
=========================================================================

This module contains functions for calculating region-based properties
of labelled images using skimage.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from scipy import ndimage

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
THREE_DIMENSIONAL = 3


def calculate_cell_statistics(
    image: napari.types.ImageData,
    labels: napari.types.LabelsData,
    pixel_spacing: tuple[float],
    id_cells: list[int] | None = None,
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
        image : napari.types.ImageData
            Timeseries of images (TYX or TZYX) for which to calculate the cell
            statistics.
        labels : napari.types.LabelsData
            Labelled input image, must be the same shape as ``image``.
            Labels with value 0 are ignored.
        pixel_spacing : tuple[float]
            The pixel spacing in each dimension of the image.
        id_cells : list[int], optional
            The cell IDs to calculate statistics for. If not provided, statistics
            will be calculated for all cells.

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
    pixel_spacing: tuple[float, ...],
    id_cells: list[int] | None = None,
) -> list[dict[str, npt.NDArray]]:
    """Calculate cell properties using skimage regionprops"""

    # Analyse the image in 3D if it is 4D
    if isinstance(image, np.ndarray) and (
        (image.ndim == FOUR_DIMENSIONAL and image.shape[1] > 1)
        or (image.ndim == THREE_DIMENSIONAL and image.shape[0] > 1)
    ):
        pixel_spacing = pixel_spacing[1:]

        # # Ensure that pixel_spacing id a sequence of floats
        # if pixel_spacing.shape != (image.ndim,):
        #     # Transform to a sequence of floats
        #     pixel_spacing = tuple(float(value) for value in pixel_spacing)

        # Properties to check in 3D
        properties = [
            "area",
            "label",
        ]
    else:
        # remove z axis if necessary
        if isinstance(image, np.ndarray) and image.ndim == FOUR_DIMENSIONAL:
            image = image[:, 0]
        if isinstance(labels, np.ndarray) and labels.ndim == FOUR_DIMENSIONAL:
            labels = labels[:, 0]

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

        id_neighbours: list[list[int] | None] = []
        for index in indices:
            new_id_neighbours = list(graph.neighbors(index))

            if len(new_id_neighbours) == 0:
                id_neighbours.append(None)
            else:
                id_neighbours.append(list(graph.neighbors(index)))

        cell_statistics[frame]["id_neighbours"] = np.array(id_neighbours, dtype=object)


def calculate_quality_metrics(
    labels: napari.types.LabelsData,
    percentage_of_zslices: float,
) -> dict[str, list[int]]:
    """Calculate quality metrics for a 3D image.

    The quality metric calculated is

    Args:
        labels : napari.types.LabelsData
            Labelled input image, must be the same shape as ``image``.
            Labels with value 0 are ignored.
        percentage_of_zslices : float
            Percentage of z-slices to use for calculating the quality metrics.

    Returns:
        dict[str, np.NDArray]
            Dictionary containing the quality metrics.

    """
    quality_metrics = {}

    # Calculate the quality metrics
    correct_cells, wrong_cells = _count_correct_cells(labels, percentage_of_zslices)
    logger.info(f"Number of correct cells: {len(correct_cells)}")
    logger.info(f"Number of wrong cells: {len(wrong_cells)}")
    quality_metrics["correct_cells"] = correct_cells
    quality_metrics["wrong_cells"] = wrong_cells

    return quality_metrics


def _count_correct_cells(
    labels: napari.types.LabelsData,
    min_percentage: float,
) -> tuple[list[int], list[int]]:
    """
    Count the number of cells that are present in a 'min_percentage' of slices.
    Originally developed by Giulia Paci

      Args:
          labels : napari.types.LabelsData
              Labelled input image, must be the same shape as ``image``.
              Labels with value 0 are ignored.
          min_percentage : float
              Percentage of z-slices to use for calculating the quality metrics.
              The percentage of slices that a cell must be present in to
              be considered correct.

      Returns:
          tuple[list[int], list[int]]
                List of the correct cells and the wrong cells.
    """
    # Define a constant for the magic value
    max_num_objects = 3

    # Define the position of the Z axis
    z_axis = 1 if labels.ndim == FOUR_DIMENSIONAL else 0

    if isinstance(labels, np.ndarray):
        z_planes = labels.shape[z_axis]

    # Minimum of number of slices for a cell to be correct
    target_n_planes = (min_percentage / 100) * z_planes

    # Count the number of good cells
    unique_ids = np.unique(labels)
    list_good = []
    list_bad = []

    # Precompute the structure array
    structure = np.ones((3, 3, 3))

    # Loop
    for cell_id in unique_ids:
        if cell_id == 0:
            continue

        # Get the voxels of the current cell
        current_img = labels == cell_id

        # Check if they are connected by using connected components
        _, num_objects = ndimage.label(current_img, structure=structure)

        # Use the constant in the comparison
        if num_objects > max_num_objects:
            # Remove small objects from current_img
            current_img = ndimage.binary_opening(current_img, structure=structure)
            # and recompute the connected components
            _, num_objects = ndimage.label(current_img, structure=structure)

            # Check again
            if num_objects > max_num_objects:
                list_bad.append(cell_id)
                continue

        # Get the unique Z position of the voxels
        unique_z_position = np.unique(np.where(current_img)[z_axis])

        # Count the number of slices that the cell is present in
        if len(unique_z_position) > target_n_planes:
            list_good.append(cell_id)
        else:
            list_bad.append(cell_id)

    return list_good, list_bad
