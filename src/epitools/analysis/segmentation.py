"""Cell segmentation --- :mod:`epitools.analysis.segmentation`
==============================================================

This module contains functions for segmenting 3D (ZYX) and 4D (TZXY)
images.

"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from skimage.filters import gaussian
from skimage.measure import label, regionprops
from skimage.morphology import local_minima
from skimage.segmentation import relabel_sequential, watershed


def local_minima_seeded_watershed(
    image: npt.NDArray[np.float64],
    spot_sigma: float = 10,
    outline_sigma: float = 0,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """
    Segment cells in images with fluorescently marked membranes.
    The two sigma parameters allow tuning the segmentation result. The first
    sigma controls how close detected cells can be (spot_sigma) and the second controls
    how precise segmented objects are outlined (outline_sigma). Under the hood, this
    filter applies two Gaussian blurs, local minima detection and a seeded watershed.
    See also
    --------
    .. [1] https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html
    """

    image = np.asarray(image)

    spot_blurred = gaussian(image, sigma=spot_sigma)

    spots = label(local_minima(spot_blurred))

    if outline_sigma == spot_sigma:
        outline_blurred = spot_blurred
    else:
        outline_blurred = gaussian(image, sigma=outline_sigma)

    return spots, watershed(outline_blurred, spots)


def thresholded_local_minima_seeded_watershed(
    image: npt.NDArray[np.float64],
    spot_sigma: float = 3,
    outline_sigma: float = 0,
    minimum_intensity: float = 500,
) -> tuple[list[list[Any]], npt.NDArray[np.uint32]]:
    """
    Segment cells in images with marked membranes that have a high signal intensity.
    The two sigma parameters allow tuning the segmentation result. The first sigma
    controls how close detected cells can be (spot_sigma) and the second controls how
    precise segmented objects are outlined (outline_sigma). Under the hood, this filter
    applies two Gaussian blurs, local minima detection and a seeded watershed.
    Afterwards, all objects are removed that have an average intensity below a given
    minimum_intensity
    """
    spots, labels = local_minima_seeded_watershed(
        image, spot_sigma=spot_sigma, outline_sigma=outline_sigma
    )

    # get seeds
    spots_stats = regionprops(spots)
    seeds = [list(r.centroid) for r in spots_stats]

    # measure intensities
    stats = regionprops(labels, image)
    intensities = [r.mean_intensity for r in stats]

    # filter labels with low intensity
    new_label_indices, _, _ = relabel_sequential(
        (np.asarray(intensities) > minimum_intensity) * np.arange(1, labels.max() + 1)
    )
    new_label_indices = np.insert(new_label_indices, 0, 0)
    new_labels = np.take(np.asarray(new_label_indices, np.uint32), labels)

    return seeds, new_labels


def calculate_segmentation(
    projection: npt.NDArray[np.float64],
    spot_sigma: float,
    outline_sigma: float,
    threshold: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Segment an image using a thresholded local-minima-seeded watershed algorith.

    This will segment cells in images with marked membranes that have a high signal
    intensity.

    The two sigma parameters allow tuning the segmentation result. Under the hood, this
    algorithm first applies two Gaussian blurs, then uses a local minima detection and
    seeded watershed. Afterwards, all objects are removed that have an average intensity
    below a given threshold.

    Args:
        projection:
            A projected image stack processed using Epitools.
        spot_sigma:
            Controls how close segmented cells can be - larger values result in more
            seeds used in the segmentation
        outline_sigma:
            Controls how precisely segmented cells are outlined - larger values result
            in more precise (rougher) cell boundaries.
        threshold:
            Cells with an average intensity below `threshold` are will be removed from
            the segmentation and treated as background.

    Returns:
        np.NDArray
            Seeds used in the segmentated
        np.NDArray
            Labels of the segmented image (0 corresponds to background)
    """
    t_size = projection.shape[0]

    seg_seeds = []
    seg_labels = []
    for t in range(t_size):
        seeds, labels = thresholded_local_minima_seeded_watershed(
            projection[t],
            spot_sigma,
            outline_sigma,
            threshold,
        )

        # seeds needs to include time dimension
        for s in seeds:
            s.insert(0, float(t))

        seg_seeds.append(np.array(seeds))
        seg_labels.append(labels)

    return np.vstack(seg_seeds), np.stack(seg_labels)
