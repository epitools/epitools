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
) -> npt.NDArray[np.float64]:
    """
    Segment cells in images with fluorescently marked membranes.
    The two sigma parameters allow tuning the segmentation result. The first
    sigma controls how close detected cells can be (spot_sigma) and the second controls
    how precise segmented objects are outlined (outline_sigma). Under the hood, this
    filter applies two Gaussian blurs, local minima detection and a seeded watershed.
    See also
    --------
    .. [1] https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html # noqa E501
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
) -> npt.NDArray[np.float64]:
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
    seeds = [r.centroid for r in spots_stats]

    # measure intensities
    stats = regionprops(labels, image)
    intensities = [r.mean_intensity for r in stats]

    # filter labels with low intensity
    new_label_indices, _, _ = relabel_sequential(
        (np.asarray(intensities) > minimum_intensity) * np.arange(labels.max())
    )
    new_label_indices = np.insert(new_label_indices, 0, 0)
    new_labels = np.take(np.asarray(new_label_indices, np.uint32), labels)

    return seeds, new_labels
