import numpy as np
from numpy import ndarray


def _get_lines(cell_labels):
    gradient = np.gradient(cell_labels)
    return (cell_labels > 0.0) & (
        np.square(gradient[0]) + np.square(gradient[1]) > 0.0
    ).astype(int)


def skeletonize(cell_labels: ndarray) -> ndarray:
    """Skeletonize a labelled image by calculating the gradient.

    Parameters
    ----------
    cell_labels : np.ndarray
        Labelled (possibly time series) image

    Returns
    -------
    np.ndarray
        The skeletonized labelled image

    Raises
    ------
    ValueError
        If the input data has more than 3 dimensions.
    """
    if cell_labels.ndim == 2:
        return _get_lines(cell_labels)

    elif cell_labels.ndim == 3:
        time_points = cell_labels.shape[0]
        cell_outlines = []
        for t in range(time_points):
            frame = cell_labels[t]
            cell_outlines.append(_get_lines(frame))

        return np.stack(cell_outlines)

    elif cell_labels.ndim > 3:
        raise ValueError(
            "Input data should be a time series with a single z plane"
        )
