import numpy as np
import numpy.typing as npt

CELL_IS_2D = 2
ZERO = 0


def _get_outlines(
    cell_labels: npt.NDArray[np.float64],
) -> npt.NDArray[np.int64]:
    """Calculate the boundary line of set of cell labels.

    Args:
        cell_labels:
            Labelled (possibly time series) image.

    Returns:
        Outlines as a numpy array.
    """
    if cell_labels.ndim > CELL_IS_2D:
        cell_labels = np.squeeze(cell_labels)

    gradient = np.gradient(cell_labels)
    not_background = cell_labels > ZERO
    positive_gradient = np.square(gradient[0]) + np.square(gradient[1]) > ZERO
    return (not_background & positive_gradient).astype(np.int64)


def skeletonize(
    cell_labels: npt.NDArray[np.float64],
) -> npt.NDArray[np.int64]:
    """Skeletonize a labelled image by calculating the gradient.

    Args:
        cell_labels:
            Labelled (possibly time series) image.

    Raises:
        ValueError:
            If the input data has more than 3 dimensions.

    Returns:
        The skeletonized labelled image.
    """
    if cell_labels.ndim == CELL_IS_2D:
        cell_outlines = _get_outlines(cell_labels)

    elif cell_labels.ndim > CELL_IS_2D:
        time_points = cell_labels.shape[0]
        cell_outlines = np.zeros(cell_labels.shape, dtype=int)
        for t in range(time_points):
            frame = cell_labels[t]
            cell_outlines[t] = _get_outlines(frame)

    return cell_outlines
