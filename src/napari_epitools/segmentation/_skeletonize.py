import numpy as np
from numpy import ndarray


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
        time_points = 1
    elif cell_labels.ndim == 3:
        time_points = cell_labels.shape[0]
    elif cell_labels.ndim > 3:
        raise ValueError(
            "Input data should be a time series with a single z plane"
        )

    cell_outlines = []
    for t in range(time_points):
        frame = cell_labels[t, :, :]
        gradient = np.gradient(frame)
        cell_outlines.append(
            (frame > 0.0)
            & (np.square(gradient[0]) + np.square(gradient[1]) > 0.0)
        )

    return np.stack(cell_outlines)


if __name__ == "__main__":
    import h5py

    f = h5py.File("sample_data/8bitDataset/Benchmark/SegResults.mat")
    cell_labels = f.get("CLabels")

    skels = skeletonize(cell_labels)
    print(f"{skels.shape=}")
    from matplotlib import pyplot as plt

    plt.figure()
    plt.imshow(skels[5])
    plt.show()
