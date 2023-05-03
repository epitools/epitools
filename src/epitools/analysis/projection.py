import itertools

import numpy as np
import numpy.typing as npt
from scipy.interpolate import griddata
from skimage.filters import gaussian

THREE_DIMENSIONAL = 3


def _smooth(
    img: npt.NDArray[np.float64], smoothing_radius: float = 0.3
) -> npt.NDArray[np.float64]:
    """Gaussian smoothing of each z plane in the image stack

    Args:
        img: The input image stack.
        smoothing_radius: Standard deviation passed to gaussian function.

    Returns:
        The smoothed image stack.
    """
    t_size, z_size = img.shape[:2]
    smoothed = np.zeros(img.shape)
    for t, z in itertools.product(range(t_size), range(z_size)):
        smoothed[t, z] = gaussian(
            img[t, z], sigma=smoothing_radius, preserve_range=True
        )

    return smoothed


def _interpolate(
    max_indices: npt.NDArray[np.float64],
    x_size: int,
    y_size: int,
    smoothness: int,
) -> npt.NDArray[np.float64]:
    """Interpolate the z coordinate map using gridfit

    Args:
        max_indices:
            z coordinate map.
        x_size:
            Image size in x dimension (number of columns)
        y_size:
            Image size in y dimension (number of rows)
        smoothness:
            How much to smooth the interpolation.

    Returns:
        Interpolated z coordinates.
    """
    indices = np.nonzero(max_indices)
    if not np.any(indices):
        return max_indices

    vals = max_indices[indices].astype(np.float64)
    x, y = np.meshgrid(np.arange(x_size), np.arange(y_size))
    interp_vals = griddata(indices, vals, (y, x), method="nearest")
    return gaussian(interp_vals, sigma=smoothness)


def _calculate_projected_image(
    imstack: npt.NDArray[np.float64], z_interp: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Create the projected image from the non-zero elements
    of the interpolated z coordinates.

    Args:
        imstack:
            The input image z stack.
        z_interp:
            Output from griddata function.

    Returns:
        The final projected image.
    """

    # make a container for the projected image
    z_size, y_size, x_size = imstack.shape
    projected_image = np.zeros((y_size, x_size), dtype=imstack.dtype)

    # take the non-zero elements of the interpolation and round
    mask = (z_interp > 0) & (z_interp < z_size - 1)
    z_coordinates = np.round(z_interp[mask]).astype(np.int64)
    projected_image[mask] = imstack[z_coordinates, mask]
    return projected_image


def calculate_projection(
    input_image: npt.NDArray[np.float64],
    smoothing_radius: float,
    surface_smoothness_1: int,
    surface_smoothness_2: int,
    cut_off_distance: int,
) -> npt.NDArray[np.float64]:
    """Z projection using image interpolation.

    Args:
        input_image:
            Numpy ndarray representation of 3D image stack.
        smoothing_radius:
            Kernel radius for gaussian blur to apply before estimating the surface.
        surface_smoothness_1:
            Surface smoothness for 1st griddata estimation, larger means smoother.
        surface_smoothness_2:
            Surface smoothness for 3nd gridFit(c) estimation, larger means smoother.
        cut_off_distance:
            Cutoff distance in z-planes from the 1st estimated surface.

    Returns:
        Stack projected onto a single plane.
    """
    if input_image.ndim == THREE_DIMENSIONAL:
        # Assume we have a stack of images at a single point in time (ZYX)
        # Expand so we have TZYX
        input_image = np.expand_dims(input_image, axis=0)

    t_size, z_size, y_size, x_size = input_image.shape
    smoothed_imstack = _smooth(input_image.astype(np.float64), smoothing_radius)

    t_interp = np.zeros((t_size, y_size, x_size))  # remove the z-axis when projecting
    for t in range(t_size):
        smoothed_t = smoothed_imstack[t]
        max_intensity = smoothed_t.max(axis=0)
        max_indices = smoothed_t.argmax(axis=0)

        confidencemap = z_size * max_intensity / np.sum(smoothed_t, axis=0)
        np.nan_to_num(confidencemap, copy=False)
        confthres = np.median(confidencemap[confidencemap > np.median(confidencemap)])

        # keep only the brightest surface points (intensity in 1 quartile)
        # assumed to be the surface of interest
        mask = confidencemap > confthres
        max_indices_confthres = max_indices * mask
        z_interp = _interpolate(
            max_indices_confthres, x_size, y_size, surface_smoothness_1
        )

        # given the height locations of the surface (z_interp) compute the difference
        # towards the 1st quartile location (max_indices_confthres), ignore the rest
        # (==0); the result reflects the distance (abs) between estimate and points.
        max_indices_diff = np.abs(z_interp - max_indices_confthres.astype("float64"))
        max_indices_diff[np.where(max_indices_confthres == 0)] = 0

        # only keep points which are relatively close to our first estimate
        mask = max_indices_diff < cut_off_distance
        max_indices_cut = max_indices_confthres * mask

        # --- 2nd iteration -
        # compute a better more detailed estimate with the filtered list
        # (max_indices_cut) this is to make sure that the highest intensity points will
        # be selected from the correct surface (The coarse grained estimate could
        # potentially approximate the origin of the point to another plane)
        z_interp = _interpolate(max_indices_cut, x_size, y_size, surface_smoothness_2)

        t_interp[t] = _calculate_projected_image(input_image[t], z_interp)

    return t_interp
