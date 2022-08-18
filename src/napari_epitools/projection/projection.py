import numpy as np
import numpy.typing as npt
from magicgui import magic_factory, widgets
from napari.qt.threading import FunctionWorker, thread_worker
from napari.types import ImageData
from scipy.interpolate import griddata
from skimage.filters import gaussian
from typing_extensions import Annotated


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
    zsize = img.shape[0]
    smoothed = np.zeros(img.shape)
    for z in range(zsize):
        smoothed[z, :, :] = gaussian(
            img[z, :, :], sigma=smoothing_radius, preserve_range=True
        )

    return smoothed


def _interpolate(
    depthmap: npt.NDArray[np.float64],
    imsize: npt.NDArray[np.float64],
    smoothness: int,
) -> npt.NDArray[np.float64]:
    """Interpolate the z coordinate map using gridfit

    Args:
        depthmap:
            z coordinate map.
        imsize:
            Input image dimensions.
        smoothness:
            How much to smooth the interpolation.

    Returns:
        Interpolated z coordinates.
    """
    indices = np.nonzero(depthmap)
    vals = depthmap[indices].astype(np.float64)
    xnodes, ynodes = imsize[2], imsize[1]
    X, Y = np.meshgrid(np.arange(xnodes), np.arange(ynodes))
    interp_vals = griddata(indices, vals, (Y, X), method="nearest")
    return gaussian(interp_vals, sigma=smoothness)


def _calculate_projected_image(
    imstack: npt.NDArray[np.float64], z_interpolation: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Create the projected image from the non-zero elements
    of the interpolated z coordinates.

    Args:
        imstack:
            The input image z stack.
        z_interpolation:
            Output from griddata function.

    Returns:
        The final projected image.
    """

    # make a container for the projected image
    imsize = imstack.shape
    projected_image = np.zeros((imsize[1], imsize[2]), dtype=imstack.dtype)

    # take the non-zero elements of the interpolation and round
    mask = (z_interpolation > 0.0) & (z_interpolation < float(imsize[0] - 1))
    z_coordinates = np.round(z_interpolation[mask]).astype(np.int64)
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
    imsize = input_image.shape
    I1 = _smooth(input_image, smoothing_radius)

    vm1 = I1.max(axis=0)
    depthmap = I1.argmax(axis=0)

    confidencemap = imsize[0] * vm1 / sum(I1, 0)
    c = confidencemap[:]
    confthres = np.median(c[c > np.median(c)])

    # keep only the brightest surface points (intensity in 1 quartile)
    # assumed to be the surface of interest
    mask = confidencemap > confthres
    depthmap2 = depthmap * mask
    zg1 = _interpolate(depthmap2, imsize, surface_smoothness_1)

    # given the hight locations of the surface (zg1) compute the difference
    # towards the 1st quartile location (depthmap2), ignore the rest (==0);
    # the result reflects the distance (abs) between estimate and points.
    depthmap3 = np.abs(zg1 - depthmap2.astype("float64"))
    depthmap3[np.where(depthmap2 == 0)] = 0

    # only keep points which are relatively close to our first estimate
    mask = depthmap3 < cut_off_distance
    depthmap4 = depthmap2 * mask

    # --- 2nd iteration -
    # compute a better more detailed estimate with the filtered list (depthmap4)
    # this is to make sure that the highest intensity points will be
    # selected from the correct surface (The coarse grained estimate could
    # potentially approximate the origin of the point to another plane)
    zg2 = _interpolate(depthmap4, imsize, surface_smoothness_2)
    return _calculate_projected_image(input_image, zg2)


@magic_factory(pbar={"visible": False, "max": 0, "label": "working..."})
def projection_widget(
    pbar: widgets.ProgressBar,
    input_image: ImageData,
    smoothing_radius: Annotated[
        float, {"min": 0.0, "max": 2.0, "step": 0.1}
    ] = 0.2,
    surface_smoothness_1: Annotated[int, {"min": 0, "max": 10, "step": 1}] = 5,
    surface_smoothness_2: Annotated[int, {"min": 0, "max": 10, "step": 1}] = 5,
    cut_off_distance: Annotated[int, {"min": 0, "max": 5, "step": 1}] = 2,
) -> FunctionWorker[ImageData]:
    """Z projection using image interpolation.

    Args:
        pbar:
            Progressbar widget
        input_image:
            Numpy ndarray representation of 3D image stack
        smoothing_radius:
            Kernel radius for gaussian blur to apply before estimating the surface.
        surface_smoothness_1:
            Surface smoothness for 1st griddata estimation, larger means smoother.
        surface_smoothness_2:
            Surface smoothness for 3nd griddata estimation, larger means smoother.
        cut_off_distance:
            Cutoff distance in z-planes from the 1st estimated surface.

    Raises:
        ValueError: When no image is loaded.

    Returns:
        Projected image as napari Image layer.
    """

    @thread_worker(connect={"returned": pbar.hide})
    def projection() -> ImageData:
        if input_image is None:
            pbar.hide()
            raise ValueError("Load an image first")

        return calculate_projection(
            input_image,
            smoothing_radius,
            surface_smoothness_1,
            surface_smoothness_2,
            cut_off_distance,
        )

    pbar.show()
    return projection()
