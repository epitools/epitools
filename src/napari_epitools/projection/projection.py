import numpy as np
from magicgui import magic_factory, widgets
from napari.qt.threading import FunctionWorker, thread_worker
from napari.types import ImageData
from skimage.filters import gaussian
from typing_extensions import Annotated

from napari_epitools.analysis._gridfit import gridfit


def _smooth(img, smoothing_radius=0.3):
    zsize = img.shape[0]
    smoothed = np.zeros(img.shape)
    for z in range(zsize):
        smoothed[z, :, :] = gaussian(img[z, :, :], sigma=smoothing_radius)

    return smoothed


def _interpolate(depthmap, imsize, smoothness):
    # plt.figure()
    # plt.imshow(depthmap)
    indices = np.nonzero(depthmap)
    vals = depthmap[indices]

    xnodes, ynodes = imsize[2], imsize[1]
    X, Y = np.meshgrid(np.arange(0, xnodes, 1), np.arange(0, ynodes, 1))

    # return X, Y, griddata(indices, vals, (X, Y), method="nearest", fill_value=0.0)
    return gridfit(indices[1], indices[0], vals, xnodes, ynodes, smoothness)


@magic_factory(pbar={"visible": False, "max": 0, "label": "working..."})
def projection_widget(
    pbar: widgets.ProgressBar,
    imstack: ImageData,
    smoothing_radius: Annotated[
        float, {"min": 0.0, "max": 2.0, "step": 0.1}
    ] = 0.2,
    surface_smoothness_1: Annotated[
        int, {"min": 0, "max": 100, "step": 1}
    ] = 50,
    surface_smoothness_2: Annotated[
        int, {"min": 0, "max": 100, "step": 1}
    ] = 50,
    cut_off_distance: Annotated[int, {"min": 0, "max": 5, "step": 1}] = 2,
) -> FunctionWorker[ImageData]:
    """Z projection using image interpolation.

    Parameters
    ----------
    pbar: magicgui.widget
        Progressbar widget
    imstack : ndarray
        numpy ndarray representation of 3D image stack
    smoothing_radius : float
        kernel radius for gaussian blur to apply before estimating the surface
    [0.1 - 5]
    surface_smoothness_1: int
        Surface smoothness for 1st gridFit(c) estimation, the smaller the smoother
        [30 - 100]
    surface_smoothness_2: int
        Surface smoothness for 3nd gridFit(c) estimation, the smaller the smoother
        [20 - 50]
    cut_off_distance : int
        Cutoff distance in z-planes from the 1st estimated surface [1 - 3]
    """

    @thread_worker(connect={"returned": pbar.hide})
    def calculate_projection() -> ImageData:
        # image dimensions are z, y, x
        imsize = imstack.shape
        I1 = _smooth(imstack, smoothing_radius)

        vm1 = I1.max(axis=0)
        depthmap = I1.argmax(axis=0)

        confidencemap = imsize[0] * vm1 / sum(I1, 0)
        c = confidencemap[:]
        confthres = np.median(c[c > np.median(c)])

        # keep only the brightest surface points (intensity in 1 quartile)
        # assumed to be the surface of interest
        mask = confidencemap > confthres
        depthmap2 = depthmap * mask
        xg1, yg1, zg1 = _interpolate(depthmap2, imsize, surface_smoothness_1)

        # given the hight locations of the surface (zg1) compute the difference
        # towards the 1st quartile location (depthmap2), ignore the rest (==0);
        # the result reflects the distance (abs) between estimate and points.
        depthmap3 = np.abs(zg1 - depthmap2.astype("float64"))
        depthmap3[np.ma.where(depthmap2 == 0)] = 0

        # only keep points which are relatively close to our first estimate,
        # i.e. below the threshold. TIP: if the first estimate is too detailed(~=smooth)
        # the points from the peripodial membrane will not be eliminated since
        # the surface approximated them well. Increase the smoothness to prevent this.
        mask = depthmap3 < cut_off_distance
        depthmap4 = depthmap2 * mask

        # TIP: depthmap4 should only contain signal of interest at this point.

        # --- 2nd iteration -
        # compute a better more detailed estimate with the filtered list (depthmap4)
        # this is to make sure that the highest intensity points will be
        # selected from the correct surface (The coarse grained estimate could
        # potentially approximate the origin of the point to another plane)
        xg2, yg2, zg2 = _interpolate(depthmap4, imsize, surface_smoothness_2)

        # ----- creating projected image from interpolated surface estimation ------

        projected_image = np.zeros((imsize[1], imsize[2]), dtype=imstack.dtype)
        z_origin_map = np.zeros((imsize[1], imsize[2]), dtype="uint8")

        # zg2_mask = zg2 > 0
        # z_coordinates = z_origin_map * zg2_mask
        # z_origin_map[zg2_mask] = np.round(zg2[zg2_mask])
        # projected_image[zg2_mask] = imstack[z_coordinates, zg2_mask]

        for y in range(imsize[1]):
            for x in range(imsize[2]):
                if zg2[y, x] > 0:
                    z_coordinate = int(round(zg2[y, x]))
                    z_origin_map[y, x] = z_coordinate
                    projected_image[y, x] = imstack[z_coordinate, y, x]

        return projected_image

    pbar.show()
    return calculate_projection()
