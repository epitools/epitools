import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import gaussian
from skimage.io import imread

from napari_epitools.projection._gridfit import gridfit


def _smooth(img, smoothing_radius=0.3):
    zsize = img.shape[0]
    smoothed = np.zeros(img.shape)
    for z in range(zsize):
        smoothed[z, :, :] = gaussian(img[z, :, :], sigma=smoothing_radius)

    return smoothed


def _interpolate(depthmap, imsize):
    # plt.figure()
    # plt.imshow(depthmap)
    indices = np.nonzero(depthmap)
    vals = depthmap[indices]

    xnodes, ynodes = imsize[2], imsize[1]
    X, Y = np.meshgrid(np.arange(0, xnodes, 1), np.arange(0, ynodes, 1))

    # return X, Y, griddata(indices, vals, (X, Y), method="nearest", fill_value=0.0)
    smoothness = 50  # <- needs to be an input
    return gridfit(indices[1], indices[0], vals, xnodes, ynodes, smoothness)


def _plot_surface(x, y, z):
    plt.figure()
    ax = plt.axes(projection="3d")

    ax.plot_surface(x, y, z, cmap="viridis", edgecolor="none")
    ax.set_title("Surface plot")
    plt.show()


def projection(imstack, smoothing_radius, depth_threshold, show_plot=False):
    """Z projection using image interpolation.

    Parameters
    ----------
    imstack : ndarray
        numpy ndarray representation of 3D image stack
    smoothing_radius : float
        kernel radius for gaussian blur to apply before estimating the surface [0.1 - 5]
    depth_threshold : int
        Cutoff distance in z-planes from the 1st estimated surface [1 - 3]
    """
    # image dimensions are z, y, x
    s = imstack.shape
    I1 = _smooth(imstack, smoothing_radius)

    vm1 = I1.max(axis=0)
    depthmap = I1.argmax(axis=0)

    confidencemap = s[0] * vm1 / sum(I1, 0)
    c = confidencemap[:]
    confthres = np.median(c[c > np.median(c)])

    # keep only the brightest surface points (intensity in 1 quartile)
    # assumed to be the surface of interest
    mask = confidencemap > confthres
    depthmap2 = depthmap * mask
    xg1, yg1, zg1 = _interpolate(depthmap2, s)

    if show_plot:
        _plot_surface(xg1, yg1, zg1)

    # given the hight locations of the surface (zg1) compute the difference
    # towards the 1st quartile location (depthmap2), ignore the rest (==0);
    # the result reflects the distance (abs) between estimate and points.
    depthmap3 = np.abs(zg1 - depthmap2.astype("float64"))
    depthmap3[np.ma.where(depthmap2 == 0)] = 0

    # only keep points which are relatively close to our first estimate,
    # i.e. below the threshold. TIP: if the first estimate is too detailed(~=smooth)
    # the points from the peripodial membrane will not be eliminated since
    # the surface approximated them well. Increase the smoothness to prevent this.
    mask = depthmap3 < depth_threshold
    depthmap4 = depthmap2 * mask

    # TIP: depthmap4 should only contain signal of interest at this point.

    # --- 2nd iteration -
    # compute a better more detailed estimate with the filtered list (depthmap4)
    # this is to make sure that the highest intensity points will be
    # selected from the correct surface (The coarse grained estimate could
    # potentially approximate the origin of the point to another plane)
    xg2, yg2, zg2 = _interpolate(depthmap4, s)

    if show_plot:
        _plot_surface(xg2, yg2, zg2)

    # ----- creating projected image from interpolated surface estimation ------

    projected_image = np.zeros((s[1], s[2]), dtype=imstack.dtype)
    z_origin_map = np.zeros((s[1], s[2]), dtype="uint8")

    # zg2_mask = zg2 > 0
    # z_coordinates = z_origin_map * zg2_mask
    # z_origin_map[zg2_mask] = np.round(zg2[zg2_mask])
    # projected_image[zg2_mask] = imstack[z_coordinates, zg2_mask]

    for y in range(s[1]):
        for x in range(s[2]):
            if zg2[y, x] > 0:
                z_coordinate = int(np.round(zg2[y, x]))
                z_origin_map[y, x] = z_coordinate
                projected_image[y, x] = imstack[z_coordinate, y, x]

    plt.figure()
    plt.imshow(projected_image)
    plt.show()

    return projected_image, z_origin_map


if __name__ == "__main__":
    sample = imread("test_data/8bitDataset/test_image.tif")

    projection(sample, 0.2, 2)
