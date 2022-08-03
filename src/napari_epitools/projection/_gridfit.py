import numpy as np
from scipy import sparse
from scipy.sparse import linalg


def _calculate_interpolation_equations(x, y, xnodes, ynodes):

    # determine which cell in the array each point lies in
    indx = np.digitize(x, xnodes) - 1
    indy = np.digitize(y, ynodes) - 1

    dx = np.diff(xnodes)
    dy = np.diff(ynodes)
    nx = np.shape(xnodes)[0]
    ny = np.shape(ynodes)[0]

    k = indx == nx - 1
    indx[k] = indx[k] - 1
    k = indy == ny - 1
    indy[k] = indy[k] - 1

    # interpolation equations for each point
    tx = np.minimum(1, np.maximum(0, (x - xnodes[indx]) / dx[indx]))
    ty = np.minimum(1, np.maximum(0, (y - ynodes[indy]) / dy[indy]))

    return tx, ty, indx, indy


def _bilinear_interpolation(n, nx, ny, tx, ty, ind):

    row = np.tile(np.arange(n)[:, np.newaxis], (1, 4))
    col = np.stack((ind, ind + 1, ind + ny, ind + ny + 1), axis=1)
    data = np.stack(
        ((1 - tx) * (1 - ty), (1 - tx) * ty, tx * (1 - ty), tx * ty), axis=1
    )
    A = sparse.csr_matrix(
        (data.flatten(), (row.flatten(), col.flatten())), shape=(n, nx * ny)
    )
    return A


def _make_flattened_indices(nrows, ncols):
    i, j = np.meshgrid(np.r_[0:ncols], np.r_[0:nrows])
    col_ind, row_ind = i.flatten(), j.flatten()
    ind = row_ind + nrows * col_ind
    return col_ind, row_ind, ind


def _create_sparse_matrix(A, B, m, ngrid, data_rows, stiffness):
    row = np.tile(np.arange(m)[:, np.newaxis], (1, 2))
    col = np.stack((A, B), axis=1)
    data = np.reshape(stiffness, (data_rows, 1)) * np.array([-1, 1])
    return sparse.csr_matrix(
        (data.flatten(), (row.flatten(), col.flatten())), shape=(m, ngrid)
    )


def _build_springs_regularizer(nx, ny, dx, dy):
    ngrid = nx * ny
    xscale = np.mean(dx)
    yscale = np.mean(dy)

    xyRelativeStiffness = np.array([[1], [1]])

    # Build regularizer
    # zero "rest length" springs
    _, row_ind, img_ind = _make_flattened_indices(ny - 1, nx)
    m = nx * (ny - 1)
    stiffness = 1 / (dy / yscale)
    stiffness = xyRelativeStiffness[1] * stiffness[row_ind]
    Areg = _create_sparse_matrix(
        img_ind, img_ind + 1, m, ngrid, row_ind.shape[0], stiffness
    )

    col_ind, _, img_ind = _make_flattened_indices(ny, nx - 1)
    m = (nx - 1) * ny
    stiffness = 1 / (dx / xscale)
    stiffness = xyRelativeStiffness[0] * stiffness[col_ind]
    Atemp = _create_sparse_matrix(
        img_ind, img_ind + ny, m, ngrid, col_ind.shape[0], stiffness
    )
    Areg = sparse.vstack((Areg, Atemp))

    col_ind, row_ind, img_ind = _make_flattened_indices(ny - 1, nx - 1)
    m = (nx - 1) * (ny - 1)
    stiffness = 1 / np.sqrt(
        np.square(dx[col_ind] / xscale / xyRelativeStiffness[0])
        + np.square(dy[row_ind] / yscale / xyRelativeStiffness[1])
    )
    Atemp = _create_sparse_matrix(
        img_ind, img_ind + ny + 1, m, ngrid, stiffness.shape[0], stiffness
    )

    Areg = sparse.vstack((Areg, Atemp))
    Atemp = _create_sparse_matrix(
        img_ind + 1, img_ind + ny, m, ngrid, stiffness.shape[0], stiffness
    )

    return sparse.vstack((Areg, Atemp))


def _least_squares_solver(A, Areg, rhs, nx, ny, smoothness):
    """Solve full system including regularizer using least squares.

    Parameters
    ----------
    A : csr_matrix
        Output of bilinear interpolation
    Areg : csr_matrix
        Output of springs regulariser
    rhs : ndarray
        Array to be solved
    nx : int
        Number of column indices where data to be interpolated is non-zero
    ny : int
        Number of row indices where data to be interpolated is non-zero
    smoothness : int
        Smoothing of interpolation - smaller value means more smoothing

    Returns
    -------
    ndarray
        The interpolated z postions
    """
    nreg = Areg.shape[0]
    NA = linalg.norm(A, np.inf)
    NR = linalg.norm(Areg, np.inf)
    A = sparse.vstack((A, Areg * (smoothness * NA / NR)))
    rhs = np.vstack((rhs[:, np.newaxis], np.zeros((nreg, 1))))

    # solve the full system, with regularizer attached
    solution = linalg.lsqr(A, rhs)
    return np.reshape(solution[0], (ny, nx))


# function [zgrid,xgrid,ygrid] = gridfit(x,y,z,xnodes,ynodes,varargin)
def gridfit(x, y, z, xnodes, ynodes, smoothness):
    """Interpolation using curve fitting. This is a port of the parts of
    https://www.mathworks.com/matlabcentral/fileexchange/8998-surface-fitting-using-gridfit
    which are relevant to this application (i.e. not all the options have been ported).

    Parameters
    ----------
    x : ndarray
        Column indices at which the z coordinate map are non-zero
    y : ndarray
        Row indices at which the z coordinate map are non-zero
    z : ndarray
        z coordinate values at x, y indices
    xnodes : int
        Number of rows in interpolation
    ynodes : int
        Number of columns in interpolation
    smoothness : int
        Smoothing of interpolation - smaller value means more smoothing

    Returns
    -------
    ndarray
        interpolated z coordinates

    Raises
    ------
    ValueError
        If x and y do not have the same number of elements
    ValueError
        If z contains insufficent (less than three) non-zero elements
    ValueError
        If the grid for interpolation is not monotone
    """

    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)

    if np.ndim(xnodes) == 0:
        xnodes = np.linspace(xmin, xmax, xnodes)

    if np.ndim(ynodes) == 0:
        ynodes = np.linspace(ymin, ymax, ynodes)

    dx = np.diff(xnodes)
    dy = np.diff(ynodes)
    nx = np.shape(xnodes)[0]
    ny = np.shape(ynodes)[0]

    # check lengths of the data
    n = x.shape[0]
    if y.shape[0] != n or z.shape[0] != n:
        raise ValueError("Data vectors are incompatible in size.")

    if n < 3:
        raise ValueError("Insufficient data for surface estimation.")

    # verify the nodes are distinct
    if np.any(np.diff(xnodes) <= 0) or np.any(np.diff(ynodes) <= 0):
        raise ValueError("xnodes and ynodes must be monotone increasing")

    # do we need to tweak the first or last node in x or y?
    if xmin < xnodes[0]:
        xnodes[0] = xmin

    if xmax > xnodes[-1]:
        xnodes[-1] = xmax

    if ymin < ynodes[0]:
        ynodes[0] = ymin

    if ymax > ynodes[-1]:
        ynodes[-1] = ymax

    # interpolation equations for each point
    tx, ty, indx, indy = _calculate_interpolation_equations(
        x, y, xnodes, ynodes
    )

    # interpolate
    ind = indy + ny * indx
    A = _bilinear_interpolation(n, nx, ny, tx, ty, ind)
    rhs = z

    # build the regulariser
    Areg = _build_springs_regularizer(nx, ny, dx, dy)

    # solve using least squares to get final interpolation
    zgrid = _least_squares_solver(A, Areg, rhs, nx, ny, smoothness)

    return zgrid
