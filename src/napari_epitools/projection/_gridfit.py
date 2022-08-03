import numpy as np
from scipy import sparse
from scipy.sparse import linalg


def _calculate_interpolation_equations(x_indices, y_indices, xnodes, ynodes):

    # determine which cell in the array each point lies in
    indx = np.digitize(x_indices, xnodes) - 1
    indy = np.digitize(y_indices, ynodes) - 1

    difference_x = np.diff(xnodes)
    difference_y = np.diff(ynodes)
    num_nodes_x = np.shape(xnodes)[0]
    num_nodes_y = np.shape(ynodes)[0]

    k = indx == num_nodes_x - 1
    indx[k] = indx[k] - 1
    k = indy == num_nodes_y - 1
    indy[k] = indy[k] - 1

    # interpolation equations for each point
    tx = np.minimum(
        1, np.maximum(0, (x_indices - xnodes[indx]) / difference_x[indx])
    )
    ty = np.minimum(
        1, np.maximum(0, (y_indices - ynodes[indy]) / difference_y[indy])
    )

    return tx, ty, indx, indy


def _bilinear_interpolation(n, num_nodes_x, num_nodes_y, tx, ty, ind):

    row = np.tile(np.arange(n)[:, np.newaxis], (1, 4))
    col = np.stack(
        (ind, ind + 1, ind + num_nodes_y, ind + num_nodes_y + 1), axis=1
    )
    data = np.stack(
        ((1 - tx) * (1 - ty), (1 - tx) * ty, tx * (1 - ty), tx * ty), axis=1
    )
    A = sparse.csr_matrix(
        (data.flatten(), (row.flatten(), col.flatten())),
        shape=(n, num_nodes_x * num_nodes_y),
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


def _build_springs_regularizer(
    num_nodes_x, num_nodes_y, difference_x, difference_y
):
    ngrid = num_nodes_x * num_nodes_y
    xscale = np.mean(difference_x)
    yscale = np.mean(difference_y)

    xyRelativeStiffness = np.array([[1], [1]])

    # Build regularizer
    # zero "rest length" springs
    _, row_ind, img_ind = _make_flattened_indices(num_nodes_y - 1, num_nodes_x)
    m = num_nodes_x * (num_nodes_y - 1)
    stiffness = 1 / (difference_y / yscale)
    stiffness = xyRelativeStiffness[1] * stiffness[row_ind]
    Areg = _create_sparse_matrix(
        img_ind, img_ind + 1, m, ngrid, row_ind.shape[0], stiffness
    )

    col_ind, _, img_ind = _make_flattened_indices(num_nodes_y, num_nodes_x - 1)
    m = (num_nodes_x - 1) * num_nodes_y
    stiffness = 1 / (difference_x / xscale)
    stiffness = xyRelativeStiffness[0] * stiffness[col_ind]
    Atemp = _create_sparse_matrix(
        img_ind, img_ind + num_nodes_y, m, ngrid, col_ind.shape[0], stiffness
    )
    Areg = sparse.vstack((Areg, Atemp))

    col_ind, row_ind, img_ind = _make_flattened_indices(
        num_nodes_y - 1, num_nodes_x - 1
    )
    m = (num_nodes_x - 1) * (num_nodes_y - 1)
    stiffness = 1 / np.sqrt(
        np.square(difference_x[col_ind] / xscale / xyRelativeStiffness[0])
        + np.square(difference_y[row_ind] / yscale / xyRelativeStiffness[1])
    )
    Atemp = _create_sparse_matrix(
        img_ind,
        img_ind + num_nodes_y + 1,
        m,
        ngrid,
        stiffness.shape[0],
        stiffness,
    )

    Areg = sparse.vstack((Areg, Atemp))
    Atemp = _create_sparse_matrix(
        img_ind + 1,
        img_ind + num_nodes_y,
        m,
        ngrid,
        stiffness.shape[0],
        stiffness,
    )

    return sparse.vstack((Areg, Atemp))


def _least_squares_solver(A, Areg, rhs, num_nodes_x, num_nodes_y, smoothness):
    """Solve full system including regularizer using least squares.

    Parameters
    ----------
    A : csr_matrix
        Output of bilinear interpolation
    Areg : csr_matrix
        Output of springs regulariser
    rhs : ndarray
        Array to be solved
    num_nodes_x : int
        Number of column indices where data to be interpolated is non-zero
    num_nodes_y : int
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
    return np.reshape(solution[0], (num_nodes_y, num_nodes_x))


def gridfit(x_indices, y_indices, z_values, xnodes, ynodes, smoothness):
    """Interpolation using curve fitting. This is a port of the parts of
    https://www.mathworks.com/matlabcentral/fileexchange/8998-surface-fitting-using-gridfit
    which are relevant to this application (i.e. not all the options have been ported).

    Parameters
    ----------
    x_indices : ndarray
        Column indices at which the z_values coordinate map are non-zero
    y_indices : ndarray
        Row indices at which the z_values coordinate map are non-zero
    z_values : ndarray
        z_values coordinate values at x_indices, y_indices indices
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
        If x_indices and y_indices do not have the same number of elements
    ValueError
        If z_values contains insufficent (less than three) non-zero elements
    ValueError
        If the grid for interpolation is not monotone
    """

    xmin = np.min(x_indices)
    xmax = np.max(x_indices)
    ymin = np.min(y_indices)
    ymax = np.max(y_indices)

    if np.ndim(xnodes) == 0:
        xnodes = np.linspace(xmin, xmax, xnodes)

    if np.ndim(ynodes) == 0:
        ynodes = np.linspace(ymin, ymax, ynodes)

    difference_x = np.diff(xnodes)
    difference_y = np.diff(ynodes)
    num_nodes_x = np.shape(xnodes)[0]
    num_nodes_y = np.shape(ynodes)[0]

    # check lengths of the data
    num_values_z = z_values.shape[0]
    if (
        x_indices.shape[0] != num_values_z
        or y_indices.shape[0] != num_values_z
    ):
        raise ValueError("Data vectors are incompatible in size.")

    if num_values_z < 3:
        raise ValueError("Insufficient data for surface estimation.")

    # verify the nodes are distinct
    if np.any(np.diff(xnodes) <= 0) or np.any(np.diff(ynodes) <= 0):
        raise ValueError("xnodes and ynodes must be monotone increasing")

    # do we need to tweak the first or last node in x_indices or y_indices?
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
        x_indices, y_indices, xnodes, ynodes
    )

    # interpolate
    ind = indy + num_nodes_y * indx
    A = _bilinear_interpolation(
        num_values_z, num_nodes_x, num_nodes_y, tx, ty, ind
    )
    rhs = z_values

    # build the regulariser
    Areg = _build_springs_regularizer(
        num_nodes_x, num_nodes_y, difference_x, difference_y
    )

    # solve using least squares to get final interpolation
    zgrid = _least_squares_solver(
        A, Areg, rhs, num_nodes_x, num_nodes_y, smoothness
    )

    return zgrid
