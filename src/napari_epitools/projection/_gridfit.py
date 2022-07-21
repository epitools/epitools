import numpy as np
from scipy import sparse


# function [zgrid,xgrid,ygrid] = gridfit(x,y,z,xnodes,ynodes,varargin)
def gridfit(x, y, z, xnodes, ynodes, smoothness):
    # x and y are the indices where depthmap > 0
    # z is the depthmap values at these indices
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)

    # did they supply a scalar for the nodes?
    if np.ndim(xnodes) == 0:
        xnodes = np.linspace(xmin, xmax, xnodes)

    if np.ndim(ynodes) == 0:
        ynodes = np.linspace(ymin, ymax, ynodes)

    dx = np.diff(xnodes)
    dy = np.diff(ynodes)
    nx = np.shape(xnodes)[0]
    ny = np.shape(ynodes)[0]
    ngrid = nx * ny

    xscale = np.mean(dx)
    yscale = np.mean(dy)

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

    # determine which cell in the array each point lies in
    indx = np.digitize(x, xnodes) - 1
    indy = np.digitize(y, ynodes) - 1
    # any point falling at the last node is taken to be
    # inside the last cell in x or y.
    k = indx == nx - 1
    indx[k] = indx[k] - 1
    k = indy == ny - 1
    indy[k] = indy[k] - 1
    ind = indy + ny * indx

    print(f"{dx[indx]=}")
    # interpolation equations for each point
    tx = np.minimum(1, np.maximum(0, (x - xnodes[indx]) / dx[indx]))
    ty = np.minimum(1, np.maximum(0, (y - ynodes[indy]) / dy[indy]))

    # bilinear interpolation in a cell
    row = np.tile(np.arange(n)[:, np.newaxis], (1, 4))
    col = np.stack((ind, ind + 1, ind + ny, ind + ny + 1), axis=1)
    data = np.stack(
        ((1 - tx) * (1 - ty), (1 - tx) * ty, tx * (1 - ty), tx * ty), axis=1
    )
    A = sparse.coo_array((data, (row, col)), shape=(n, ngrid))
    print(f"{np.max(A.data)=}")
    rhs = z

    # do we have relative smoothing parameters?
    xyRelativeStiffness = np.array([[1], [1]])

    # Build regularizer. Add del^4 regularizer one day.
    # zero "rest length" springs
    i, j = np.mgrid[0:nx, 0 : (ny - 1)]
    ind = j.flatten() + ny * (i.flatten() - 1)
    m = nx * (ny - 1)
    stiffness = 1 / (dy / yscale)
    Ireg = np.tile(np.arange(m)[:, np.newaxis], (1, 2))
    Jreg = np.array([ind, ind + 1])
    Vreg = xyRelativeStiffness[1] * stiffness[j.flatten()] * np.array([-1, 1])
    Areg = sparse.coo_array(Ireg, (Jreg, Vreg), shape=(m, ngrid))

    i, j = np.mgrid[0 : (nx - 1), 0:ny]
    ind = j.flatten() + ny * (i.flatten() - 1)
    m = (nx - 1) * ny
    stiffness = 1 / dx / xscale
    row = np.tile(np.arange(m)[:, np.newaxis], (1, 2))
    col = np.array([ind, ind + ny])
    data = xyRelativeStiffness[0] * stiffness[i.flatten()] * np.array([-1, 1])
    temp = sparse.coo_array((data, (row, col)), shape=(m, ngrid))
    Areg = np.vstack((Areg, temp))

    i, j = np.mgrid[0 : (nx - 1), 1 : (ny - 1)]
    ind = j.flatten() + ny * (i.flatten() - 1)
    m = (nx - 1) * (ny - 1)
    stiffness = 1 / np.sqrt(
        np.square(dx[i.flatten()] / xscale / xyRelativeStiffness[0])
        + np.squre(dy[j.flatten()] / yscale / xyRelativeStiffness[1])
    )

    Itemp = np.tile(np.arange(m)[:, np.newaxis], (1, 2))
    Jtemp = np.array([ind, ind + ny + 1])
    Vtemp = stiffness * np.array([-1, 1])
    temp = sparse.coo_array(Itemp, (Jtemp, Vtemp), shape=(m, ngrid))
    Areg = np.vstack((Areg, temp))

    Itemp = np.tile(np.arange(m)[:, np.newaxis], (1, 2))
    Jtemp = np.array([ind + 1, ind + ny])
    Vtemp = stiffness * np.array([-1, 1])
    temp = sparse.coo_array(Itemp, (Jtemp, Vtemp), shape=(m, ngrid))
    Areg = np.vstack((Areg, temp))

    nreg = Areg.shape[0]

    # Append the regularizer to the interpolation equations,
    # scaling the problem first. Use the 1-norm for speed.
    NA = sparse.linalg.norm(A, 1)
    NR = sparse.linalg.norm(Areg, 1)
    A = np.vstack((A, Areg * (smoothness * NA / NR)))
    rhs = np.vstack((rhs, np.zeros((nreg, 1))))

    # solve the full system, with regularizer attached
    zgrid = np.reshape(sparse.linalg.lstsq(A, rhs), (ny, nx))

    # only generate xgrid and ygrid if requested.
    xgrid, ygrid = np.meshgrid(np.arange(xnodes), np.arange(ynodes))

    return xgrid, ygrid, zgrid
