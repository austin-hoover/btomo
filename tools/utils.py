import numpy as np


def rotation_matrix(angle):
    """2x2 clockwise rotation matrix."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s], [-s, c]])


def get_bin_centers(bin_edges):
    """Return bin centers from bin edges."""
    return 0.5 * (bin_edges[:-1] + bin_edges[1:])


def get_bin_edges(bin_centers):
    """Return bin edges from bin centers."""
    w = 0.5 * np.diff(bin_centers)[0]
    return np.hstack([bin_centers - w, [bin_centers[-1] + w]])


def cov2corr(cov_mat):
    """Correlation matrix from covariance matrix."""
    D = np.sqrt(np.diag(cov_mat.diagonal()))
    Dinv = np.linalg.inv(D)
    corr_mat = np.linalg.multi_dot([Dinv, cov_mat, Dinv])
    return corr_mat


def symmetrize(M):
    """Return a symmetrized version of M.
    
    M : A square upper or lower triangular matrix.
    """
    return M + M.T - np.diag(M.diagonal())


def rand_rows(X, n):
    """Return n random elements of X."""
    Xsamp = np.copy(X)
    if n < len(X):
        idx = np.random.choice(Xsamp.shape[0], n, replace=False)
        Xsamp = Xsamp[idx]
    return Xsamp


def apply(M, X):
    """Apply M to each row of X."""
    return np.apply_along_axis(lambda x: np.matmul(M, x), 1, X)


def project(array, axis=0):
    """Project array onto one or more axes."""
    if type(axis) is int:
        axis = [axis]
    axis_sum = tuple([i for i in range(array.ndim) if i not in axis])
    proj = np.sum(array, axis=axis_sum)
    # Handle out of order projection. Right now it just handles 2D.
    if proj.ndim == 2 and axis[0] > axis[1]:
        proj = np.moveaxis(proj, 0, 1)
    return proj


def make_slice(n, axis=0, ind=0):
    """Return a slice index."""
    if type(axis) is int:
        axis = [axis]
    if type(ind) is int:
        ind = [ind]
    idx = n * [slice(None)]
    for k, i in zip(axis, ind):
        if i is None:
            continue
        idx[k] = slice(i[0], i[1]) if type(i) in [tuple, list, np.ndarray] else i
    return tuple(idx)


def hist(X, bins='auto'):
    edges = [np.histogram_bin_edges(X[:, i], bins) for i in range(X.shape[1])]
    return np.histogramdd(X, edges)