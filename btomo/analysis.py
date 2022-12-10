import numpy as np
import numpy.linalg as la
from tqdm import tqdm
from tqdm import trange
from .utils import cov2corr
from .utils import symmetrize


def mat2vec(cov):
    """Return vector of N*(N-1)/2 independent elements in NxN symmetric matrix."""
    return cov[np.triu_indices(cov.shape[0])]


def vec2mat(moment_vec):
    """Inverse of `mat2vec`."""
    cov = np.zeros((4, 4))
    indices = np.triu_indices(4)
    for moment, (i, j) in zip(moment_vec, zip(*indices)):
        cov[i, j] = moment
    return symmetrize(cov)


def rms_ellipse_angle(sig_xx, sig_yy, sig_xy):
    """Return tilt angle of rms ellipse in x-y plane."""
    return -0.5 * np.arctan2(2 * sig_xy, sig_xx - sig_yy)


def rms_ellipse_semiaxes(sig_xx, sig_yy, sig_xy):
    """Return semi-axes rms ellipse in x-y plane."""
    angle = rms_ellipse_angle(sig_xx, sig_yy, sig_xy)
    sn, cs = np.sin(angle), np.cos(angle)
    cx = np.sqrt(abs(sig_xx * cs ** 2 + sig_yy * sn ** 2 - 2 * sig_xy * sn * cs))
    cy = np.sqrt(abs(sig_xx * sn ** 2 + sig_yy * cs ** 2 + 2 * sig_xy * sn * cs))
    return cx, cy


def rms_ellipse_dims(cov, x1="x", x2="y"):
    """Return (angle, c1, c2) of rms ellipse in x1-x2 plane, where angle is the
    clockwise tilt angle and c1/c2 are the semi-axes.
    """
    str_to_int = {"x": 0, "xp": 1, "y": 2, "yp": 3}
    i, j = str_to_int[x1], str_to_int[x2]
    sii, sjj, sij = cov[i, i], cov[j, j], cov[i, j]
    angle = rms_ellipse_angle(sii, sjj, sij)
    c1, c2 = rms_ellipse_semiaxes(sii, sjj, sij)
    return angle, c1, c2


def intrinsic_emittances(cov, order=False):
    """Return intrinsic emittances from covariance matrix."""
    U = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )
    trSU2 = np.trace(la.matrix_power(np.matmul(cov, U), 2))
    detS = la.det(cov)
    eps_1 = 0.5 * np.sqrt(-trSU2 + np.sqrt(trSU2 ** 2 - 16.0 * detS))
    eps_2 = 0.5 * np.sqrt(-trSU2 - np.sqrt(trSU2 ** 2 - 16.0 * detS))
    return eps_1, eps_2


def apparent_emittances(cov):
    """Return apparent emittances from covariance matrix."""
    eps_x = np.sqrt(la.det(cov[:2, :2]))
    eps_y = np.sqrt(la.det(cov[2:, 2:]))
    return eps_x, eps_y


def emittances(cov, order=False):
    """Return apparent and intrinsic emittances from covariance matrix."""
    eps_x, eps_y = apparent_emittances(cov)
    eps_1, eps_2 = intrinsic_emittances(cov, order=order)
    return eps_x, eps_y, eps_1, eps_2


def coupling_coefficient(cov):
    eps_1, eps_2 = intrinsic_emittances(cov)
    eps_x, eps_y = apparent_emittances(cov)
    return 1.0 - np.sqrt((eps_1 * eps_2) / (eps_x * eps_y))


def twiss2D(cov):
    """Return 2D Twiss parameters from covariance matrix."""
    eps_x, eps_y = apparent_emittances(cov)
    beta_x = cov[0, 0] / eps_x
    beta_y = cov[2, 2] / eps_y
    alpha_x = -cov[0, 1] / eps_x
    alpha_y = -cov[2, 3] / eps_y
    return np.array([alpha_x, alpha_y, beta_x, beta_y])


def beam_stats(cov):
    """Return dictionary of parameters from NxN covariance matrix.
    
    Parameters
    ----------
    cov : ndarray, shape (N, N)
        The NxN covariance matrix.
    
    Returns
    -------
    dict
        cov : ndarray, shape (N, N)
            The covariance matrix.
        corr : ndarray, shape (N, N)
            The correlation matrix.
        'eps_x': apparent emittance in x-x' plane
        'eps_y': apparent emittance in y-y' plane
        'eps_1': intrinsic emittance
        'eps_2': intrinsic emittance
        'eps_4D': 4D emittance = eps_1 * eps_2
        'eps_4D_app': apparent 4D emittance = eps_x * eps_y
        'C': coupling coefficient = sqrt((eps_1 * eps_2) / (eps_x * eps_y))
        'beta_x': beta_x = <x^2> / eps_x
        'beta_y': beta_y = <y^2> / eps_y
        'alpha_x': alpha_x = -<xx'> / eps_x
        'alpha_y': alpha_y = -<yy'> / eps_y
    """
    stats = dict()
    stats["cov"] = cov
    stats["corr"] = cov2corr(cov)
    stats["alpha_x"], stats["alpha_y"], stats["beta_x"], stats["beta_y"] = twiss2D(cov)
    stats["eps_x"], stats["eps_y"] = apparent_emittances(cov)
    stats["eps_1"], stats["eps_2"] = intrinsic_emittances(cov)
    stats["eps_4D"] = stats["eps_1"] * stats["eps_2"]
    stats["eps_4D_app"] = stats["eps_x"] * stats["eps_y"]
    stats["C"] = np.sqrt(stats["eps_4D"] / stats["eps_4D_app"])
    return stats


def dist_cov(f, coords, verbose=False):
    """Compute the distribution covariance matrix.
    
    This will take a while for large arrays.
    
    Parameters
    ----------
    f : ndarray
        The distribution function.
    coords : list[ndarray]
        List of coordinates along each axis of `H`. Can also
        provide meshgrid coordinates.
    verbose : bool
        Whether to print progress.
        
    Returns
    -------
    cov : ndarray, shape (n, n)
        The distribution covariance matrix.
    mu : ndarray, shape (n,)
        The distribution centroid.
    """
    if verbose:
        print(f"Forming {f.shape} meshgrid...")
    if coords[0].ndim == 1:
        COORDS = np.meshgrid(*coords, indexing="ij")
    else:
        COORDS = coords
    n = f.ndim
    f_sum = np.sum(f)
    if f_sum == 0:
        return np.zeros((n, n)), np.zeros((n,))
    if verbose:
        print("Averaging...")
    mu = np.array([np.average(C, weights=f) for C in COORDS])
    cov = np.zeros((n, n))
    _range = trange if disp else range
    for i in _range(Sigma.shape[0]):
        for j in _range(i + 1):
            X = COORDS[i] - means[i]
            Y = COORDS[j] - means[j]
            EX = np.sum(X * f) / f_sum
            EY = np.sum(Y * f) / f_sum
            EXY = np.sum(X * Y * f) / f_sum
            cov[i, j] = EXY - EX * EY
    cov = utils.symmetrize(cov)
    if verbose:
        print("Done.")
    return cov, mu
