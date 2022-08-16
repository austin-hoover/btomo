import numpy as np


def rotation_matrix(angle):
    """2x2 clockwise rotation matrix."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s], [-s, c]])


def rotation_matrix_4x4(angle):
    """4x4 matrix to rotate [x, x', y, y'] clockwise in the x-y plane."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s, 0], [0, c, 0, s], [-s, 0, c, 0], [0, -s, 0, c]])


def phase_adv_matrix(mu):
    """2x2 matrix to advance phase in x-x' or y-y'."""
    return rotation_matrix(mu)


def phase_adv_matrix_4x4(mu1, mu2):
    """4x4 matrix to advance x-x' by mu1 and y-y' by mu2."""
    P = np.zeros((4, 4))
    P[:2, :2] = phase_adv_matrix(mu1)
    P[2:, 2:] = phase_adv_matrix(mu2)
    return P


def norm_matrix(alpha, beta):
    """2x2 normalization matrix for x-x' or y-y'."""
    return np.array([[beta, 0], [-alpha, 1]]) / np.sqrt(beta)
    
    
def norm_matrix_4x4_uncoupled(alpha_x, alpha_y, beta_x, beta_y):
    """4x4 normalization matrix for x-x' and y-y', i.e., no off-block-diagonal elements."""
    V = np.zeros((4, 4))
    V[:2, :2] = norm_matrix(alpha_x, beta_x)
    V[2:, 2:] = norm_matrix(alpha_y, beta_y)
    return V


def cov_from_twiss2D(alpha_x, alpha_y, beta_x, beta_y, eps_x, eps_y):
    """Construct covariance matrix from 2D Twiss parameters."""
    gamma_x = (1.0 + alpha_x**2) / beta_x
    gamma_y = (1.0 + alpha_y**2) / beta_y
    Sigma = np.zeros((4, 4))
    Sigma[:2, :2] = eps_x * np.array([[beta_x, -alpha_x], [-alpha_x, gamma_x]])
    Sigma[2:, 2:] = eps_y * np.array([[beta_y, -alpha_y], [-alpha_y, gamma_y]])
    return Sigma