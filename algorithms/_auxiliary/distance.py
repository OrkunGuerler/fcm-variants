import numpy as np
import numpy.typing as npt


# -------------------------------------------------------------------------------

def sqrd_euclidean_distance(x: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
    return np.sum(np.power(x - y, 2))


def pairwise_sqrd_euclidean_distance(x: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
    return np.sum(np.power(x[:, np.newaxis] - y, 2), axis=2)


def euclidean_distance(x: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
    return np.sqrt(sqrd_euclidean_distance(x, y))


def pairwise_euclidean_distance(x: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
    return np.sqrt(pairwise_sqrd_euclidean_distance(x, y))


# -------------------------------------------------------------------------------

def manhattan_distance(x: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
    return np.sum(np.abs(x - y))


# -------------------------------------------------------------------------------

def mahalanobis_distance(x: npt.NDArray, y: npt.NDArray, cov: npt.NDArray) -> npt.NDArray:
    inv_cov = np.linalg.inv(cov)
    diff = x - y
    return np.sqrt(np.dot(np.dot(diff.T, inv_cov), diff))


# -------------------------------------------------------------------------------


def rbf_kernel_distance(x: npt.NDArray, y: npt.NDArray, bandwidth: npt.NDArray) -> npt.NDArray:
    return np.exp(- sqrd_euclidean_distance(x, y) / (bandwidth ** 2))
