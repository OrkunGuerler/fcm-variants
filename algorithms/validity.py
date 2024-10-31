import numpy as np
import numpy.typing as npt
from algorithms._auxiliary.distance import pairwise_sqrd_euclidean_distance


def partition_coefficient(weights: npt.NDArray, fuzziness: float = 2) -> npt.NDArray:
    """Partition Coefficient: Measures the degree of overlap between clusters.
    Higher values are better; means less overlap between clusters.

    Args:
        weights (npt.NDArray): _description_

    Returns:
        npt.NDArray: _description_
    """
    return np.sum(np.power(weights, fuzziness)) / weights.shape[0]


def partition_entropy(weights: npt.NDArray, base: float = np.e) -> npt.NDArray:
    """Partition Entropy: Quantifies the fuzziness or uncertainty in the clustering. It is calculated using the entropy of the membership values.
    Lower values are better; means less fuzziness and clearer distinctions between clusters.

    Args:
        weights (npt.NDArray): _description_

    Returns:
        npt.NDArray: _description_
    """
    return - np.sum(np.multiply(weights, np.emath.logn(base, weights))) / weights.shape[0]


def fuzzy_silhouette(dataset: npt.NDArray, weights: npt.NDArray) -> npt.NDArray:
    """Fuzzy Silhouette: Combines membership degrees with the silhouette value, which measures how similar an object is to its own cluster compared to other clusters.
    Higher values are better; means well-defined clusters.

    Args:
        dataset (npt.NDArray): _description_
        w (npt.NDArray): _description_

    Returns:
        npt.NDArray: returns fs value
    """
    n_points, n_clusters = weights.shape
    dists = pairwise_sqrd_euclidean_distance(dataset, dataset)

    a = np.empty(n_points)
    for j in range(n_points):
        aj = []
        for i in range(n_clusters):
            numer, denom = 0, 0
            for k in range(n_points):
                if j != k:
                    numer += (weights[j, i] and weights[k, i]) * dists[j, k]
                    denom += (weights[j, i] and weights[k, i])
            aj.append(numer / denom)
        a[j] = np.min(np.array(aj))

    b = np.empty(n_points)
    for j in range(n_points):
        bj = []
        for r in range(n_clusters - 1):
            numer, denom = 0, 0
            for s in range(r + 1, n_clusters):
                for k in range(n_points):
                    if j != k:
                        lhs = (weights[j, r] and weights[k, s])
                        rhs = (weights[j, s] and weights[k, r])
                        numer += (lhs or rhs) * dists[j, k]
                        denom += (lhs or rhs)
            bj.append(numer/denom)
        b[j] = np.min(np.array(bj))

    s = np.empty(n_points)
    for j in range(n_points):
        s[j] = (b[j] - a[j]) / np.max((a[j], b[j]))

    return np.sum(s) / n_points


def xie_beni(dataset: npt.NDArray, weights: npt.NDArray, centroids: npt.NDArray,
             fuzziness: float = 2) -> npt.NDArray:
    """Xie/Beni: Measures the compactness and separation of the clusters. It considers the distances between data points and their cluster centers as well as the distances between different cluster centers.
    Lower values are better; means compact and well-separated clusters.

    Args:
        dataset (npt.NDArray): _description_
        weights (npt.NDArray): _description_
        centroids (npt.NDArray): _description_
        fuzziness (float, optional): _description_. Defaults to 2.

    Returns:
        npt.NDArray: _description_
    """
    x_dists = pairwise_sqrd_euclidean_distance(dataset, centroids)
    v_dists = pairwise_sqrd_euclidean_distance(centroids, centroids)
    v_dists[np.diag_indices_from(v_dists)] = np.inf
    return np.sum(x_dists * np.power(weights, fuzziness)) / (dataset.shape[0] * np.min(v_dists))
