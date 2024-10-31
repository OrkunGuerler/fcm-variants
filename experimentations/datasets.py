import numpy as np
import numpy.typing as npt
from sklearn import datasets
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

N_SAMPLES: int = 500
# -------------------------------------------------------------------------------
# Gen Datasets


def noisy_circles(n_samples: int = N_SAMPLES, random_state: int = 30
                  ) -> tuple[npt.NDArray, npt.NDArray, int]:
    data, target = datasets.make_circles(
        n_samples=n_samples, random_state=random_state,
        factor=0.5, noise=0.05
    )
    return {"data": data, "target": target, "n_clusters": 2}


def noisy_moons(n_samples: int = N_SAMPLES, random_state: int = 30
                ) -> tuple[npt.NDArray, npt.NDArray, int]:
    data, target = datasets.make_moons(
        n_samples=n_samples, noise=0.05, random_state=random_state
    )
    return {"data": data, "target": target, "n_clusters": 2}


def blobs(n_samples: int = N_SAMPLES, random_state: int = 30
          ) -> tuple[npt.NDArray, npt.NDArray, int]:
    data, target = datasets.make_blobs(
        n_samples=n_samples, random_state=random_state)
    return {"data": data, "target": target, "n_clusters": 3}


def aniso(n_samples: int = N_SAMPLES, random_state: int = 170
          ) -> tuple[npt.NDArray, npt.NDArray, int]:
    X, target = datasets.make_blobs(
        n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    return {"data": X_aniso, "target": target, "n_clusters": 3}


def varied(n_samples: int = N_SAMPLES, random_state: int = 170
           ) -> tuple[npt.NDArray, npt.NDArray, int]:
    data, target = datasets.make_blobs(
        n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
    )
    return {"data": data, "target": target, "n_clusters": 3}


# -------------------------------------------------------------------------------
# Toy Datasets

def iris() -> tuple[npt.NDArray, npt.NDArray, int]:
    return {"data": load_iris().get("data"),
            "target": load_iris().get("target"),
            "n_clusters": 3}


def wine() -> tuple[npt.NDArray, npt.NDArray, int]:
    return {"data": load_wine().get("data"),
            "target": load_wine().get("target"),
            "n_clusters": 3}


def breast_cancer() -> tuple[npt.NDArray, npt.NDArray, int]:
    return {"data": load_breast_cancer().get("data"),
            "target": load_breast_cancer().get("target"),
            "n_clusters": 2}
