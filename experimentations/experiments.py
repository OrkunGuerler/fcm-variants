import os
import time

import numpy as np
import matplotlib.pyplot as plt

from experimentations.datasets import (
    noisy_circles,
    noisy_moons,
    aniso,
    varied,
    blobs,
    iris,
    wine,
    breast_cancer
)
from algorithms.validity import (
    partition_coefficient,
    partition_entropy,
    fuzzy_silhouette,
    xie_beni
)
from algorithms import (
    fuzzy_c_means,
    gustafson_kessel,
    noise_clustering,
    possibilistic_c_means,
    possibilistic_fuzzy_c_means,
    credibilistic_fuzzy_c_means,
    kernel_fuzzy_c_means
)


np.set_printoptions(suppress=True, precision=4)

EXP_DIR_PATH = os.path.dirname(__file__)
IMG_DIR_PATH = os.path.join(EXP_DIR_PATH, "images")
os.makedirs(IMG_DIR_PATH, exist_ok=True)


datasets = {
    "noisy_circles": noisy_circles,
    "noisy_moons": noisy_moons,
    "aniso": aniso,
    "varied": varied,
    "blobs": blobs,
    "iris": iris,
    "wine": wine,
    "breast_cancer": breast_cancer
}


algorithms = {
    "fcm": fuzzy_c_means,
    "gk": gustafson_kessel,
    "nc": noise_clustering,
    "pcm": possibilistic_c_means,
    "pfcm": possibilistic_fuzzy_c_means,
    "cfcm": credibilistic_fuzzy_c_means,
    "kfcm": kernel_fuzzy_c_means,
}

cmap = "viridis"

for d_name, dataset in datasets.items():
    data = dataset().get("data")
    targets = dataset().get("target")

    if d_name == "iris":
        plt.scatter(data[:, 2], data[:, 3], c=targets, cmap=cmap)
    elif d_name == "wine":
        plt.scatter(data[:, 11], data[:, 12], c=targets, cmap=cmap)
    else:
        plt.scatter(data[:, 0], data[:, 1], c=targets, cmap=cmap)
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    DATA_IMG = os.path.join(IMG_DIR_PATH, f"{d_name}.png")
    plt.savefig(f"{DATA_IMG}", format="png", dpi=600, transparent=True)
    plt.clf()

count = 100
results = {}
for i, (a_name, algorithm) in enumerate(algorithms.items()):
    print(f"{i}. {a_name}:")
    ALG_PATH = os.path.join(IMG_DIR_PATH, f"{a_name}")
    os.makedirs(ALG_PATH, exist_ok=True)
    results[a_name] = {}
    for j, (d_name, dataset) in enumerate(datasets.items()):
        print(f"\t{i}.{j}. {d_name}:")
        results[a_name][d_name] = {}

        data = dataset().get("data")
        n_clusters = dataset().get("n_clusters")

        pt, pc, pe, fs, xb = 0, 0, 0, 0, 0
        for k in range(count):
            print(f"{i}.{j}.{k}.")
            perf_start = time.perf_counter_ns()
            returns: dict = algorithm(data, n_clusters)
            perf_end = time.perf_counter_ns()

            weights = returns.get("weights")
            if a_name in ["pcm", "pfcm"]:
                weights = returns.get("typicalities")
            centroids = returns.get("centroids")

            pt += (perf_end - perf_start) * 1e-9
            pc += partition_coefficient(weights)
            pe += partition_entropy(weights)
            fs += fuzzy_silhouette(data, weights)
            xb += xie_beni(data, weights, centroids)

        results[a_name][d_name] = {
            "avg_partition_coefficient": format(pc / count, ".4f"),
            "avg_partition_entropy": format(pe / count, ".4f"),
            "avg_fuzzy_silhouette": format(fs / count, ".4f"),
            "avg_xie_beni": format(xb / count, ".4f"),
            "avg_performance_time": format(pt / count, ".4f")
        }

        pred_targets = np.max(weights, axis=1)
        pred_targets_ind = np.argmax(weights, axis=1)
        EXP_IMG = os.path.join(ALG_PATH, f"{d_name}.png")
        if d_name == "iris":
            plt.scatter(data[:, 2], data[:, 3], alpha=pred_targets, c=pred_targets_ind,
                        linewidths=None, edgecolors=None, cmap=cmap)
        elif d_name == "wine":
            plt.scatter(data[:, 11], data[:, 12], alpha=pred_targets, c=pred_targets_ind,
                        linewidths=None, edgecolors=None, cmap=cmap)
        else:
            plt.scatter(data[:, 0], data[:, 1], alpha=pred_targets, c=pred_targets_ind,
                        linewidths=None, edgecolors=None, cmap=cmap)
        plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.savefig(f"{EXP_IMG}", format="png", dpi=300, transparent=True)
        plt.clf()
RESULTS_PATH = os.path.join(EXP_DIR_PATH, "results_final.json")
with open(f"{RESULTS_PATH}", "w") as results_file:
    import json
    json.dump(results, results_file)
print(f"{'#'*50} BİTTİ {'#'*50}")
