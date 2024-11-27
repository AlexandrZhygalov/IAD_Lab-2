from sklearn.datasets import make_moons, make_blobs
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score, pairwise_distances
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import time
import numpy as np


def visual_dataset(): # task 1
    X_moons, y_moons = make_moons(n_samples=1000, noise=0.05, random_state=0)
    plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap='viridis', edgecolor='k', s=40)
    plt.title("Набір даних 'make_moons'")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

    X_blobs, y_blobs = make_blobs(n_samples=20000, n_features=2, centers=4, cluster_std=1, center_box=(-10.0, 10.0),
                                  shuffle=True, random_state=1)
    plt.figure(figsize=(8, 8))
    plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_blobs, cmap='viridis', s=5, alpha=0.6)
    plt.title("Набір даних 'make_blobs'")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(label='Cluster Label')
    plt.grid(True)
    plt.show()
    return X_moons, y_moons, X_blobs, y_blobs


def spectral_clustering_and_plot(X_set, n_clusters, dataset_name, size): # task 2, 3, 4
    start_time = time.time()
    model = SpectralClustering(n_clusters=n_clusters, affinity='rbf', gamma=1.0, random_state=42)
    labels = model.fit_predict(X_set)
    end_time = time.time()

    # Час виконання
    print(f"{dataset_name} - Час кластеризації (Розмір вибірки - {size}): {end_time - start_time:.4f} секунд")

    plt.scatter(X_set[:, 0], X_set[:, 1], c=labels, cmap='viridis', edgecolor='k', s=30)
    plt.title(f"Набір даних - '{dataset_name}'")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


def diff_size_clust(): # task 5
    size_array = [10000, 20000, 30000, 40000, 50000]

    for size in size_array:
        print("")
        X_moons, y_moons = make_moons(n_samples=size, noise=0.05, random_state=0)
        X_blobs, y_blobs = make_blobs(n_samples=size, n_features=2, centers=4, cluster_std=1, center_box=(-10.0, 10.0),
                                      shuffle=True, random_state=1)
        spectral_clustering_and_plot(X_moons, 2, "make_moons", size)
        spectral_clustering_and_plot(X_blobs, 4, "make_blobs", size)


if __name__ == "__main__":
    # X_moons, y_moons, X_blobs, y_blobs = visual_dataset()
    #
    # spectral_clustering_and_plot(X_moons, 2, "make_moons", 1000)
    # spectral_clustering_and_plot(X_blobs, 4, "make_blobs", 20000)

    diff_size_clust()
