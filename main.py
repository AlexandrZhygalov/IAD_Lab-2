from sklearn.datasets import make_moons, make_blobs
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score, pairwise_distances, make_scorer
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import time
import random
import numpy as np
from sklearn.model_selection import GridSearchCV
import numpy as np
import warnings
warnings.filterwarnings("ignore")


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
    plt.title(f"Кластерізація - '{dataset_name}'")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
    return labels


def diff_size_clust(): # task 5
    size_array = [10000, 20000, 30000, 40000, 50000]

    for size in size_array:
        print("")
        X_moons, y_moons = make_moons(n_samples=size, noise=0.05, random_state=0)
        X_blobs, y_blobs = make_blobs(n_samples=size, n_features=2, centers=4, cluster_std=1, center_box=(-10.0, 10.0),
                                      shuffle=True, random_state=1)
        spectral_clustering_and_plot(X_moons, 2, "make_moons", size)
        spectral_clustering_and_plot(X_blobs, 4, "make_blobs", size)


def best_model_search(X, dataset_name):
    # Власний скорер для кластеризації
    def custom_scorer(estimator, X):
        labels = estimator.fit_predict(X)  # Кластеризація
        return silhouette_score(X, labels)  # Повернення оцінки
    # Параметри для пошуку
    param_grid = {'n_clusters': [2, 3, 4, 5],
                  'assign_labels': ['kmeans', 'discretize', 'cluster_qr'],
                  'eigen_solver': ['arpack', 'lobpcg']}
    # Модель SpectralClustering
    sc_model = SpectralClustering(affinity='rbf', random_state=42)
        # Пошук найкращих параметрів
    grid_search = GridSearchCV(estimator=sc_model,
                               param_grid=param_grid,
                               scoring=custom_scorer,  # Передаємо скорер без y_true
                               cv=3,  # Кількість фолдів для крос-валідації
                               verbose=0,  # Вимикає вивід
                               n_jobs=-1)  # Використання всіх доступних ядер процесора
    grid_search.fit(X)
    # Вивід результатів
    print(f"Найкращі параметри для набору даних '{dataset_name}': {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    labels = best_model.fit_predict(X)
    # Побудова графіка
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)
    plt.title(f"Найкраща кластеризація для набору даних '{dataset_name}'")
    plt.show()


def evaluate_clustering(X, Y, labels): # Task 7
    print(f"Estimated number of clusters: {len(set(labels))}")
    print(f"Adjusted Rand Index: {adjusted_rand_score(Y, labels):.3f}")
    print(f"Silhouette Coefficient: {silhouette_score(X, labels):.3f}")
    print(f"Davies-Bouldin Index: {davies_bouldin_score(X, labels):.3f}")


def stability_test(X_set, y_set, model, name):
    remove_data = [0.1, 0.5, 0.9]

    for rem in remove_data:
        print(f"Data lose: {rem*100}%")
        n_remove = int(len(X_set) * rem)
        indices_to_remove = random.sample(range(len(X_set)), n_remove)
        y_set_removed = np.delete(y_set, indices_to_remove, axis=0)
        X_set_removed = np.delete(X_set, indices_to_remove, axis=0)
        labels_removed = model.fit_predict(X_set_removed)

        evaluate_clustering(X_set_removed, y_set_removed, labels_removed)
        print("\n")
        plt.scatter(X_set_removed[:, 0], X_set_removed[:, 1], c=labels_removed, cmap='viridis', edgecolor='k', s=30)
        plt.title(f"Набір даних та модель - '{name}', 'removed - {rem*100}%'")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()


def alt_models(X_moons, X_blobs):
    model_moons_1 = SpectralClustering(n_clusters=2, affinity='rbf', assign_labels='kmeans', eigen_solver='arpack',
                                       gamma=1.0, random_state=42)
    model_moons_2 = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', assign_labels='discretize',
                                       eigen_solver='lobpcg', gamma=1.0, random_state=42)
    model_moons_3 = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='kmeans',
                                       eigen_solver='arpack', gamma=1.0, random_state=42)
    model_blobs_1 = SpectralClustering(n_clusters=4, affinity='nearest_neighbors', assign_labels='kmeans',
                                       eigen_solver='arpack', gamma=1.0, random_state=42)
    model_blobs_2 = SpectralClustering(n_clusters=5, affinity='rbf', assign_labels='discretize',
                                       eigen_solver='lobpcg', gamma=1.0, random_state=42)
    model_blobs_3 = SpectralClustering(n_clusters=4, affinity='rbf', assign_labels='kmeans',
                                       eigen_solver='arpack', gamma=1.0, random_state=42)

    label_moon_1 = model_moons_1.fit_predict(X_moons)
    label_moon_2 = model_moons_2.fit_predict(X_moons)
    label_moon_3 = model_moons_3.fit_predict(X_moons)
    label_blobs_1 = model_blobs_1.fit_predict(X_blobs)
    label_blobs_2 = model_blobs_2.fit_predict(X_blobs)
    label_blobs_3 = model_blobs_3.fit_predict(X_blobs)

    return (model_moons_1, model_moons_2, model_moons_3, model_blobs_1, model_blobs_2, model_blobs_3,
            label_moon_1, label_moon_2, label_moon_3, label_blobs_1, label_blobs_2, label_blobs_3)


if __name__ == "__main__":
    # Task 1
    X_moons, y_moons, X_blobs, y_blobs = visual_dataset()

    # Task 2, 3, 4
    label_moons = spectral_clustering_and_plot(X_moons, 2, "make_moons", 1000)
    label_blobs = spectral_clustering_and_plot(X_blobs, 4, "make_blobs", 20000)

    # Task 5
    # diff_size_clust()

    # Task 6
    X_blobs_new, y_blobs_new = make_blobs(n_samples=2000, n_features=2, centers=4, cluster_std=1, center_box=(-10.0, 10.0),
                                  shuffle=True, random_state=1)
    (model_moons_1, model_moons_2, model_moons_3,
     model_blobs_1, model_blobs_2, model_blobs_3,
     label_moon_1, label_moon_2, label_moon_3,
     label_blobs_1, label_blobs_2, label_blobs_3) = alt_models(X_moons, X_blobs_new)

    # Task 7
    params = ["moons (n_clusters: 2, affinity: 'rbf', assign_labels: 'kmeans', eigen_solver: 'arpack')",
              "moons (n_clusters: 3, affinity: 'nearest_neighbors', assign_labels: 'discretize', eigen_solver: 'lobpcg')",
              "moons (n_clusters: 2, affinity: 'nearest_neighbors', assign_labels: 'kmeans', eigen_solver: 'arpack')",
              "blobs (n_clusters: 4, affinity: 'nearest_neighbors', assign_labels: 'kmeans', eigen_solver: 'arpack')",
              "blobs (n_clusters: 5, affinity: 'rbf', assign_labels: 'discretize', eigen_solver: 'lobpcg')",
              "blobs (n_clusters: 4, affinity: 'rbf', assign_labels: 'kmeans', eigen_solver: 'arpack')"]
    names = ["moons_1", "moons_2", "moons_3", "blobs_1", "blobs_2", "blobs_3"]
    labels = [label_moon_1, label_moon_2, label_moon_3, label_blobs_1, label_blobs_2, label_blobs_3]
    models = [model_moons_1, model_moons_2, model_moons_3, model_blobs_1, model_blobs_2, model_blobs_3]
    datasets_X = [X_moons, X_moons, X_moons, X_blobs_new, X_blobs_new, X_blobs_new]
    datasets_Y = [y_moons, y_moons, y_moons, y_blobs_new, y_blobs_new, y_blobs_new]

    for label, X, Y, param in zip(labels, datasets_X, datasets_Y, params):
        print(f"\nMetrics for {param}")
        evaluate_clustering(X, Y, label)

    # Task 8
    for model, X, Y, param, name in zip(models, datasets_X, datasets_Y, params, names):
        print(f"\nStability test for model {param}")
        stability_test(X, Y, model, name)

    # Task 11
    X_blobs_for_best, y_blobs_for_best = make_blobs(n_samples=2000, n_features=2, centers=4, cluster_std=1, center_box=(-10.0, 10.0),
                                  shuffle=True, random_state=1)
    print("Найкращі параметри для наборів даних")
    best_model_search(X_moons,  "make_moons")
    best_model_search(X_blobs_for_best, "make_blobs")
