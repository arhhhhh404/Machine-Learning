import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

sample = int(input("Entrer le nombre de sample que vous voulez : "))
X, y_true = make_classification(n_samples=sample, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, n_classes=2, random_state=42)

def K_means(X, k, max_iter, tol=1e-4):
    n_sample, n_feature = X.shape

    rng = np.random.default_rng(42)
    centroid = X[rng.choice(n_sample, k, replace=False)]

    for i in range(max_iter):
        distance = np.linalg.norm(X[:, np.newaxis] - centroid, axis=2)
        label = np.argmin(distance, axis=1)

        new_centroid = np.array([X[label == i].mean(axis=0) for i in range(k)])
        
        if np.all(np.linalg.norm(new_centroid - centroid, axis=1) < tol):
            break

        centroid = new_centroid

    return centroid, label

k = 2
centroid, label = K_means(X, k, 100)

plt.figure(figsize=(10, 5))
plt.scatter(X[:, 0], X[:, 1], c=label, cmap="bwr", edgecolor="k", alpha=0.6)
plt.scatter(centroid[:, 0], centroid[:, 1], c="yellow", marker="X", s=200, label="centroid")
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.title("K_means clustering")
plt.legend()
plt.show()
