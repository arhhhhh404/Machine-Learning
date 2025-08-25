import numpy as np
import matplotlib.pyplot as plt
from  sklearn.datasets import make_classification
from  collections import Counter

sample = int(input("Entrer le nombre d'échantillons: "))

X, y = make_classification(n_samples=sample, n_features=2, n_redundant=0, n_clusters_per_class=1, n_classes=2, random_state=42)

def euclidian_distance(a, b):
             return np.sqrt(np.sum((a-b) ** 2, axis=1))

def KNN_predict(X_train, y_train, X_test, k):
             prediction = []
             for test_point in X_test:
                distance = euclidian_distance(X_train, test_point)
                k_indices = distance.argsort()[:k]
                k_labels = y_train[k_indices]
                most_common = Counter(k_labels).most_common(1)[0][0]
                prediction.append(most_common)
             return np.array(prediction)

k = 5

x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
grid_point = np.c_[xx.ravel(), yy.ravel()]

Z = KNN_predict(X, y, grid_point, k)
Z = Z.reshape(xx.shape)


plt.figure(figsize=(15, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
plt.scatter(X[:,0], X[:,1], c=y, cmap="coolwarm", edgecolor="k")
plt.title(f"KNN Decision Boundary (k={k})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
