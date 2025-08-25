import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

sample = int(input("Entrer le nombre d'échantillons : "))
X, y = make_classification(n_samples=sample, n_features=2, n_redundant=0, n_clusters_per_class=1, n_classes=2, random_state=42)

def gini(y):
    classe = np.unique(y)
    impurty = 1
    for c in classe:
        p = np.sum(y == c)/len(y)
        impurty -= p ** 2
    return impurty

def best_split(X, y):
    n_sample, n_features = X.shape
    best_feat, best_thresh = None, None
    best_gini, best_sets = 1, None

    for feat in range(n_features):
        thresholds = np.unique(X[:, feat])
        for threshold in thresholds:
            left = X[:, feat] <= threshold
            right = ~left
            if np.sum(left) == 0 or np.sum(right) == 0:
                continue
            impurty_left = gini(left)
            impurty_right = gini(right)
            impurty = (len(y[left]) * impurty_left + len(y[right]) * impurty_right) / len(y)
            if impurty < best_gini:
                best_gini = impurty
                best_feat = feat
                best_thresh = threshold
                best_sets = (left, right)

    return best_feat, best_thresh, best_sets
            

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def build_tree(X, y, depth=0, max_depth=5):
    feat, thresh, sets = best_split(X, y)
    if len(np.unique(y)) == 1 or depth >= max_depth:
        values, counts = np.unique(y, return_counts=True)
        return Node(value=values[np.argmax(counts)])

    if feat is None:
        values, counts = np.unique(y, return_counts=True)
        return Node(value=values[np.argmax(counts)])
    
    left, right = sets
    left_child = build_tree(X[:, left), y[:, left], depth+=1, max_depth)
    right_child = build_tree(X[:, right), y[:, left], depth+=1, max_depth)
    return Node(feature = feat, threshold = thresh, left = left_child, right = right_child)

def predict_one(node, x):
    if node.value is not None:
           return node.value
    if x[node.feature] <= node.threshold:
           return predict_one(node.left, x)
    else:
           return predict_one(node.right, x)

def predict(node, x):
    return np.array([predict_one(node, x) for x in X])

tree = build_tree(X, y)

h = 0.05
x_min, y_min = X[:,0].min() - 1, X[:,0].min() + 1
x_max, y_max = X[:,1].max() - 1, X[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = predict(tree, np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(15, 7))
plt.contour(xx, yy, Z, alpha=0.3, cmap="coolwarm")
plt.scatter(X[:,0], X[:,1], c=y, cmap="coolwarm", edgecolor="k")
plt.title("Decision Tree (2D classification")
plt.xlabel("feature1")
plt.ylabel("feature2")
plt.show()
