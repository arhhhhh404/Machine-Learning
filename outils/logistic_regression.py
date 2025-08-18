import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

sample = int(input("Entrer le nombre d'échantillons : "))

X, y = make_classification(n_samples=sample, n_features=1, n_informative=1, n_redundant=0, n_clusters_per_class=1)
y = y.reshape(sample, 1)

X_b = np.hstack((X, np.ones(X.shape)))
theta = np.random.randn(2, 1)

def model(X, theta):
    return 1 / (1 + np.exp(-X.dot(theta)))

def cost_function(model, y):
    m = len(y)
    h = model(X, theta)
    return -1/m * np.sum(y*np.log(h) + (1-y)*np.log(1-h))

def grad(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(model(X, theta) - y)

def gradient_descent(X, y, theta, learning_rate, n_iterations):
    for i in range(n_iterations):
        theta = theta - learning_rate * grad(X, y, theta)
    return theta

theta_final = gradient_descent(X_b, y, theta, learning_rate=0.1, n_iterations=1000)

prediction = model(X_b, theta_final)

plt.figure(figsize=(15, 10))
plt.scatter(X, y, label="Données")
plt.scatter(X, prediction, color='r', marker='x', label="Prédictions")
plt.title("courbe de régression Logistique")
plt.legend()
plt.show()