import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

sample = int(input("Entrer le nombre d'échantillons : "))
noise = int(input("Entrer le niveau de bruit ajouté aux données : "))

x, y = make_regression(n_samples=sample, n_features=1, noise=noise)
y = y.reshape(sample, 1)

X = np.hstack((x, np.ones(x.shape)))
theta = np.random.randn(2, 1)

def model(X, theta):
    return X.dot(theta)

def cost_function(model, y):
    m = len(y)
    return 1/(2*m)*np.sum((model(X, theta) - y)**2)

def grad(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(model(X, theta) - y)

def gradient_descent(X, y, theta, learning_rate, n_iterations):
    for i in range(0, n_iterations):
        theta = theta - learning_rate * grad(X, y, theta)
    return theta

theta_final = gradient_descent(X, y, theta, learning_rate = 0.001, n_iterations=1000)

prediction = model(X, theta_final)

plt.figure(figsize=(15, 10))
plt.scatter(x, y)
plt.plot(x, prediction, c="r")
<<<<<<< HEAD
plt.title("courbe de régression")
plt.show()
=======
plt.title("courbe de régression linéaire")
plt.show()
>>>>>>> d2bcad8 (commit8)
