import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

#initialisation des arguments
sample = int(input("Entrer le nombre d'échantillons : "))
noise = int(input("Entrer le niveau de bruit ajouté aux données : "))

#création des données
x, y = make_regression(n_samples=sample, n_features=1, noise=noise)
y = y.reshape(sample, 1)

#ajout d'une colone de biais + initialisation du paramètre theta
X = np.hstack((x, np.ones(x.shape)))
theta = np.random.randn(2, 1)

#model de regression linéaire
def model(X, theta):
    return X.dot(theta)

#fonction de cout (MSE - erreur moyenne)
def cost_function(model, y):
    m = len(y)
    return 1/(2*m)*np.sum((model(X, theta) - y)**2)

#calcul de gradient
def grad(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(model(X, theta) - y)

#descente de gradient
def gradient_descent(X, y, theta, learning_rate, n_iterations):
    for i in range(0, n_iterations):
        theta = theta - learning_rate * grad(X, y, theta)
    return theta

#execution de la descente de gradient
theta_final = gradient_descent(X, y, theta, learning_rate = 0.001, n_iterations=1000)

#prédiction des paramètre optimisé
prediction = model(X, theta_final)

#visualisation
plt.figure(figsize=(15, 10))
plt.scatter(x, y)
plt.plot(x, prediction, c="r")
plt.title("courbe de régression")
plt.show()