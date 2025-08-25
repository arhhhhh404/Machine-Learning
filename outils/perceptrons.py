import numpy as np
import matplotlib.pyplot as plt

X, y = 
y = y.reshape((y.shape[0], 1))

def initialisation(X):
    W = np.random.randn(
    b = np.random.randn(1)
    return (W, b)

def model(X, W, b):
    Z = X.dot(W) + b
    sigmoid = 1 / (1 + np.exp(-Z))
    return sigmoid

def log_loss(A, y):
    return (1 / len(y)) * (np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A)))

def gradients(A, X, y):
    dW = (1/len(y)) * np.dot(X.T, A - y)
    db = (1/len(y)) * np.sum(A - y)
    return (dW, db)

def update(dW, db, W, b, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return (W, b)

def artificial_neuron(X, y, learning_rate = 0.1, n_iterations = 100):
    W, b = initialisation(X)
    Loss = []
    for i in range(n_iterations):
        A = model(X, W, b)
        Loss.append(log_loss(A, y))
        dW, db = gradients(A, X, y)
        W, b = update(dW, db, W, b, learning_rate)

    plt.plot(Loss)
    plt.show()

artificial_neuron(X, y)
