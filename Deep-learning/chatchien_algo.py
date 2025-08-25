import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score

def load_h5py(name):


X_train, y_train = load_h5py("trainset.hdf5")
X_train_reshape = X_train.reshape(X_train[0], X_train[1] * X_train[2]) / X_train.max()
X_test, y_test = load_h5py("testset.hdf5")
X_test_reshape = X_test_reshape(X_test[0], X_test[1] * X_test[2]) / X_train.max()

def initialisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)

def model(X, W, b):
    Z = X.dot(W) + b
    sigmoid = 1 / (1 + np.exp(-Z))
    return sigmoid

def log_loss(A, y, epsilon=1e-15):
    return (1 / len(y)) * (np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon)))

def gradients(A, X, y):
    dW = (1/len(y)) * np.dot(X.T, A - y)
    db = (1/len(y)) * np.sum(A - y)
    return (dW, db)

def update(dW, db, W, b, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return (W, b)

def artificial_neuron(X_train, y_train,X_test, y_test learning_rate, n_iterations):
    W, b = initialisation(X_train)
    Loss_train = []
    Acc_train = []
    Loss_test = []
    Acc_test = []
    for i in tqdm(range(n_iterations)):
        A = model(X_train, W, b)

        if i % 10 == 0:
            Loss_train.append(log_loss(A, y_train))
            y_pred = predict(X_train, W, b)
            Acc_train.append(accuracy_score(y_train, y_pred)
            
            A_test = model(X_test, W, b)
            Loss_test.append(log_loss(A_test, y_test))
            y_pred = predict(X_test, W, b)
            Acc_test.append(accuracy_score(y_test, y_pred)

        dW, db = gradients(A, X_train, y_train)
        W, b = update(dW, db, W, b, learning_rate)

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(Loss_train, label="train loss")
    plt.plot(Loss_test, label="test loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(Acc_train, label="train accuracy")
    plt.plot(Acc_test, label="test accuracy")
    plt.legend()
    plt.show()

    return (W, b)

W, b = artificial_neuron(X_train_reshape, y_train, X_test_reshape, y_test, 0.01, 1000)
