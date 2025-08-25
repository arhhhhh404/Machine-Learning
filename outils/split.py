import numpy as np
import matplotlib.pyplot as plt

pourcent = float(input("Entrer le nombre de pourcentage du dataset que vous voulez pour le trainset (0.x) : \n"))

np.random.seed(0)
X = np.random.rand(100, 3)
y = np.random.randint(0, 2, size=100)

n = len(X)
train_size = int(pourcent * n)

indices = np.arange(n)
np.random.shuffle(indices)

train_index = indices[:train_size]
test_index = indices[train_size:]

X_train, y_train = X[train_index], y[train_index]
X_test, y_test = X[test_index], y[test_index]

plt.figure(figsize=(15, 7))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, label="train set", alpha=0.7)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, label="test set", alpha=0.7)
plt.legend()
plt.title("Train / Test split")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
