import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Perceptron:
    def __init__(self, eta=0.05, max_iter=10000, tol=0.01):
        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        self.w = np.random.normal(0, 0.1, X.shape[1] + 1)
        errors = []

        for i in range(self.max_iter):
            delta_w = np.zeros(len(self.w))
            for xi, target in zip(X, y):
                pred = self.predict_proba(xi)
                error = target - pred
                xi_bias = np.append(1, xi)
                delta_w += self.eta * error * xi_bias

            self.w += delta_w
            mse = np.mean([(target - self.predict_proba(xi)) ** 2 for xi, target in zip(X, y)])
            errors.append(mse)

            if i % 1000 == 0:
                print(f"Iteration {i}: MSE = {mse:.6f}")

            if mse < self.tol:
                print(f"Convergence après {i} iterations")
                break

        return errors

    def predict_proba(self, X):
        return 1 / (1 + np.exp(-np.dot(np.append(1, X), self.w)))

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)


def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = np.array([model.predict(np.array([x, y])) for x, y in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, edgecolors='k', s=80)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Frontière de décision')
    plt.grid(True)
    plt.show()


def plot_learning_curve(errors):
    plt.figure(figsize=(10, 6))
    plt.plot(errors)
    plt.xlabel('Itération')
    plt.ylabel('Erreur quadratique moyenne')
    plt.title('Courbe d\'apprentissage')
    plt.grid(True)
    plt.show()


def load_data(filename):
    df = pd.read_csv(filename)
    X = df.iloc[:, :-1].values  # Caractéristiques
    y_orig = df.iloc[:, -1].values  # Étiquettes
    y = np.where(y_orig == 1, 1, 0)  # Conversion pour sigmoïde
    return X, y
