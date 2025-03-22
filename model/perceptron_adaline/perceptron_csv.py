import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Perceptron:
    def __init__(self, eta=0.05, max_iter=10000, tol=0.01):
        self.eta = eta  # Taux d'apprentissage
        self.max_iter = max_iter  # Nombre maximum d'itérations
        self.tol = tol  # Seuil de tolérance pour l'erreur

    def fit(self, X, y):
        # 1. Initialiser les poids synaptiques avec des valeurs aléatoires
        self.w = np.random.normal(0, 0.1, X.shape[1] + 1)  # +1 pour le biais
        errors = []

        # Compteur d'itérations
        iteration = 0

        while iteration < self.max_iter:
            # 2. Pour chaque exemple du jeu d'apprentissage
            for k in range(len(X)):
                # (a) Évaluer la sortie du Perceptron
                xi = X[k]
                target = y[k]
                y_pred = self.predict_proba(xi)

                # (b) Calculer l'erreur
                error = target - y_pred

                # (c) Corriger les poids synaptiques selon la règle de Widrow-Hoff
                xi_bias = np.append(1, xi)  # Ajout du biais
                self.w = self.w + self.eta * error * xi_bias

            # 3. Calculer l'erreur quadratique moyenne
            mse = np.mean([(y[k] - self.predict_proba(X[k])) ** 2 for k in range(len(X))])
            errors.append(mse)

            # Afficher l'erreur toutes les 1000 itérations
            if iteration % 1000 == 0:
                print(f"Iteration {iteration}: MSE = {mse:.6f}")

            # 4. Vérifier si l'erreur est inférieure au seuil de tolérance
            if mse < self.tol:
                print(f"Convergence après {iteration} iterations")
                break

            iteration += 1

        if iteration == self.max_iter:
            print(f"Nombre maximum d'itérations atteint sans convergence")

        return errors

    def predict_proba(self, X):
        # Fonction sigmoïde pour calculer la sortie du Perceptron
        return 1 / (1 + np.exp(-np.dot(np.append(1, X), self.w)))

    def predict(self, X):
        # Classification binaire basée sur un seuil de 0.5
        return (self.predict_proba(X) > 0.5).astype(int)


def plot_decision_boundary(model, X, y):
    # Création d'une grille pour visualiser la frontière de décision
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
    # Visualisation de la courbe d'apprentissage
    plt.figure(figsize=(10, 6))
    plt.plot(errors)
    plt.xlabel('Itération')
    plt.ylabel('Erreur quadratique moyenne')
    plt.title('Courbe d\'apprentissage')
    plt.grid(True)
    plt.show()


def load_data(filename):
    # Chargement des données depuis un fichier CSV
    df = pd.read_csv(filename)
    X = df.iloc[:, :-1].values  # Caractéristiques
    y_orig = df.iloc[:, -1].values  # Étiquettes
    y = np.where(y_orig == 1, 1, 0)  # Conversion pour sigmoïde
    return X, y
