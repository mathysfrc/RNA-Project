import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

class LinearRegression:
    def __init__(self, eta=0.001, max_iter=10000, tol=0.01):
        self.eta = eta  # Taux d'apprentissage
        self.max_iter = max_iter  # Nombre maximal d'itérations
        self.tol = tol  # Tolérance pour la convergence

    def fit(self, X, y):
        # Initialisation des poids avec une petite valeur aléatoire
        self.w = np.random.normal(0, 0.01, X.shape[1] + 1)
        errors = []

        for i in range(self.max_iter):
            delta_w = np.zeros(len(self.w))
            for xi, target in zip(X, y):
                pred = self.predict(xi)
                error = target - pred
                xi_bias = np.append(1, xi)  # Ajouter un biais
                delta_w += self.eta * error * xi_bias  # Calcul de la mise à jour des poids

            self.w += delta_w  # Mise à jour des poids

            # Calcul de l'erreur quadratique moyenne
            mse = np.mean([(target - self.predict(xi)) ** 2 for xi, target in zip(X, y)])
            errors.append(mse)

            if i % 1000 == 0:
                print(f"Iteration {i}: MSE = {mse:.6f}")

            if mse < self.tol:  # Convergence si l'erreur est suffisamment faible
                print(f"Convergence après {i} itérations")
                break

            # Vérification des valeurs infinies ou NaN dans les poids
            if np.any(np.isnan(self.w)) or np.any(np.isinf(self.w)):
                print(f"Problème avec les poids après l'itération {i}")
                break

        return errors

    def predict(self, X):
        xi_bias = np.append(1, X)  # Ajouter un biais (1) pour le calcul du produit scalaire
        return np.dot(xi_bias, self.w)  # Prédiction linéaire

def plot_learning_curve(errors):
    plt.figure(figsize=(10, 6))
    plt.plot(errors)
    plt.xlabel('Itération')
    plt.ylabel('Erreur quadratique moyenne')
    plt.title('Courbe d\'apprentissage')
    plt.grid(True)
    plt.show()

def plot_regression_curve(X, y, model):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], y, color='blue', label='Données')
    plt.plot(X[:, 0], model.predict(X), color='red', label='Régression')
    plt.xlabel('Caractéristique')
    plt.ylabel('Valeur cible')
    plt.title('Courbe de régression linéaire')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_classification_curve(X, y, model):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='o', edgecolor='k', s=100)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    XX, YY = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = np.array([model.predict(np.array([xx, yy])) for xx, yy in zip(XX.ravel(), YY.ravel())])
    Z = Z.reshape(XX.shape)
    plt.contourf(XX, YY, Z, cmap='viridis', alpha=0.3)
    plt.title('Courbe de classification')
    plt.grid(True)
    plt.show()

def load_data(filename):
    df = pd.read_csv(filename)
    X = df.iloc[:, :-1].values  # Caractéristiques
    y = df.iloc[:, -1].values  # Valeurs cibles
    return X, y

# Chargement des données depuis un fichier CSV (exemple 2_9)
filename = "../../datas/table_2_9.csv"
X, y = load_data(filename)

# Normalisation des données
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Apprentissage
model = LinearRegression(eta=0.001)
errors = model.fit(X, y)

# Évaluation
predictions = np.array([model.predict(xi) for xi in X])
mse = np.mean((y - predictions) ** 2)
print(f"\nPoids finaux: {model.w}")
print(f"Erreur quadratique moyenne: {mse:.6f}")

# Visualisation des courbes
plot_learning_curve(errors)
plot_regression_curve(X, y, model)  # Affiche la courbe de régression
plot_classification_curve(X, y, model)  # Affiche la courbe de classification
