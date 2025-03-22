import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Perceptron:
    def __init__(self, eta=0.9, max_iter=10000, tol=0.01):
        self.eta = eta  # Taux d'apprentissage
        self.max_iter = max_iter  # Nombre maximum d'itérations
        self.tol = tol  # Seuil de tolérance pour l'erreur

    def fit(self, X, y):
        # Initialiser les poids avec de petites valeurs aléatoires
        self.w = np.random.randn(X.shape[1] + 1) * 0.1  # +1 pour le biais
        self.errors = []  # Liste des erreurs pendant l'apprentissage
        self.X = X
        self.y = y

        for iteration in range(self.max_iter):
            error_count = 0

            # Apprentissage par rétropropagation
            for xi, target in zip(X, y):
                xi_bias = np.append(1, xi)  # Ajouter le biais à l'entrée
                output = self.sigmoid(np.dot(xi_bias, self.w))  # Sortie du perceptron
                error = target - output  # Calcul de l'erreur
                if error != 0:
                    error_count += 1

                # Mise à jour des poids
                self.w += self.eta * error * xi_bias  # Règle de mise à jour des poids

            # Calcul de l'erreur quadratique moyenne
            mse = np.mean([(target - self.sigmoid(np.dot(np.append(1, xi), self.w))) ** 2 for xi, target in zip(X, y)])
            self.errors.append(mse)

            # Affichage de l'erreur tous les 1000 itérations
            if iteration % 1000 == 0:
                print(f"Iteration {iteration}: MSE = {mse:.6f}, erreurs = {error_count}")

            # Vérifier si l'erreur est en dessous du seuil de tolérance
            if mse < self.tol:
                print(f"Convergence après {iteration} itérations")
                break

        return self.errors

    def sigmoid(self, z):
        # Fonction sigmoïde
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        # Prédiction binaire (seuil de 0.5)
        return (self.sigmoid(np.dot(np.append(1, X), self.w)) > 0.5).astype(int)


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


# Chargement des données depuis un fichier CSV
filename = "../../datas/table_2_10.csv"
X, y = load_data(filename)

# Apprentissage
model = Perceptron(eta=0.05, max_iter=10000, tol=0.01)
errors = model.fit(X, y)

# Évaluation
accuracy = sum(model.predict(xi) == yi for xi, yi in zip(X, y)) / len(X)
print(f"\nPoids finaux: {model.w}")
print(f"Précision: {accuracy * 100:.2f}%")

# Visualisations
plot_decision_boundary(model, X, y)
plot_learning_curve(errors)
