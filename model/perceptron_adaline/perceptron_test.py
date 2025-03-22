import numpy as np
import matplotlib.pyplot as plt


class Adaline:
    def __init__(self, eta=0.01, max_iter=10000, tol=0.0001):
        self.eta = eta  # Taux d'apprentissage
        self.max_iter = max_iter  # Nombre maximum d'itérations
        self.tol = tol  # Seuil de tolérance pour l'erreur

    def fit(self, X, y):
        # Initialisation des poids avec des valeurs aléatoires
        self.w = np.random.normal(0, 0.1, X.shape[1] + 1)  # +1 pour le biais
        errors = []

        # Boucle d'apprentissage
        for i in range(self.max_iter):
            # Calcul des sorties pour tous les exemples
            net_outputs = np.array([self.net_input(xi) for xi in X])

            # Calcul de l'erreur quadratique moyenne
            mse = np.mean((y - net_outputs) ** 2)
            errors.append(mse)

            # Affichage périodique
            if i % 1000 == 0:
                print(f"Iteration {i}: MSE = {mse:.6f}")

            # Vérification de la convergence
            if mse < self.tol:
                print(f"Convergence après {i} iterations")
                break

            # Mise à jour des poids selon la règle de Widrow-Hoff
            for xi, target, output in zip(X, y, net_outputs):
                xi_bias = np.append(1, xi)  # Ajout du biais
                self.w += self.eta * (target - output) * xi_bias

        # Si max_iter est atteint sans convergence
        if i == self.max_iter - 1:
            print(f"Nombre maximum d'itérations atteint sans convergence")

        return errors

    def net_input(self, X):
        # Calcul de la somme pondérée (inclut le biais)
        return np.dot(np.append(1, X), self.w)

    def predict(self, X):
        # Pour un ensemble de données
        if len(X.shape) > 1:
            return np.array([self.predict(xi) for xi in X])
        # Pour un exemple unique
        return 1 if self.net_input(X) >= 0.0 else -1


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
    plt.title('Frontière de décision Adaline')
    plt.grid(True)
    plt.show()


def plot_learning_curve(errors):
    plt.figure(figsize=(10, 6))
    plt.plot(errors)
    plt.xlabel('Itération')
    plt.ylabel('Erreur quadratique moyenne')
    plt.title('Courbe d\'apprentissage Adaline')
    plt.grid(True)
    plt.show()


# Données du tableau 2.9
X = np.array([
    [1, 6], [7, 9], [1, 9], [7, 10], [2, 5],
    [2, 7], [2, 8], [6, 8], [6, 9], [3, 5],
    [3, 6], [3, 8], [3, 9], [5, 7], [5, 8],
    [5, 10], [5, 11], [4, 6], [4, 7], [4, 9],
    [4, 10]
])

# Étiquettes correspondantes
y = np.array([
    1, -1, 1, -1, -1,
    1, 1, -1, -1, -1,
    -1, 1, 1, -1, -1,
    1, 1, -1, -1, 1,
    1
])

# Apprentissage
model = Adaline(eta=0.005, max_iter=10000, tol=0.0001)
errors = model.fit(X, y)

# Évaluation
accuracy = np.mean(model.predict(X) == y)
print(f"\nPoids finaux: {model.w}")
print(f"Précision: {accuracy * 100:.2f}%")

# Visualisations
plot_decision_boundary(model, X, y)
plot_learning_curve(errors)