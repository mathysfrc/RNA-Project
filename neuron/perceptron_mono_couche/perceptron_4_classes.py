import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class MulticlassPerceptron:
    def __init__(self, num_classes=4, eta=0.05, max_iter=3000, tol=0.01):
        self.eta = eta  # Taux d'apprentissage
        self.max_iter = max_iter  # Nombre maximum d'itérations
        self.tol = tol  # Seuil de tolérance pour l'erreur
        self.num_classes = num_classes  # Nombre de classes

    def softmax(self, z):
        # Fonction softmax pour la classification multiclasse
        exp_z = np.exp(z - np.max(z))  # Soustraction du max pour éviter l'overflow
        return exp_z / exp_z.sum()

    def one_hot_encode(self, y):
        # Encodage one-hot personnalisé (sans scikit-learn)
        n_samples = len(y)
        encoded = np.zeros((n_samples, self.num_classes))
        for i, label in enumerate(y):
            encoded[i, int(label)] = 1
        return encoded

    def fit(self, X, y):
        # Convertir les étiquettes en format one-hot
        y_one_hot = self.one_hot_encode(y)

        # Initialiser les poids avec de petites valeurs aléatoires (un ensemble de poids par classe)
        self.w = np.random.randn(X.shape[1] + 1, self.num_classes) * 0.1  # +1 pour le biais
        self.errors = []  # Liste des erreurs pendant l'apprentissage

        for iteration in range(self.max_iter):
            total_error = 0

            # Parcourir chaque exemple
            for xi, target_one_hot in zip(X, y_one_hot):
                xi_bias = np.append(1, xi)  # Ajouter le biais à l'entrée

                # Calculer les scores pour chaque classe
                scores = np.dot(xi_bias, self.w)

                # Appliquer softmax pour obtenir les probabilités
                output_probs = self.softmax(scores)

                # Calculer l'erreur
                errors = target_one_hot - output_probs

                # Mise à jour des poids (pour chaque classe)
                for j in range(self.num_classes):
                    self.w[:, j] += self.eta * errors[j] * xi_bias

                # Accumuler l'erreur quadratique
                total_error += np.sum(errors ** 2)

            # Erreur moyenne
            mse = total_error / (len(X) * self.num_classes)
            self.errors.append(mse)

            # Affichage de l'erreur tous les 1000 itérations
            if iteration % 1000 == 0:
                print(f"Iteration {iteration}: MSE = {mse:.6f}")

            # Vérifier si l'erreur est en dessous du seuil de tolérance
            if mse < self.tol:
                print(f"Convergence après {iteration} itérations")
                break

        return self.errors

    def predict_proba(self, X):
        """Retourne les probabilités pour chaque classe"""
        if X.ndim == 1:
            X = X.reshape(1, -1)

        probas = []
        for xi in X:
            xi_bias = np.append(1, xi)
            scores = np.dot(xi_bias, self.w)
            probas.append(self.softmax(scores))

        return np.array(probas)

    def predict(self, X):
        """Retourne la classe prédite"""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)


def plot_decision_boundary(model, X, y):
    # Création d'une grille pour visualiser la frontière de décision
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Prédiction pour chaque point de la grille
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # Visualisation avec la nouvelle syntaxe pour obtenir une colormap
    plt.figure(figsize=(10, 8))
    # Utiliser la nouvelle syntaxe recommandée pour les colormaps
    cmap = plt.colormaps['tab10'].resampled(model.num_classes)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=cmap)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k', s=80)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Frontière de décision pour classification à 4 classes')
    plt.grid(True)
    plt.colorbar(ticks=range(model.num_classes), label='Classe')
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
    y = df.iloc[:, -1].values  # Étiquettes (supposées être 0, 1, 2, 3 pour 4 classes)
    return X, y


# Exemple d'utilisation avec des données synthétiques pour 4 classes
def generate_synthetic_data(n_samples=100, n_features=2, n_classes=4):
    """Génère des données synthétiques pour tester le perceptron multiclasse"""
    np.random.seed(42)
    X = np.random.randn(n_samples * n_classes, n_features) * 2

    # Créer des clusters pour chaque classe
    for i in range(n_classes):
        X[i * n_samples:(i + 1) * n_samples, 0] += i * 3
        X[i * n_samples:(i + 1) * n_samples, 1] += (i % 2) * 3

    y = np.zeros(n_samples * n_classes)
    for i in range(n_classes):
        y[i * n_samples:(i + 1) * n_samples] = i

    return X, y


# Génération de données synthétiques
X, y = generate_synthetic_data(n_samples=100, n_features=2, n_classes=4)

# Apprentissage
model = MulticlassPerceptron(num_classes=4, eta=0.03, max_iter=4000, tol=0.001)
errors = model.fit(X, y)

# Évaluation
predictions = model.predict(X)
accuracy = np.mean(predictions == y)
print(f"\nPrécision: {accuracy * 100:.2f}%")

# Visualisations
plot_decision_boundary(model, X, y)
plot_learning_curve(errors)

# Pour charger des données depuis un fichier CSV, décommentez ces lignes
filename = "../../datas/table_3_5.csv"
X, y = load_data(filename)