import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Fonction d'activation Sigmoid et sa dérivée
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Classe pour le Perceptron multicouche
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialisation des poids et des biais
        self.W1 = np.random.randn(input_size, hidden_size)  # Poids de la couche cachée
        self.b1 = np.zeros((1, hidden_size))  # Biais de la couche cachée
        self.W2 = np.random.randn(hidden_size, output_size)  # Poids de la couche de sortie
        self.b2 = np.zeros((1, output_size))  # Biais de la couche de sortie

    def forward(self, X):
        # Propagation avant
        self.hidden_input = np.dot(X, self.W1) + self.b1  # Entrée cachée
        self.hidden_output = sigmoid(self.hidden_input)  # Sortie cachée
        self.output_input = np.dot(self.hidden_output, self.W2) + self.b2  # Entrée de sortie
        self.output = sigmoid(self.output_input)  # Sortie
        return self.output

    def backward(self, X, y, learning_rate):
        # Rétropropagation de l'erreur
        output_error = y - self.output  # Erreur de la sortie
        output_delta = output_error * sigmoid_derivative(self.output)  # Delta de la sortie

        hidden_error = np.dot(output_delta, self.W2.T)  # Erreur de la couche cachée
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)  # Delta de la couche cachée

        # Mise à jour des poids et biais
        self.W2 += np.dot(self.hidden_output.T, output_delta) * learning_rate
        self.b2 += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

        self.W1 += np.dot(X.T, hidden_delta) * learning_rate
        self.b1 += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        # Entraînement du modèle
        for epoch in range(epochs):
            self.forward(X)  # Propagation avant
            self.backward(X, y, learning_rate)  # Rétropropagation
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - self.output))  # Erreur quadratique moyenne
                print(f"Epoch {epoch}, Loss: {loss}")

# Chargement des données depuis le fichier CSV
data = pd.read_csv("../../datas/table_4_12.csv")

# Séparation des entrées et des sorties
X = data.iloc[:, :-1].values  # Entrées (colonnes 1 et 2)
y = data.iloc[:, -1].values  # Sortie (colonne 3)

# Normalisation manuelle des données
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X = (X - X_min) / (X_max - X_min)  # Normalisation min-max

# Reshape de y pour correspondre à la dimension de sortie
y = y.reshape(-1, 1)

# Initialisation du perceptron multicouche
input_size = X.shape[1]  # Nombre d'entrées
hidden_size = 5  # Nombre de neurones dans la couche cachée
output_size = 1  # Sortie binaire

mlp = MLP(input_size, hidden_size, output_size)

# Entraînement du modèle
epochs = 1000  # Nombre d'époques
learning_rate = 0.1  # Taux d'apprentissage

mlp.train(X, y, epochs, learning_rate)

# Test du modèle
y_pred = mlp.forward(X)
y_pred_class = (y_pred > 0.5).astype(int)  # Classe prédite (0 ou 1)
accuracy = np.mean(y_pred_class == y)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Visualisation des résultats

# Tracer les points de données
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='coolwarm', s=50, edgecolor='k', marker='o', label='Données')

# Créer un meshgrid pour visualiser la frontière de décision
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Prédictions sur le meshgrid
Z = mlp.forward(grid_points)
Z = (Z > 0.5).astype(int)  # Classification binaire

# Afficher la frontière de décision
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

# Légendes et titres
plt.title("Classification avec Perceptron Multicouche")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar()
plt.show()
