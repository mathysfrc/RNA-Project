import numpy as np
import matplotlib.pyplot as plt

# Fonction sigmoïde
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Dérivée de la sigmoïde
def sigmoid_derivative(x):
    return x * (1 - x)

# Initialisation de la classe MLP
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialisation des poids et des biais
        self.W1 = np.random.randn(input_size, hidden_size)  # Poids de la couche d'entrée à la couche cachée
        self.b1 = np.zeros((1, hidden_size))  # Biais de la couche cachée
        self.W2 = np.random.randn(hidden_size, output_size)  # Poids de la couche cachée à la couche de sortie
        self.b2 = np.zeros((1, output_size))  # Biais de la couche de sortie

    def forward(self, X):
        # Propagation avant
        self.hidden_input = np.dot(X, self.W1) + self.b1
        self.hidden_output = sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.W2) + self.b2
        self.final_output = sigmoid(self.final_input)  # Sortie après activation sigmoïde
        return self.final_output

    def backward(self, X, y, learning_rate=0.01):
        # Rétropropagation
        output_error = y - self.final_output  # Erreur de sortie
        d_output = output_error * sigmoid_derivative(self.final_output)  # Gradient de l'erreur

        hidden_error = np.dot(d_output, self.W2.T)  # Propagation de l'erreur vers la couche cachée
        d_hidden = hidden_error * sigmoid_derivative(self.hidden_output)  # Gradient de l'erreur de la couche cachée

        # Mise à jour des poids et biais
        self.W2 += np.dot(self.hidden_output.T, d_output) * learning_rate
        self.b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        self.W1 += np.dot(X.T, d_hidden) * learning_rate
        self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs=1000, learning_rate=0.01):
        accuracies = []
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, learning_rate)

            # Calcul de l'accuracy en utilisant un seuil de 0.5 pour la classification
            correct_predictions = np.sum((output > 0.5) == (y > 0.5))
            accuracy = correct_predictions / len(y) * 100

            # Affichage de l'Epoch et de l'Accuracy tous les 10 epochs
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Accuracy: {accuracy:.2f}%")

            accuracies.append(accuracy)

        return accuracies


# Chargement des données depuis le CSV
data = np.loadtxt("../../datas/table_4_17.csv", delimiter=',')
X = data[:, :-1]  # Données d'entrée (toutes les colonnes sauf la dernière)
y = data[:, -1:]  # Cibles (dernière colonne)

# Initialisation du MLP avec une couche cachée de taille 5 et une sortie de taille 1
mlp = MLP(input_size=X.shape[1], hidden_size=5, output_size=1)

# Entraînement du MLP
accuracies = mlp.train(X, y, epochs=1000, learning_rate=0.01)

# Affichage du graphique des accuracies
plt.figure(figsize=(8, 6))
plt.plot(accuracies)
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy')
plt.show()

# Affichage du diagramme de régression non linéaire
predictions = mlp.forward(X)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], y, color='blue', label='Vraies valeurs')  # Points réels
plt.scatter(X[:, 0], predictions, color='red', label='Prédictions')  # Prédictions du modèle
plt.xlabel('Feature')
plt.ylabel('Output')
plt.title('Régression Non Linéaire')
plt.legend()
plt.show()

# Affichage limité des prédictions et des statistiques
print("Predictions (First 10):", predictions[:10])
print("Predictions (Last 10):", predictions[-10:])
print(f"Predictions - Mean: {np.mean(predictions):.4f}, Min: {np.min(predictions):.4f}, Max: {np.max(predictions):.4f}")
