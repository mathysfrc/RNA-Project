import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Fonction d'activation sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Dérivée de la fonction d'activation sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialisation des poids et des biais
        self.W1 = np.random.randn(input_size, hidden_size)  # Poids de la couche d'entrée à cachée
        self.b1 = np.zeros((1, hidden_size))  # Biais de la couche cachée
        self.W2 = np.random.randn(hidden_size, output_size)  # Poids de la couche cachée à sortie
        self.b2 = np.zeros((1, output_size))  # Biais de la couche de sortie

    # Propagation avant
    def forward(self, X):
        self.hidden_input = np.dot(X, self.W1) + self.b1  # Entrée cachée
        self.hidden_output = sigmoid(self.hidden_input)  # Sortie cachée
        self.final_input = np.dot(self.hidden_output, self.W2) + self.b2  # Entrée de la couche de sortie
        self.final_output = sigmoid(self.final_input)  # Sortie finale
        return self.final_output

    # Rétropropagation (Backpropagation)
    def backward(self, X, Y, learning_rate=0.01):
        # Calcul de l'erreur
        output_error = Y - self.final_output
        d_output = output_error * sigmoid_derivative(self.final_output)  # Dérivée de la sortie

        # Propagation de l'erreur dans la couche cachée
        hidden_error = d_output.dot(self.W2.T)
        d_hidden = hidden_error * sigmoid_derivative(self.hidden_output)  # Dérivée de la couche cachée

        # Mise à jour des poids et des biais
        self.W2 += self.hidden_output.T.dot(d_output) * learning_rate
        self.b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        self.W1 += X.T.dot(d_hidden) * learning_rate
        self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    # Entraînement
    def train(self, X, Y, epochs=1000, learning_rate=0.01):
        losses = []
        for epoch in range(epochs):
            self.forward(X)  # Propagation avant
            self.backward(X, Y, learning_rate)  # Rétropropagation
            if epoch % 100 == 0:
                loss = np.mean(np.square(Y - self.final_output))  # Erreur quadratique moyenne (MSE)
                losses.append(loss)
                print(f"Epoch {epoch} - Loss: {loss:.4f}")
        return losses

# Charger les données CSV
data = pd.read_csv("../../datas/table_4_14.csv", header=None)

# Séparer les caractéristiques (X) et les labels (Y)
X = data.iloc[:, :-3].values  # caractéristiques (2 colonnes)
Y = data.iloc[:, -3:].values  # labels (3 classes)

# Définir les dimensions de l'entrée, cachée et sortie
input_size = 2  # 2 caractéristiques d'entrée
hidden_size = 5  # Taille de la couche cachée (exemple)
output_size = 3  # 3 classes de sortie

# Créer le modèle MLP
mlp = MLP(input_size, hidden_size, output_size)

# Entraîner le modèle et récupérer les valeurs de perte
losses = mlp.train(X, Y, epochs=1000, learning_rate=0.01)

# Afficher la courbe de perte
plt.figure(figsize=(10, 6))
plt.plot(range(0, 1000, 100), losses, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Loss vs Epochs")
plt.legend()
plt.grid(True)
plt.show()

# Tester le modèle sur les points d'entraînement
predictions = mlp.forward(X)

# Convertir les prédictions en classes (la classe avec la probabilité la plus élevée)
predicted_classes = np.argmax(predictions, axis=1)

# Calculer l'accuracy
correct_predictions = np.sum(predicted_classes == np.argmax(Y, axis=1))
accuracy = correct_predictions / len(Y) * 100

# Afficher l'accuracy
print(f"Accuracy: {accuracy:.2f}%")

# Visualiser les points d'entraînement avec les couleurs correspondant aux classes
plt.figure(figsize=(10, 6))

# Créer une grille de points pour tracer les contours des zones de classification
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Prédire la classe pour chaque point de la grille
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_predictions = mlp.forward(grid_points)
grid_classes = np.argmax(grid_predictions, axis=1)
grid_classes = grid_classes.reshape(xx.shape)

# Afficher les zones de classification
plt.contourf(xx, yy, grid_classes, cmap=plt.cm.RdYlBu, alpha=0.3)

# Choisir des couleurs pour chaque classe et tracer les points d'entraînement
colors = ['red', 'green', 'blue']

# Tracer les points d'entraînement
for i in range(3):
    plt.scatter(X[predicted_classes == i, 0], X[predicted_classes == i, 1],
                color=colors[i], label=f'Classe {i + 1}', edgecolors='black')

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Points d'entraînement et zones de classification")
plt.legend()
plt.grid(True)
plt.show()
