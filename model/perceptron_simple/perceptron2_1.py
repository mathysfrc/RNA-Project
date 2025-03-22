import numpy as np


# Fonction d'activation (fonction seuil)
def activation(x):
    return 1 if x >= 0 else 0


# Initialisation des poids à 0
def init_weights(n):
    return np.zeros(n)


# Algorithme d'apprentissage du perceptron
def perceptron(X, d, eta=1, max_epochs=100):
    n_samples, n_features = X.shape
    weights = init_weights(n_features + 1)  # +1 pour le biais
    nb_erreurs = -1
    epoch = 0

    while nb_erreurs != 0 and epoch < max_epochs:
        nb_erreurs = 0
        for i in range(n_samples):
            x_i = np.insert(X[i], 0, 1)  # Ajout du biais
            y_pred = activation(np.dot(weights, x_i))
            erreur = d[i] - y_pred
            if erreur != 0:
                nb_erreurs += 1
                weights += eta * erreur * x_i  # Mise à jour des poids
        epoch += 1
        print(f"Époque {epoch}, erreurs: {nb_erreurs}")

    return weights


# Jeu de données pour la porte logique ET
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
d = np.array([0, 0, 0, 1])  # Sorties attendues

# Entraînement du perceptron
weights = perceptron(X, d)

print("Poids finaux:", weights)

# Test du perceptron sur la porte ET
for i, x in enumerate(X, start=1):
    x_i = np.insert(x, 0, 1)  # Ajout du biais
    y = activation(np.dot(weights, x_i))
    print(f"k={i}, Entrée: {x}, Sortie prédite: {y}")
