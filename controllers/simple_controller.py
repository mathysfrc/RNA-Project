import numpy as np
import model.perceptron_simple.perceptron as simple

def start(filename):
    # Jeu de données pour la porte logique ET
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    d = np.array([0, 0, 0, 1])  # Sorties attendues

    # Entraînement du perceptron
    weights = simple.perceptron(X, d)
    print("Poids finaux:", weights)

    # Test du perceptron sur la porte ET
    for x in X:
        x_i = np.insert(x, 0, 1)  # Ajout du biais
        y = simple.activation(np.dot(weights, x_i))
        print(f"Entrée: {x}, Sortie prédite: {y}")

    # Affichage graphique des résultats
    simple.plot_decision_boundary(X, d, weights)
