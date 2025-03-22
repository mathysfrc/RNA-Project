import numpy as np
import matplotlib.pyplot as plt


# Fonction d'activation (fonction seuil)
def activation(x):
    return 1 if x >= 0 else 0


# Initialisation des poids avec une distribution gaussienne centrée en 0
def init_weights(n):
    return np.random.normal(0, 0.1, n)


# Algorithme d'apprentissage du perceptron
def perceptron(X, d, eta=0.1, max_epochs=100):
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


# Affichage graphique des résultats
def plot_decision_boundary(X, d, weights):
    plt.figure(figsize=(6, 6))

    for i in range(len(X)):
        if d[i] == 1:
            plt.scatter(X[i][0], X[i][1], marker='o', color='blue')
        else:
            plt.scatter(X[i][0], X[i][1], marker='x', color='red')

    x_values = np.linspace(-0.1, 1.1, 100)
    y_values = -(weights[1] * x_values + weights[0]) / weights[2]
    plt.plot(x_values, y_values, 'g')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Frontière de décision du Perceptron")
    plt.show()
