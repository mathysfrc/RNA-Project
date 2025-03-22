import model.perceptron_adaline.perceptron_csv as adaline

def start(filename):
    filename = "datas/table_2_9.csv"

    # Chargement des données depuis un fichier CSV
    X, y = adaline.load_data(filename)

    # Apprentissage
    model = adaline.Perceptron(eta=0.05, max_iter=10000, tol=0.01)
    errors = model.fit(X, y)

    # Évaluation
    accuracy = sum(model.predict(xi) == yi for xi, yi in zip(X, y)) / len(X)
    print(f"\nPoids finaux: {model.w}")
    print(f"Précision: {accuracy * 100:.2f}%")

    # Visualisations
    adaline.plot_decision_boundary(model, X, y)
    adaline.plot_learning_curve(errors)
