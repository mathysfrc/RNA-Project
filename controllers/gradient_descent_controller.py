import model.perceptron_descente_gradient.perceptron_csv as gradient

def start(filename):
    X, y = gradient.load_data(filename)

    # Apprentissage
    model = gradient.Perceptron()
    errors = model.fit(X, y)

    # Évaluation
    accuracy = sum(model.predict(xi) == yi for xi, yi in zip(X, y)) / len(X)
    print(f"\nPoids finaux: {model.w}")
    print(f"Précision: {accuracy * 100:.2f}%")

    # Visualisations
    gradient.plot_decision_boundary(model, X, y)
    gradient.plot_learning_curve(errors)
