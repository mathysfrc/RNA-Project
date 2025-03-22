def ask_perceptron():
    print("Veuillez choisir un perceptron en entrant le numéro correspondant.\n")
    options = {
        1: "1 = Perceptron simple",
        2: "2 = Perceptron à descente de gradient",
        3: "3 = Perceptron Adaline",
        4: "4 = Réseau de neurones monocouches",
        5: "5 = Réseau de neurones multicouches"
    }
    for key, value in options.items():
        print(value)

    while True:
        try:
            choice = int(input("Entrez un nombre entre 1 et 5 : "))
            if 1 <= choice <= 5:
                return choice
            else:
                print("Veuillez entrer un nombre valide entre 1 et 5.")
        except ValueError:
            print("Entrée invalide. Veuillez entrer un nombre entier.")
