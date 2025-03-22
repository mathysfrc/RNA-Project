import controllers.adaline_controller
import controllers.gradient_descent_controller
import controllers.simple_controller
import utils
from colorama import Fore, Back, Style, init
import os
from dotenv import load_dotenv

init(autoreset=True)

load_dotenv()
filename = os.getenv("TABLE_2_9")

def display_welcome():
    print(Fore.CYAN + Style.BRIGHT + "Bienvenue dans le projet RNA !")
    print(Fore.GREEN + f"Fichier utilisé : {filename}")
    print(Fore.YELLOW + "-" * 50)
    print()


def launch_perceptron(perceptron_id):
    perceptrons = {
        1: controllers.simple_controller.start,
        2: controllers.gradient_descent_controller.start,
        3: controllers.adaline_controller.start,
        4: controllers.simple_controller.start,
        5: controllers.simple_controller.start}

    perceptrons[perceptron_id](filename)


def main():
    display_welcome()
    choice = utils.ask_perceptron()
    launch_perceptron(choice)


if __name__ == "__main__":
    main()
