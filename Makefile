OS := $(shell uname)
VENV_DIR := venv
PYTHON := python3

ifeq ($(OS), Windows_NT)
    VENV_ACTIVATE := $(VENV_DIR)\Scripts\activate
else
    VENV_ACTIVATE := source $(VENV_DIR)/bin/activate
endif

install:
	@echo "Creating virtual environment..."
	@$(PYTHON) -m venv $(VENV_DIR)
	@echo "Installing dependencies..."
	@$(VENV_ACTIVATE) && pip install -r requirements.txt
	@echo "Installation complete."

run:
	@echo "Activating virtual environment and running main.py..."
	@$(VENV_ACTIVATE) && $(PYTHON) main.py
