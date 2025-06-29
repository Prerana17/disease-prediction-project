# Makefile for Disease Predictor Project
# --------------------------------------
# This file automates environment setup, testing, formatting, CLI interaction, and more.

# === VARIABLES ===
PYTHON = python
PIP = pip
VENV = venv
SRC_DIR = disease_predictor/src
CLI = $(SRC_DIR)/cli.py
TEST_FILE = $(SRC_DIR)/test_pipeline_models.py
REQUIREMENTS = requirements.txt

# === COMMANDS ===

# Set up virtual environment and install dependencies
setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/Scripts/activate && $(PIP) install --upgrade pip && $(PIP) install -r $(REQUIREMENTS)

# Run unit tests with verbose output
test:
	$(VENV)/Scripts/activate && pytest $(TEST_FILE) -v

# Format code using Black
format:
	$(VENV)/Scripts/activate && black $(SRC_DIR)

# Run ML pipeline directly (skips CLI)
run-model:
	$(VENV)/Scripts/activate && $(PYTHON) $(SRC_DIR)/ml_models.py

# Run EDA pipeline directly
run-eda:
	$(VENV)/Scripts/activate && $(PYTHON) $(SRC_DIR)/eda.py

# Run pipeline via CLI interface
cli:
	$(VENV)/Scripts/activate && $(PYTHON) -m cli $(CMD)

# Clean Python cache and compiled files
clean:
	del /S /Q *.pyc
	del /S /Q *.pyo
	del /S /Q *__pycache__*

# Freeze current env to requirements.txt
freeze:
	$(VENV)/Scripts/activate && pip freeze > $(REQUIREMENTS)

# === Example CLI usage ===
# make cli CMD=eda       → runs EDA pipeline
# make cli CMD=train     → runs model training pipeline
