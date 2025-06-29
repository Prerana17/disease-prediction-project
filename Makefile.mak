# Makefile for Disease Predictor Project

# === VARIABLES ===
PYTHON = python
PIP = pip
VENV = venv
SRC_DIR = disease_predictor/src
TEST_FILE = $(SRC_DIR)/test_pipeline_models.py

# === COMMANDS ===

# Set up virtual environment and install dependencies
setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/Scripts/activate && $(PIP) install --upgrade pip && $(PIP) install -r requirements.txt

# Run unit tests with verbose output
test:
	$(VENV)/Scripts/activate && pytest $(TEST_FILE) -v

# Format code with Black
format:
	$(VENV)/Scripts/activate && black $(SRC_DIR)

# Run the ML model pipeline
run-model:
	$(VENV)/Scripts/activate && $(PYTHON) $(SRC_DIR)/ml_model.py

# Run the full data analysis pipeline (data + visualizations)
run-pipeline:
	$(VENV)/Scripts/activate && $(PYTHON) $(SRC_DIR)/data_pipeline.py

# Clean .pyc and cache files
clean:
	del /S *.pyc
	del /S *.pyo
	del /S *__pycache__*

# Freeze requirements to file
freeze:
	$(VENV)/Scripts/activate && pip freeze > requirements.txt
