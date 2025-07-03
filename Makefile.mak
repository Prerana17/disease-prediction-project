# Makefile for Disease Predictor Project

# This file automates environment setup, testing, formatting, CLI interaction, and more.

# --- Variables ---
PYTHON := python
PIP := pip
VENV := venv
SRC_DIR := disease-analysis-cli/src
RESULTS_DIR := disease-analysis-cli/results
TEST_DIR := disease-analysis-cli/test
APP_FILE := $(SRC_DIR)/app.py
EDA_FILE := $(SRC_DIR)/eda.py
ML_FILE := $(SRC_DIR)/ml_models.py
REQUIREMENTS := requirements.txt

# Platform-specific VENV Scripts path
ACTIVATE := $(VENV)/Scripts/activate

# --- Default target ---
all: help

# --- Help Menu ---
help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  setup          Create virtual environment and install dependencies"
	@echo "  format         Format code with black and isort"
	@echo "  lint           Lint code using flake8 (optional)"
	@echo "  test           Run pytest unit tests"
	@echo "  eda            Run EDA analysis and save plots"
	@echo "  model          Run machine learning training and evaluation"
	@echo "  streamlit      Launch Streamlit dashboard"
	@echo "  clean          Remove __pycache__, *.pyc, *.pyo and results"
	@echo ""

# --- Create virtualenv and install dependencies ---
setup:
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE) && $(PIP) install --upgrade pip && $(PIP) install -r $(REQUIREMENTS)

# --- Format Code ---
format:
	$(ACTIVATE) && black $(SRC_DIR) && isort $(SRC_DIR)

# --- Lint Code  ---
lint:
	$(ACTIVATE) && flake8 $(SRC_DIR)

# --- Run Pytest ---
test:
	$(ACTIVATE) && pytest $(TEST_DIR) -v

# --- Run EDA analysis ---
eda:
	$(ACTIVATE) && $(PYTHON) $(EDA_FILE)

# --- Run ML Model Training ---
model:
	$(ACTIVATE) && $(PYTHON) $(ML_FILE)

# --- Launch Streamlit Dashboard ---
streamlit:
	$(ACTIVATE) && streamlit run $(APP_FILE)

# --- Clean build artifacts ---
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf $(RESULTS_DIR) || true

.PHONY: all help setup format lint test eda model streamlit clean

