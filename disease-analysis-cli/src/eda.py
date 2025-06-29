# -*- coding: utf-8 -*-

"""
SPDX-FileCopyrightText: Group-3_RSE_TeamProject
SPDX-License-Identifier: CC0: Public Domain

This script loads, preprocesses, analyzes, and visualizes the Disease Symptoms
and Patient Profile dataset using a structured data science pipeline.
"""

# Loading required libraries
import os
import shutil
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import kagglehub


# Download latest version of the dataset
path = kagglehub.dataset_download(
    "uom190346a/disease-symptoms-and-patient-profile-dataset"
)

print("Path to dataset files:", path)

# Define paths
DATA_DIR = Path("data")
RAW_DATA_FILE = DATA_DIR / "Disease_symptom_and_patient_profile_dataset.csv"
RESULT_DIR = Path("results")


# Data loading
def download_and_load_dataset():
    """Load dataset from KaggleHub and return as DataFrame."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_DATA_FILE.exists():
        print("\U0001f4e5 Downloading dataset using kagglehub...")
        kaggle_path = kagglehub.dataset_download(
            "uom190346a/disease-symptoms-and-patient-profile-dataset"
        )
        source_file = (
            Path(kaggle_path) / "Disease_symptom_and_patient_profile_dataset.csv"
        )
        if not source_file.exists():
            raise FileNotFoundError("❌ CSV not found in KaggleHub download path.")
        shutil.copy(source_file, RAW_DATA_FILE)
        print("✅ Dataset downloaded and copied to data directory.")
    else:
        print(f"\U0001f4c1 Using cached dataset: {RAW_DATA_FILE}")

    return pd.read_csv(RAW_DATA_FILE)


# Data Preprocessing
def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and encode the disease symptom dataset."""
    print("\U0001f9f9 Preprocessing dataset...")
    df.columns = df.columns.str.strip()

    binary_cols = ["Fever", "Cough", "Fatigue", "Difficulty Breathing"]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})
    df["Blood Pressure"] = df["Blood Pressure"].map({"Low": 0, "Normal": 1, "High": 2})
    df["Cholesterol Level"] = df["Cholesterol Level"].map(
        {"Low": 0, "Normal": 1, "High": 2}
    )
    df["Outcome Variable"] = df["Outcome Variable"].map({"Negative": 0, "Positive": 1})

    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    return df


# Analysis Functions
def analyze_distribution(df: pd.DataFrame):
    """Prints basic distribution stats."""
    print("\n\U0001f50d Dataset Overview")
    print(df.describe(include="all"))
    print("\nOutcome distribution:")
    print(df["Outcome Variable"].value_counts())


def plot_symptom_distribution(df: pd.DataFrame):
    """Generates and saves a heatmap of symptom correlation."""
    RESULT_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    heatmap_path = RESULT_DIR / "correlation_heatmap.png"
    plt.savefig(heatmap_path)
    plt.close()
    print(f"\U0001f4ca Correlation heatmap saved to: {heatmap_path}")


def plot_violin_distributions(df: pd.DataFrame):
    """
    Generate and save violin plots of Age vs. various categorical features.
    """
    RESULT_DIR.mkdir(exist_ok=True)
    cols = [
        "Fever",
        "Cough",
        "Fatigue",
        "Difficulty Breathing",
        "Gender",
        "Blood Pressure",
        "Cholesterol Level",
        "Outcome Variable",
    ]

    fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(18, 10))
    ax = ax.flatten()

    for i, col in enumerate(cols):
        sns.violinplot(data=df, x=col, y="Age", ax=ax[i], scale="count", palette="Set2")
        ax[i].set_title(col)
        ax[i].set_xlabel("Age")

    fig.suptitle("Age Distribution by Objective Value", fontsize=20)
    plt.tight_layout()
    violin_path = RESULT_DIR / "age_violin_distribution.png"
    plt.savefig(violin_path)
    plt.close()
    print(f"\U0001f4ca Violin plot saved to: {violin_path}")


def apply_custom_theme() -> None:
    """
    Apply a custom Seaborn and Matplotlib theme for consistent and professional plots.
    """
    sns.set_theme(style="whitegrid")
    mpl.rcParams.update(
        {
            "axes.edgecolor": "gray",
            "axes.linewidth": 1.2,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.figsize": (8, 5),
            "figure.dpi": 100,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


# Main execution
def perform_analysis():
    df = download_and_load_dataset()
    df = preprocess_dataset(df)
    analyze_distribution(df)
    plot_symptom_distribution(df)
    plot_violin_distributions(df)


if __name__ == "__main__":
    perform_analysis()
