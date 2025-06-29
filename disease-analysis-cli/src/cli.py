"""
SPDX-FileCopyrightText: Group-3_RSE_TeamProject
SPDX-License-Identifier: CC0: Public Domain

This script provides a command-line interface (CLI) for performing:
1. Exploratory Data Analysis (EDA)
2. Machine Learning Model Training and Evaluation

Run it as:
$ python -m src.cli eda
$ python -m src.cli train
"""

import click
from eda import download_and_load_dataset, preprocess_dataset, perform_analysis
from ml_models import run_modeling_pipeline


@click.group()
def cli():
    """📊 Disease Analysis CLI Tool

    Use this tool to explore the dataset and train ML models.
    """
    pass


@cli.command("eda")
def run_eda():
    """
    Run Exploratory Data Analysis (EDA)

    This will generate plots, statistics, and save visualizations
    in the 'results/' folder.
    """
    click.echo("🔍 Starting EDA...")
    perform_analysis()
    click.echo("✅ EDA complete. Results saved to 'results/' folder.")


@cli.command("train")
def train_model():
    """
    Train ML Models and Evaluate

    This will run preprocessing, model training, and metrics evaluation.
    """
    click.echo("📥 Loading and preprocessing data...")
    df = preprocess_dataset(download_and_load_dataset())

    click.echo("📈 Starting model training and evaluation...")
    run_modeling_pipeline(df)

    click.echo("✅ Model training and evaluation complete.")


if __name__ == "__main__":
    cli()
