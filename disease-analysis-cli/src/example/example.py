"""
Demonstration of how to use the disease_predictor_cli package programmatically.
"""

from eda import download_and_load_dataset, preprocess_dataset, perform_analysis
from ml_models import run_modeling_pipeline

def main():
    print("🔍 Downloading and preprocessing dataset...")
    df = download_and_load_dataset()
    df = preprocess_dataset(df)

    print("📊 Performing Exploratory Data Analysis...")
    perform_analysis()

    print("🧠 Training models and evaluating performance...")
    _, train_metrics, test_metrics = run_modeling_pipeline(df)

    print("\n✅ Training Metrics:")
    print(train_metrics)

    print("\n✅ Test Metrics:")
    print(test_metrics)

if __name__ == "__main__":
    main()
