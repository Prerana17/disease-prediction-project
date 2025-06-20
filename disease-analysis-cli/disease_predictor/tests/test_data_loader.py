import pandas as pd
import pytest
from disease_predictor.data_loader import load_and_preprocess_data
import os


@pytest.mark.unit
def test_load_and_preprocess_data_kaggle_or_sample():
    # First try to use real Kaggle dataset (if Kaggle CLI is set up)
    dataset_path = "tests/Disease_symptom_and_patient_profile_dataset.csv"
    
    try:
        if not os.path.exists(dataset_path):
            print("Downloading dataset from Kaggle...")
            os.system("kaggle datasets download -d uom190346a/disease-symptoms-and-patient-profile-dataset -p tests --unzip")

        if os.path.exists(dataset_path):
            X, y = load_and_preprocess_data(dataset_path)
            assert X.shape[0] == y.shape[0]
            assert 'Outcome Variable' not in X.columns
            print("Kaggle dataset loaded and processed successfully.")
            return  # Exit test early if Kaggle version worked

    except Exception as e:
        print(f"Kaggle download failed: {e}")

    # Fallback: use manual sample dataset
    sample_data = {
        'Disease': ['Asthma'],
        'Fever': ['Yes'],
        'Cough': ['No'],
        'Fatigue': ['Yes'],
        'Difficulty Breathing': ['No'],
        'Age': [25],
        'Gender': ['Female'],
        'Blood Pressure': ['Normal'],
        'Cholesterol Level': ['High'],
        'Outcome Variable': ['Positive']
    }
    fallback_path = "tests/temp_data.csv"
    pd.DataFrame(sample_data).to_csv(fallback_path, index=False)

    X, y = load_and_preprocess_data(fallback_path)
    assert X.shape[0] == y.shape[0]
    assert 'Outcome Variable' not in X.columns
    print("Sample fallback dataset processed successfully.")

