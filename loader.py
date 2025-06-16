import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

# Step 1: Set up Kaggle credentials (for demo purposes only — don't commit to GitHub!)
os.environ['KAGGLE_USERNAME'] = 'biswajyotiroy'
os.environ['KAGGLE_KEY'] = 'd4ae563409803899f19a636c2d12020a'

# Step 2: Define cross-platform base directory to download the dataset
base_dir = os.path.join(os.path.expanduser("~"), "kaggle_data", "disease_symptom_data")
os.makedirs(base_dir, exist_ok=True)

def download_data_from_kaggle(dataset_slug: str = "uom190346a/disease-symptoms-and-patient-profile-dataset",
                               download_dir: str = base_dir) -> str:
    """
    Download and extract the dataset from Kaggle using the Kaggle API.

    Args:
        dataset_slug (str): Kaggle dataset slug.
        download_dir (str): Directory to store the downloaded file.

    Returns:
        str: Path to the extracted CSV file.
    """
    api = KaggleApi()
    api.authenticate()

    print(f"Downloading from Kaggle: {dataset_slug}")
    api.dataset_download_files(dataset_slug, path=download_dir, unzip=True)
    print("Download completed and extracted.")

    for file in os.listdir(download_dir):
        if file.endswith(".csv"):
            csv_path = os.path.join(download_dir, file)
            print(f"CSV file found: {csv_path}")
            return csv_path

    raise FileNotFoundError("No CSV file found after download.")


def load_data_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file into a pandas DataFrame.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_csv(csv_path)
    print(f"✅ Data loaded successfully with shape: {df.shape}\n")

    print("First 5 rows of the dataset:")
    print(df.head())

    # Optional: Display entire dataset (for small datasets)
    #print("\nFull dataset (if not too large):")
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #    print(df)

    return df
