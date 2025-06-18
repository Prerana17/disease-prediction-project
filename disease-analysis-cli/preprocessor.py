import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the disease symptom dataset.

    Args:
        df (pd.DataFrame): Raw dataset.

    Returns:
        pd.DataFrame: Cleaned and preprocessed dataset.
    """

    print("Starting preprocessing...")

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Convert binary symptom columns ('Yes'/'No') to 1/0
    binary_cols = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    # Encode categorical features
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    if 'Blood Pressure' in df.columns:
        df['Blood Pressure'] = df['Blood Pressure'].map({'Low': 0, 'Normal': 1, 'High': 2})
    if 'Cholesterol Level' in df.columns:
        df['Cholesterol Level'] = df['Cholesterol Level'].map({'Low': 0, 'Normal': 1, 'High': 2})
    
    # Encode outcome variable
    if 'Outcome Variable' in df.columns:
        df['Outcome Variable'] = df['Outcome Variable'].map({'Negative': 0, 'Positive': 1})
    
    # Drop duplicates
    original_len = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {original_len - len(df)} duplicate rows.")

    # Drop missing values
    missing_count = df.isnull().sum().sum()
    df = df.dropna()
    print(f"Dropped {missing_count} missing values.")

    # Final dataset shape
    print(f"Preprocessing complete. Final shape: {df.shape}")

    return df
