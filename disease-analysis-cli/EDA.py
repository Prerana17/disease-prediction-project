import pandas as pd

def perform_eda(df: pd.DataFrame) -> None:
    """
    Perform detailed exploratory data analysis (EDA) leveraging the preprocessing.

    Args:
        df (pd.DataFrame): The cleaned and preprocessed dataset.
    """

    print("BASIC DATA STRUCTURE OVERVIEW")
    print("-" * 50)
    print("DataFrame shape (rows, columns):", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())

    print("\n\nCOLUMN-WISE INFORMATION")
    print("-" * 50)
    print("Data types and null counts:")
    print(df.info())

    print("\n\nDESCRIPTIVE STATISTICS")
    print("-" * 50)
    print("Numerical Columns:")
    print(df.describe())

    print("\nCategorical Columns:")
    print(df.describe(include=['object']))

    print("\n\nMISSING VALUE CHECK")
    print("-" * 50)
    missing = df.isnull().sum()
    print("Missing values per column:")
    print(missing[missing > 0] if missing.any() else "✅ No missing values detected.")

    print("\n\nUNIQUE VALUE COUNTS")
    print("-" * 50)
    for col in df.columns:
        print(f"{col}: {df[col].nunique()} unique value(s)")

    print("\n\nOUTLIER ANALYSIS (using ±3 std)")
    print("-" * 50)
    for col in df.select_dtypes(include='number'):
        mean = df[col].mean()
        std = df[col].std()
        outliers = df[(df[col] > mean + 3 * std) | (df[col] < mean - 3 * std)]
        if not outliers.empty:
            print(f"{col}: {len(outliers)} outlier(s)")

    print("\n\nSYMPTOM ANALYSIS BY OUTCOME")
    print("-" * 50)
    symptom_cols = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']
    available_symptoms = [col for col in symptom_cols if col in df.columns]
    if available_symptoms and 'Outcome Variable' in df.columns:
        print(df.groupby('Outcome Variable')[available_symptoms].mean())

    print("\n\nDISTRIBUTION OF KEY CATEGORICAL VARIABLES")
    print("-" * 50)
    if 'Disease' in df.columns:
        print("\nTop 10 most frequent diseases:")
        print(df['Disease'].value_counts().head(10))

    if 'Gender' in df.columns:
        print("\nGender distribution:")
        print(df['Gender'].value_counts().rename({0: 'Female', 1: 'Male'}))

    if 'Outcome Variable' in df.columns:
        print("\nHealth outcome distribution:")
        print(df['Outcome Variable'].value_counts().rename({0: 'Negative', 1: 'Positive'}))

    print("\n\nCORRELATION WITH OUTCOME")
    print("-" * 50)
    if 'Outcome Variable' in df.columns:
        correlation = df.corr(numeric_only=True)['Outcome Variable'].sort_values(ascending=False)
        print(correlation)

    print("\n\nGROUPED INSIGHTS")
    print("-" * 50)
    if 'Blood Pressure' in df.columns and 'Outcome Variable' in df.columns:
        print("\nAverage Outcome Rate by Blood Pressure:")
        print(df.groupby('Blood Pressure')['Outcome Variable'].mean())

    if 'Cholesterol Level' in df.columns and 'Outcome Variable' in df.columns:
        print("\nAverage Outcome Rate by Cholesterol Level:")
        print(df.groupby('Cholesterol Level')['Outcome Variable'].mean())

    print("\n✅ EDA complete.")
