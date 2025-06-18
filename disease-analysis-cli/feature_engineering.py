# feature_engineering.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def run_feature_engineering(df: pd.DataFrame, output_path="data/processed_with_features.csv") -> None:
    """
    Perform feature engineering and generate correlation heatmap.

    Args:
        df (pd.DataFrame): Cleaned DataFrame from preprocessing
        output_path (str): Where to save the enriched dataset
    """

    print(f"\nStarting Feature Engineering → initial shape: {df.shape}")

    # Step 1: Create symptom_count
    symptom_cols = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']
    df['symptom_count'] = df[symptom_cols].sum(axis=1)

    # Step 2: is_elderly (Age > 60)
    df['is_elderly'] = (df['Age'] > 60).astype(int)

    # Step 3: risk_score = BP + Cholesterol + is_elderly
    df['risk_score'] = df['Blood Pressure'] + df['Cholesterol Level'] + df['is_elderly']

    # Step 4: age_group (binned age)
    df['age_group'] = pd.cut(df['Age'], bins=[0, 18, 35, 50, 65, 100], labels=False)

    # Step 5: Interaction features
    df['fever_fatigue'] = df['Fever'] & df['Fatigue']
    df['cough_difficulty'] = df['Cough'] & df['Difficulty Breathing']
    df['bp_high_and_old'] = ((df['Blood Pressure'] == 2) & (df['Age'] > 60)).astype(int)
    df['multi_symptom_flag'] = (df['symptom_count'] >= 3).astype(int)
    df['cholesterol_high_and_age'] = ((df['Cholesterol Level'] == 2) & (df['Age'] > 55)).astype(int)

    print("Features added: symptom_count, is_elderly, risk_score, age_group, and interactions")

    # Step 6: Correlation with Outcome Variable
    print("\nCorrelation with Outcome Variable:")
    if 'Outcome Variable' in df.columns:
        corr_with_outcome = df.corr(numeric_only=True)['Outcome Variable'].drop('Outcome Variable')
        corr_with_outcome = corr_with_outcome.sort_values(ascending=False)

        if not corr_with_outcome.empty:
            print(corr_with_outcome)

            # Step 7: Plot and save heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr_with_outcome.to_frame(), annot=True, cmap="coolwarm", cbar=False)
            plt.title("Correlation of Features with Outcome Variable")
            plt.tight_layout()

            # Save heatmap
            heatmap_path = Path("output/final_feature_correlation_heatmap.png")
            heatmap_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(heatmap_path)
            plt.show()
            plt.close()

            print(f"Heatmap saved to: {heatmap_path}")
        else:
            print("No correlations found to plot.")
    else:
        print("'Outcome Variable' not found in DataFrame!")

    # Step 8: Save final DataFrame
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Enriched dataset saved to: {output_path}")
    print(f"Final shape: {df.shape}")
    return df  # RETURN the enriched DataFrame

