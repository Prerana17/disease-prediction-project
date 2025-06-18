import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

def apply_custom_theme() -> None:
    """
    Apply a custom Seaborn and Matplotlib theme for consistent and professional plots.
    """
    sns.set_theme(style="whitegrid")
    mpl.rcParams.update({
        'axes.edgecolor': 'gray',
        'axes.linewidth': 1.2,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.figsize': (8, 5),
        'figure.dpi': 100,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False
    })

def plot_full_feature_insights(df: pd.DataFrame) -> None:
    """
    Generate detailed visual insights for selected features in the DataFrame.
    
    Includes:
    - KDE plots for distribution by outcome
    - Boxplots for numerical comparison
    - Countplots for binary indicators
    - Stacked bar plots for proportions
    - Histograms
    - Pairplots for multivariate relationships
    - Rare category analysis

    Args:
        df (pd.DataFrame): Feature-engineered DataFrame with 'Outcome Variable' and engineered features
    """
    apply_custom_theme()

    def show(title: str = "") -> None:
        """
        Utility function to display and clear plots.
        """
        plt.title(title)
        plt.tight_layout()
        plt.show()
        plt.clf()

    # KDE Plots for distributional comparison
    if 'Age' in df.columns:
        sns.kdeplot(data=df, x="Age", hue="Outcome Variable", fill=True)
        show("KDE Plot: Age Distribution by Outcome")

    if 'risk_score' in df.columns:
        sns.kdeplot(data=df, x="risk_score", hue="Outcome Variable", fill=True)
        show("KDE Plot: Risk Score by Outcome")

    # Boxplots: Visualize numerical distribution by Outcome
    if 'symptom_count' in df.columns:
        sns.boxplot(x='Outcome Variable', y='symptom_count', data=df, palette="Set2")
        show("Boxplot: Symptom Count by Outcome")

    if 'Age' in df.columns:
        sns.boxplot(x='Outcome Variable', y='Age', data=df, palette="Set2")
        show("Boxplot: Age by Outcome")

    if 'risk_score' in df.columns:
        sns.boxplot(x='Outcome Variable', y='risk_score', data=df, palette="Set2")
        show("Boxplot: Risk Score by Outcome")

    # Countplots for binary features
    binary_cols: list[str] = [
        'Fever', 'Fatigue', 'Cough', 'Difficulty Breathing',
        'multi_symptom_flag', 'fever_fatigue',
        'cholesterol_high_and_age', 'is_elderly'
    ]
    for col in binary_cols:
        if col in df.columns:
            sns.countplot(x=col, hue='Outcome Variable', data=df, palette="coolwarm")
            show(f"Countplot: {col} vs Outcome")

    # Stacked Bar Plot: Proportion of outcomes per Blood Pressure level
    if 'Blood Pressure' in df.columns:
        ctab = pd.crosstab(df['Blood Pressure'], df['Outcome Variable'], normalize='index')
        ctab.plot(kind='bar', stacked=True, colormap='viridis')
        plt.ylabel("Proportion")
        plt.xlabel("Blood Pressure")
        plt.title("Stacked Bar: Outcome Rate by Blood Pressure")
        plt.tight_layout()
        plt.show()
        plt.clf()

    # Histogram with KDE overlay
    if 'Age' in df.columns:
        sns.histplot(
            df['Age'],
            bins=20,
            kde=True,
            color="skyblue",
            edgecolor="black",
            linewidth=1.2,
            alpha=0.8
        )
        show("Histogram: Age Distribution")

    # Pairplot for selected numeric features
    pair_cols: list[str] = [col for col in ['Age', 'risk_score', 'symptom_count'] if col in df.columns]
    if len(pair_cols) >= 2:
        sns.pairplot(df, vars=pair_cols, hue='Outcome Variable', palette="husl")
        plt.suptitle("Pairplot: Age, Risk Score, Symptom Count", y=1.02)
        plt.tight_layout()
        plt.show()

    # Print rare category insights
    print("\n🔍 Rare Category Checks:")
    for col in ['Blood Pressure', 'Cholesterol Level']:
        if col in df.columns:
            print(f"\n📊 {col} value distribution:")
            print(df[col].value_counts(normalize=True).round(3))
