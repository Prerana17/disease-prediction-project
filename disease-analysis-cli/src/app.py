import streamlit as st
import pandas as pd
from eda import download_and_load_dataset, preprocess_dataset, perform_analysis
from ml_models import run_modeling_pipeline
import os
from PIL import Image
import warnings

st.set_page_config(page_title="Disease Predictor", layout="wide")

st.title("🩺 Disease Predictor Dashboard")
st.markdown("A Streamlit interface for exploring and predicting health outcomes.")

#  Sidebar 
st.sidebar.header("Options")
task = st.sidebar.radio("Choose Task", ["View Data", "Run Exploratory Data Analysis", "Train Models and check metrics comparison"])

#  Load Dataset
@st.cache_data
def load_data():
    df = download_and_load_dataset()
    return preprocess_dataset(df)

# Main Panel
df = load_data()

# Show raw data
if task == "View Data":
    st.subheader("Raw Processed Dataset")
    st.dataframe(df)

# Show EDA section
if task == "Run Exploratory Data Analysis":
    st.subheader("Exploratory Data Analysis")

    with st.spinner("Running EDA..."):
        # Temporarily suppress warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            perform_analysis()
        st.success("EDA completed! Showing saved plots...")

    RESULTS_DIR = "results"
    plot_files = ["correlation_heatmap.png", "age_violin_distribution.png"]  # files names based on our results

    for file in plot_files:
        full_path = os.path.join(RESULTS_DIR, file)
        if os.path.exists(full_path):
            st.image(Image.open(full_path), caption=file, use_container_width=True)
        else:
            st.warning(f"{file} not found in {RESULTS_DIR}")


# Show ML training and results
if task == "Train Models and check metrics comparison":
    st.subheader(" Model Training and Evaluation")
    if st.button("Start Training"):
        with st.spinner("Training ML models..."):
            _, train_metrics, test_metrics = run_modeling_pipeline(df)
            st.success("✅ Model training complete!")

        st.markdown("### Training Metrics")
        st.dataframe(train_metrics)

        st.markdown("### Test Metrics")
        st.dataframe(test_metrics)



