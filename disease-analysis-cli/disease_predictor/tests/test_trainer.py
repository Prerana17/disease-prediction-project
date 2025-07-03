from disease_predictor.model_registry import get_models
from disease_predictor.trainer import split_and_scale, train_best_model
from disease_predictor.data_loader import load_and_preprocess_data
import yaml
import pytest

@pytest.mark.integration
def test_train_model():
    X, y = load_and_preprocess_data("data/disease_dataset.csv")
    X_train, _, y_train, _ = split_and_scale(X, y, test_size=0.25, seed=42)
    models = get_models()
    model = train_best_model(X_train, y_train, models["Logistic Regression"])
    assert hasattr(model, "predict")
